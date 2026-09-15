//! The develop render graph: five passes from a photo to an 8-bit sRGB
//! picture, all on one device.
//!
//! 1. The source upload: the photo's RGBA bytes in a plain 8-bit format.
//!    The sRGB curve is decoded in the shader rather than by the sampler:
//!    the sampler's decode is implementation-defined and measured 1 to 2
//!    codes off on dark channels on this machine's driver, and every M1
//!    source shares the sRGB curve anyway; only the matrix differs.
//! 2. `input_transform.wgsl`: samples the source at the render size and
//!    writes linear Rec.2020 into the working texture.
//! 3. `blur.wgsl` twice: the luminance of the working texture under a
//!    gaussian, the base layer that highlights and shadows split on.
//! 4. `develop.wgsl`: the six Basic operators, from a uniform that mirrors
//!    `PhotoEdit`, into the developed texture.
//! 5. `output.wgsl`: the crop out of the developed texture, Rec.2020 to
//!    sRGB, clipped, into an 8-bit sRGB texture that egui draws and
//!    `Readback` reads.
//!
//! Passes 2 and 3 depend only on the source and the render size, so they
//! run again only when one of those changes. A slider change runs 4 and 5.

use bytemuck::{Pod, Zeroable};
use slate_color::SourceSpace;
use slate_color::basic;
use slate_color::matrices;
use slate_core::{CropRect, ExportPreset, PhotoEdit};
use slate_media::Photo;

use crate::{FULLSCREEN_VERTICES, Readback, fullscreen_primitive};

/// The working-space format: linear Rec.2020 in half floats.
pub const WORKING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The base layer format: one half float of luminance.
pub const BASE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// The output format: 8-bit sRGB, encoded by the hardware on write.
pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The most samples per axis the input transform averages per output pixel.
pub const MAX_TAPS: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InputUniform {
    matrix: [[f32; 4]; 3],
    render_size: [f32; 2],
    decode_srgb: u32,
    taps: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurUniform {
    direction: [i32; 2],
    radius: i32,
    luma: u32,
    sigma: f32,
    _pad: [f32; 3],
}

/// Mirrors `PhotoEdit` field for field after the white balance matrix.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DevelopUniform {
    white_balance: [[f32; 4]; 3],
    white_balance_temperature: f32,
    white_balance_tint: f32,
    exposure: f32,
    contrast: f32,
    highlights: f32,
    shadows: f32,
    whites: f32,
    blacks: f32,
    vibrance: f32,
    saturation: f32,
    _pad: [f32; 2],
}

impl DevelopUniform {
    fn new(edit: &PhotoEdit) -> Self {
        let wb =
            basic::white_balance_matrix(edit.white_balance_temperature, edit.white_balance_tint);
        Self {
            white_balance: wb.to_wgsl_columns(),
            white_balance_temperature: edit.white_balance_temperature,
            white_balance_tint: edit.white_balance_tint,
            exposure: edit.exposure,
            contrast: edit.contrast,
            highlights: edit.highlights,
            shadows: edit.shadows,
            whites: edit.whites,
            blacks: edit.blacks,
            vibrance: edit.vibrance,
            saturation: edit.saturation,
            _pad: [0.0; 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OutputUniform {
    matrix: [[f32; 4]; 3],
    crop: [f32; 4],
}

/// A pipeline and the layout of its one bind group.
struct Pass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

/// A render target and its view.
struct Target {
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

struct Source {
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    space: SourceSpace,
    generation: u64,
}

/// Everything that depends on the source and the render size.
struct Frame {
    width: u32,
    height: u32,
    generation: u64,
    working: Target,
    ping: Target,
    base: Target,
    developed: Target,
    input_bind: wgpu::BindGroup,
    blur_h_bind: wgpu::BindGroup,
    blur_v_bind: wgpu::BindGroup,
    develop_bind: wgpu::BindGroup,
}

struct Output {
    target: Target,
    bind: wgpu::BindGroup,
    frame_generation: u64,
}

/// The develop graph on one device. Build it once, set a source, render as
/// often as the edit or the view changes.
pub struct Develop {
    device: wgpu::Device,
    queue: wgpu::Queue,
    input: Pass,
    blur: Pass,
    develop: Pass,
    output: Pass,
    input_uniform: wgpu::Buffer,
    blur_h_uniform: wgpu::Buffer,
    blur_v_uniform: wgpu::Buffer,
    develop_uniform: wgpu::Buffer,
    output_uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    readback: Readback,
    source: Option<Source>,
    frame: Option<Frame>,
    out: Option<Output>,
    generation: u64,
    frame_generation: u64,
    out_generation: u64,
}

impl Develop {
    /// Builds the passes on the shared device.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let input = make_pass(
            device,
            "input transform",
            include_str!("shaders/input_transform.wgsl"),
            WORKING_FORMAT,
            &[uniform_entry(0), texture_entry(1), sampler_entry(2)],
        );
        let blur = make_pass(
            device,
            "blur",
            include_str!("shaders/blur.wgsl"),
            BASE_FORMAT,
            &[uniform_entry(0), texture_entry(1)],
        );
        let develop = make_pass(
            device,
            "develop",
            include_str!("shaders/develop.wgsl"),
            WORKING_FORMAT,
            &[uniform_entry(0), texture_entry(1), texture_entry(2)],
        );
        let output = make_pass(
            device,
            "output",
            include_str!("shaders/output.wgsl"),
            OUTPUT_FORMAT,
            &[uniform_entry(0), texture_entry(1), sampler_entry(2)],
        );
        let uniform = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("develop sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            device: device.clone(),
            queue: queue.clone(),
            input,
            blur,
            develop,
            output,
            input_uniform: uniform("input uniform", size_of::<InputUniform>() as u64),
            blur_h_uniform: uniform("blur h uniform", size_of::<BlurUniform>() as u64),
            blur_v_uniform: uniform("blur v uniform", size_of::<BlurUniform>() as u64),
            develop_uniform: uniform("develop uniform", size_of::<DevelopUniform>() as u64),
            output_uniform: uniform("output uniform", size_of::<OutputUniform>() as u64),
            sampler,
            readback: Readback::new(device),
            source: None,
            frame: None,
            out: None,
            generation: 0,
            frame_generation: 0,
            out_generation: 0,
        }
    }

    /// Uploads a photo as the source. A photo wider or taller than the
    /// device allows is refused with a log line and the old source stays.
    pub fn set_source(&mut self, photo: &Photo) {
        let limit = self.device.limits().max_texture_dimension_2d;
        if photo.width > limit || photo.height > limit {
            log::error!(
                "photo of {}x{} exceeds the {limit} pixel texture limit",
                photo.width,
                photo.height
            );
            return;
        }
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let size = wgpu::Extent3d {
            width: photo.width,
            height: photo.height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("develop source"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            &photo.rgba8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(photo.width * 4),
                rows_per_image: Some(photo.height),
            },
            size,
        );
        self.generation += 1;
        self.source = Some(Source {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            width: photo.width,
            height: photo.height,
            space: photo.source,
            generation: self.generation,
        });
        self.frame = None;
        log::info!(
            "develop source set: {}x{} {:?} as {format:?}",
            photo.width,
            photo.height,
            photo.source
        );
    }

    /// The size of the source photo, when one is set.
    pub fn source_size(&self) -> Option<(u32, u32)> {
        self.source.as_ref().map(|s| (s.width, s.height))
    }

    /// Renders the whole photo at `render_size`, then writes the part of it
    /// under `crop` into an output of `output_size` pixels. Returns the
    /// output view, or `None` when no source is set. The work is submitted
    /// but not waited for.
    pub fn render(
        &mut self,
        edit: &PhotoEdit,
        crop: CropRect,
        render_size: (u32, u32),
        output_size: (u32, u32),
    ) -> Option<&wgpu::TextureView> {
        let source = self.source.as_ref()?;
        let (width, height) = (render_size.0.max(1), render_size.1.max(1));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("develop encoder"),
            });

        let stale = self.frame.as_ref().is_none_or(|f| {
            (f.width, f.height, f.generation) != (width, height, source.generation)
        });
        if stale {
            let frame = self.build_frame(source, width, height);
            let taps = (source.width as f32 / width as f32)
                .max(source.height as f32 / height as f32)
                .ceil()
                .clamp(1.0, MAX_TAPS as f32) as u32;
            self.queue.write_buffer(
                &self.input_uniform,
                0,
                bytemuck::bytes_of(&InputUniform {
                    matrix: matrices::input_matrix(source.space).to_wgsl_columns(),
                    render_size: [width as f32, height as f32],
                    decode_srgb: 1,
                    taps,
                }),
            );
            let sigma = basic::base_sigma(width, height);
            let radius = basic::blur_radius(sigma);
            for (buffer, direction, luma) in [
                (&self.blur_h_uniform, [1, 0], 1),
                (&self.blur_v_uniform, [0, 1], 0),
            ] {
                self.queue.write_buffer(
                    buffer,
                    0,
                    bytemuck::bytes_of(&BlurUniform {
                        direction,
                        radius,
                        luma,
                        sigma,
                        _pad: [0.0; 3],
                    }),
                );
            }
            draw(
                &mut encoder,
                "input transform",
                &self.input.pipeline,
                &frame.input_bind,
                &frame.working.view,
            );
            draw(
                &mut encoder,
                "blur h",
                &self.blur.pipeline,
                &frame.blur_h_bind,
                &frame.ping.view,
            );
            draw(
                &mut encoder,
                "blur v",
                &self.blur.pipeline,
                &frame.blur_v_bind,
                &frame.base.view,
            );
            self.frame_generation += 1;
            self.frame = Some(frame);
        }
        let frame = self.frame.as_ref().expect("frame built above");

        self.queue.write_buffer(
            &self.develop_uniform,
            0,
            bytemuck::bytes_of(&DevelopUniform::new(edit)),
        );
        draw(
            &mut encoder,
            "develop",
            &self.develop.pipeline,
            &frame.develop_bind,
            &frame.developed.view,
        );

        let (out_width, out_height) = (output_size.0.max(1), output_size.1.max(1));
        let out_stale = self.out.as_ref().is_none_or(|o| {
            (o.target.width, o.target.height) != (out_width, out_height)
                || o.frame_generation != self.frame_generation
        });
        if out_stale {
            let target = create_target(
                &self.device,
                "develop output",
                OUTPUT_FORMAT,
                out_width,
                out_height,
            );
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("output bind group"),
                layout: &self.output.layout,
                entries: &[
                    buffer_binding(0, &self.output_uniform),
                    texture_binding(1, &frame.developed.view),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.out = Some(Output {
                target,
                bind,
                frame_generation: self.frame_generation,
            });
            self.out_generation += 1;
        }
        let out = self.out.as_ref().expect("output built above");
        self.queue.write_buffer(
            &self.output_uniform,
            0,
            bytemuck::bytes_of(&OutputUniform {
                matrix: matrices::rec2020_to_srgb().to_wgsl_columns(),
                crop: [crop.x, crop.y, crop.width, crop.height],
            }),
        );
        draw(
            &mut encoder,
            "output",
            &self.output.pipeline,
            &out.bind,
            &out.target.view,
        );
        self.queue.submit(Some(encoder.finish()));
        Some(&out.target.view)
    }

    /// The last output view, when a render has happened.
    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.out.as_ref().map(|o| &o.target.view)
    }

    /// The size of the last output.
    pub fn output_size(&self) -> Option<(u32, u32)> {
        self.out.as_ref().map(|o| (o.target.width, o.target.height))
    }

    /// Counts up every time the output texture is replaced, so a caller
    /// that registered the view elsewhere knows to register it again.
    pub fn output_generation(&self) -> u64 {
        self.out_generation
    }

    /// Renders `crop` at the full source resolution, resamples it to the
    /// preset's size with linear filtering (halving first while the ratio
    /// is above 2, so no step skips pixels) and reads it back as sRGB RGBA
    /// bytes. Returns `None` when no source is set. The viewer's frame is
    /// replaced by the full-size one, so the next viewer render rebuilds.
    pub fn render_export(
        &mut self,
        edit: &PhotoEdit,
        crop: CropRect,
        preset: ExportPreset,
    ) -> Option<Vec<u8>> {
        let (source_width, source_height) = self.source_size()?;
        let crop_size = crop.pixel_size(source_width, source_height);
        self.render(edit, crop, (source_width, source_height), crop_size)?;
        let target = preset.size();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("export encoder"),
            });
        let mut view = self.out.as_ref()?.target.view.clone();
        let mut size = crop_size;
        let mut steps = Vec::new();
        while size.0 > target.0 * 2 && size.1 > target.1 * 2 {
            size = (size.0.div_ceil(2), size.1.div_ceil(2));
            let step = self.resample(&mut encoder, &view, size);
            view = step.0.view.clone();
            steps.push(step);
        }
        let last = self.resample(&mut encoder, &view, target);
        steps.push(last);
        self.queue.submit(Some(encoder.finish()));
        let (final_target, _, _) = steps.last().expect("at least one step");
        log::info!(
            "exported {crop_size:?} of {source_width}x{source_height} to {target:?} in {} steps",
            steps.len()
        );
        Some(self.readback.read(
            &self.device,
            &self.queue,
            &final_target.view,
            target.0,
            target.1,
        ))
    }

    /// One linear resample of `source` into a new sRGB target of `size`,
    /// through the output pass with the identity matrix.
    fn resample(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        size: (u32, u32),
    ) -> (Target, wgpu::Buffer, wgpu::BindGroup) {
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resample uniform"),
            size: size_of::<OutputUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &uniform,
            0,
            bytemuck::bytes_of(&OutputUniform {
                matrix: matrices::Mat3::IDENTITY.to_wgsl_columns(),
                crop: [0.0, 0.0, 1.0, 1.0],
            }),
        );
        let target = create_target(
            &self.device,
            "resample target",
            OUTPUT_FORMAT,
            size.0,
            size.1,
        );
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resample bind group"),
            layout: &self.output.layout,
            entries: &[
                buffer_binding(0, &uniform),
                texture_binding(1, source),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        draw(
            encoder,
            "resample",
            &self.output.pipeline,
            &bind,
            &target.view,
        );
        (target, uniform, bind)
    }

    fn build_frame(&self, source: &Source, width: u32, height: u32) -> Frame {
        let device = &self.device;
        let working = create_target(device, "working", WORKING_FORMAT, width, height);
        let ping = create_target(device, "blur ping", BASE_FORMAT, width, height);
        let base = create_target(device, "base", BASE_FORMAT, width, height);
        let developed = create_target(device, "developed", WORKING_FORMAT, width, height);
        let input_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("input bind group"),
            layout: &self.input.layout,
            entries: &[
                buffer_binding(0, &self.input_uniform),
                texture_binding(1, &source.view),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let blur_h_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur h bind group"),
            layout: &self.blur.layout,
            entries: &[
                buffer_binding(0, &self.blur_h_uniform),
                texture_binding(1, &working.view),
            ],
        });
        let blur_v_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur v bind group"),
            layout: &self.blur.layout,
            entries: &[
                buffer_binding(0, &self.blur_v_uniform),
                texture_binding(1, &ping.view),
            ],
        });
        let develop_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("develop bind group"),
            layout: &self.develop.layout,
            entries: &[
                buffer_binding(0, &self.develop_uniform),
                texture_binding(1, &working.view),
                texture_binding(2, &base.view),
            ],
        });
        Frame {
            width,
            height,
            generation: source.generation,
            working,
            ping,
            base,
            developed,
            input_bind,
            blur_h_bind,
            blur_v_bind,
            develop_bind,
        }
    }
}

/// The size to render the whole photo at so that `crop` lands on an output
/// of `output` pixels one to one.
pub fn render_size_for_crop(crop: CropRect, output: (u32, u32)) -> (u32, u32) {
    (
        (output.0 as f32 / crop.width.max(1e-6)).round().max(1.0) as u32,
        (output.1 as f32 / crop.height.max(1e-6)).round().max(1.0) as u32,
    )
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn buffer_binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn texture_binding(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn make_pass(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    format: wgpu::TextureFormat,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Pass {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: fullscreen_primitive(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    Pass { pipeline, layout }
}

fn create_target(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Target {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    Target {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        width,
        height,
    }
}

fn draw(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..FULLSCREEN_VERTICES, 0..1);
}
