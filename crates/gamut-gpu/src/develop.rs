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
//! 4. `develop.wgsl`: the whole develop chain of gamut-color's
//!    `develop_pixel_with`, from a uniform that mirrors `PhotoEdit`, into
//!    the developed texture.
//! 5. `output.wgsl`: the crop out of the developed texture, Rec.2020 to
//!    sRGB, clipped, into an 8-bit sRGB texture that egui draws and
//!    `Readback` reads.
//!
//! Passes 2 and 3 depend only on the source and the render size, so they
//! run again only when one of those changes. A slider change runs 4 and 5.
//!
//! M3 adds three products the develop pass reads, each skipped while its
//! operator is at rest, so an edit that does not use them costs nothing:
//!
//! - The texture layer: `blur.wgsl` twice more at the finer sigma, into its
//!   own R16Float texture. A head-pass product: it runs once when texture
//!   leaves 0 and again only when pass 2 runs again.
//! - The transmission map of dehaze: `minimum.wgsl` twice (the dark channel
//!   over the atmospheric light under a square minimum filter), then
//!   `blur.wgsl` twice to smooth it, into its own R16Float texture. Also a
//!   head-pass product, run once when dehaze leaves 0. The atmospheric light
//!   is read from the photo on the CPU in `set_source`; a video frame takes
//!   white.
//! - The tone curve table: 1024 by 1 in Rgba32Float, baked by gamut-color
//!   and uploaded only when a curve changes. It is read with `textureLoad`,
//!   two loads and a mix, so it needs no filtering.
//!
//! M2 lets a decoded video frame stand in for the photo: the source is
//! then the two plane textures of [`crate::video::VideoSource`] and pass 2
//! is `yuv_to_working.wgsl`, which writes the same linear Rec.2020 working
//! texture. Passes 3 to 5 do not know the difference. A new frame of the
//! same size rewrites the planes and reruns passes 2 and 3 without
//! rebuilding any texture.

use bytemuck::{Pod, Zeroable};
use gamut_color::SourceSpace;
use gamut_color::basic;
use gamut_color::curve::{self, TABLE_SIZE};
use gamut_color::dehaze;
use gamut_color::hsl::HslParams;
use gamut_color::local;
use gamut_color::mask::{self as mask_twin, Geometry};
use gamut_color::matrices;
use gamut_color::video::VideoColour;
use gamut_color::wheels::Cdl;
use gamut_core::look::ToneCurves;
use gamut_core::mask::{MAX_COMPONENTS, MAX_MASKS, Mask, MaskOp, MaskShape, MaskSource};
use gamut_core::{Adjustments, CropRect, ExportPreset, PhotoEdit};
use gamut_media::{FramePlanes, Photo};

use crate::video::{VideoSource, VideoUniform};
use crate::{FULLSCREEN_VERTICES, Readback, fullscreen_primitive};

/// The working-space format: linear Rec.2020 in half floats.
pub const WORKING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The base layer format: one half float of luminance.
pub const BASE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// The output format: 8-bit sRGB, encoded by the hardware on write.
pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The tone curve table format: full floats, read with `textureLoad`.
pub const TABLE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// The alpha of a mask: one byte per pixel.
pub const ALPHA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// The rows of the tone curve table: the global edit, then one per mask.
pub const TABLE_ROWS: u32 = 1 + MAX_MASKS as u32;

/// The most samples per axis the input transform averages per output pixel.
pub const MAX_TAPS: u32 = 8;

/// The `flags` bits of the develop uniform.
const FLAG_CURVES: u32 = 1;
const FLAG_HSL: u32 = 2;
const FLAG_WHEELS: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InputUniform {
    matrix: [[f32; 4]; 3],
    render_size: [f32; 2],
    decode_srgb: u32,
    taps: u32,
    window: [f32; 4],
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

/// The uniform of `minimum.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MinimumUniform {
    direction: [i32; 2],
    radius: i32,
    stage: u32,
    atmosphere: [f32; 4],
}

/// Mirrors the `Uniform` of `develop.wgsl` member for member; a test holds
/// the two layouts equal.
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
    texture: f32,
    clarity: f32,
    dehaze: f32,
    flags: u32,
    table_row: u32,
    opacity: f32,
    atmosphere: [f32; 4],
    cdl_slope: [f32; 4],
    cdl_offset: [f32; 4],
    cdl_power: [f32; 4],
    hsl: [[f32; 4]; 8],
}

impl DevelopUniform {
    /// The uniform of the global pass. `atmosphere` is the atmospheric light
    /// of the source.
    fn new(edit: &Adjustments, atmosphere: [f32; 3]) -> Self {
        let wb =
            basic::white_balance_matrix(edit.white_balance_temperature, edit.white_balance_tint);
        let look = &edit.look;
        let mut flags = 0;
        if !look.curves.is_identity() {
            flags |= FLAG_CURVES;
        }
        let mut hsl = [[0.0; 4]; 8];
        if !look.hsl_is_identity() {
            flags |= FLAG_HSL;
            hsl = HslParams::new(&look.hsl).ranges;
        }
        let mut cdl = Cdl::IDENTITY;
        if !look.wheels.is_identity() {
            flags |= FLAG_WHEELS;
            cdl = Cdl::new(&look.wheels);
        }
        let wide = |v: [f32; 3]| [v[0], v[1], v[2], 1.0];
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
            texture: edit.texture,
            clarity: edit.clarity,
            dehaze: edit.dehaze,
            flags,
            table_row: 0,
            opacity: 1.0,
            atmosphere: wide(basic::exposed_atmosphere(&wb, atmosphere, edit.exposure)),
            cdl_slope: wide(cdl.slope),
            cdl_offset: wide(cdl.offset),
            cdl_power: wide(cdl.power),
            hsl,
        }
    }
}

impl DevelopUniform {
    /// The uniform of the pass of the mask at `index` of the list: its
    /// effective adjustments, its row of the curve table, its opacity. The
    /// curves run when the global edit or the mask has one.
    fn for_mask(global: &Adjustments, mask: &Mask, index: usize, atmosphere: [f32; 3]) -> Self {
        let effective = mask_twin::effective_adjustments(global, &mask.adjust);
        let mut uniform = Self::new(&effective, atmosphere);
        if !mask.adjust.look.curves.is_identity() {
            uniform.flags |= FLAG_CURVES;
        }
        uniform.table_row = index as u32 + 1;
        uniform.opacity = mask.opacity / 100.0;
        uniform
    }
}

/// One component of `mask.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskComponentUniform {
    header: [u32; 4],
    a: [f32; 4],
    b: [f32; 4],
}

/// Mirrors the `Uniform` of `mask.wgsl`; a test holds the two layouts equal.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskUniform {
    window: [f32; 4],
    render_size: [f32; 2],
    aspect: [f32; 2],
    count: u32,
    invert: u32,
    _pad: [u32; 2],
    components: [MaskComponentUniform; MAX_COMPONENTS],
}

impl MaskUniform {
    /// `mask` is sanitised. Everything the twin computes once per mask is
    /// computed here with the same arithmetic: the sine and the cosine of a
    /// rotation, the inner share of a feather, the floored falloffs.
    fn new(mask: &Mask, geometry: &Geometry) -> Self {
        let mut components = [MaskComponentUniform::zeroed(); MAX_COMPONENTS];
        for (slot, component) in components.iter_mut().zip(&mask.components) {
            let (kind, a, b) = match component.source {
                MaskSource::Linear(g) => {
                    (0, [g.start[0], g.start[1], g.end[0], g.end[1]], [0.0; 4])
                }
                MaskSource::Radial(g) => {
                    let (sin, cos) = g.rotation.to_radians().sin_cos();
                    let inner = (1.0 - g.feather / 100.0).min(mask_twin::RADIAL_INNER_CEILING);
                    (
                        1,
                        [g.centre[0], g.centre[1], g.radius[0], g.radius[1]],
                        [sin, cos, inner, 0.0],
                    )
                }
                MaskSource::Luminance(r) => (
                    2,
                    [
                        r.low,
                        r.high,
                        r.falloff.max(mask_twin::LUMINANCE_FALLOFF_FLOOR),
                        0.0,
                    ],
                    [0.0; 4],
                ),
                MaskSource::Colour(r) => (
                    3,
                    [
                        r.hue,
                        r.hue_width / 2.0,
                        r.chroma_low,
                        r.falloff.max(mask_twin::HUE_FALLOFF_FLOOR),
                    ],
                    [0.0; 4],
                ),
            };
            let op = match component.op {
                MaskOp::Add => 0,
                MaskOp::Subtract => 1,
                MaskOp::Intersect => 2,
            };
            *slot = MaskComponentUniform {
                header: [kind, op, u32::from(component.invert), 0],
                a,
                b,
            };
        }
        let window = geometry.window;
        MaskUniform {
            window: [window.x, window.y, window.width, window.height],
            render_size: [geometry.size.0 as f32, geometry.size.1 as f32],
            aspect: geometry.aspect(),
            count: mask.components.len().min(MAX_COMPONENTS) as u32,
            invert: u32::from(mask.invert),
            _pad: [0; 2],
            components,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OutputUniform {
    matrix: [[f32; 4]; 3],
    crop: [f32; 4],
    overlay: [f32; 4],
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

/// What the head of the graph reads.
enum SourceKind {
    /// A photo's RGBA texture, through the input transform.
    Photo {
        view: wgpu::TextureView,
        space: SourceSpace,
    },
    /// A video frame's planes, through the YUV pass. Boxed because the
    /// planes are far larger than the photo variant.
    Video(Box<VideoSource>),
}

struct Source {
    kind: SourceKind,
    /// The display size: the photo's, or the frame's after its rotation.
    width: u32,
    height: u32,
    /// Counts up when the textures are replaced.
    generation: u64,
    /// Counts up when the content changes but the textures stay.
    content: u64,
    /// The atmospheric light dehaze works against, in linear Rec.2020.
    atmosphere: [f32; 3],
}

/// Everything that depends on the source and the render size.
struct Frame {
    width: u32,
    height: u32,
    generation: u64,
    /// The source content the working and base textures hold.
    content: u64,
    /// The part of the source this render covers.
    window: CropRect,
    /// The size the blur sigma was taken from.
    sigma_size: (u32, u32),
    working: Target,
    ping: Target,
    base: Target,
    /// The texture layer; holds nothing until `texture_ready`.
    texture_base: Target,
    /// The transmission map; holds nothing until `transmission_ready`.
    transmission: Target,
    developed: Target,
    /// The texture layer matches the working texture and `products_region`.
    texture_ready: bool,
    /// The transmission map matches the working texture and
    /// `products_region`.
    transmission_ready: bool,
    /// The scissor the two products above were last drawn under.
    products_region: Option<(u32, u32, u32, u32)>,
    input_bind: wgpu::BindGroup,
    blur_h_bind: wgpu::BindGroup,
    blur_v_bind: wgpu::BindGroup,
    texture_h_bind: wgpu::BindGroup,
    texture_v_bind: wgpu::BindGroup,
    minimum_h_bind: wgpu::BindGroup,
    minimum_v_bind: wgpu::BindGroup,
    smooth_h_bind: wgpu::BindGroup,
    smooth_v_bind: wgpu::BindGroup,
    develop_bind: wgpu::BindGroup,
    /// What each mask of the list has on this frame, by its index in the
    /// list; nothing until the mask first runs.
    masks: [Option<FrameMask>; MAX_MASKS],
}

/// The head-pass product of one mask: its alpha, cached like the texture
/// layer and the transmission map.
struct FrameMask {
    alpha: Target,
    /// The shape `alpha` holds, or `None` when it has to be drawn again: the
    /// head passes reran or the scissor moved. A slider of the mask's
    /// adjustments is no part of the shape, so it never redraws the alpha.
    shape: Option<MaskShape>,
    raster_bind: wgpu::BindGroup,
    develop_bind: wgpu::BindGroup,
}

struct Output {
    target: Target,
    bind: wgpu::BindGroup,
    frame_generation: u64,
    /// The mask whose alpha the bind group holds for the overlay.
    overlay: Option<usize>,
}

/// The develop graph on one device. Build it once, set a source, render as
/// often as the edit or the view changes.
pub struct Develop {
    device: wgpu::Device,
    queue: wgpu::Queue,
    input: Pass,
    video: Pass,
    blur: Pass,
    minimum: Pass,
    develop: Pass,
    /// `mask.wgsl`: draws the alpha of one mask.
    mask: Pass,
    /// `develop.wgsl` through `fs_masked` with alpha blending: develops one
    /// mask over the developed texture.
    masked: Pass,
    output: Pass,
    input_uniform: wgpu::Buffer,
    video_uniform: wgpu::Buffer,
    blur_h_uniform: wgpu::Buffer,
    blur_v_uniform: wgpu::Buffer,
    texture_h_uniform: wgpu::Buffer,
    texture_v_uniform: wgpu::Buffer,
    minimum_h_uniform: wgpu::Buffer,
    minimum_v_uniform: wgpu::Buffer,
    smooth_h_uniform: wgpu::Buffer,
    smooth_v_uniform: wgpu::Buffer,
    develop_uniform: wgpu::Buffer,
    /// Per mask of the list: the uniform of its alpha and of its develop.
    mask_uniforms: Vec<wgpu::Buffer>,
    masked_uniforms: Vec<wgpu::Buffer>,
    /// The tone curve table and the curves it was baked from: row 0 from
    /// the global curves, row 1 + i from the global curves and those of
    /// mask i.
    curve_table: wgpu::Texture,
    curve_table_view: wgpu::TextureView,
    curves_uploaded: Option<ToneCurves>,
    mask_curves_uploaded: [Option<(ToneCurves, ToneCurves)>; MAX_MASKS],
    /// How many mask alphas have been drawn, for the cache test.
    alpha_builds: u64,
    output_uniform: wgpu::Buffer,
    /// The mask of the list shown as a red overlay, when one is.
    overlay: Option<usize>,
    /// One zero texel: the overlay alpha of an output that shows none.
    no_overlay: Target,
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
        let video = make_pass(
            device,
            "video",
            include_str!("shaders/yuv_to_working.wgsl"),
            WORKING_FORMAT,
            &[
                uniform_entry(0),
                texture_entry(1),
                texture_entry(2),
                sampler_entry(3),
            ],
        );
        let blur = make_pass(
            device,
            "blur",
            include_str!("shaders/blur.wgsl"),
            BASE_FORMAT,
            &[uniform_entry(0), texture_entry(1), sampler_entry(2)],
        );
        let minimum = make_pass(
            device,
            "minimum",
            include_str!("shaders/minimum.wgsl"),
            BASE_FORMAT,
            &[uniform_entry(0), texture_entry(1)],
        );
        let develop = make_pass(
            device,
            "develop",
            include_str!("shaders/develop.wgsl"),
            WORKING_FORMAT,
            &[
                uniform_entry(0),
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                table_entry(5),
            ],
        );
        let mask = make_pass(
            device,
            "mask",
            include_str!("shaders/mask.wgsl"),
            ALPHA_FORMAT,
            &[uniform_entry(0), texture_entry(1)],
        );
        let masked = make_pass_with(
            device,
            "masked develop",
            include_str!("shaders/develop.wgsl"),
            WORKING_FORMAT,
            &[
                uniform_entry(0),
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                table_entry(5),
                texture_entry(6),
            ],
            "fs_masked",
            wgpu::BlendState::ALPHA_BLENDING,
        );
        let curve_table = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tone curve table"),
            size: wgpu::Extent3d {
                width: TABLE_SIZE as u32,
                height: TABLE_ROWS,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TABLE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let curve_table_view = curve_table.create_view(&wgpu::TextureViewDescriptor::default());
        let output = make_pass(
            device,
            "output",
            include_str!("shaders/output.wgsl"),
            OUTPUT_FORMAT,
            &[
                uniform_entry(0),
                texture_entry(1),
                sampler_entry(2),
                texture_entry(3),
            ],
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
            video,
            blur,
            minimum,
            develop,
            mask,
            masked,
            output,
            input_uniform: uniform("input uniform", size_of::<InputUniform>() as u64),
            video_uniform: uniform("video uniform", size_of::<VideoUniform>() as u64),
            blur_h_uniform: uniform("blur h uniform", size_of::<BlurUniform>() as u64),
            blur_v_uniform: uniform("blur v uniform", size_of::<BlurUniform>() as u64),
            texture_h_uniform: uniform("texture h uniform", size_of::<BlurUniform>() as u64),
            texture_v_uniform: uniform("texture v uniform", size_of::<BlurUniform>() as u64),
            minimum_h_uniform: uniform("minimum h uniform", size_of::<MinimumUniform>() as u64),
            minimum_v_uniform: uniform("minimum v uniform", size_of::<MinimumUniform>() as u64),
            smooth_h_uniform: uniform("smooth h uniform", size_of::<BlurUniform>() as u64),
            smooth_v_uniform: uniform("smooth v uniform", size_of::<BlurUniform>() as u64),
            develop_uniform: uniform("develop uniform", size_of::<DevelopUniform>() as u64),
            mask_uniforms: (0..MAX_MASKS)
                .map(|_| uniform("mask uniform", size_of::<MaskUniform>() as u64))
                .collect(),
            masked_uniforms: (0..MAX_MASKS)
                .map(|_| uniform("masked develop uniform", size_of::<DevelopUniform>() as u64))
                .collect(),
            curve_table,
            curve_table_view,
            curves_uploaded: None,
            mask_curves_uploaded: Default::default(),
            alpha_builds: 0,
            output_uniform: uniform("output uniform", size_of::<OutputUniform>() as u64),
            overlay: None,
            no_overlay: zero_texel(device, queue),
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
            kind: SourceKind::Photo {
                view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
                space: photo.source,
            },
            width: photo.width,
            height: photo.height,
            generation: self.generation,
            content: self.generation,
            atmosphere: dehaze::atmosphere_rgba8(
                &photo.rgba8,
                photo.width,
                photo.height,
                photo.source,
            ),
        });
        self.frame = None;
        log::info!(
            "develop source set: {}x{} {:?} as {format:?}",
            photo.width,
            photo.height,
            photo.source
        );
    }

    /// Uploads a decoded video frame as the source. The plane textures are
    /// kept between frames of one size and format; a frame of another
    /// size, format, colour or rotation replaces them.
    pub fn set_video_frame(&mut self, frame: &dyn FramePlanes, colour: VideoColour, rotation: u32) {
        let reuse = matches!(
            &self.source,
            Some(Source {
                kind: SourceKind::Video(planes),
                ..
            }) if planes.accepts(frame) && planes.colour == colour && planes.rotation == rotation
        );
        if !reuse {
            let planes = VideoSource::new(
                &self.device,
                frame.width(),
                frame.height(),
                frame.format(),
                colour,
                rotation,
            );
            let (width, height) = planes.display_size();
            self.generation += 1;
            self.source = Some(Source {
                kind: SourceKind::Video(Box::new(planes)),
                width,
                height,
                generation: self.generation,
                content: 0,
                // A frame's own atmospheric light waits for per-clip
                // analysis; until then haze is measured against white.
                atmosphere: dehaze::WHITE_ATMOSPHERE,
            });
            self.frame = None;
            log::info!(
                "develop video source set: {}x{} {:?} rotation {rotation} {colour:?}",
                frame.width(),
                frame.height(),
                frame.format()
            );
        }
        let source = self.source.as_mut().expect("set above");
        let SourceKind::Video(planes) = &source.kind else {
            unreachable!("the source is the video planes");
        };
        planes.upload(&self.queue, frame);
        source.content += 1;
    }

    /// Whether the source is a video frame.
    pub fn has_video(&self) -> bool {
        matches!(
            &self.source,
            Some(Source {
                kind: SourceKind::Video(_),
                ..
            })
        )
    }

    /// Shows the mask at this index of the edit's list as a red overlay on
    /// every render from now on, or none. The overlay is for the window and
    /// for a screenshot that asks for it: [`render_export`](Self::render_export)
    /// never draws it.
    pub fn set_overlay(&mut self, mask: Option<usize>) {
        self.overlay = mask;
    }

    /// The mask shown as an overlay, when one is.
    pub fn overlay(&self) -> Option<usize> {
        self.overlay
    }

    /// How many mask alphas this graph has drawn since it was built. A test
    /// reads it to hold that a slider of a mask never redraws its alpha.
    #[doc(hidden)]
    pub fn mask_alpha_builds(&self) -> u64 {
        self.alpha_builds
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
        self.render_window(
            edit,
            crop,
            render_size,
            output_size,
            CropRect::FULL,
            render_size,
        )
    }

    /// Renders only what `crop` needs at the scale where it lands on
    /// `output_size` one to one: the crop plus the reach of the base blur
    /// on every side, so the picture inside the crop is the one
    /// [`render`](Self::render) would give at the full size. The rest of
    /// the frame is never drawn, which is what makes a Reel frame cheap.
    pub fn render_crop(
        &mut self,
        edit: &PhotoEdit,
        crop: CropRect,
        output_size: (u32, u32),
    ) -> Option<&wgpu::TextureView> {
        let full = render_size_for_crop(crop, output_size);
        let (full_w, full_h) = (full.0 as f32, full.1 as f32);
        let reach = basic::blur_radius(basic::base_sigma(full.0, full.1)) as f32;
        let x0 = (crop.x * full_w - reach).floor().max(0.0);
        let y0 = (crop.y * full_h - reach).floor().max(0.0);
        let x1 = ((crop.x + crop.width) * full_w + reach).ceil().min(full_w);
        let y1 = ((crop.y + crop.height) * full_h + reach).ceil().min(full_h);
        let window = CropRect {
            x: x0 / full_w,
            y: y0 / full_h,
            width: (x1 - x0) / full_w,
            height: (y1 - y0) / full_h,
        };
        let inner = CropRect {
            x: (crop.x - window.x) / window.width,
            y: (crop.y - window.y) / window.height,
            width: crop.width / window.width,
            height: crop.height / window.height,
        };
        let window_size = ((x1 - x0).round() as u32, (y1 - y0).round() as u32);
        self.render_window(edit, inner, window_size, output_size, window, full)
    }

    /// The shared body: renders `window` of the source at `render_size`
    /// with the blur sigma of `sigma_size`, then crops `crop` of that
    /// render into the output.
    fn render_window(
        &mut self,
        edit: &PhotoEdit,
        crop: CropRect,
        render_size: (u32, u32),
        output_size: (u32, u32),
        window: CropRect,
        sigma_size: (u32, u32),
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
                || f.window != window
                || f.sigma_size != sigma_size
        });
        let rerun = stale
            || self
                .frame
                .as_ref()
                .is_some_and(|f| f.content != source.content);
        if stale {
            self.frame = Some(self.build_frame(source, width, height, window, sigma_size));
            self.frame_generation += 1;
        }
        if rerun {
            let frame = self.frame.as_mut().expect("frame built above");
            frame.content = source.content;
            let window_uniform = [window.x, window.y, window.width, window.height];
            let (head_pipeline, head_label) = match &source.kind {
                SourceKind::Photo { space, .. } => {
                    let taps = (source.width as f32 * window.width / width as f32)
                        .max(source.height as f32 * window.height / height as f32)
                        .ceil()
                        .clamp(1.0, MAX_TAPS as f32) as u32;
                    self.queue.write_buffer(
                        &self.input_uniform,
                        0,
                        bytemuck::bytes_of(&InputUniform {
                            matrix: matrices::input_matrix(*space).to_wgsl_columns(),
                            render_size: [width as f32, height as f32],
                            decode_srgb: 1,
                            taps,
                            window: window_uniform,
                        }),
                    );
                    (&self.input.pipeline, "input transform")
                }
                SourceKind::Video(planes) => {
                    self.queue.write_buffer(
                        &self.video_uniform,
                        0,
                        bytemuck::bytes_of(&planes.uniform(window_uniform)),
                    );
                    (&self.video.pipeline, "video")
                }
            };
            let sigma = basic::base_sigma(sigma_size.0, sigma_size.1);
            write_blur(&self.queue, &self.blur_h_uniform, [1, 0], 1, sigma);
            write_blur(&self.queue, &self.blur_v_uniform, [0, 1], 0, sigma);
            draw(
                &mut encoder,
                head_label,
                head_pipeline,
                &frame.input_bind,
                &frame.working.view,
                None,
            );
            // Only the columns under the crop are read by the vertical
            // pass, and only the crop by the develop pass, so the blur and
            // develop passes are scissored to the crop plus a margin.
            let columns = scissor_for(crop, (width, height), true);
            let region = scissor_for(crop, (width, height), false);
            draw(
                &mut encoder,
                "blur h",
                &self.blur.pipeline,
                &frame.blur_h_bind,
                &frame.ping.view,
                Some(columns),
            );
            draw(
                &mut encoder,
                "blur v",
                &self.blur.pipeline,
                &frame.blur_v_bind,
                &frame.base.view,
                Some(region),
            );
        }
        // The head-pass products of M3. Each is drawn once when its slider
        // leaves 0 and again only after the head passes ran or the scissor
        // moved, never on a plain slider change.
        let frame = self.frame.as_mut().expect("frame built above");
        let columns = scissor_for(crop, (width, height), true);
        let region = scissor_for(crop, (width, height), false);
        if rerun || frame.products_region != Some(region) {
            frame.texture_ready = false;
            frame.transmission_ready = false;
            frame.products_region = Some(region);
            for mask in frame.masks.iter_mut().flatten() {
                mask.shape = None;
            }
        }
        // The masks that change the picture, and what each develops with. A
        // mask that alone turns on texture or dehaze asks for the product
        // the same way the global slider does.
        let masks = mask_twin::active_masks(edit);
        // The overlay shows a mask whether or not it adjusts anything yet,
        // so its alpha is drawn even when the mask itself is not.
        let shown = self.overlay.filter(|index| *index < MAX_MASKS);
        let overlaid: Option<(usize, Mask)> = shown.and_then(|index| {
            let mask = edit.masks.get(index)?.sanitised();
            Some((index, mask))
        });
        let effective: Vec<Adjustments> = masks
            .iter()
            .map(|(_, mask)| mask_twin::effective_adjustments(&edit.adjust, &mask.adjust))
            .collect();
        let wants_texture = edit.texture != 0.0 || effective.iter().any(|e| e.texture != 0.0);
        let wants_transmission = edit.dehaze != 0.0 || effective.iter().any(|e| e.dehaze != 0.0);
        if wants_texture && !frame.texture_ready {
            let sigma = local::texture_sigma(sigma_size.0, sigma_size.1);
            write_blur(&self.queue, &self.texture_h_uniform, [1, 0], 1, sigma);
            write_blur(&self.queue, &self.texture_v_uniform, [0, 1], 0, sigma);
            draw(
                &mut encoder,
                "texture blur h",
                &self.blur.pipeline,
                &frame.texture_h_bind,
                &frame.ping.view,
                Some(columns),
            );
            draw(
                &mut encoder,
                "texture blur v",
                &self.blur.pipeline,
                &frame.texture_v_bind,
                &frame.texture_base.view,
                Some(region),
            );
            frame.texture_ready = true;
        }
        if wants_transmission && !frame.transmission_ready {
            let radius = dehaze::patch_radius(sigma_size.0, sigma_size.1);
            let a = source.atmosphere;
            for (buffer, direction, stage) in [
                (&self.minimum_h_uniform, [1, 0], 0),
                (&self.minimum_v_uniform, [0, 1], 1),
            ] {
                self.queue.write_buffer(
                    buffer,
                    0,
                    bytemuck::bytes_of(&MinimumUniform {
                        direction,
                        radius,
                        stage,
                        atmosphere: [a[0], a[1], a[2], 1.0],
                    }),
                );
            }
            let sigma = dehaze::smoothing_sigma(sigma_size.0, sigma_size.1);
            write_blur(&self.queue, &self.smooth_h_uniform, [1, 0], 0, sigma);
            write_blur(&self.queue, &self.smooth_v_uniform, [0, 1], 0, sigma);
            // The minimum runs over the whole render: the smoothing reads
            // the map beyond the scissor on every side.
            draw(
                &mut encoder,
                "minimum h",
                &self.minimum.pipeline,
                &frame.minimum_h_bind,
                &frame.ping.view,
                None,
            );
            draw(
                &mut encoder,
                "minimum v",
                &self.minimum.pipeline,
                &frame.minimum_v_bind,
                &frame.transmission.view,
                None,
            );
            draw(
                &mut encoder,
                "transmission blur h",
                &self.blur.pipeline,
                &frame.smooth_h_bind,
                &frame.ping.view,
                Some(columns),
            );
            draw(
                &mut encoder,
                "transmission blur v",
                &self.blur.pipeline,
                &frame.smooth_v_bind,
                &frame.transmission.view,
                Some(region),
            );
            frame.transmission_ready = true;
        }
        let curves = &edit.look.curves;
        if !curves.is_identity() && self.curves_uploaded.as_ref() != Some(curves) {
            write_table_row(&self.queue, &self.curve_table, 0, &curve::bake(curves));
            self.curves_uploaded = Some(curves.clone());
        }
        // The alpha of each mask, drawn once and kept until its shape, the
        // head passes or the scissor change; its row of the curve table,
        // uploaded only when the composed curves change; its uniform.
        let geometry = Geometry {
            window,
            size: (width, height),
            photo: (source.width, source.height),
        };
        let only_shown = overlaid
            .iter()
            .filter(|(index, _)| masks.iter().all(|(active, _)| active != index));
        for (index, mask, develops) in masks
            .iter()
            .map(|(index, mask)| (*index, mask, true))
            .chain(only_shown.map(|(index, mask)| (*index, mask, false)))
        {
            let slot = frame.masks[index].get_or_insert_with(|| {
                let alpha = create_target(&self.device, "mask alpha", ALPHA_FORMAT, width, height);
                let raster_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mask bind group"),
                    layout: &self.mask.layout,
                    entries: &[
                        buffer_binding(0, &self.mask_uniforms[index]),
                        texture_binding(1, &frame.working.view),
                    ],
                });
                let develop_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("masked develop bind group"),
                    layout: &self.masked.layout,
                    entries: &[
                        buffer_binding(0, &self.masked_uniforms[index]),
                        texture_binding(1, &frame.working.view),
                        texture_binding(2, &frame.base.view),
                        texture_binding(3, &frame.texture_base.view),
                        texture_binding(4, &frame.transmission.view),
                        texture_binding(5, &self.curve_table_view),
                        texture_binding(6, &alpha.view),
                    ],
                });
                FrameMask {
                    alpha,
                    shape: None,
                    raster_bind,
                    develop_bind,
                }
            });
            let shape = mask.shape();
            if slot.shape.as_ref() != Some(&shape) {
                self.queue.write_buffer(
                    &self.mask_uniforms[index],
                    0,
                    bytemuck::bytes_of(&MaskUniform::new(mask, &geometry)),
                );
                draw(
                    &mut encoder,
                    "mask alpha",
                    &self.mask.pipeline,
                    &slot.raster_bind,
                    &slot.alpha.view,
                    Some(region),
                );
                slot.shape = Some(shape);
                self.alpha_builds += 1;
            }
            if !develops {
                continue;
            }
            let (ours, theirs) = (&edit.look.curves, &mask.adjust.look.curves);
            let composed = (!ours.is_identity() || !theirs.is_identity())
                .then(|| (ours.clone(), theirs.clone()));
            if composed.is_some() && self.mask_curves_uploaded[index] != composed {
                let tables = curve::bake_composed(ours, theirs);
                write_table_row(&self.queue, &self.curve_table, index as u32 + 1, &tables);
                self.mask_curves_uploaded[index] = composed;
            }
            self.queue.write_buffer(
                &self.masked_uniforms[index],
                0,
                bytemuck::bytes_of(&DevelopUniform::for_mask(
                    &edit.adjust,
                    mask,
                    index,
                    source.atmosphere,
                )),
            );
        }
        let frame = self.frame.as_ref().expect("frame built above");

        self.queue.write_buffer(
            &self.develop_uniform,
            0,
            bytemuck::bytes_of(&DevelopUniform::new(&edit.adjust, source.atmosphere)),
        );
        draw(
            &mut encoder,
            "develop",
            &self.develop.pipeline,
            &frame.develop_bind,
            &frame.developed.view,
            Some(region),
        );
        // The ordered blend: each mask in list order over what is there.
        for (index, _) in &masks {
            let slot = frame.masks[*index].as_ref().expect("built above");
            draw_over(
                &mut encoder,
                "masked develop",
                &self.masked.pipeline,
                &slot.develop_bind,
                &frame.developed.view,
                Some(region),
            );
        }

        let (out_width, out_height) = (output_size.0.max(1), output_size.1.max(1));
        let overlay = overlaid.as_ref().map(|(index, _)| *index);
        let out_stale = self.out.as_ref().is_none_or(|o| {
            (o.target.width, o.target.height) != (out_width, out_height)
                || o.frame_generation != self.frame_generation
                || o.overlay != overlay
        });
        if out_stale {
            let target = create_target(
                &self.device,
                "develop output",
                OUTPUT_FORMAT,
                out_width,
                out_height,
            );
            let overlay_alpha = overlay
                .and_then(|index| frame.masks[index].as_ref())
                .map_or(&self.no_overlay.view, |slot| &slot.alpha.view);
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
                    texture_binding(3, overlay_alpha),
                ],
            });
            self.out = Some(Output {
                target,
                bind,
                frame_generation: self.frame_generation,
                overlay,
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
                overlay: [
                    overlay.map_or(0.0, |_| mask_twin::OVERLAY_STRENGTH),
                    0.0,
                    0.0,
                    0.0,
                ],
            }),
        );
        draw(
            &mut encoder,
            "output",
            &self.output.pipeline,
            &out.bind,
            &out.target.view,
            None,
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
        // An export is the picture, never the view of a mask.
        let overlay = self.overlay.take();
        let rendered = self
            .render(edit, crop, (source_width, source_height), crop_size)
            .is_some();
        self.overlay = overlay;
        if !rendered {
            return None;
        }
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
                overlay: [0.0; 4],
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
                texture_binding(3, &self.no_overlay.view),
            ],
        });
        draw(
            encoder,
            "resample",
            &self.output.pipeline,
            &bind,
            &target.view,
            None,
        );
        (target, uniform, bind)
    }

    fn build_frame(
        &self,
        source: &Source,
        width: u32,
        height: u32,
        window: CropRect,
        sigma_size: (u32, u32),
    ) -> Frame {
        let device = &self.device;
        let working = create_target(device, "working", WORKING_FORMAT, width, height);
        let ping = create_target(device, "blur ping", BASE_FORMAT, width, height);
        let base = create_target(device, "base", BASE_FORMAT, width, height);
        let texture_base = create_target(device, "texture base", BASE_FORMAT, width, height);
        let transmission = create_target(device, "transmission", BASE_FORMAT, width, height);
        let developed = create_target(device, "developed", WORKING_FORMAT, width, height);
        let input_bind = match &source.kind {
            SourceKind::Photo { view, .. } => {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("input bind group"),
                    layout: &self.input.layout,
                    entries: &[
                        buffer_binding(0, &self.input_uniform),
                        texture_binding(1, view),
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            }
            SourceKind::Video(planes) => device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("video bind group"),
                layout: &self.video.layout,
                entries: &[
                    buffer_binding(0, &self.video_uniform),
                    texture_binding(1, &planes.luma_view),
                    texture_binding(2, &planes.chroma_view),
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            }),
        };
        let blur_h_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur h bind group"),
            layout: &self.blur.layout,
            entries: &[
                buffer_binding(0, &self.blur_h_uniform),
                texture_binding(1, &working.view),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let blur_v_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur v bind group"),
            layout: &self.blur.layout,
            entries: &[
                buffer_binding(0, &self.blur_v_uniform),
                texture_binding(1, &ping.view),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let blur_bind = |label: &str, uniform: &wgpu::Buffer, view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.blur.layout,
                entries: &[
                    buffer_binding(0, uniform),
                    texture_binding(1, view),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let texture_h_bind = blur_bind(
            "texture h bind group",
            &self.texture_h_uniform,
            &working.view,
        );
        let texture_v_bind = blur_bind("texture v bind group", &self.texture_v_uniform, &ping.view);
        let smooth_h_bind = blur_bind(
            "smooth h bind group",
            &self.smooth_h_uniform,
            &transmission.view,
        );
        let smooth_v_bind = blur_bind("smooth v bind group", &self.smooth_v_uniform, &ping.view);
        let minimum_bind = |label: &str, uniform: &wgpu::Buffer, view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.minimum.layout,
                entries: &[buffer_binding(0, uniform), texture_binding(1, view)],
            })
        };
        let minimum_h_bind = minimum_bind(
            "minimum h bind group",
            &self.minimum_h_uniform,
            &working.view,
        );
        let minimum_v_bind =
            minimum_bind("minimum v bind group", &self.minimum_v_uniform, &ping.view);
        let develop_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("develop bind group"),
            layout: &self.develop.layout,
            entries: &[
                buffer_binding(0, &self.develop_uniform),
                texture_binding(1, &working.view),
                texture_binding(2, &base.view),
                texture_binding(3, &texture_base.view),
                texture_binding(4, &transmission.view),
                texture_binding(5, &self.curve_table_view),
            ],
        });
        Frame {
            width,
            height,
            generation: source.generation,
            // The head passes have not run into these textures yet.
            content: source.content.wrapping_sub(1),
            window,
            sigma_size,
            working,
            ping,
            base,
            texture_base,
            transmission,
            developed,
            texture_ready: false,
            transmission_ready: false,
            products_region: None,
            input_bind,
            blur_h_bind,
            blur_v_bind,
            texture_h_bind,
            texture_v_bind,
            minimum_h_bind,
            minimum_v_bind,
            smooth_h_bind,
            smooth_v_bind,
            develop_bind,
            masks: Default::default(),
        }
    }
}

/// A 1 by 1 alpha texture holding 0.
fn zero_texel(device: &wgpu::Device, queue: &wgpu::Queue) -> Target {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("no overlay"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ALPHA_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &[0],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(1),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    Target {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        width: 1,
        height: 1,
    }
}

/// Writes one row of the tone curve table.
fn write_table_row(queue: &wgpu::Queue, table: &wgpu::Texture, row: u32, tables: &curve::Tables) {
    let texels: Vec<[f32; 4]> = (0..TABLE_SIZE)
        .map(|i| [tables.red[i], tables.green[i], tables.blue[i], 1.0])
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: table,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: row, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TABLE_SIZE as u32 * 16),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: TABLE_SIZE as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

/// The pixels the blur must reach: the crop with a margin of two pixels
/// for the output pass's sampling, over the whole height when
/// `full_height` (the horizontal pass feeds every row the vertical pass
/// reads). As x, y, width, height of a render of `size`.
fn scissor_for(crop: CropRect, size: (u32, u32), full_height: bool) -> (u32, u32, u32, u32) {
    const MARGIN: f32 = 2.0;
    let (w, h) = (size.0 as f32, size.1 as f32);
    let x0 = (crop.x * w - MARGIN).floor().clamp(0.0, w - 1.0);
    let x1 = ((crop.x + crop.width) * w + MARGIN)
        .ceil()
        .clamp(x0 + 1.0, w);
    let (y0, y1) = if full_height {
        (0.0, h)
    } else {
        let y0 = (crop.y * h - MARGIN).floor().clamp(0.0, h - 1.0);
        (
            y0,
            ((crop.y + crop.height) * h + MARGIN)
                .ceil()
                .clamp(y0 + 1.0, h),
        )
    };
    (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
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

/// The tone curve table: full floats, which are not filterable without a
/// device feature, read with `textureLoad` only.
fn table_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Writes the uniform of one blur pass.
fn write_blur(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    direction: [i32; 2],
    luma: u32,
    sigma: f32,
) {
    queue.write_buffer(
        buffer,
        0,
        bytemuck::bytes_of(&BlurUniform {
            direction,
            radius: basic::blur_radius(sigma),
            luma,
            sigma,
            _pad: [0.0; 3],
        }),
    );
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
    make_pass_with(
        device,
        label,
        source,
        format,
        entries,
        "fs_main",
        wgpu::BlendState::REPLACE,
    )
}

/// [`make_pass`] with the fragment entry point and the blend named.
fn make_pass_with(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    format: wgpu::TextureFormat,
    entries: &[wgpu::BindGroupLayoutEntry],
    fragment: &str,
    blend: wgpu::BlendState,
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
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
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
    scissor: Option<(u32, u32, u32, u32)>,
) {
    let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
    draw_loading(encoder, label, pipeline, bind_group, target, scissor, clear);
}

/// [`draw`] over what the target holds, for a pipeline that blends.
fn draw_over(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    scissor: Option<(u32, u32, u32, u32)>,
) {
    let load = wgpu::LoadOp::Load;
    draw_loading(encoder, label, pipeline, bind_group, target, scissor, load);
}

fn draw_loading(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    scissor: Option<(u32, u32, u32, u32)>,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
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
    if let Some((x, y, w, h)) = scissor {
        pass.set_scissor_rect(x, y, w, h);
    }
    pass.draw(0..FULLSCREEN_VERTICES, 0..1);
}

#[cfg(test)]
mod tests {
    use super::{
        DevelopUniform, FLAG_CURVES, Geometry, MaskComponentUniform, MaskUniform, MinimumUniform,
        OutputUniform, scissor_for,
    };
    use gamut_core::mask::{Component, LinearGradient, MaskOp, MaskSource, RadialGradient};
    use gamut_core::{Adjustments, CropRect, Mask};
    use std::mem::offset_of;
    use wgpu::naga;

    /// The size of the struct named `Uniform` in a shader and the offset of
    /// every member, as naga lays them out.
    fn wgsl_uniform(source: &str) -> (u32, Vec<(String, u32)>) {
        wgsl_struct(source, "Uniform")
    }

    /// The same for the struct of any name.
    fn wgsl_struct(source: &str, name: &str) -> (u32, Vec<(String, u32)>) {
        let module = naga::front::wgsl::parse_str(source).expect("the shader parses");
        for (_, ty) in module.types.iter() {
            if ty.name.as_deref() != Some(name) {
                continue;
            }
            if let naga::TypeInner::Struct { members, span } = &ty.inner {
                let offsets = members
                    .iter()
                    .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
                    .collect();
                return (*span, offsets);
            }
        }
        panic!("the shader has no struct named {name}");
    }

    #[test]
    fn the_develop_uniform_matches_the_wgsl_struct() {
        let (size, offsets) = wgsl_uniform(include_str!("shaders/develop.wgsl"));
        assert_eq!(size as usize, size_of::<DevelopUniform>());
        let offset = |name: &str| {
            offsets
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("no member {name}"))
                .1 as usize
        };
        assert_eq!(
            offset("white_balance"),
            offset_of!(DevelopUniform, white_balance)
        );
        assert_eq!(
            offset("white_balance_temperature"),
            offset_of!(DevelopUniform, white_balance_temperature)
        );
        assert_eq!(offset("saturation"), offset_of!(DevelopUniform, saturation));
        assert_eq!(
            offset("texture_amount"),
            offset_of!(DevelopUniform, texture)
        );
        assert_eq!(offset("clarity"), offset_of!(DevelopUniform, clarity));
        assert_eq!(offset("dehaze"), offset_of!(DevelopUniform, dehaze));
        assert_eq!(offset("flags"), offset_of!(DevelopUniform, flags));
        assert_eq!(offset("table_row"), offset_of!(DevelopUniform, table_row));
        assert_eq!(offset("opacity"), offset_of!(DevelopUniform, opacity));
        assert_eq!(offset("atmosphere"), offset_of!(DevelopUniform, atmosphere));
        assert_eq!(offset("cdl_slope"), offset_of!(DevelopUniform, cdl_slope));
        assert_eq!(offset("cdl_offset"), offset_of!(DevelopUniform, cdl_offset));
        assert_eq!(offset("cdl_power"), offset_of!(DevelopUniform, cdl_power));
        assert_eq!(offset("hsl"), offset_of!(DevelopUniform, hsl));
        assert_eq!(offsets.len(), 22, "a member was added without a check here");
        assert_eq!(size, 304, "the two new members took the old padding");
    }

    #[test]
    fn the_output_uniform_matches_the_wgsl_struct() {
        let (size, offsets) = wgsl_uniform(include_str!("shaders/output.wgsl"));
        assert_eq!(size as usize, size_of::<OutputUniform>());
        let expected = [
            ("matrix", offset_of!(OutputUniform, matrix)),
            ("crop", offset_of!(OutputUniform, crop)),
            ("overlay", offset_of!(OutputUniform, overlay)),
        ];
        assert_eq!(offsets.len(), expected.len());
        for ((name, offset), (wanted_name, wanted)) in offsets.iter().zip(expected) {
            assert_eq!(name, wanted_name);
            assert_eq!(*offset as usize, wanted, "{name}");
        }
    }

    #[test]
    fn the_mask_uniform_matches_the_wgsl_struct() {
        let source = include_str!("shaders/mask.wgsl");
        let (size, offsets) = wgsl_uniform(source);
        assert_eq!(size as usize, size_of::<MaskUniform>());
        let expected = [
            ("window", offset_of!(MaskUniform, window)),
            ("render_size", offset_of!(MaskUniform, render_size)),
            ("aspect", offset_of!(MaskUniform, aspect)),
            ("count", offset_of!(MaskUniform, count)),
            ("invert", offset_of!(MaskUniform, invert)),
            ("components", offset_of!(MaskUniform, components)),
        ];
        assert_eq!(offsets.len(), expected.len());
        for ((name, offset), (wanted_name, wanted)) in offsets.iter().zip(expected) {
            assert_eq!(name, wanted_name);
            assert_eq!(*offset as usize, wanted, "{name}");
        }
        let (size, offsets) = wgsl_struct(source, "Component");
        assert_eq!(size as usize, size_of::<MaskComponentUniform>());
        let expected = [
            ("header", offset_of!(MaskComponentUniform, header)),
            ("a", offset_of!(MaskComponentUniform, a)),
            ("b", offset_of!(MaskComponentUniform, b)),
        ];
        assert_eq!(offsets.len(), expected.len());
        for ((name, offset), (wanted_name, wanted)) in offsets.iter().zip(expected) {
            assert_eq!(name, wanted_name);
            assert_eq!(*offset as usize, wanted, "{name}");
        }
    }

    #[test]
    fn a_mask_uniform_carries_what_the_twin_computes_once() {
        let mut mask = Mask::new(
            "Both",
            MaskSource::Radial(RadialGradient {
                centre: [0.4, 0.6],
                radius: [0.3, 0.2],
                rotation: 90.0,
                feather: 25.0,
            }),
        );
        mask.invert = true;
        mask.components.push(Component {
            op: MaskOp::Intersect,
            source: MaskSource::Linear(LinearGradient {
                start: [0.1, 0.2],
                end: [0.3, 0.4],
            }),
            invert: true,
        });
        let geometry = Geometry {
            window: CropRect {
                x: 0.25,
                y: 0.0,
                width: 0.5,
                height: 1.0,
            },
            size: (200, 100),
            photo: (4000, 2000),
        };
        let uniform = MaskUniform::new(&mask, &geometry);
        assert_eq!(uniform.window, [0.25, 0.0, 0.5, 1.0]);
        assert_eq!(uniform.render_size, [200.0, 100.0]);
        assert_eq!(uniform.aspect, [1.0, 0.5]);
        assert_eq!((uniform.count, uniform.invert), (2, 1));
        let radial = uniform.components[0];
        assert_eq!(radial.header, [1, 0, 0, 0]);
        assert_eq!(radial.a, [0.4, 0.6, 0.3, 0.2]);
        assert!((radial.b[0] - 1.0).abs() < 1e-6 && radial.b[1].abs() < 1e-6);
        assert_eq!(radial.b[2], 0.75);
        let linear = uniform.components[1];
        assert_eq!(linear.header, [0, 2, 1, 0]);
        assert_eq!(linear.a, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(uniform.components[2].header, [0; 4]);
    }

    #[test]
    fn a_mask_pass_reads_its_own_row_and_runs_the_curves_of_either_side() {
        let global = Adjustments {
            exposure: 0.5,
            ..Adjustments::default()
        };
        let mut mask = Mask::new("Sky", MaskSource::default());
        mask.adjust.exposure = -1.25;
        mask.opacity = 40.0;
        let uniform = DevelopUniform::for_mask(&global, &mask, 3, [1.0; 3]);
        assert_eq!(uniform.table_row, 4);
        assert_eq!(uniform.opacity, 0.4);
        assert_eq!(uniform.exposure, -0.75, "the mask adds to the global edit");
        assert_eq!(uniform.flags & FLAG_CURVES, 0);
        mask.adjust.look.curves.red.points.insert(1, [0.5, 0.6]);
        let uniform = DevelopUniform::for_mask(&global, &mask, 3, [1.0; 3]);
        assert_eq!(uniform.flags & FLAG_CURVES, FLAG_CURVES);
        let plain = DevelopUniform::new(&global, [1.0; 3]);
        assert_eq!((plain.table_row, plain.opacity), (0, 1.0));
    }

    #[test]
    fn the_minimum_uniform_matches_the_wgsl_struct() {
        let (size, offsets) = wgsl_uniform(include_str!("shaders/minimum.wgsl"));
        assert_eq!(size as usize, size_of::<MinimumUniform>());
        assert_eq!(offsets[3], ("atmosphere".to_string(), 16));
    }

    /// The list after `name` in a shader, as numbers.
    fn wgsl_list(source: &str, name: &str) -> Vec<f32> {
        let start = source.find(name).expect("the constant is in the shader");
        let rest = &source[start..];
        let open = rest.find(">(").expect("a constructor follows") + 2;
        let close = rest[open..].find(')').expect("the constructor closes") + open;
        rest[open..close]
            .split(',')
            .map(|n| n.trim().parse().expect("a number"))
            .collect()
    }

    #[test]
    fn the_shader_hue_centres_equal_the_gamut_color_list() {
        let shader = wgsl_list(
            include_str!("shaders/develop.wgsl"),
            "const HUE_CENTRES: array<f32, 8> =",
        );
        let twin = gamut_color::hue::range_centres();
        assert_eq!(shader.len(), twin.len());
        for (k, (s, t)) in shader.iter().zip(twin).enumerate() {
            assert!(
                (s - t).abs() < 1e-6,
                "centre {k}: shader {s}, gamut-color {t}"
            );
        }
    }

    #[test]
    fn the_shader_constants_equal_the_gamut_color_constants() {
        let source = include_str!("shaders/develop.wgsl");
        let value = |name: &str| -> f32 {
            let start = source
                .find(&format!("const {name}: f32 = "))
                .unwrap_or_else(|| panic!("no constant {name}"));
            let rest = &source[start..];
            let from = rest.find("= ").expect("an equals sign") + 2;
            let to = rest.find(';').expect("a semicolon");
            rest[from..to].trim().parse().expect("a number")
        };
        use gamut_color::{acescct, dehaze, hsl, local, mask, wheels};
        assert_eq!(value("ACES_LINEAR_CUT"), acescct::LINEAR_CUT);
        assert_eq!(value("ACES_ENCODED_CUT"), acescct::ENCODED_CUT);
        assert_eq!(value("ACES_SLOPE"), acescct::SLOPE);
        assert_eq!(value("ACES_OFFSET"), acescct::OFFSET);
        assert_eq!(value("ACES_LOG_SHIFT"), acescct::LOG_SHIFT);
        assert_eq!(value("ACES_LOG_SCALE"), acescct::LOG_SCALE);
        assert_eq!(value("LOCAL_STRENGTH"), local::STRENGTH);
        assert_eq!(value("CLARITY_HALF_WIDTH"), local::CLARITY_HALF_WIDTH);
        assert_eq!(value("TRANSMISSION_FLOOR"), dehaze::TRANSMISSION_FLOOR);
        assert_eq!(value("RECOVERY_FLOOR"), dehaze::RECOVERY_FLOOR);
        assert_eq!(value("CHROMA_FLOOR"), hsl::CHROMA_FLOOR);
        assert_eq!(value("CHROMA_FULL"), hsl::CHROMA_FULL);
        assert_eq!(value("SHADOWS_END"), wheels::SHADOWS_END);
        assert_eq!(value("HIGHLIGHTS_START"), wheels::HIGHLIGHTS_START);

        // mask.wgsl repeats what it needs of acescct.rs and adds mask.rs.
        let source = include_str!("shaders/mask.wgsl");
        let value = |name: &str| -> f32 {
            let start = source
                .find(&format!("const {name}: f32 = "))
                .unwrap_or_else(|| panic!("no constant {name}"));
            let rest = &source[start..];
            let from = rest.find("= ").expect("an equals sign") + 2;
            let to = rest.find(';').expect("a semicolon");
            rest[from..to].trim().parse().expect("a number")
        };
        assert_eq!(value("ACES_LINEAR_CUT"), acescct::LINEAR_CUT);
        assert_eq!(value("ACES_SLOPE"), acescct::SLOPE);
        assert_eq!(value("ACES_OFFSET"), acescct::OFFSET);
        assert_eq!(value("ACES_LOG_SHIFT"), acescct::LOG_SHIFT);
        assert_eq!(value("ACES_LOG_SCALE"), acescct::LOG_SCALE);
        assert_eq!(value("CHROMA_RAMP"), mask::CHROMA_RAMP);
        assert_eq!(value("LINEAR_LENGTH_FLOOR"), mask::LINEAR_LENGTH_FLOOR);
        assert_eq!(value("DEGREES"), 1f32.to_degrees());
    }

    #[test]
    fn the_scissor_covers_the_crop_and_a_margin_inside_the_render() {
        let crop = CropRect {
            x: 0.25,
            y: 0.0,
            width: 0.5,
            height: 1.0,
        };
        assert_eq!(scissor_for(crop, (400, 200), true), (98, 0, 204, 200));
        assert_eq!(scissor_for(crop, (400, 200), false), (98, 0, 204, 200));
        assert_eq!(scissor_for(CropRect::FULL, (64, 64), false), (0, 0, 64, 64));
        let tiny = CropRect {
            x: 0.5,
            y: 0.5,
            width: 0.01,
            height: 0.01,
        };
        let (x, y, w, h) = scissor_for(tiny, (100, 100), false);
        assert!(x <= 50 && y <= 50 && w >= 1 && h >= 1 && x + w <= 100 && y + h <= 100);
    }
}
