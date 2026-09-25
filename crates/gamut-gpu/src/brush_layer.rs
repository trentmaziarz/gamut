//! The painted sources of one mask on the GPU: a layer for each brush
//! component, stamped by instanced dabs and read by `mask.wgsl` as a fifth
//! source.
//!
//! A layer covers the window of the frame it belongs to and is laid out on
//! photo coordinates like every mask alpha. It is a product of that window.
//! A layer with no auto stroke reads no pixel of the source, so it is kept
//! while the frame lives and its strokes are the ones it holds. Only the dabs
//! that touch the window are stamped. A brush that only grew (points added to
//! its last stroke, strokes added after it) is stamped by its new dabs alone,
//! which is exact because dabs build in order.
//!
//! A dab of an auto stroke paints only the colour under its centre. Its
//! fragment stage reads the source pixel of the working texture and its
//! vertex stage the reference, from the proxy of the whole source. A layer
//! that holds such a stroke is a product of the source content too:
//! [`Layers::forget_auto`] has it stamped again after the head passes ran.
//! Dabs with no gate go through the pipelines and the entry points they
//! always did.
//!
//! The layer is a half float, not the r8unorm of a mask alpha: a dab of a low
//! flow adds less than half of an 8 bit code and would stop building. The
//! blend runs in the layer, so the layer holds what is built; `mask.wgsl`
//! stores the mask's alpha as r8unorm like every other source.

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use gamut_color::brush as twin;
use gamut_color::mask::{Geometry, RADIAL_INNER_CEILING};
use gamut_core::brush::Brush;

/// The format of a brush layer.
pub(crate) const LAYER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// The most dabs one draw of a layer stamps; past it the rest is left out
/// with a log line. Four million dabs inside one window is no real brush.
const MAX_LAYER_DABS: usize = 1 << 22;

const QUAD_VERTICES: u32 = 4;

/// Mirrors the `Uniform` of `brush.wgsl`; a test holds the two layouts equal.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct BrushUniform {
    window: [f32; 4],
    render_size: [f32; 2],
    aspect: [f32; 2],
}

impl BrushUniform {
    pub(crate) fn new(geometry: &Geometry) -> Self {
        let window = geometry.window;
        BrushUniform {
            window: [window.x, window.y, window.width, window.height],
            render_size: [geometry.size.0 as f32, geometry.size.1 as f32],
            aspect: geometry.aspect(),
        }
    }
}

/// One dab as `brush.wgsl` reads it: the centre normalised to the photo, then
/// the radius, the inner share where the feather starts and the flow share,
/// then the pass distance of its colour gate, which only the auto entry
/// points read and which is 0 on a dab with no gate.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct DabInstance {
    centre: [f32; 2],
    brush: [f32; 3],
    pass_distance: f32,
}

/// Dabs drawn through one pipeline, in the order they were painted.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Run {
    erase: bool,
    auto: bool,
    range: Range<u32>,
}

/// What the dabs of an auto stroke read: the proxy of the whole source and
/// the working texture of the frame.
#[derive(Clone, Copy)]
pub(crate) struct AutoInputs<'a> {
    pub(crate) proxy: &'a wgpu::TextureView,
    pub(crate) working: &'a wgpu::TextureView,
    /// Counts up when either view is another texture.
    pub(crate) generation: u64,
}

/// Whether a brush holds an auto stroke.
pub(crate) fn has_auto(brush: &Brush) -> bool {
    brush.strokes.iter().any(|stroke| stroke.auto)
}

/// What a draw stamps: the dabs, their runs, and the render pixels they
/// reach (x, y, width, height), or nothing when no dab touches the window.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Stamp {
    dabs: Vec<DabInstance>,
    runs: Vec<Run>,
    pub(crate) reach: Option<(u32, u32, u32, u32)>,
}

/// The dabs of `brush` from dab `first.1` of stroke `first.0` on that touch
/// the window of `geometry`.
pub(crate) fn stamp(brush: &Brush, geometry: &Geometry, first: (usize, usize)) -> Stamp {
    let aspect = geometry.aspect();
    let window = geometry.window;
    let (width, height) = (geometry.size.0 as f32, geometry.size.1 as f32);
    // One render pixel on the photo: the margin the vertex stage adds.
    let pixel = [window.width / width, window.height / height];
    let mut out = Stamp::default();
    let mut bounds: Option<[f32; 4]> = None;
    for (index, stroke) in brush.strokes.iter().enumerate().skip(first.0) {
        let skip = if index == first.0 { first.1 } else { 0 };
        let inner = (1.0 - stroke.feather / 100.0).min(RADIAL_INNER_CEILING);
        let pass_distance = if stroke.auto {
            twin::gate_pass(stroke.sensitivity)
        } else {
            0.0
        };
        let start = out.dabs.len() as u32;
        for placed in twin::placed_dabs(stroke, aspect).into_iter().skip(skip) {
            let centre = placed.centre;
            let (radius, flow) = twin::dab_size_and_flow(stroke, placed.pressure);
            let reach = [radius / aspect[0] + pixel[0], radius / aspect[1] + pixel[1]];
            let (x0, x1) = (centre[0] - reach[0], centre[0] + reach[0]);
            let (y0, y1) = (centre[1] - reach[1], centre[1] + reach[1]);
            let outside = x1 < window.x
                || y1 < window.y
                || x0 > window.x + window.width
                || y0 > window.y + window.height;
            if outside {
                continue;
            }
            if out.dabs.len() >= MAX_LAYER_DABS {
                log::warn!("a brush layer holds {MAX_LAYER_DABS} dabs; the rest is left out");
                break;
            }
            out.dabs.push(DabInstance {
                centre,
                brush: [radius, inner, flow],
                pass_distance,
            });
            bounds = Some(match bounds {
                None => [x0, y0, x1, y1],
                Some(b) => [b[0].min(x0), b[1].min(y0), b[2].max(x1), b[3].max(y1)],
            });
        }
        let end = out.dabs.len() as u32;
        if end > start {
            match out.runs.last_mut() {
                Some(run) if run.erase == stroke.erase && run.auto == stroke.auto => {
                    run.range.end = end
                }
                _ => out.runs.push(Run {
                    erase: stroke.erase,
                    auto: stroke.auto,
                    range: start..end,
                }),
            }
        }
    }
    out.reach = bounds.map(|b| {
        let x0 = ((b[0] - window.x) / pixel[0])
            .floor()
            .clamp(0.0, width - 1.0);
        let y0 = ((b[1] - window.y) / pixel[1])
            .floor()
            .clamp(0.0, height - 1.0);
        let x1 = ((b[2] - window.x) / pixel[0]).ceil().clamp(x0 + 1.0, width);
        let y1 = ((b[3] - window.y) / pixel[1])
            .ceil()
            .clamp(y0 + 1.0, height);
        (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
    });
    out
}

/// What stamping `dab` into a layer of `geometry` costs a slice of a frame
/// built ahead, in texels read and written, as
/// [`crate::develop::AHEAD_SLICE_TEXELS`] counts them: each pixel of the
/// dab's quad inside the frame counts 2 (the blend reads the layer and writes
/// it), and 3 for an auto dab, whose fragment also reads the working texture.
/// The clear of a layer stamped whole counts 1 a pixel of the frame, and is
/// counted by [`Layers::stamp_batch`] with the first batch.
fn dab_texels(dab: &DabInstance, auto: bool, geometry: &Geometry) -> u64 {
    let aspect = geometry.aspect();
    let window = geometry.window;
    let (width, height) = (geometry.size.0 as f32, geometry.size.1 as f32);
    let pixel = [window.width / width, window.height / height];
    let per_pixel = if auto { 3 } else { 2 };
    let radius = dab.brush[0];
    let reach = [radius / aspect[0] + pixel[0], radius / aspect[1] + pixel[1]];
    let span = |centre: f32, reach: f32, start: f32, pixel: f32, size: f32| {
        let low = ((centre - reach - start) / pixel).floor().clamp(0.0, size);
        let high = ((centre + reach - start) / pixel).ceil().clamp(0.0, size);
        (high - low) as u64
    };
    let across = span(dab.centre[0], reach[0], window.x, pixel[0], width);
    let down = span(dab.centre[1], reach[1], window.y, pixel[1], height);
    across * down * per_pixel
}

/// How a brush differs from the one a layer holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Change {
    Same,
    /// Everything held is still there in order; what is new starts at this
    /// dab of this stroke.
    Grown(usize, usize),
    Other,
}

/// `held_last_dabs` is how many dabs the last held stroke had, culled or not.
pub(crate) fn change(held: &Brush, held_last_dabs: usize, now: &Brush) -> Change {
    if held == now {
        return Change::Same;
    }
    let Some((last, before)) = held.strokes.split_last() else {
        return Change::Grown(0, 0);
    };
    let n = before.len();
    let grown = now.strokes.len() > n
        && now.strokes[..n] == *before
        && now.strokes[n].grown_from(last).is_some();
    if grown {
        Change::Grown(n, held_last_dabs)
    } else {
        Change::Other
    }
}

/// What a layer did on a render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Drawn {
    Nothing,
    /// New dabs over what was there, and the render pixels they reach.
    Appended(Option<(u32, u32, u32, u32)>),
    Whole,
}

/// The pipelines of `brush.wgsl` and the uniform they share: paint and erase
/// for a dab with no gate, and the same two through the auto entry points,
/// which also bind the proxy and the working texture.
pub(crate) struct BrushPass {
    paint: wgpu::RenderPipeline,
    erase: wgpu::RenderPipeline,
    auto_paint: wgpu::RenderPipeline,
    auto_erase: wgpu::RenderPipeline,
    auto_layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    written: Option<BrushUniform>,
}

impl BrushPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brush"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brush.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let texture = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let auto_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush auto"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture(1, wgpu::ShaderStages::VERTEX),
                texture(2, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let auto_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush auto"),
            bind_group_layouts: &[Some(&auto_layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3];
        let auto_attributes =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3, 2 => Float32];
        let pipeline = |label: &str, source: wgpu::BlendFactor, auto: bool| {
            // Paint: s + a (1 - s). Erase: a (1 - s).
            let component = wgpu::BlendComponent {
                src_factor: source,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(if auto {
                    &auto_pipeline_layout
                } else {
                    &pipeline_layout
                }),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(if auto { "vs_auto" } else { "vs_main" }),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<DabInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: if auto { &auto_attributes } else { &attributes },
                    })],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if auto { "fs_auto" } else { "fs_main" }),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: LAYER_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: component,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush uniform"),
            size: size_of::<BrushUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brush bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        BrushPass {
            paint: pipeline("brush paint", wgpu::BlendFactor::One, false),
            erase: pipeline("brush erase", wgpu::BlendFactor::Zero, false),
            auto_paint: pipeline("brush auto paint", wgpu::BlendFactor::One, true),
            auto_erase: pipeline("brush auto erase", wgpu::BlendFactor::Zero, true),
            auto_layout,
            uniform,
            bind,
            written: None,
        }
    }

    fn set_geometry(&mut self, queue: &wgpu::Queue, geometry: &Geometry) {
        let uniform = BrushUniform::new(geometry);
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
    }
}

/// The key of a layer: the brush it holds and the dabs of its last stroke.
pub(crate) type LayerHeld = (Brush, usize);

struct Layer {
    view: wgpu::TextureView,
    /// The brush the layer holds and the dabs of its last stroke, or `None`
    /// when nothing was drawn yet.
    held: Option<LayerHeld>,
    dabs: Option<(wgpu::Buffer, u64)>,
    /// A stamp drawn a batch of dabs at a time and not finished, while the
    /// key claims nothing.
    stamping: Option<Stamping>,
}

/// A stamp of a layer of a frame built ahead, drawn a batch of dabs at a
/// time by [`Layers::stamp_batch`], each batch in a submit of its own. The
/// dabs build in order in the layer, so the batches drawn one after another
/// leave it as one draw of every dab does.
struct Stamping {
    /// The key the layer holds once every dab is drawn.
    held: LayerHeld,
    stamp: Stamp,
    /// The layer is cleared before the first dab: the brush is stamped whole
    /// and not by the dabs added since the brush it holds.
    whole: bool,
    /// Whether the first batch was drawn, with the clear of a whole stamp.
    begun: bool,
    /// How many dabs the batches drew, in order.
    drawn: u32,
}

/// What a batch of a stamp drew: the layer's draw once its last dab is
/// drawn and [`Drawn::Nothing`] before, whether it recorded a pass, what it
/// cost in texels read and written, and whether every dab is drawn.
pub(crate) struct Batch {
    pub(crate) drawn: Drawn,
    pub(crate) recorded: bool,
    pub(crate) texels: u64,
    pub(crate) done: bool,
}

/// The layers of the brush components of one mask on one frame: one array
/// texture, a layer for each brush in component order.
pub(crate) struct Layers {
    /// What `mask.wgsl` binds.
    pub(crate) array: wgpu::TextureView,
    layers: Vec<Layer>,
    /// The bind group of the auto pipelines and the generation of the
    /// inputs it holds.
    auto_bind: Option<(wgpu::BindGroup, u64)>,
}

impl Layers {
    /// `count` layers of the size of the frame. With no brush the array is
    /// one texel that is never drawn and reads 0.
    pub(crate) fn new(device: &wgpu::Device, count: usize, width: u32, height: u32) -> Self {
        let (width, height) = if count == 0 { (1, 1) } else { (width, height) };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brush layers"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: count.max(1) as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LAYER_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("brush layers"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layers = (0..count as u32)
            .map(|layer| Layer {
                view: texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("brush layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                }),
                held: None,
                dabs: None,
                stamping: None,
            })
            .collect();
        Layers {
            array,
            layers,
            auto_bind: None,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether layer `index` holds `brush`, so an update would stamp
    /// nothing.
    pub(crate) fn holds(&self, index: usize, brush: &Brush) -> bool {
        self.layers[index]
            .held
            .as_ref()
            .is_some_and(|(held, _)| held == brush)
    }

    /// Takes the key of layer `index` out, so the layer claims nothing
    /// until [`Layers::hold`] sets it again: a frame built ahead stamps a
    /// layer in a slice and sets the key once the slice's commands are
    /// submitted.
    pub(crate) fn take_held(&mut self, index: usize) -> Option<LayerHeld> {
        self.layers[index].held.take()
    }

    /// Sets the key [`Layers::take_held`] took out.
    pub(crate) fn hold(&mut self, index: usize, held: LayerHeld) {
        self.layers[index].held = Some(held);
    }

    /// Forgets what every layer holds and every stamp not finished, so the
    /// next update stamps each layer whole: the frame they belong to renders
    /// another window, where the dabs land on other pixels.
    pub(crate) fn forget(&mut self) {
        for layer in &mut self.layers {
            layer.held = None;
            layer.stamping = None;
        }
    }

    /// Forgets what every layer with an auto stroke holds, so the next
    /// update stamps it whole: its dabs read the source, and the source
    /// content is another.
    pub(crate) fn forget_auto(&mut self) {
        for layer in &mut self.layers {
            if layer
                .held
                .as_ref()
                .is_some_and(|(brush, _)| has_auto(brush))
            {
                layer.held = None;
            }
            // A stamp not finished of such a brush is drawn again whole.
            if layer
                .stamping
                .as_ref()
                .is_some_and(|stamping| has_auto(&stamping.held.0))
            {
                layer.stamping = None;
            }
        }
    }

    /// Brings layer `index` to `brush`: nothing when it holds it, the new
    /// dabs when the brush only grew, everything otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update(
        &mut self,
        index: usize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pass: &mut BrushPass,
        brush: &Brush,
        geometry: &Geometry,
        auto: Option<AutoInputs>,
    ) -> Drawn {
        // A render's update draws the layer whole or by its new dabs; a stamp
        // a frame built ahead left unfinished is drawn again from the start.
        self.layers[index].stamping = None;
        let layer = &mut self.layers[index];
        let how = match &layer.held {
            Some((held, last_dabs)) => change(held, *last_dabs, brush),
            None => Change::Other,
        };
        let first = match how {
            Change::Same => return Drawn::Nothing,
            Change::Grown(stroke, dab) => (stroke, dab),
            Change::Other => (0, 0),
        };
        let stamped = stamp(brush, geometry, first);
        let last_dabs = brush.strokes.last().map_or(0, |stroke| {
            twin::placed_dabs(stroke, geometry.aspect()).len()
        });
        layer.held = Some((brush.clone(), last_dabs));

        let whole = how == Change::Other;
        if !whole && stamped.dabs.is_empty() {
            return Drawn::Appended(None);
        }
        self.upload(index, device, queue, pass, &stamped, geometry, auto);
        let dabs = 0..stamped.dabs.len() as u32;
        self.record(index, encoder, pass, &stamped, dabs, whole);
        if whole {
            Drawn::Whole
        } else {
            Drawn::Appended(stamped.reach)
        }
    }

    /// Brings layer `index` of a frame built ahead to `brush` by a batch of
    /// its dabs: those the texels of `budget` pay for, at least one when
    /// `first` (the slice has drawn nothing yet), counted as
    /// [`dab_texels`] counts them and, before the first dab of a whole stamp,
    /// 1 a pixel of the frame for the clear. The first batch lays the stamp
    /// out as [`update`](Self::update) would draw it, whole or by the dabs
    /// added since, and takes the key out, so the layer claims nothing until
    /// the last batch sets it again; each later batch draws the next dabs in
    /// order over what the batches before left. A brush other than the one
    /// being stamped starts again. `None` when the budget pays for nothing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stamp_batch(
        &mut self,
        index: usize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pass: &mut BrushPass,
        brush: &Brush,
        geometry: &Geometry,
        auto: Option<AutoInputs>,
        budget: u64,
        first: bool,
    ) -> Option<Batch> {
        let layer = &mut self.layers[index];
        if layer
            .stamping
            .as_ref()
            .is_some_and(|stamping| stamping.held.0 != *brush)
        {
            layer.stamping = None;
        }
        if layer.stamping.is_none() {
            let how = match &layer.held {
                Some((held, last_dabs)) => change(held, *last_dabs, brush),
                None => Change::Other,
            };
            let first_dab = match how {
                Change::Same => {
                    return Some(Batch {
                        drawn: Drawn::Nothing,
                        recorded: false,
                        texels: 0,
                        done: true,
                    });
                }
                Change::Grown(stroke, dab) => (stroke, dab),
                Change::Other => (0, 0),
            };
            let stamped = stamp(brush, geometry, first_dab);
            let last_dabs = brush.strokes.last().map_or(0, |stroke| {
                twin::placed_dabs(stroke, geometry.aspect()).len()
            });
            let held = (brush.clone(), last_dabs);
            let whole = how == Change::Other;
            if !whole && stamped.dabs.is_empty() {
                layer.held = Some(held);
                return Some(Batch {
                    drawn: Drawn::Appended(None),
                    recorded: false,
                    texels: 0,
                    done: true,
                });
            }
            // What the layer holds is being drawn again: it claims nothing
            // until the last batch.
            layer.held = None;
            layer.stamping = Some(Stamping {
                held,
                stamp: stamped,
                whole,
                begun: false,
                drawn: 0,
            });
        }
        let stamping = layer.stamping.as_ref().expect("laid out above");
        let total = stamping.stamp.dabs.len() as u32;
        let clear = if stamping.whole && !stamping.begun {
            u64::from(geometry.size.0) * u64::from(geometry.size.1)
        } else {
            0
        };
        if !first && clear > budget {
            return None;
        }
        let auto_at = |dab: u32| {
            stamping
                .stamp
                .runs
                .iter()
                .any(|run| run.auto && run.range.contains(&dab))
        };
        let (mut texels, mut end) = (clear, stamping.drawn);
        while end < total {
            let cost = dab_texels(&stamping.stamp.dabs[end as usize], auto_at(end), geometry);
            // A batch forced to draw draws one dab at least.
            let forced = first && end == stamping.drawn;
            if !forced && texels + cost > budget {
                break;
            }
            texels += cost;
            end += 1;
        }
        if stamping.begun && end == stamping.drawn {
            return None;
        }
        let stamping = self.layers[index].stamping.take().expect("laid out above");
        if stamping.begun {
            // The dabs are in the buffer since the first batch. A render
            // between two batches may have written another geometry, and
            // the inputs of the auto dabs may be others.
            pass.set_geometry(queue, geometry);
            self.auto_inputs(device, pass, &stamping.stamp, auto);
        } else {
            self.upload(index, device, queue, pass, &stamping.stamp, geometry, auto);
        }
        let clear = stamping.whole && !stamping.begun;
        self.record(
            index,
            encoder,
            pass,
            &stamping.stamp,
            stamping.drawn..end,
            clear,
        );
        let done = end == total;
        let layer = &mut self.layers[index];
        let drawn = if !done {
            layer.stamping = Some(Stamping {
                begun: true,
                drawn: end,
                ..stamping
            });
            Drawn::Nothing
        } else {
            layer.held = Some(stamping.held);
            if stamping.whole {
                Drawn::Whole
            } else {
                Drawn::Appended(stamping.stamp.reach)
            }
        };
        Some(Batch {
            drawn,
            recorded: true,
            texels,
            done,
        })
    }

    /// Writes the dabs of `stamped` into the dab buffer of layer `index`,
    /// the geometry into the uniform, and makes the bind group of the auto
    /// pipelines when the stamp holds an auto dab and the inputs changed.
    #[allow(clippy::too_many_arguments)]
    fn upload(
        &mut self,
        index: usize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut BrushPass,
        stamped: &Stamp,
        geometry: &Geometry,
        auto: Option<AutoInputs>,
    ) {
        let layer = &mut self.layers[index];
        let bytes: &[u8] = bytemuck::cast_slice(&stamped.dabs);
        if layer
            .dabs
            .as_ref()
            .is_none_or(|(_, size)| *size < bytes.len() as u64)
        {
            let size = (bytes.len() as u64).next_power_of_two().max(1 << 12);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brush dabs"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            layer.dabs = Some((buffer, size));
        }
        let (buffer, _) = layer.dabs.as_ref().expect("made above");
        queue.write_buffer(buffer, 0, bytes);
        pass.set_geometry(queue, geometry);
        self.auto_inputs(device, pass, stamped, auto);
    }

    /// Makes the bind group of the auto pipelines when `stamped` holds an
    /// auto dab and the one held was made for other inputs.
    fn auto_inputs(
        &mut self,
        device: &wgpu::Device,
        pass: &BrushPass,
        stamped: &Stamp,
        auto: Option<AutoInputs>,
    ) {
        if stamped.runs.iter().any(|run| run.auto) {
            let inputs = auto.expect("a brush with an auto stroke is given its inputs");
            if self
                .auto_bind
                .as_ref()
                .is_none_or(|(_, generation)| *generation != inputs.generation)
            {
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("brush auto bind group"),
                    layout: &pass.auto_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: pass.uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(inputs.proxy),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(inputs.working),
                        },
                    ],
                });
                self.auto_bind = Some((bind, inputs.generation));
            }
        }
    }

    /// Records into layer `index` the dabs `dabs` of `stamped`, which
    /// [`upload`](Self::upload) wrote, in one pass that clears the layer
    /// first when `clear` and draws over what it holds otherwise.
    fn record(
        &self,
        index: usize,
        encoder: &mut wgpu::CommandEncoder,
        pass: &BrushPass,
        stamped: &Stamp,
        dabs: Range<u32>,
        clear: bool,
    ) {
        let layer = &self.layers[index];
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };
        let mut render = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("brush layer"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &layer.view,
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
        if !stamped.dabs.is_empty() {
            let (buffer, _) = layer.dabs.as_ref().expect("uploaded");
            let bytes = std::mem::size_of_val(stamped.dabs.as_slice()) as u64;
            render.set_vertex_buffer(0, buffer.slice(..bytes));
            for run in &stamped.runs {
                // The dabs of this run inside the batch.
                let range = run.range.start.max(dabs.start)..run.range.end.min(dabs.end);
                if range.is_empty() {
                    continue;
                }
                let (pipeline, bind) = match (run.auto, run.erase) {
                    (false, false) => (&pass.paint, &pass.bind),
                    (false, true) => (&pass.erase, &pass.bind),
                    (true, erase) => {
                        let (bind, _) = self.auto_bind.as_ref().expect("made above");
                        let pipeline = if erase {
                            &pass.auto_erase
                        } else {
                            &pass.auto_paint
                        };
                        (pipeline, bind)
                    }
                };
                render.set_pipeline(pipeline);
                render.set_bind_group(0, bind, &[]);
                render.draw(0..QUAD_VERTICES, range);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamut_core::CropRect;
    use gamut_core::brush::{SharedStroke, Stroke};

    fn stroke(points: &[[f32; 2]], size: f32, erase: bool) -> SharedStroke {
        SharedStroke::new(&Stroke {
            points: points.to_vec(),
            size,
            feather: 40.0,
            flow: 50.0,
            erase,
            ..Stroke::default()
        })
    }

    fn zoomed() -> Geometry {
        Geometry {
            window: CropRect {
                x: 0.5,
                y: 0.25,
                width: 0.25,
                height: 0.25,
            },
            size: (200, 100),
            photo: (800, 400),
        }
    }

    #[test]
    fn the_brush_uniform_matches_the_wgsl_struct() {
        // window vec4 at 0, render_size vec2 at 16, aspect vec2 at 24.
        assert_eq!(size_of::<BrushUniform>(), 32);
        assert_eq!(std::mem::offset_of!(BrushUniform, render_size), 16);
        assert_eq!(std::mem::offset_of!(BrushUniform, aspect), 24);
        // centre at location 0, brush at location 1, and for the auto entry
        // points the pass distance at location 2.
        assert_eq!(size_of::<DabInstance>(), 24);
        assert_eq!(std::mem::offset_of!(DabInstance, brush), 8);
        assert_eq!(std::mem::offset_of!(DabInstance, pass_distance), 20);
    }

    #[test]
    fn only_the_dabs_that_touch_the_window_are_stamped() {
        // A line across the whole photo at a radius of 0.01: a dab every
        // 0.0025, and the window shows the quarter from 0.5 to 0.75.
        let brush = Brush {
            strokes: vec![stroke(&[[0.0, 0.375], [1.0, 0.375]], 0.01, false)],
        };
        let all = twin::dab_centres(&brush.strokes[0], zoomed().aspect()).len();
        let stamped = stamp(&brush, &zoomed(), (0, 0));
        assert!(
            all >= 400 && stamped.dabs.len() < all / 3,
            "{all} {}",
            stamped.dabs.len()
        );
        assert!(stamped.dabs.len() > all / 4, "the quarter and its rim");
        let reach = 0.01 + 0.25 / 200.0;
        assert!(
            stamped
                .dabs
                .iter()
                .all(|d| d.centre[0] > 0.5 - reach - 1e-6)
        );
        assert!(
            stamped
                .dabs
                .iter()
                .all(|d| d.centre[0] < 0.75 + reach + 1e-6)
        );
        assert_eq!(stamped.dabs[0].brush, [0.01, 0.6, 0.5]);
        // The whole width of the window, and the height of the line.
        let (x, y, w, h) = stamped.reach.expect("dabs");
        assert_eq!((x, w), (0, 200));
        assert!(y >= 35 && y + h <= 65 && h >= 18, "{y} {h}");

        let far = Brush {
            strokes: vec![stroke(&[[0.1, 0.1], [0.3, 0.9]], 0.01, false)],
        };
        assert_eq!(stamp(&far, &zoomed(), (0, 0)), Stamp::default());
    }

    #[test]
    fn strokes_are_drawn_in_order_in_runs_of_one_blend() {
        let brush = Brush {
            strokes: vec![
                stroke(&[[0.6, 0.3]], 0.02, false),
                stroke(&[[0.6, 0.4], [0.621, 0.4]], 0.02, false),
                stroke(&[[0.1, 0.1]], 0.02, true),
                stroke(&[[0.6, 0.3]], 0.02, true),
                stroke(&[[0.7, 0.3]], 0.02, false),
            ],
        };
        let stamped = stamp(&brush, &zoomed(), (0, 0));
        let runs: Vec<(bool, u32, u32)> = stamped
            .runs
            .iter()
            .map(|r| (r.erase, r.range.start, r.range.end))
            .collect();
        // 1 dab, then 5 (0.021 long at 0.005 apart), the far eraser culled.
        assert_eq!(runs, [(false, 0, 6), (true, 6, 7), (false, 7, 8)]);
        // From the third dab of the second stroke on.
        let later = stamp(&brush, &zoomed(), (1, 2));
        assert_eq!(later.dabs[..], stamped.dabs[3..]);
        assert_eq!(later.runs[0].range, 0..3);
    }

    #[test]
    fn an_auto_stroke_is_a_run_of_its_own_and_carries_its_pass_distance() {
        let auto = |erase: bool, sensitivity: f32| {
            SharedStroke::new(&Stroke {
                auto: true,
                sensitivity,
                ..(*stroke(&[[0.6, 0.3]], 0.02, erase)).clone()
            })
        };
        let brush = Brush {
            strokes: vec![
                stroke(&[[0.6, 0.3]], 0.02, false),
                auto(false, 0.0),
                auto(false, 100.0),
                auto(true, 50.0),
                stroke(&[[0.6, 0.3]], 0.02, true),
            ],
        };
        assert!(has_auto(&brush));
        assert!(!has_auto(&Brush {
            strokes: vec![stroke(&[[0.6, 0.3]], 0.02, false)],
        }));
        let stamped = stamp(&brush, &zoomed(), (0, 0));
        let runs: Vec<(bool, bool, u32, u32)> = stamped
            .runs
            .iter()
            .map(|r| (r.auto, r.erase, r.range.start, r.range.end))
            .collect();
        assert_eq!(
            runs,
            [
                (false, false, 0, 1),
                (true, false, 1, 3),
                (true, true, 3, 4),
                (false, true, 4, 5),
            ]
        );
        let distances: Vec<f32> = stamped.dabs.iter().map(|d| d.pass_distance).collect();
        assert_eq!(
            distances,
            [
                0.0,
                twin::GATE_PASS_LOOSE,
                twin::gate_pass(100.0),
                twin::gate_pass(50.0),
                0.0
            ]
        );
        assert!((twin::gate_pass(100.0) - twin::GATE_PASS_STRICT).abs() < 1e-6);
    }

    #[test]
    fn pressure_scales_the_radius_and_the_flow_of_a_dab_and_what_it_reaches() {
        let pen = |size: bool, flow: bool| Brush {
            strokes: vec![SharedStroke::new(&Stroke {
                pressure: vec![0.5],
                pressure_size: size,
                pressure_flow: flow,
                ..(*stroke(&[[0.6, 0.3]], 0.02, false)).clone()
            })],
        };
        let plain = stamp(&pen(false, false), &zoomed(), (0, 0));
        assert_eq!(plain.dabs[0].brush, [0.02, 0.6, 0.5]);
        let lighter = stamp(&pen(false, true), &zoomed(), (0, 0));
        assert_eq!(lighter.dabs[0].brush, [0.02, 0.6, 0.25]);
        assert_eq!(lighter.reach, plain.reach);
        let smaller = stamp(&pen(true, false), &zoomed(), (0, 0));
        assert_eq!(smaller.dabs[0].brush, [0.02 * 0.6, 0.6, 0.5]);
        let (wide, narrow) = (plain.reach.expect("a dab"), smaller.reach.expect("a dab"));
        assert!(
            narrow.2 < wide.2 && narrow.3 < wide.3,
            "{narrow:?} {wide:?}"
        );
    }

    #[test]
    fn a_brush_that_only_grew_is_told_from_one_that_changed() {
        let first = stroke(&[[0.6, 0.3], [0.62, 0.3]], 0.02, false);
        let held = Brush {
            strokes: vec![stroke(&[[0.55, 0.3]], 0.02, false), first.clone()],
        };
        assert_eq!(change(&held, 5, &held.clone()), Change::Same);
        assert_eq!(change(&Brush::default(), 0, &held), Change::Grown(0, 0));

        let mut longer = held.clone();
        longer.strokes[1].push([0.64, 0.3], None);
        assert_eq!(change(&held, 5, &longer), Change::Grown(1, 5));
        longer.strokes.push(stroke(&[[0.7, 0.3]], 0.01, true));
        assert_eq!(change(&held, 5, &longer), Change::Grown(1, 5));
        let mut added = held.clone();
        added.strokes.push(stroke(&[[0.7, 0.3]], 0.01, true));
        assert_eq!(change(&held, 5, &added), Change::Grown(1, 5));

        // An undo, a cleared brush and a changed earlier stroke start again.
        let mut undone = held.clone();
        undone.strokes.pop();
        assert_eq!(change(&held, 5, &undone), Change::Other);
        assert_eq!(change(&held, 5, &Brush::default()), Change::Other);
        let mut earlier = longer.clone();
        earlier.strokes[0] = stroke(&[[0.56, 0.3]], 0.02, false);
        assert_eq!(change(&held, 5, &earlier), Change::Other);
        let mut other_brush = held.clone();
        other_brush.strokes[1] = stroke(&[[0.6, 0.3], [0.62, 0.3], [0.64, 0.3]], 0.03, false);
        assert_eq!(change(&held, 5, &other_brush), Change::Other);
    }
}
