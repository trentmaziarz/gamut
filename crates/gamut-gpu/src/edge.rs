//! The edge controls on the GPU: the pass family of `edge.wgsl`, the targets
//! it works in, and the finished alpha of one mask.
//!
//! The arithmetic and the plan of a render (the runs of the octagon, the
//! cells and the kernel of Feather, the gain of Contrast) are gamut-color's
//! [`Plan`]; this module only lays the work out.
//!
//! Each control that is on keeps its output in its own texture of the mask,
//! so new dabs of a brush redraw each stage over its own reach only: the
//! shifted alpha (r8unorm, the frame's size), the blurred cells of Feather
//! (r32float, the cell grid's size) and the finished alpha (r8unorm, the
//! frame's size), which Feather and Contrast write in one pass. The runs of
//! Shift edge work through three r8unorm textures of the frame's size and
//! Feather through two r32float cell targets, all shared by the masks of the
//! frame and held only while a mask needs them. The shifted alpha is written
//! by the last pass of the last run alone and is never a place the runs work
//! in, so what it holds outside the pixels a pass redraws stays valid.

use bytemuck::{Pod, Zeroable};
use gamut_color::edge::{Plan, Run, doubling};

use crate::{FULLSCREEN_VERTICES, fullscreen_primitive};

/// The format of the alphas: the input, the runs, the shifted and the
/// finished alpha.
pub(crate) const RUN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// The format of the cells of Feather.
pub(crate) const CELL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// A rectangle of pixels or of cells: x, y, width, height.
pub(crate) type Rect = (u32, u32, u32, u32);

/// Mirrors the `Uniform` of `edge.wgsl`; a test holds the two layouts equal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct EdgeUniform {
    pub(crate) direction: [i32; 2],
    pub(crate) offset: i32,
    pub(crate) grow: u32,
    pub(crate) origin: [u32; 2],
    pub(crate) size: [u32; 2],
    pub(crate) grid_first: [u32; 2],
    pub(crate) grid_count: [u32; 2],
    pub(crate) step: u32,
    pub(crate) radius: i32,
    pub(crate) sigma: f32,
    pub(crate) feathered: u32,
    pub(crate) contrast: f32,
    pub(crate) gain: f32,
}

impl EdgeUniform {
    /// The numbers every pass of a plan shares.
    fn of(plan: &Plan) -> Self {
        let (grid_first, grid_count) = plan.grid();
        EdgeUniform {
            direction: [0, 0],
            offset: 0,
            grow: u32::from(plan.grow),
            origin: [plan.origin.0, plan.origin.1],
            size: [plan.size.0, plan.size.1],
            grid_first: [grid_first.0, grid_first.1],
            grid_count: [grid_count.0, grid_count.1],
            step: plan.step,
            radius: plan.radius_cells as i32,
            sigma: plan.sigma_cells,
            feathered: u32::from(plan.feathers()),
            contrast: plan.contrast,
            gain: plan.gain,
        }
    }
}

/// Where a pass of Shift edge reads and writes: the input of the chain, the
/// three textures the runs work in, or the shifted alpha.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    Input,
    Work(usize),
    Shifted,
}

/// One pass of Shift edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RunPass {
    /// The run it belongs to, by its place among the plan's runs.
    pub(crate) run: usize,
    pub(crate) direction: (i32, i32),
    pub(crate) offset: u32,
    /// The last backward pass: it takes the forward run in from `other`.
    pub(crate) last: bool,
    pub(crate) read: Place,
    pub(crate) other: Option<Place>,
    pub(crate) write: Place,
}

/// The passes of Shift edge for these runs: each run forward by doubling,
/// then backward by doubling, whose last pass takes the forward run in. The
/// first run reads the input; each later one reads the run before. Only the
/// last pass of the last run writes the shifted alpha.
pub(crate) fn schedule(runs: &[Run]) -> Vec<RunPass> {
    let mut passes = Vec::new();
    let mut input = Place::Input;
    for (index, run) in runs.iter().enumerate() {
        let offsets = doubling(run.half);
        let (dx, dy) = run.direction;
        // The work textures the input of this run is not in.
        let free: Vec<usize> = (0..3).filter(|w| input != Place::Work(*w)).collect();
        // Forward: back and forth between the first two free textures.
        let mut read = input;
        let mut forward = Place::Input;
        for (k, offset) in offsets.iter().enumerate() {
            let write = Place::Work(free[k % 2]);
            passes.push(RunPass {
                run: index,
                direction: (dx, dy),
                offset: *offset,
                last: false,
                read,
                other: None,
                write,
            });
            read = write;
            forward = write;
        }
        // Backward: the two work textures the forward run is not in. The
        // input is read by the first pass only, so it may be written after.
        let Place::Work(f) = forward else {
            unreachable!("a run takes a step")
        };
        let others: Vec<usize> = (0..3)
            .filter(|w| *w != f && input != Place::Work(*w))
            .collect();
        let spare = match input {
            Place::Work(w) => w,
            _ => others[1],
        };
        let pair = [others[0], spare];
        let mut read = input;
        for (k, offset) in offsets.iter().enumerate() {
            let last = k + 1 == offsets.len();
            let write = if last && index + 1 == runs.len() {
                Place::Shifted
            } else {
                Place::Work(pair[k % 2])
            };
            passes.push(RunPass {
                run: index,
                direction: (-dx, -dy),
                offset: *offset,
                last,
                read,
                other: last.then_some(forward),
                write,
            });
            read = write;
        }
        input = read;
    }
    passes
}

struct Texture {
    view: wgpu::TextureView,
    size: (u32, u32),
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    size: (u32, u32),
) -> Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    Texture {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        size,
    }
}

/// The textures of a frame the edge passes work in, shared by its masks.
#[derive(Default)]
pub(crate) struct EdgeScratch {
    /// The three r8unorm textures the runs of Shift edge work in.
    runs: Option<[Texture; 3]>,
    /// The means of the cells and the blur across, r32float.
    cells: Option<[Texture; 2]>,
}

impl EdgeScratch {
    /// Drops what no mask of the frame needs.
    pub(crate) fn keep(&mut self, runs: bool, cells: bool) {
        if !runs {
            self.runs = None;
        }
        if !cells {
            self.cells = None;
        }
    }

    /// How many textures it holds.
    pub(crate) fn held(&self) -> usize {
        self.runs.as_ref().map_or(0, |r| r.len()) + self.cells.as_ref().map_or(0, |c| c.len())
    }
}

/// The controls the products of a mask hold, and what they were made from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Held {
    pub(crate) plan: Plan,
    /// The input was the refined alpha (and not the alpha of the components).
    pub(crate) refined: bool,
}

/// The edge products of one mask and what draws them.
pub(crate) struct Edged {
    shifted: Option<Texture>,
    /// The blurred cells of Feather.
    cells: Option<Texture>,
    finished: Option<Texture>,
    uniform: wgpu::Buffer,
    /// How many passes the uniform buffer holds.
    slots: u32,
    /// What the products hold, or `None` when they must be drawn whole.
    pub(crate) held: Option<Held>,
}

impl Edged {
    /// The alpha the blend and the overlay read: the finished alpha while
    /// Feather or Contrast is on, the shifted alpha while Shift edge alone
    /// is, or `None` while all three are at rest.
    pub(crate) fn product(&self) -> Option<&wgpu::TextureView> {
        self.finished
            .as_ref()
            .or(self.shifted.as_ref())
            .map(|t| &t.view)
    }

    /// Whether it holds a shifted alpha: Shift edge is on.
    pub(crate) fn shifts(&self) -> bool {
        self.shifted.is_some()
    }

    /// Whether it holds the cells of Feather: Feather is on.
    pub(crate) fn feathers(&self) -> bool {
        self.cells.is_some()
    }

    /// How many textures it holds.
    pub(crate) fn held_textures(&self) -> usize {
        [&self.shifted, &self.cells, &self.finished]
            .iter()
            .filter(|t| t.is_some())
            .count()
    }
}

/// What one render redraws of the edge products of a mask. Each is a
/// rectangle of the render, or `None` to leave that product as it is.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EdgeWork {
    /// The pixels of the shifted alpha to draw again.
    pub(crate) shift: Option<Rect>,
    /// The pixels whose cells Feather draws again.
    pub(crate) feather: Option<Rect>,
    /// The pixels of the finished alpha to draw again.
    pub(crate) finish: Option<Rect>,
}

/// The products of a mask [`EdgePass::prepare`] made again: each holds
/// nothing yet and is drawn whole.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Made {
    pub(crate) shifted: bool,
    pub(crate) cells: bool,
    pub(crate) finished: bool,
}

/// How many passes of each kind an edge pass family has drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgePasses {
    pub shift: u64,
    pub feather: u64,
    pub finish: u64,
}

/// The pipelines of `edge.wgsl`.
pub(crate) struct EdgePass {
    layout: wgpu::BindGroupLayout,
    run: wgpu::RenderPipeline,
    run_last: wgpu::RenderPipeline,
    cells: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    finish: wgpu::RenderPipeline,
    /// The distance between two passes in a uniform buffer.
    stride: u32,
    pub(crate) passes: EdgePasses,
}

pub(crate) const SHADER: &str = include_str!("shaders/edge.wgsl");

/// `rect` with `x` more pixels left and right and `y` more above and below,
/// inside a render of `size`.
fn grow_xy(rect: Rect, x: u32, y: u32, size: (u32, u32)) -> Rect {
    let (x0, y0) = (rect.0.saturating_sub(x), rect.1.saturating_sub(y));
    let x1 = (rect.0 + rect.2 + x).min(size.0).max(x0);
    let y1 = (rect.1 + rect.3 + y).min(size.1).max(y0);
    (x0, y0, x1 - x0, y1 - y0)
}

/// The cells, as texels of a cell target, that the pixels of `over` read
/// bilinearly: the cell before the first pixel's centre to the one after the
/// last's, inside the grid.
fn cells_under(plan: &Plan, over: Rect) -> Rect {
    let ((first_x, first_y), (columns, rows)) = plan.grid();
    let axis = |origin: u32, start: u32, length: u32, first: u32, count: u32| {
        let step = plan.step;
        // The twin's floor((X + 0.5) / step - 0.5), held at the grid.
        let before = |pixel: u32| (2 * (origin + pixel) + 1).saturating_sub(step) / (2 * step);
        let low = before(start).max(first) - first;
        let high = (before(start + length.max(1) - 1) + 1).min(first + count - 1) - first;
        (low, high.max(low) - low + 1)
    };
    let (x, w) = axis(plan.origin.0, over.0, over.2, first_x, columns);
    let (y, h) = axis(plan.origin.1, over.1, over.3, first_y, rows);
    (x, y, w, h)
}

impl EdgePass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edge"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let size = std::mem::size_of::<EdgeUniform>() as u64;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("edge"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(size),
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("edge"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |fragment: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
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
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        EdgePass {
            run: pipeline("fs_run", RUN_FORMAT),
            run_last: pipeline("fs_run_last", RUN_FORMAT),
            cells: pipeline("fs_cells", CELL_FORMAT),
            blur: pipeline("fs_blur", CELL_FORMAT),
            finish: pipeline("fs_finish", RUN_FORMAT),
            layout,
            stride: (size as u32).div_ceil(alignment) * alignment,
            passes: EdgePasses::default(),
        }
    }

    /// The edge products of a mask, none drawn yet.
    pub(crate) fn edged(&self, device: &wgpu::Device) -> Edged {
        Edged {
            shifted: None,
            cells: None,
            finished: None,
            uniform: self.uniform_buffer(device, 1),
            slots: 1,
            held: None,
        }
    }

    fn uniform_buffer(&self, device: &wgpu::Device, slots: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge uniforms"),
            size: u64::from(self.stride) * u64::from(slots),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Makes and drops the textures of a mask and of the frame for `plan`:
    /// each product while its stage is on and none while it is at rest.
    /// Returns the products it made again, which hold nothing yet. The
    /// others keep what they hold: new cells of Feather leave the shifted
    /// alpha as it is.
    pub(crate) fn prepare(
        &self,
        device: &wgpu::Device,
        scratch: &mut EdgeScratch,
        edged: &mut Edged,
        plan: &Plan,
    ) -> Made {
        let frame = plan.size;
        let (_, grid) = plan.grid();
        let mut made = Made::default();
        if plan.shifts() {
            if edged.shifted.as_ref().is_none_or(|t| t.size != frame) {
                edged.shifted = Some(texture(device, "shifted alpha", RUN_FORMAT, frame));
                made.shifted = true;
            }
            if scratch.runs.as_ref().is_none_or(|r| r[0].size != frame) {
                scratch.runs =
                    Some([0; 3].map(|_| texture(device, "edge runs", RUN_FORMAT, frame)));
            }
        } else {
            edged.shifted = None;
        }
        if plan.feathers() {
            let fits = |t: &Texture| t.size.0 >= grid.0 && t.size.1 >= grid.1;
            if edged.cells.as_ref().is_none_or(|t| !fits(t)) {
                edged.cells = Some(texture(device, "feather cells", CELL_FORMAT, grid));
                made.cells = true;
            }
            if scratch.cells.as_ref().is_none_or(|c| !fits(&c[0])) {
                scratch.cells =
                    Some([0; 2].map(|_| texture(device, "feather scratch", CELL_FORMAT, grid)));
            }
        } else {
            edged.cells = None;
        }
        if plan.feathers() || plan.contrasts() {
            if edged.finished.as_ref().is_none_or(|t| t.size != frame) {
                edged.finished = Some(texture(device, "finished alpha", RUN_FORMAT, frame));
                made.finished = true;
            }
        } else {
            edged.finished = None;
        }
        made
    }

    /// Draws what `work` asks of the edge products of one mask from `input`,
    /// the alpha Refine edges hands on. [`EdgePass::prepare`] ran for the
    /// same plan.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scratch: &EdgeScratch,
        edged: &mut Edged,
        input: &wgpu::TextureView,
        plan: &Plan,
        work: &EdgeWork,
    ) {
        let frame = plan.size;
        let base = EdgeUniform::of(plan);
        // Every pass: its pipeline, its uniform, what it reads (source,
        // other, cells), what it writes, and where.
        struct Draw<'a> {
            kind: Kind,
            uniform: EdgeUniform,
            source: &'a wgpu::TextureView,
            other: &'a wgpu::TextureView,
            cells: &'a wgpu::TextureView,
            target: &'a wgpu::TextureView,
            area: Rect,
        }
        #[derive(Clone, Copy)]
        enum Kind {
            Run,
            RunLast,
            Cells,
            Blur,
            Finish,
        }
        let mut draws: Vec<Draw> = Vec::new();
        let mut counts = EdgePasses::default();
        let shifted = edged.shifted.as_ref().map(|t| &t.view);
        // What Feather and the finished alpha read: the shifted alpha while
        // Shift edge is on, the input otherwise.
        let stage_input = shifted.unwrap_or(input);
        if let (Some(out), Some(shifted), Some(runs)) = (work.shift, shifted, scratch.runs.as_ref())
        {
            let runs_of_plan = plan.runs();
            // The pixels each run must be right over, from the last back:
            // the last over `out`, each earlier one over what the one after
            // it reads.
            let mut rights = vec![out; runs_of_plan.len()];
            for k in (1..runs_of_plan.len()).rev() {
                let run = runs_of_plan[k];
                let (dx, dy) = run.direction;
                rights[k - 1] = grow_xy(
                    rights[k],
                    run.half * dx.unsigned_abs(),
                    run.half * dy.unsigned_abs(),
                    frame,
                );
            }
            let view = |place: Place| match place {
                Place::Input => input,
                Place::Work(w) => &runs[w].view,
                Place::Shifted => shifted,
            };
            for pass in schedule(&runs_of_plan) {
                let run = runs_of_plan[pass.run];
                let (dx, dy) = run.direction;
                // A pass is right where every sample the run takes for the
                // pixels after it is; the pass that writes the shifted alpha
                // writes the pixels asked for and no more.
                let area = if pass.write == Place::Shifted {
                    rights[pass.run]
                } else {
                    grow_xy(
                        rights[pass.run],
                        run.half * dx.unsigned_abs(),
                        run.half * dy.unsigned_abs(),
                        frame,
                    )
                };
                draws.push(Draw {
                    kind: if pass.last { Kind::RunLast } else { Kind::Run },
                    uniform: EdgeUniform {
                        direction: [pass.direction.0, pass.direction.1],
                        offset: pass.offset as i32,
                        ..base
                    },
                    source: view(pass.read),
                    other: view(pass.other.unwrap_or(pass.read)),
                    cells: view(pass.read),
                    target: view(pass.write),
                    area,
                });
                counts.shift += 1;
            }
        }
        if let (Some(over), Some(held), Some(spare)) =
            (work.feather, edged.cells.as_ref(), scratch.cells.as_ref())
        {
            let (_, grid) = plan.grid();
            let radius = plan.radius_cells;
            // The cells the pixels read, their blur down, and across.
            let down = cells_under(plan, over);
            let across = grow_xy(down, 0, radius, grid);
            let means = grow_xy(down, radius, radius, grid);
            draws.push(Draw {
                kind: Kind::Cells,
                uniform: base,
                source: stage_input,
                other: stage_input,
                cells: stage_input,
                target: &spare[0].view,
                area: means,
            });
            draws.push(Draw {
                kind: Kind::Blur,
                uniform: EdgeUniform {
                    direction: [1, 0],
                    ..base
                },
                source: stage_input,
                other: stage_input,
                cells: &spare[0].view,
                target: &spare[1].view,
                area: across,
            });
            draws.push(Draw {
                kind: Kind::Blur,
                uniform: EdgeUniform {
                    direction: [0, 1],
                    ..base
                },
                source: stage_input,
                other: stage_input,
                cells: &spare[1].view,
                target: &held.view,
                area: down,
            });
            counts.feather += 3;
        }
        if let (Some(over), Some(finished)) = (work.finish, edged.finished.as_ref()) {
            let cells = edged.cells.as_ref().map_or(stage_input, |t| &t.view);
            draws.push(Draw {
                kind: Kind::Finish,
                uniform: base,
                source: stage_input,
                other: stage_input,
                cells,
                target: &finished.view,
                area: over,
            });
            counts.finish += 1;
        }
        if draws.is_empty() {
            return;
        }
        if edged.slots < draws.len() as u32 {
            edged.slots = (draws.len() as u32).next_power_of_two();
            edged.uniform = self.uniform_buffer(device, edged.slots);
        }
        // One bind group for each set of textures a pass reads.
        let mut groups: Vec<([usize; 3], wgpu::BindGroup)> = Vec::new();
        let key =
            |d: &Draw| [d.source, d.other, d.cells].map(|v| v as *const wgpu::TextureView as usize);
        for d in &draws {
            let k = key(d);
            if groups.iter().any(|(held, _)| *held == k) {
                continue;
            }
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("edge"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &edged.uniform,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<EdgeUniform>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(d.source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(d.other),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(d.cells),
                    },
                ],
            });
            groups.push((k, group));
        }
        for (slot, d) in draws.iter().enumerate() {
            let offset = slot as u32 * self.stride;
            queue.write_buffer(
                &edged.uniform,
                u64::from(offset),
                bytemuck::bytes_of(&d.uniform),
            );
            let pipeline = match d.kind {
                Kind::Run => &self.run,
                Kind::RunLast => &self.run_last,
                Kind::Cells => &self.cells,
                Kind::Blur => &self.blur,
                Kind::Finish => &self.finish,
            };
            let k = key(d);
            let group = &groups
                .iter()
                .find(|(held, _)| *held == k)
                .expect("made above")
                .1;
            if d.area.2 == 0 || d.area.3 == 0 {
                continue;
            }
            draw(encoder, pipeline, group, offset, d.target, d.area);
        }
        self.passes.shift += counts.shift;
        self.passes.feather += counts.feather;
        self.passes.finish += counts.finish;
    }
}

/// One pass over `area` of its target, which keeps what it holds outside it.
/// Every pass writes every pixel of its area.
fn draw(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind: &wgpu::BindGroup,
    offset: u32,
    target: &wgpu::TextureView,
    area: Rect,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("edge"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[offset]);
    pass.set_scissor_rect(area.0, area.1, area.2, area.3);
    pass.draw(0..FULLSCREEN_VERTICES, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamut_color::edge::{self as twin, CONTRAST_MIDDLE};
    use gamut_core::mask::{Edge, MAX_EDGE_CONTRAST};
    use std::mem::offset_of;
    use wgpu::naga;

    #[test]
    fn the_edge_uniform_matches_the_wgsl_struct() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("the shader parses");
        let (span, members) = module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { members, span }
                    if ty.name.as_deref() == Some("Uniform") =>
                {
                    Some((*span, members.clone()))
                }
                _ => None,
            })
            .expect("the shader has a Uniform");
        assert_eq!(span as usize, size_of::<EdgeUniform>());
        let offset = |name: &str| {
            members
                .iter()
                .find(|m| m.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("no member {name}"))
                .offset as usize
        };
        assert_eq!(offset("direction"), offset_of!(EdgeUniform, direction));
        assert_eq!(offset("offset"), offset_of!(EdgeUniform, offset));
        assert_eq!(offset("grow"), offset_of!(EdgeUniform, grow));
        assert_eq!(offset("origin"), offset_of!(EdgeUniform, origin));
        assert_eq!(offset("size"), offset_of!(EdgeUniform, size));
        assert_eq!(offset("grid_first"), offset_of!(EdgeUniform, grid_first));
        assert_eq!(offset("grid_count"), offset_of!(EdgeUniform, grid_count));
        assert_eq!(offset("step"), offset_of!(EdgeUniform, step));
        assert_eq!(offset("radius"), offset_of!(EdgeUniform, radius));
        assert_eq!(offset("sigma"), offset_of!(EdgeUniform, sigma));
        assert_eq!(offset("feathered"), offset_of!(EdgeUniform, feathered));
        assert_eq!(offset("contrast"), offset_of!(EdgeUniform, contrast));
        assert_eq!(offset("gain"), offset_of!(EdgeUniform, gain));
        assert_eq!(members.len(), 13);
    }

    /// The value of `const NAME: f32 = value;` in the shader.
    fn wgsl_constant(name: &str) -> f32 {
        let start = SHADER
            .find(&format!("const {name}: f32 = "))
            .unwrap_or_else(|| panic!("no constant {name}"));
        let text = &SHADER[start..];
        let value = &text[text.find("= ").expect("an equals sign") + 2..];
        value[..value.find(';').expect("a semicolon")]
            .parse()
            .expect("a number")
    }

    #[test]
    fn the_shader_constants_equal_the_gamut_color_constants() {
        assert_eq!(wgsl_constant("CONTRAST_MIDDLE"), CONTRAST_MIDDLE);
        assert_eq!(wgsl_constant("MAX_CONTRAST"), MAX_EDGE_CONTRAST);
        // The runs of the octagon, their widths, the cells and the kernel are
        // the plan's: the shader holds no number of its own for them.
        for name in ["OCTAGON", "0.398", "0.281", "FEATHER"] {
            assert!(!SHADER.contains(name), "{name}");
        }
        // The cells are 32 bit floats and every alpha 8 bits.
        assert_eq!(CELL_FORMAT.target_pixel_byte_cost(), Some(4));
        assert_eq!(RUN_FORMAT.target_pixel_byte_cost(), Some(1));
    }

    #[test]
    fn the_uniform_carries_the_plan() {
        let edge = Edge {
            shift: -0.01,
            feather: 0.02,
            contrast: 80.0,
        };
        let plan = twin::Plan::new(&edge, (6000, 4000), (1021, 513), (900, 700));
        let uniform = EdgeUniform::of(&plan);
        let (first, count) = plan.grid();
        assert_eq!(uniform.grow, 0);
        assert_eq!(uniform.origin, [1021, 513]);
        assert_eq!(uniform.size, [900, 700]);
        assert_eq!(uniform.grid_first, [first.0, first.1]);
        assert_eq!(uniform.grid_count, [count.0, count.1]);
        assert_eq!(uniform.step, plan.step);
        assert_eq!(uniform.radius, plan.radius_cells as i32);
        assert_eq!(uniform.sigma, plan.sigma_cells);
        assert_eq!(uniform.feathered, 1);
        assert_eq!((uniform.contrast, uniform.gain), (plan.contrast, plan.gain));
    }

    /// The schedule run on the CPU texture by texture, as the GPU runs it:
    /// what lands in the shifted alpha is the twin's Shift edge, and no pass
    /// reads what it writes or a texture another pass spoiled.
    #[test]
    fn the_schedule_of_the_runs_gives_the_twin_and_keeps_its_textures_apart() {
        let size = (41u32, 33u32);
        let input: Vec<f32> = (0..size.0 * size.1)
            .map(|i| {
                let (x, y) = (i % size.0, i / size.0);
                (((x * 7 + y * 13) % 17) as f32 / 16.0 * 255.0).round() / 255.0
            })
            .collect();
        let at = |x: i32, y: i32| {
            let x = x.clamp(0, size.0 as i32 - 1) as u32;
            let y = y.clamp(0, size.1 as i32 - 1) as u32;
            (y * size.0 + x) as usize
        };
        for (axis, diagonal) in [(1, 0), (2, 1), (3, 2), (7, 5), (8, 6), (24, 17), (5, 0)] {
            for grow in [true, false] {
                let plan = twin::Plan {
                    grow,
                    axis,
                    diagonal,
                    ..twin::Plan::new(&Edge::default(), size, (0, 0), size)
                };
                let runs = plan.runs();
                let passes = schedule(&runs);
                let mut work: [Vec<f32>; 3] = [vec![], vec![], vec![]];
                let mut shifted = vec![-1.0f32; input.len()];
                let mut writes_to_shifted = 0;
                for pass in &passes {
                    assert_ne!(Some(pass.write), Some(pass.read));
                    assert_ne!(Some(pass.write), pass.other);
                    assert_ne!(pass.write, Place::Input);
                    let read = |place: Place, work: &[Vec<f32>; 3]| match place {
                        Place::Input => input.clone(),
                        Place::Work(w) => work[w].clone(),
                        Place::Shifted => panic!("the shifted alpha is never read"),
                    };
                    let source = read(pass.read, &work);
                    assert_eq!(source.len(), input.len(), "{pass:?} reads what was written");
                    let other = pass.other.map(|o| read(o, &work));
                    let pick = |a: f32, b: f32| if grow { a.max(b) } else { a.min(b) };
                    let out: Vec<f32> = (0..input.len())
                        .map(|i| {
                            let (x, y) = ((i as u32 % size.0) as i32, (i as u32 / size.0) as i32);
                            let o = pass.offset as i32;
                            let far =
                                source[at(x + o * pass.direction.0, y + o * pass.direction.1)];
                            let v = pick(source[i], far);
                            other.as_ref().map_or(v, |f| pick(f[i], v))
                        })
                        .collect();
                    match pass.write {
                        Place::Work(w) => work[w] = out,
                        Place::Shifted => {
                            shifted = out;
                            writes_to_shifted += 1;
                        }
                        Place::Input => unreachable!(),
                    }
                }
                assert_eq!(writes_to_shifted, 1, "{axis} {diagonal}");
                assert_eq!(passes.last().map(|p| p.write), Some(Place::Shifted));
                let want = twin::shift(&input, &plan);
                assert!(
                    shifted
                        .iter()
                        .zip(&want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{axis} {diagonal} {grow}"
                );
                // Two passes a doubling level, both ways, for each run.
                let levels: usize = runs.iter().map(|r| 2 * doubling(r.half).len()).sum();
                assert_eq!(passes.len(), levels);
            }
        }
    }

    #[test]
    fn the_cells_under_a_rect_are_the_ones_its_pixels_read() {
        let edge = Edge {
            feather: 0.01,
            ..Edge::default()
        };
        // Cells of 15 pixels on a render that begins inside a cell.
        let plan = twin::Plan::new(&edge, (6000, 4000), (1021, 513), (900, 700));
        assert_eq!(plan.step, 15);
        let ((first_x, first_y), (columns, rows)) = plan.grid();
        for over in [
            (0, 0, 900, 700),
            (100, 200, 1, 1),
            (897, 3, 3, 697),
            (7, 8, 30, 45),
        ] {
            let (x, y, w, h) = cells_under(&plan, over);
            assert!(x + w <= columns && y + h <= rows, "{over:?}");
            for (px, py) in [(over.0, over.1), (over.0 + over.2 - 1, over.1 + over.3 - 1)] {
                let place = |p: u32, origin: u32, first: u32| {
                    ((origin + p) as f32 + 0.5) / plan.step as f32 - 0.5 - first as f32
                };
                let (ux, uy) = (place(px, 1021, first_x), place(py, 513, first_y));
                for (u, low, count, len) in [(ux, x, columns, w), (uy, y, rows, h)] {
                    let before = (u.floor().max(0.0) as u32).min(count - 1);
                    let after = ((u.floor() + 1.0).max(0.0) as u32).min(count - 1);
                    assert!(before >= low && after < low + len, "{over:?}: {u}");
                }
            }
        }
    }
}
