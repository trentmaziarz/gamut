//! Refine edges on the GPU: the pass family of `refine.wgsl`, the float
//! targets it works in, and the refined alpha of one mask.
//!
//! The arithmetic and the plan of a render (the side of a cell, the radius of
//! the box, eps) are gamut-color's [`Plan`]; this module only lays the work
//! out. The moments of the filter are differences of near-equal numbers, so
//! they live in Rgba32Float targets read with `textureLoad`. Those targets
//! belong to the frame and not to a mask, exist only once an edit holds a
//! refined mask, and hold one tile of the cell grid at a time within a budget
//! of bytes, so their size is bounded whatever the render: a grid the budget
//! holds is one tile, and a larger render is refined a tile after another. A
//! default device draws into at most 32 bytes a sample, which is two of them
//! a pass; Gamut asks the adapter for up to 64
//! ([`crate::video::wanted_limits`]), which is four.
//!
//! A tile is drawn in 21 passes. Six take the moments of the source and
//! their box means; they depend on no mask and no gather, so they are kept
//! while the source, the radius and the tile stay the same, and a tile then
//! costs 15. They are taken over the cells the gathers work over, joined
//! with the cells already held, so a refine over cells they cover draws
//! none, and a refine of a small mask on a large grid draws few. Each of the
//! three gathers takes five: the moments of `q` over the cells, their box
//! means across and down, the solve in one pass of four targets, and the
//! move at full resolution. The move of the last gather writes the r8unorm
//! refined alpha; the two before it write `q` into a 32 bit single channel
//! target, so no store rounds it between gathers.
//!
//! A device that draws into no more than 32 bytes a sample, such as one made
//! with wgpu's default limits, draws the solve in two passes of two targets:
//! a gather takes six, and a tile 24, or 18 with the moments of the source
//! held. The RTX 4080 Laptop GPU on Vulkan and DX12 WARP both offer 128
//! bytes, so both draw the fused solve.
//!
//! A box of [`BLOCK_TAPS`] cells or more (a radius of 12 cells or more)
//! has a block pass before it, which sums [`BLOCK`] cells from every cell on
//! along the axis; the box then adds those sums [`BLOCK`] cells apart and the
//! cells left over, in place of every cell. The ten box passes of a tile gain
//! ten block passes: 31 passes, and 21 with the moments of the source held
//! (34 and 24 with the solve in two passes). A box keeps its cells, each held
//! at the edges as before; only the order of the sum changes. A smaller box
//! sums every cell in the direct loop, which measured no slower (see
//! [`BLOCK_TAPS`]): a block pass is a pass of its own, and a box of few cells
//! does not win it back.

use bytemuck::{Pod, Zeroable};
use gamut_color::refine::{GATHERS, Plan};

use crate::{FULLSCREEN_VERTICES, fullscreen_primitive};

/// The format of the float targets of the cells.
pub(crate) const MOMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// The format of the target that holds `q` between two gathers.
pub(crate) const MOVED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// The bytes a cell of each format costs a target.
const MOMENT_BYTES: u64 = 16;
const MOVED_BYTES: u64 = 4;

/// How many Rgba32Float targets of cells a scratch holds of each kind: the
/// box means of the source's moments, the far side of every box mean, and
/// what a gather solved. The far side of the mask's box mean is one more.
const SOURCE_TARGETS: usize = 3;
const FAR_TARGETS: usize = 3;
const SOLVED_TARGETS: usize = 4;

/// The most bytes the scratch of a frame takes, the targets of cells and `q`
/// together, unless one box needs more.
pub(crate) const SCRATCH_BUDGET_BYTES: u64 = 384 * 1024 * 1024;

/// The longest side of a texture a device of default limits makes, which is
/// what the headless device and the app ask for: `q` of one tile over a
/// whole grid stays inside it.
pub(crate) const TEXTURE_SIDE_LIMIT: u32 = 8192;

/// How many cells a block pass sums, as `BLOCK` in `refine.wgsl`.
pub(crate) const BLOCK: u32 = 8;

/// The fewest cells a box (2 cells + 1) adds from block sums, as
/// `BLOCK_TAPS` in `refine.wgsl`; a box of fewer sums every cell in the
/// direct loop. Measured on timing_24mp.jpg at 100 percent, the block box
/// against the direct loop, three runs each: at 107 cells (Radius 0.05) the
/// blocks are 0.9 to 2.4 ms faster a slider step, and at 23 cells (Radius
/// 0.01) 0.6 to 1.2 ms slower. On the 4K30 clip at 15 cells two measurements
/// put them 0.3 ms slower and 0.7 ms faster, inside the noise. So the blocks
/// start at the first box past 23 cells.
pub(crate) const BLOCK_TAPS: u32 = 24;

/// How many blocks of [`BLOCK`] cells a box of `cells` either side adds: 0
/// under [`BLOCK_TAPS`] cells, where the box sums every cell in the direct
/// loop.
pub(crate) fn box_blocks(cells: u32) -> u32 {
    let taps = 2 * cells + 1;
    if taps < BLOCK_TAPS { 0 } else { taps / BLOCK }
}

/// The bytes one cell of a tile costs the scratch, at `step` pixels a cell.
/// Counted are the eleven Rgba32Float targets of cells, 16 bytes a cell each:
/// the three `s` (the box means of the source's moments), the three `f` (the
/// far side of every box mean), `g` (the far side of the box mean of the
/// mask's moments) and the four `v` (what a gather solved). Then the R32Float
/// target `q` (the moved alpha between two gathers), which holds the `step`
/// by `step` pixels of a cell at 4 bytes each. The refined alpha belongs to
/// its mask and is not counted. At a step of 4 a cell costs 240 bytes. The
/// block passes before the boxes add no target: they write `v[0]` and
/// `v[1]`, which no pass reads between a move and the next solve, and every
/// box lies between the two (see [`RefinePass::run`]).
pub(crate) fn bytes_a_cell(step: u32) -> u64 {
    let moments = (SOURCE_TARGETS + FAR_TARGETS + 1 + SOLVED_TARGETS) as u64;
    moments * MOMENT_BYTES + u64::from(step * step) * MOVED_BYTES
}

/// The most cells a side a square tile of `step` pixels a cell holds within
/// `budget` bytes.
pub(crate) fn budget_side(budget: u64, step: u32) -> u32 {
    u32::try_from((budget / bytes_a_cell(step)).isqrt()).unwrap_or(u32::MAX)
}

/// A rectangle of pixels or of cells: x, y, width, height.
pub(crate) type Rect = (u32, u32, u32, u32);

/// Mirrors the `Uniform` of `refine.wgsl`; a test holds the two layouts equal.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct RefineUniform {
    pub(crate) origin: [u32; 2],
    pub(crate) size: [u32; 2],
    pub(crate) grid_first: [u32; 2],
    pub(crate) grid_count: [u32; 2],
    pub(crate) tile_first: [u32; 2],
    pub(crate) tile_count: [u32; 2],
    pub(crate) q_first: [u32; 2],
    pub(crate) step: u32,
    pub(crate) cells: u32,
    pub(crate) eps: f32,
    pub(crate) amount: f32,
    /// How many blocks of [`BLOCK`] cells a box adds, 0 for the direct loop.
    pub(crate) blocks: u32,
    /// The struct of the shader is 8 byte aligned.
    pub(crate) pad: u32,
}

/// One tile of a refine: the pixels of the render it writes and the cells
/// the float targets hold while it is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tile {
    /// The pixels of the render the last move writes.
    pub(crate) out: Rect,
    /// The first cell held and how many, on each axis.
    pub(crate) first: (u32, u32),
    pub(crate) count: (u32, u32),
    /// The cells the gathers are drawn over, as texels of the targets: the
    /// ones the pixels of `out` lie among and the margin around them.
    pub(crate) work: Rect,
}

impl Tile {
    /// The pixel of the render the first texel of the `q` target holds: the
    /// first pixel of the first cell.
    fn q_first(&self, plan: &Plan) -> (u32, u32) {
        let axis = |first: u32, origin: u32| (first * plan.step).max(origin) - origin;
        (
            axis(self.first.0, plan.origin.0),
            axis(self.first.1, plan.origin.1),
        )
    }

    /// The pixels of the render inside the cells of `work`.
    fn work_pixels(&self, plan: &Plan) -> Rect {
        let axis = |first: u32, start: u32, count: u32, origin: u32, size: u32| {
            let low = ((first + start) * plan.step).max(origin) - origin;
            let high = ((first + start + count) * plan.step).min(origin + size) - origin;
            (low, high - low)
        };
        let (x, width) = axis(
            self.first.0,
            self.work.0,
            self.work.2,
            plan.origin.0,
            plan.size.0,
        );
        let (y, height) = axis(
            self.first.1,
            self.work.1,
            self.work.3,
            plan.origin.1,
            plan.size.1,
        );
        (x, y, width, height)
    }

    pub(crate) fn uniform(&self, plan: &Plan) -> RefineUniform {
        let (grid_first, grid_count) = plan.grid();
        let q_first = self.q_first(plan);
        RefineUniform {
            origin: [plan.origin.0, plan.origin.1],
            size: [plan.size.0, plan.size.1],
            grid_first: [grid_first.0, grid_first.1],
            grid_count: [grid_count.0, grid_count.1],
            tile_first: [self.first.0, self.first.1],
            tile_count: [self.count.0, self.count.1],
            q_first: [q_first.0, q_first.1],
            step: plan.step,
            cells: plan.cells,
            eps: plan.eps,
            amount: plan.amount,
            blocks: box_blocks(plan.cells),
            pad: 0,
        }
    }
}

/// The most cells a side a tile of this plan may hold. The whole grid is one
/// tile when its cells fit `budget` bytes and its `q` fits a texture on each
/// side; the side is then the larger side of the grid. Otherwise as many as
/// a square of `budget` bytes holds and no more than the larger side of the
/// grid. Either way at least what the boxes of the gathers and a few pixels
/// around them need.
pub(crate) fn tile_side(plan: &Plan, budget: u64) -> u32 {
    let (_, grid) = plan.grid();
    let bytes = u64::from(grid.0) * u64::from(grid.1) * bytes_a_cell(plan.step);
    let whole = bytes <= budget
        && u64::from(grid.0) * u64::from(plan.step) <= u64::from(TEXTURE_SIDE_LIMIT)
        && u64::from(grid.1) * u64::from(plan.step) <= u64::from(TEXTURE_SIDE_LIMIT);
    let side = if whole {
        grid.0.max(grid.1)
    } else {
        budget_side(budget, plan.step).min(grid.0.max(grid.1))
    };
    side.max(plan.margin() * 2 + 64)
}

/// The cells a block pass writes for a box pass over `area` along one axis
/// (down when `down`): the area and `reach` cells either way along the axis,
/// inside the `count` cells the tile holds. A box reads the block that
/// starts `reach` cells before a cell, and blocks up to its far end.
fn block_area(area: Rect, down: bool, reach: u32, count: (u32, u32)) -> Rect {
    let grow = |start: u32, length: u32, count: u32| {
        let low = start.saturating_sub(reach);
        let high = (start + length).saturating_add(reach).min(count);
        (low, high.max(low) - low)
    };
    if down {
        let (y, height) = grow(area.1, area.3, count.1);
        (area.0, y, area.2, height)
    } else {
        let (x, width) = grow(area.0, area.2, count.0);
        (x, area.1, width, area.3)
    }
}

/// The cells one axis of a span of pixels of the render needs: the cells its
/// pixels lie among (the one before the first centre, the one after the
/// last) and the margin around them, inside the grid. First cell and count.
fn cells_for(plan: &Plan, origin: u32, start: u32, length: u32, grid: (u32, u32)) -> (u32, u32) {
    let step = plan.step;
    // The cell whose centre lies at or before the centre of a pixel: the
    // twin's floor((X + 0.5) / step - 0.5), which is -1 before the first
    // centre and held at the grid there.
    let before = |pixel: u32| (2 * (origin + pixel) + 1).saturating_sub(step) / (2 * step);
    let low = before(start).saturating_sub(plan.margin()).max(grid.0);
    let high = (before(start + length - 1) + 1 + plan.margin()).min(grid.0 + grid.1 - 1);
    (low, high.max(low) - low + 1)
}

/// The tiles that refine `over` of a render, each inside `side` cells a side.
/// A grid that fits in one tile is always held whole, whatever `over` is, so
/// the moments of the source held for it serve a patch under new dabs too;
/// the gathers are then drawn over the cells the patch needs alone.
pub(crate) fn tiles(plan: &Plan, over: Rect, side: u32) -> Vec<Tile> {
    let (grid_first, grid_count) = plan.grid();
    if grid_count.0 <= side && grid_count.1 <= side {
        let (first_x, columns) = cells_for(
            plan,
            plan.origin.0,
            over.0,
            over.2,
            (grid_first.0, grid_count.0),
        );
        let (first_y, rows) = cells_for(
            plan,
            plan.origin.1,
            over.1,
            over.3,
            (grid_first.1, grid_count.1),
        );
        return vec![Tile {
            out: over,
            first: grid_first,
            count: grid_count,
            work: (
                first_x - grid_first.0,
                first_y - grid_first.1,
                columns,
                rows,
            ),
        }];
    }
    // The pixels a tile may write along one axis so that its cells fit: the
    // margin on both sides and the two cells around the span come off.
    let span = (side.saturating_sub(2 * plan.margin() + 3)).max(1) * plan.step;
    let cuts = |start: u32, length: u32| -> Vec<(u32, u32)> {
        (0..length.div_ceil(span))
            .map(|k| (start + k * span, span.min(length - k * span)))
            .collect()
    };
    let mut out = Vec::new();
    for (y, height) in cuts(over.1, over.3) {
        for (x, width) in cuts(over.0, over.2) {
            let (first_x, columns) =
                cells_for(plan, plan.origin.0, x, width, (grid_first.0, grid_count.0));
            let (first_y, rows) =
                cells_for(plan, plan.origin.1, y, height, (grid_first.1, grid_count.1));
            out.push(Tile {
                out: (x, y, width, height),
                first: (first_x, first_y),
                count: (columns, rows),
                work: (0, 0, columns, rows),
            });
        }
    }
    out
}

struct FloatTarget {
    view: wgpu::TextureView,
}

/// The moments of the source the targets `s` hold: the plan they were taken
/// with, but for eps and the amount, the tile, and the cells of the tile
/// they were taken over.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Held {
    full: (u32, u32),
    origin: (u32, u32),
    size: (u32, u32),
    step: u32,
    cells: u32,
    first: (u32, u32),
    count: (u32, u32),
    area: Rect,
}

impl Held {
    /// Whether these moments were taken with the plan and the tile of
    /// `other`, over whichever cells.
    fn takes_as(&self, other: &Held) -> bool {
        Held {
            area: other.area,
            ..*self
        } == *other
    }

    /// Whether they were taken over every cell of `area`. The box means
    /// near the edge of the cells taken read cells not taken, as the box
    /// means of a gather near the edge of its work do, and the margin of
    /// the work keeps both from the pixels a tile writes.
    fn covers(&self, area: Rect) -> bool {
        let (x, y, width, height) = self.area;
        area.0 >= x && area.1 >= y && area.0 + area.2 <= x + width && area.1 + area.3 <= y + height
    }
}

/// The smallest rectangle that holds both.
fn join(a: Rect, b: Rect) -> Rect {
    let (x, y) = (a.0.min(b.0), a.1.min(b.1));
    let right = (a.0 + a.2).max(b.0 + b.2);
    let bottom = (a.1 + a.3).max(b.1 + b.3);
    (x, y, right - x, bottom - y)
}

/// The float targets of a frame, eleven of cells and one of pixels.
pub(crate) struct Scratch {
    /// The box means of the source's moments: (I, rr), (rg, rb, gg, gb), (bb).
    s: [FloatTarget; SOURCE_TARGETS],
    /// The far side of every box mean. While a mask is gathered `f[0]` holds
    /// the means of (q, q I) and `f[2]` those of (p, p p).
    f: [FloatTarget; FAR_TARGETS],
    /// The far side of the box mean of (p, p p).
    g: FloatTarget,
    /// What a gather solved: 15 numbers a cell.
    v: [FloatTarget; SOLVED_TARGETS],
    /// `q` between two gathers, at the pixels of the cells.
    q: FloatTarget,
    /// The cells a side the targets of cells hold, and the pixels `q` holds.
    size: (u32, u32),
    q_size: (u32, u32),
    held: Option<Held>,
    /// Told apart from every scratch before it, for the bind groups that
    /// hold its views.
    id: u64,
}

impl Scratch {
    /// The working texture holds other pixels: the moments of the source
    /// are taken again by the next refine.
    pub(crate) fn forget_source(&mut self) {
        self.held = None;
    }
}

/// The refined alpha of one mask and what draws it.
pub(crate) struct Refined {
    pub(crate) alpha: wgpu::TextureView,
    uniform: wgpu::Buffer,
    /// How many tiles the uniform buffer holds.
    slots: u32,
    /// The bind groups, and the scratch they were made for.
    binds: Option<(u64, Binds)>,
}

/// One bind group for each set of textures a pass reads. A texture is never
/// in the group of a pass that writes it; `v` fills the places a pass does
/// not read. The groups a box reads hold v0 and v1 third and fourth, where a
/// box of blocks reads the sums of its block pass.
struct Binds {
    /// f1, g: the passes over the pixels of a cell, which read no target of
    /// cells, and the box down of a gather.
    cells: wgpu::BindGroup,
    /// s0, s1 and s2: the box across of the source's moments.
    source_pair: wgpu::BindGroup,
    source_one: wgpu::BindGroup,
    /// f0, f1 and f2: the box down of the source's moments.
    far_pair: wgpu::BindGroup,
    far_one: wgpu::BindGroup,
    /// f0, f2: the box across of a gather.
    gather_across: wgpu::BindGroup,
    /// s0, s1, s2, f0, f2: the solve.
    solve: wgpu::BindGroup,
    /// v0 to v3, and no `q`: the move.
    moving: wgpu::BindGroup,
    /// What the block pass before each box reads, the same targets first
    /// and never v0 or v1, which it writes.
    block_cells: wgpu::BindGroup,
    block_source_pair: wgpu::BindGroup,
    block_source_one: wgpu::BindGroup,
    block_far_pair: wgpu::BindGroup,
    block_far_one: wgpu::BindGroup,
    block_gather_across: wgpu::BindGroup,
}

/// The block pass that draws before a box pass of [`BLOCK_TAPS`] cells or
/// more.
struct Block<'a> {
    label: &'a str,
    pipeline: &'a wgpu::RenderPipeline,
    bind: &'a wgpu::BindGroup,
    /// Down the targets, or across.
    down: bool,
}

impl<'a> Block<'a> {
    fn across(
        label: &'a str,
        pipeline: &'a wgpu::RenderPipeline,
        bind: &'a wgpu::BindGroup,
    ) -> Option<Self> {
        Some(Block {
            label,
            pipeline,
            bind,
            down: false,
        })
    }

    fn down(
        label: &'a str,
        pipeline: &'a wgpu::RenderPipeline,
        bind: &'a wgpu::BindGroup,
    ) -> Option<Self> {
        Some(Block {
            label,
            pipeline,
            bind,
            down: true,
        })
    }
}

/// The pipelines of `refine.wgsl`.
pub(crate) struct RefinePass {
    layout: wgpu::BindGroupLayout,
    source_a: wgpu::RenderPipeline,
    source_b: wgpu::RenderPipeline,
    gather_first: wgpu::RenderPipeline,
    gather: wgpu::RenderPipeline,
    box_h2: wgpu::RenderPipeline,
    box_v2: wgpu::RenderPipeline,
    box_h1: wgpu::RenderPipeline,
    box_v1: wgpu::RenderPipeline,
    block_h2: wgpu::RenderPipeline,
    block_v2: wgpu::RenderPipeline,
    block_h1: wgpu::RenderPipeline,
    block_v1: wgpu::RenderPipeline,
    /// The solve in one pass of four targets, on a device that draws into
    /// 64 bytes a sample; `None` on one that draws into less.
    solve: Option<wgpu::RenderPipeline>,
    /// The solve in two passes of two targets, when there is no `solve`.
    solve_a: Option<wgpu::RenderPipeline>,
    solve_b: Option<wgpu::RenderPipeline>,
    moving: wgpu::RenderPipeline,
    apply: wgpu::RenderPipeline,
    /// The distance between two tiles in the uniform buffer.
    stride: u32,
    scratches: u64,
    /// How many times the moments of the source were taken, in tiles.
    pub(crate) source_builds: u64,
    /// The most bytes the scratch takes: [`SCRATCH_BUDGET_BYTES`], unless a
    /// test sets less to draw a render in more tiles.
    scratch_budget: u64,
    /// How many passes and how many tiles this pass family has drawn.
    pub(crate) refine_passes: u32,
    pub(crate) refine_tiles: u32,
    /// Every box sums its cells in the direct loop, whatever its size, so a
    /// test can hold the block sums to it.
    direct_box: bool,
}

const SHADER: &str = include_str!("shaders/refine.wgsl");

impl RefinePass {
    pub(crate) fn new(device: &wgpu::Device, alpha_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("refine"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let size = std::mem::size_of::<RefineUniform>() as u64;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("refine"),
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
                texture(1),
                texture(2),
                texture(3),
                texture(4),
                texture(5),
                texture(6),
                texture(7),
                texture(8),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("refine"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |fragment: &str, formats: &[wgpu::TextureFormat]| {
            let targets: Vec<Option<wgpu::ColorTargetState>> = formats
                .iter()
                .map(|format| {
                    Some(wgpu::ColorTargetState {
                        format: *format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                })
                .collect();
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
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pair = [MOMENT_FORMAT; 2];
        let one = [MOMENT_FORMAT];
        // The fused solve draws into every solved target at once.
        let fused = u64::from(device.limits().max_color_attachment_bytes_per_sample)
            >= SOLVED_TARGETS as u64 * MOMENT_BYTES;
        let split = |fragment: &str| (!fused).then(|| pipeline(fragment, &pair));
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        RefinePass {
            source_a: pipeline("fs_source_a", &pair),
            source_b: pipeline("fs_source_b", &one),
            gather_first: pipeline("fs_gather_first", &pair),
            gather: pipeline("fs_gather", &one),
            box_h2: pipeline("fs_box_h2", &pair),
            box_v2: pipeline("fs_box_v2", &pair),
            box_h1: pipeline("fs_box_h1", &one),
            box_v1: pipeline("fs_box_v1", &one),
            block_h2: pipeline("fs_block_h2", &pair),
            block_v2: pipeline("fs_block_v2", &pair),
            block_h1: pipeline("fs_block_h1", &one),
            block_v1: pipeline("fs_block_v1", &one),
            solve: fused.then(|| pipeline("fs_solve", &[MOMENT_FORMAT; SOLVED_TARGETS])),
            solve_a: split("fs_solve_a"),
            solve_b: split("fs_solve_b"),
            moving: pipeline("fs_move", &[MOVED_FORMAT]),
            apply: pipeline("fs_apply", &[alpha_format]),
            layout,
            stride: (size as u32).div_ceil(alignment) * alignment,
            scratches: 0,
            source_builds: 0,
            scratch_budget: SCRATCH_BUDGET_BYTES,
            refine_passes: 0,
            refine_tiles: 0,
            direct_box: false,
        }
    }

    /// Whether the solve is drawn in one pass of four targets, on a device
    /// that draws into 64 bytes a sample, rather than in two of two.
    pub(crate) fn solve_fused(&self) -> bool {
        self.solve.is_some()
    }

    /// The same pass family with a scratch of at most `bytes`, so a test can
    /// draw a render in more tiles than the default budget gives.
    #[doc(hidden)]
    pub fn with_scratch_budget(self, bytes: u64) -> Self {
        RefinePass {
            scratch_budget: bytes,
            ..self
        }
    }

    /// The same pass family with every box summed in the direct loop when
    /// `on`, and no block pass, so a test can hold the block sums to it.
    #[doc(hidden)]
    pub fn with_direct_box(self, on: bool) -> Self {
        RefinePass {
            direct_box: on,
            ..self
        }
    }

    /// The refined alpha of a mask on a render of this size, not drawn yet.
    pub(crate) fn refined(
        &self,
        device: &wgpu::Device,
        alpha_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Refined {
        Refined {
            alpha: target(device, "refined alpha", alpha_format, width, height).view,
            uniform: self.uniform_buffer(device, 1),
            slots: 1,
            binds: None,
        }
    }

    fn uniform_buffer(&self, device: &wgpu::Device, slots: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("refine uniforms"),
            size: u64::from(self.stride) * u64::from(slots),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Draws the refined alpha of one mask over `over` of the render, from
    /// its finished alpha and the working texture. `scratch` is the frame's:
    /// it is made here when there is none or when the plan needs a larger
    /// one. How many tiles were drawn.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scratch: &mut Option<Scratch>,
        refined: &mut Refined,
        working: &wgpu::TextureView,
        alpha: &wgpu::TextureView,
        plan: &Plan,
        over: Rect,
    ) -> u32 {
        let (_, grid) = plan.grid();
        let side = tile_side(plan, self.scratch_budget);
        let wanted = (grid.0.min(side), grid.1.min(side));
        let wanted_q = (
            (wanted.0 * plan.step).min(plan.size.0),
            (wanted.1 * plan.step).min(plan.size.1),
        );
        if scratch.as_ref().is_none_or(|held| {
            held.size.0 < wanted.0
                || held.size.1 < wanted.1
                || held.q_size.0 < wanted_q.0
                || held.q_size.1 < wanted_q.1
        }) {
            let (size, q_size) = scratch.as_ref().map_or((wanted, wanted_q), |held| {
                (
                    (held.size.0.max(wanted.0), held.size.1.max(wanted.1)),
                    (held.q_size.0.max(wanted_q.0), held.q_size.1.max(wanted_q.1)),
                )
            });
            self.scratches += 1;
            let float = |label: &str| FloatTarget {
                view: target(device, label, MOMENT_FORMAT, size.0, size.1).view,
            };
            *scratch = Some(Scratch {
                s: [0; SOURCE_TARGETS].map(|_| float("refine source means")),
                f: [0; FAR_TARGETS].map(|_| float("refine means")),
                g: float("refine mask means"),
                v: [0; SOLVED_TARGETS].map(|_| float("refine solved")),
                q: FloatTarget {
                    view: target(device, "refine moved", MOVED_FORMAT, q_size.0, q_size.1).view,
                },
                size,
                q_size,
                held: None,
                id: self.scratches,
            });
        }
        let scratch = scratch.as_mut().expect("made above");
        let tiles = tiles(plan, over, side);
        if refined.slots < tiles.len() as u32 {
            refined.slots = (tiles.len() as u32).next_power_of_two();
            refined.uniform = self.uniform_buffer(device, refined.slots);
            refined.binds = None;
        }
        if refined
            .binds
            .as_ref()
            .is_none_or(|(id, _)| *id != scratch.id)
        {
            let group = |label: &str, moved: &wgpu::TextureView, textures: [&FloatTarget; 5]| {
                fn view(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
                    wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::TextureView(view),
                    }
                }
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &refined.uniform,
                                offset: 0,
                                size: wgpu::BufferSize::new(
                                    std::mem::size_of::<RefineUniform>() as u64
                                ),
                            }),
                        },
                        view(1, working),
                        view(2, alpha),
                        view(3, moved),
                        view(4, &textures[0].view),
                        view(5, &textures[1].view),
                        view(6, &textures[2].view),
                        view(7, &textures[3].view),
                        view(8, &textures[4].view),
                    ],
                })
            };
            let (s, f, g, v, q) = (&scratch.s, &scratch.f, &scratch.g, &scratch.v, &scratch.q);
            refined.binds = Some((
                scratch.id,
                Binds {
                    cells: group("refine cells", &q.view, [&f[1], g, &v[0], &v[1], &v[2]]),
                    source_pair: group(
                        "refine source pair",
                        &q.view,
                        [&s[0], &s[1], &v[0], &v[1], &v[2]],
                    ),
                    source_one: group(
                        "refine source one",
                        &q.view,
                        [&s[2], &v[2], &v[0], &v[1], &v[3]],
                    ),
                    far_pair: group(
                        "refine far pair",
                        &q.view,
                        [&f[0], &f[1], &v[0], &v[1], &v[2]],
                    ),
                    far_one: group(
                        "refine far one",
                        &q.view,
                        [&f[2], &v[2], &v[0], &v[1], &v[3]],
                    ),
                    gather_across: group(
                        "refine gather across",
                        &q.view,
                        [&f[0], &f[2], &v[0], &v[1], &v[2]],
                    ),
                    solve: group("refine solve", &q.view, [&s[0], &s[1], &s[2], &f[0], &f[2]]),
                    // The move writes `q`, so the alpha stands in its place.
                    moving: group("refine move", alpha, [&v[0], &v[1], &v[2], &v[3], &f[1]]),
                    block_cells: group(
                        "refine block cells",
                        &q.view,
                        [&f[1], g, &v[2], &v[3], &s[2]],
                    ),
                    block_source_pair: group(
                        "refine block source pair",
                        &q.view,
                        [&s[0], &s[1], &v[2], &v[3], g],
                    ),
                    block_source_one: group(
                        "refine block source one",
                        &q.view,
                        [&s[2], g, &v[2], &v[3], &f[0]],
                    ),
                    block_far_pair: group(
                        "refine block far pair",
                        &q.view,
                        [&f[0], &f[1], &v[2], &v[3], g],
                    ),
                    block_far_one: group(
                        "refine block far one",
                        &q.view,
                        [&f[2], g, &v[2], &v[3], &f[0]],
                    ),
                    block_gather_across: group(
                        "refine block gather across",
                        &q.view,
                        [&f[0], &f[2], &v[2], &v[3], g],
                    ),
                },
            ));
        }
        let (_, binds) = refined.binds.as_ref().expect("made above");
        // The blocks each box adds, and none when every box sums its cells
        // in the direct loop.
        let blocks = if self.direct_box {
            0
        } else {
            box_blocks(plan.cells)
        };
        let mut drawn = 0u32;
        for (slot, tile) in tiles.iter().enumerate() {
            let offset = slot as u32 * self.stride;
            let uniform = RefineUniform {
                blocks,
                ..tile.uniform(plan)
            };
            queue.write_buffer(
                &refined.uniform,
                u64::from(offset),
                bytemuck::bytes_of(&uniform),
            );
            let (s, f, g, v) = (&scratch.s, &scratch.f, &scratch.g, &scratch.v);
            // A box pass of blocks has its block pass first, into v0 (and v1
            // for a pair). Both are free at every box: the solve writes them
            // after the last box of a gather, and the move reads them before
            // the first box of the next.
            let sums = [&v[0].view, &v[1].view];
            let count = tile.count;
            let mut pass = |label: &str,
                            pipeline: &wgpu::RenderPipeline,
                            bind: &wgpu::BindGroup,
                            targets: &[&wgpu::TextureView],
                            area: Rect,
                            block: Option<Block>| {
                if let Some(block) = block.filter(|_| blocks > 0) {
                    draw(
                        encoder,
                        block.label,
                        block.pipeline,
                        block.bind,
                        offset,
                        &sums[..targets.len()],
                        block_area(area, block.down, plan.cells, count),
                    );
                    drawn += 1;
                }
                draw(encoder, label, pipeline, bind, offset, targets, area);
                drawn += 1;
            };
            fn pair<'a>(a: &'a FloatTarget, b: &'a FloatTarget) -> [&'a wgpu::TextureView; 2] {
                [&a.view, &b.view]
            }
            // The moments of the source over the cells the gathers work
            // over, unless the targets hold them there already; joined with
            // the cells held when those were taken with this plan and tile.
            let held = Held {
                full: plan.full,
                origin: plan.origin,
                size: plan.size,
                step: plan.step,
                cells: plan.cells,
                first: tile.first,
                count: tile.count,
                area: tile.work,
            };
            let take = match scratch.held {
                Some(had) if had.takes_as(&held) && had.covers(tile.work) => None,
                Some(had) if had.takes_as(&held) => Some(join(had.area, tile.work)),
                _ => Some(tile.work),
            };
            if let Some(all) = take {
                let cells = &binds.cells;
                pass(
                    "refine source a",
                    &self.source_a,
                    cells,
                    &pair(&s[0], &s[1]),
                    all,
                    None,
                );
                pass(
                    "refine source b",
                    &self.source_b,
                    cells,
                    &[&s[2].view],
                    all,
                    None,
                );
                pass(
                    "refine source h a",
                    &self.box_h2,
                    &binds.source_pair,
                    &pair(&f[0], &f[1]),
                    all,
                    Block::across(
                        "refine source h a block",
                        &self.block_h2,
                        &binds.block_source_pair,
                    ),
                );
                pass(
                    "refine source h b",
                    &self.box_h1,
                    &binds.source_one,
                    &[&f[2].view],
                    all,
                    Block::across(
                        "refine source h b block",
                        &self.block_h1,
                        &binds.block_source_one,
                    ),
                );
                pass(
                    "refine source v a",
                    &self.box_v2,
                    &binds.far_pair,
                    &pair(&s[0], &s[1]),
                    all,
                    Block::down(
                        "refine source v a block",
                        &self.block_v2,
                        &binds.block_far_pair,
                    ),
                );
                pass(
                    "refine source v b",
                    &self.box_v1,
                    &binds.far_one,
                    &[&s[2].view],
                    all,
                    Block::down(
                        "refine source v b block",
                        &self.block_v1,
                        &binds.block_far_one,
                    ),
                );
                scratch.held = Some(Held { area: all, ..held });
                self.source_builds += 1;
            }
            let work = tile.work;
            let q_first = tile.q_first(plan);
            let work_pixels = tile.work_pixels(plan);
            let moved_area = (
                work_pixels.0 - q_first.0,
                work_pixels.1 - q_first.1,
                work_pixels.2,
                work_pixels.3,
            );
            for gather in 0..GATHERS {
                if gather == 0 {
                    // With the moments of the mask itself, which the solve
                    // of every gather reads from f2.
                    pass(
                        "refine gather first",
                        &self.gather_first,
                        &binds.cells,
                        &pair(&f[0], &f[2]),
                        work,
                        None,
                    );
                    pass(
                        "refine gather h2",
                        &self.box_h2,
                        &binds.gather_across,
                        &pair(&f[1], g),
                        work,
                        Block::across(
                            "refine gather h2 block",
                            &self.block_h2,
                            &binds.block_gather_across,
                        ),
                    );
                    pass(
                        "refine gather v2",
                        &self.box_v2,
                        &binds.cells,
                        &pair(&f[0], &f[2]),
                        work,
                        Block::down("refine gather v2 block", &self.block_v2, &binds.block_cells),
                    );
                } else {
                    pass(
                        "refine gather",
                        &self.gather,
                        &binds.cells,
                        &[&f[0].view],
                        work,
                        None,
                    );
                    pass(
                        "refine gather h1",
                        &self.box_h1,
                        &binds.gather_across,
                        &[&f[1].view],
                        work,
                        Block::across(
                            "refine gather h1 block",
                            &self.block_h1,
                            &binds.block_gather_across,
                        ),
                    );
                    pass(
                        "refine gather v1",
                        &self.box_v1,
                        &binds.cells,
                        &[&f[0].view],
                        work,
                        Block::down("refine gather v1 block", &self.block_v1, &binds.block_cells),
                    );
                }
                if let Some(solve) = &self.solve {
                    pass(
                        "refine solve",
                        solve,
                        &binds.solve,
                        &[&v[0].view, &v[1].view, &v[2].view, &v[3].view],
                        work,
                        None,
                    );
                } else {
                    pass(
                        "refine solve a",
                        self.solve_a.as_ref().expect("made without a fused solve"),
                        &binds.solve,
                        &pair(&v[0], &v[1]),
                        work,
                        None,
                    );
                    pass(
                        "refine solve b",
                        self.solve_b.as_ref().expect("made without a fused solve"),
                        &binds.solve,
                        &pair(&v[2], &v[3]),
                        work,
                        None,
                    );
                }
                if gather + 1 < GATHERS {
                    pass(
                        "refine move",
                        &self.moving,
                        &binds.moving,
                        &[&scratch.q.view],
                        moved_area,
                        None,
                    );
                } else {
                    pass(
                        "refine apply",
                        &self.apply,
                        &binds.moving,
                        &[&refined.alpha],
                        tile.out,
                        None,
                    );
                }
            }
        }
        self.refine_passes = self.refine_passes.wrapping_add(drawn);
        self.refine_tiles = self.refine_tiles.wrapping_add(tiles.len() as u32);
        tiles.len() as u32
    }
}

struct MadeTarget {
    view: wgpu::TextureView,
}

fn target(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> MadeTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    MadeTarget {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
    }
}

/// One pass over `area` of its targets, which keep what they hold outside
/// it. Every pass of the filter writes every pixel of its area.
fn draw(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind: &wgpu::BindGroup,
    offset: u32,
    targets: &[&wgpu::TextureView],
    area: Rect,
) {
    let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = targets
        .iter()
        .copied()
        .map(|view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })
        })
        .collect();
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &attachments,
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

    /// Moments of the source serve a later refine over cells they were
    /// taken over, with the same plan and tile; a refine over other cells
    /// takes them over both.
    #[test]
    fn the_held_moments_serve_the_cells_they_were_taken_over() {
        let held = Held {
            full: (6000, 4000),
            origin: (960, 0),
            size: (4032, 4000),
            step: 4,
            cells: 53,
            first: (240, 0),
            count: (1008, 1000),
            area: (100, 50, 600, 700),
        };
        assert!(held.covers((100, 50, 600, 700)));
        assert!(held.covers((150, 60, 20, 30)));
        assert!(!held.covers((99, 50, 10, 10)));
        assert!(!held.covers((650, 700, 51, 10)));
        assert!(held.takes_as(&Held {
            area: (0, 0, 1008, 1000),
            ..held
        }));
        assert!(!held.takes_as(&Held { cells: 52, ..held }));
        assert!(!held.takes_as(&Held {
            first: (0, 0),
            ..held
        }));
        assert_eq!(
            join((100, 50, 600, 700), (650, 700, 51, 10)),
            (100, 50, 601, 700)
        );
        assert_eq!(join((10, 20, 5, 5), (0, 0, 3, 3)), (0, 0, 15, 25));
    }
    use gamut_color::acescct;
    use gamut_color::refine as twin;
    use gamut_core::mask::Refine;
    use std::mem::offset_of;
    use wgpu::naga;

    fn plan(radius: f32, full: (u32, u32), origin: (u32, u32), size: (u32, u32)) -> Plan {
        let refine = Refine {
            amount: 100.0,
            radius,
            sensitivity: 50.0,
        };
        Plan::new(&refine, full, origin, size)
    }

    #[test]
    fn the_refine_uniform_matches_the_wgsl_struct() {
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
        assert_eq!(span as usize, size_of::<RefineUniform>());
        let offset = |name: &str| {
            members
                .iter()
                .find(|m| m.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("no member {name}"))
                .offset as usize
        };
        assert_eq!(offset("origin"), offset_of!(RefineUniform, origin));
        assert_eq!(offset("size"), offset_of!(RefineUniform, size));
        assert_eq!(offset("grid_first"), offset_of!(RefineUniform, grid_first));
        assert_eq!(offset("grid_count"), offset_of!(RefineUniform, grid_count));
        assert_eq!(offset("tile_first"), offset_of!(RefineUniform, tile_first));
        assert_eq!(offset("tile_count"), offset_of!(RefineUniform, tile_count));
        assert_eq!(offset("q_first"), offset_of!(RefineUniform, q_first));
        assert_eq!(offset("step"), offset_of!(RefineUniform, step));
        assert_eq!(offset("cells"), offset_of!(RefineUniform, cells));
        assert_eq!(offset("eps"), offset_of!(RefineUniform, eps));
        assert_eq!(offset("amount"), offset_of!(RefineUniform, amount));
        assert_eq!(offset("blocks"), offset_of!(RefineUniform, blocks));
        // The pad fills the shader struct to its 8 byte alignment.
        assert_eq!(offset_of!(RefineUniform, pad), 76);
        assert_eq!(members.len(), 12);
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

    /// The value of `const NAME: u32 = valueu;` in the shader.
    fn wgsl_u32_constant(name: &str) -> u32 {
        let start = SHADER
            .find(&format!("const {name}: u32 = "))
            .unwrap_or_else(|| panic!("no constant {name}"));
        let text = &SHADER[start..];
        let value = &text[text.find("= ").expect("an equals sign") + 2..];
        value[..value.find("u;").expect("a u32 literal")]
            .parse()
            .expect("a number")
    }

    #[test]
    fn the_shader_constants_equal_the_gamut_color_constants() {
        assert_eq!(wgsl_constant("ACES_LINEAR_CUT"), acescct::LINEAR_CUT);
        assert_eq!(wgsl_constant("ACES_SLOPE"), acescct::SLOPE);
        assert_eq!(wgsl_constant("ACES_OFFSET"), acescct::OFFSET);
        assert_eq!(wgsl_constant("ACES_LOG_SHIFT"), acescct::LOG_SHIFT);
        assert_eq!(wgsl_constant("ACES_LOG_SCALE"), acescct::LOG_SCALE);
        assert_eq!(wgsl_constant("COVER_LOW"), twin::COVER_LOW);
        assert_eq!(wgsl_constant("COVER_HIGH"), twin::COVER_HIGH);
        assert_eq!(wgsl_constant("HARD_LOW"), twin::HARD_LOW);
        assert_eq!(wgsl_constant("HARD_HIGH"), twin::HARD_HIGH);
        assert_eq!(wgsl_constant("SEPARATE_LOW"), twin::SEPARATE_LOW);
        assert_eq!(wgsl_constant("SEPARATE_HIGH"), twin::SEPARATE_HIGH);
        assert_eq!(wgsl_constant("LINE_LOW"), twin::LINE_LOW);
        assert_eq!(wgsl_constant("LINE_HIGH"), twin::LINE_HIGH);
        assert_eq!(wgsl_constant("OUT_LOW"), twin::OUT_LOW);
        assert_eq!(wgsl_constant("OUT_HIGH"), twin::OUT_HIGH);
        assert_eq!(wgsl_constant("IN_LOW"), twin::IN_LOW);
        assert_eq!(wgsl_constant("IN_HIGH"), twin::IN_HIGH);
        assert_eq!(wgsl_constant("SHARE_FLOOR"), twin::SHARE_FLOOR);
        assert_eq!(wgsl_constant("SEPARATION_FLOOR"), twin::SEPARATION_FLOOR);
        assert_eq!(wgsl_u32_constant("BLOCK"), BLOCK);
        assert_eq!(wgsl_u32_constant("BLOCK_TAPS"), BLOCK_TAPS);
        // The shader writes the three gathers out as the loop of `run`, the
        // smoothstep as its polynomial, and never calls the builtin.
        assert_eq!(GATHERS, 3);
        assert!(
            !SHADER.contains("smoothstep("),
            "the builtin is never called"
        );
        // The moments of the source, of a gather and of the mask, and what a
        // gather solves, fit their float targets: 3, 1, 1 and 4 of four
        // channels, two a pass at most, 32 bytes a sample, the most a default
        // device draws into.
        assert_eq!(twin::SOURCE_MOMENTS.div_ceil(4), 3);
        assert_eq!(twin::GATHER_MOMENTS.div_ceil(4), 1);
        assert_eq!(twin::MASK_MOMENTS.div_ceil(4), 1);
        assert_eq!(twin::SOLVED.div_ceil(4), 4);
        assert_eq!(MOMENT_FORMAT.target_pixel_byte_cost(), Some(16));
        assert_eq!(MOVED_FORMAT.target_pixel_byte_cost(), Some(4));
        assert_eq!(MOMENT_BYTES, 16);
        assert_eq!(MOVED_BYTES, 4);
        assert_eq!(SOURCE_TARGETS, twin::SOURCE_MOMENTS.div_ceil(4));
        assert_eq!(SOLVED_TARGETS, twin::SOLVED.div_ceil(4));
        assert_eq!(FAR_TARGETS, 3);
    }

    #[test]
    fn a_render_that_fits_is_one_tile_of_its_whole_grid() {
        let plan = plan(0.01, (1280, 1600), (0, 0), (1280, 1600));
        assert_eq!((plan.step, plan.cells), (2, 6), "16 pixels");
        // The grid is 640 by 800 cells: one tile of the budget holds it, and
        // more than one of a budget of 512 cells a side.
        let tiles = tiles(
            &plan,
            (0, 0, 1280, 1600),
            tile_side(&plan, SCRATCH_BUDGET_BYTES),
        );
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].count, (640, 800));
        let small = 512 * 512 * bytes_a_cell(plan.step);
        assert_eq!(tile_side(&plan, small), 512);
        let tiles = self::tiles(&plan, (0, 0, 1280, 1600), tile_side(&plan, small));
        assert!(tiles.len() > 1);
        let plan = self::plan(0.05, (1280, 1600), (0, 0), (1280, 1600));
        assert_eq!((plan.step, plan.cells), (4, 14), "80 pixels");
        let tiles = self::tiles(
            &plan,
            (0, 0, 1280, 1600),
            tile_side(&plan, SCRATCH_BUDGET_BYTES),
        );
        assert_eq!(
            tiles,
            vec![Tile {
                out: (0, 0, 1280, 1600),
                first: (0, 0),
                count: (320, 400),
                work: (0, 0, 320, 400),
            }]
        );
        let uniform = tiles[0].uniform(&plan);
        assert_eq!(uniform.grid_count, [320, 400]);
        assert_eq!(uniform.tile_count, [320, 400]);
        assert_eq!(uniform.q_first, [0, 0]);
        assert_eq!((uniform.step, uniform.cells), (4, 14));
        // A patch of the same render keeps the tile and works on the cells
        // it needs alone: the cells its pixels lie among, 49 to 62 across and
        // 74 to 85 down, and 44 cells of margin around them.
        let patch = self::tiles(
            &plan,
            (200, 300, 50, 40),
            tile_side(&plan, SCRATCH_BUDGET_BYTES),
        );
        assert_eq!(patch.len(), 1);
        assert_eq!((patch[0].first, patch[0].count), ((0, 0), (320, 400)));
        assert_eq!(plan.margin(), 44);
        assert_eq!(patch[0].work, (5, 30, 102, 100));
        assert_eq!(patch[0].work_pixels(&plan), (20, 120, 408, 400));
    }

    #[test]
    fn a_window_that_begins_inside_a_cell_places_q_on_its_first_pixel() {
        let plan = plan(0.05, (6000, 4000), (1021, 513), (900, 700));
        assert_eq!(plan.step, 4);
        let tile = tiles(
            &plan,
            (0, 0, 900, 700),
            tile_side(&plan, SCRATCH_BUDGET_BYTES),
        )[0];
        // The first cell, 255, begins at pixel 1020, before the render.
        assert_eq!(tile.first, (255, 128));
        assert_eq!(tile.q_first(&plan), (0, 0));
        let tile = Tile {
            first: (300, 140),
            ..tile
        };
        assert_eq!(tile.q_first(&plan), (1200 - 1021, 560 - 513));
    }

    /// Every pixel of what is refined is written by exactly one tile, and a
    /// tile holds, and works on, every cell the twin reads for its pixels:
    /// the two cells a pixel lies among and the margin of the three gathers
    /// around them, inside the grid of the render.
    #[test]
    fn the_tiles_write_every_pixel_once_and_hold_every_cell_they_read() {
        let window = ((6000, 4000), (1021, 513), (3104, 2005));
        let cases = [
            // Cells of one pixel on a large window: many tiles.
            (
                plan(0.0012, window.0, window.1, window.2),
                (3, 2, 3098, 2001),
            ),
            // The default radius at 100 percent.
            (plan(0.01, window.0, window.1, window.2), (3, 2, 3098, 2001)),
            // The widest box: 53 cells either side, three times.
            (
                plan(0.05, (6000, 4000), (0, 0), (6000, 4000)),
                (0, 0, 6000, 4000),
            ),
            // A patch under new dabs, on a grid of many tiles and on one
            // that fits in a tile.
            (
                plan(0.02, window.0, window.1, window.2),
                (1500, 700, 301, 277),
            ),
            (
                plan(0.05, (1280, 1600), (0, 0), (1280, 1600)),
                (600, 700, 301, 277),
            ),
        ];
        // The budget, and 16 MiB, which cuts every grid into more tiles.
        let budgets = [SCRATCH_BUDGET_BYTES, 16 * 1024 * 1024];
        for ((plan, over), budget) in cases
            .into_iter()
            .flat_map(|case| budgets.map(|budget| (case, budget)))
        {
            let side = tile_side(&plan, budget);
            let tiles = tiles(&plan, over, side);
            let (grid_first, grid_count) = plan.grid();
            let mut written = vec![0u8; (over.2 * over.3) as usize];
            for tile in &tiles {
                assert!(tile.count.0 <= side && tile.count.1 <= side, "{tile:?}");
                assert!(
                    tile.work.0 + tile.work.2 <= tile.count.0
                        && tile.work.1 + tile.work.3 <= tile.count.1,
                    "{tile:?}"
                );
                for y in tile.out.1..tile.out.1 + tile.out.3 {
                    for x in tile.out.0..tile.out.0 + tile.out.2 {
                        written[((y - over.1) * over.2 + x - over.0) as usize] += 1;
                    }
                }
                // Start and length of the pixels, the origin of the render,
                // the grid, and what the tile works on, on each axis.
                let axes = [
                    (
                        (tile.out.0, tile.out.2, plan.origin.0),
                        (grid_first.0, grid_count.0),
                        (tile.first.0 + tile.work.0, tile.work.2),
                    ),
                    (
                        (tile.out.1, tile.out.3, plan.origin.1),
                        (grid_first.1, grid_count.1),
                        (tile.first.1 + tile.work.1, tile.work.3),
                    ),
                ];
                for ((start, length, origin), (first, count), (held_first, held_count)) in axes {
                    for pixel in [start, start + length - 1] {
                        // The twin's place of a pixel among the cell centres.
                        let place = ((origin + pixel) as f32 + 0.5) / plan.step as f32 - 0.5;
                        let low = place.floor() as i64;
                        let reach = i64::from(plan.margin());
                        let clamp =
                            |cell: i64| cell.clamp(i64::from(first), i64::from(first + count - 1));
                        let (need_low, need_high) = (clamp(low - reach), clamp(low + 1 + reach));
                        assert!(
                            i64::from(held_first) <= need_low
                                && need_high <= i64::from(held_first + held_count - 1),
                            "{tile:?}: pixel {pixel} reads cells {need_low} to {need_high}"
                        );
                    }
                }
                // The q target holds the pixels of every cell worked on.
                let q_first = tile.q_first(&plan);
                let pixels = tile.work_pixels(&plan);
                assert!(pixels.0 >= q_first.0 && pixels.1 >= q_first.1, "{tile:?}");
                assert!(
                    pixels.0 - q_first.0 + pixels.2 <= (tile.count.0 * plan.step).min(plan.size.0)
                        && pixels.1 - q_first.1 + pixels.3
                            <= (tile.count.1 * plan.step).min(plan.size.1),
                    "{tile:?}"
                );
            }
            assert!(written.iter().all(|count| *count == 1), "{plan:?}");
            println!(
                "{} tiles of at most {side} cells a side at a budget of {budget} bytes",
                tiles.len()
            );
        }
    }

    #[test]
    fn a_box_wider_than_a_tile_gets_a_tile_that_holds_it() {
        // A picture of 40,000 pixels: a radius of 0.05 is 2,000 pixels, 355
        // cells either side for each of the three gathers.
        let plan = plan(0.05, (40_000, 30_000), (0, 0), (40_000, 30_000));
        assert_eq!((plan.step, plan.cells), (4, 355));
        let side = tile_side(&plan, SCRATCH_BUDGET_BYTES);
        assert!(side > budget_side(SCRATCH_BUDGET_BYTES, plan.step));
        assert_eq!(side, plan.margin() * 2 + 64);
        let tiles = tiles(&plan, (0, 0, 4000, 3000), side);
        assert!(tiles.iter().all(|tile| tile.out.2 >= 4 && tile.out.3 >= 4));
    }

    /// A box of 2 cells + 1 cells adds (2 cells + 1) / 8 blocks and the
    /// cells left over from 24 cells on, and a box under 24 cells sums
    /// every cell directly.
    #[test]
    fn a_box_of_24_cells_or_more_adds_blocks_and_the_cells_left_over() {
        assert_eq!((BLOCK, BLOCK_TAPS), (8, 24));
        // Cells either side, blocks, and single cells left over.
        for (cells, blocks, singles) in [
            (1, 0, 3),
            (2, 0, 5),
            (3, 0, 7),
            (4, 0, 9),
            (6, 0, 13),
            (7, 0, 15),
            (9, 0, 19),
            (11, 0, 23),
            (12, 3, 1),
            (18, 4, 5),
            (53, 13, 3),
        ] {
            assert_eq!(box_blocks(cells), blocks, "{cells} cells");
            assert_eq!(2 * cells + 1 - blocks * BLOCK, singles, "{cells} cells");
        }
        // The uniform carries the blocks of its plan: none at a box of 5
        // cells, 13 at the box of Radius 0.05 at 100 percent.
        let small = plan(0.05, (64, 64), (0, 0), (64, 64));
        assert_eq!(small.cells, 2);
        let tile = tiles(
            &small,
            (0, 0, 64, 64),
            tile_side(&small, SCRATCH_BUDGET_BYTES),
        )[0];
        assert_eq!(tile.uniform(&small).blocks, 0);
        let wide = plan(0.05, (6000, 4000), (960, 0), (4032, 4000));
        let tile = tiles(
            &wide,
            (0, 0, 4032, 4000),
            tile_side(&wide, SCRATCH_BUDGET_BYTES),
        )[0];
        assert_eq!(tile.uniform(&wide).blocks, 13);
    }

    /// A block pass writes the cells of its box and `cells` more either way
    /// along the axis, inside the tile.
    #[test]
    fn a_block_pass_writes_the_blocks_its_box_reads() {
        assert_eq!(
            block_area((0, 0, 100, 50), false, 9, (100, 50)),
            (0, 0, 100, 50)
        );
        assert_eq!(
            block_area((20, 5, 30, 10), false, 9, (100, 50)),
            (11, 5, 48, 10)
        );
        assert_eq!(
            block_area((20, 5, 30, 10), true, 9, (100, 50)),
            (20, 0, 30, 24)
        );
        assert_eq!(
            block_area((90, 40, 10, 10), true, 53, (100, 50)),
            (90, 0, 10, 50)
        );
    }

    #[test]
    fn a_cell_costs_the_scratch_its_eleven_float_targets_and_its_pixels_of_q() {
        // 11 targets of 16 bytes, and step by step pixels of 4 bytes.
        assert_eq!(bytes_a_cell(1), 180);
        assert_eq!(bytes_a_cell(2), 192);
        assert_eq!(bytes_a_cell(3), 212);
        assert_eq!(bytes_a_cell(4), 240);
        assert_eq!(SCRATCH_BUDGET_BYTES, 402_653_184);
        assert_eq!(budget_side(SCRATCH_BUDGET_BYTES, 4), 1295);
        assert_eq!(budget_side(SCRATCH_BUDGET_BYTES, 1), 1495);
    }

    /// The window of timing_24mp.jpg at 100 percent at Radius 0.05: 2624 by
    /// 3264 pixels padded to a frame of 4032 by 4000 that begins at column
    /// 960, 1008 by 1000 cells of 4 pixels.
    #[test]
    fn the_100_percent_window_at_radius_0_05_is_one_tile() {
        let plan = plan(0.05, (6000, 4000), (960, 0), (4032, 4000));
        assert_eq!((plan.step, plan.cells), (4, 53));
        assert_eq!(plan.grid(), ((240, 0), (1008, 1000)));
        let side = tile_side(&plan, SCRATCH_BUDGET_BYTES);
        assert_eq!(side, 1008);
        assert!(1008 * 1000 * bytes_a_cell(plan.step) <= SCRATCH_BUDGET_BYTES);
        // The product region a Refine slider refines, and the whole frame
        // an edge control refines.
        for over in [(702, 382, 2628, 3268), (0, 0, 4032, 4000)] {
            let tiles = tiles(&plan, over, side);
            assert_eq!(tiles.len(), 1, "{over:?}");
            assert_eq!(tiles[0].first, (240, 0));
            assert_eq!(tiles[0].count, (1008, 1000));
            assert_eq!(tiles[0].out, over);
        }
    }

    /// An export of 6000 by 4000 pixels at Radius 0.05 is 1500 by 1000 cells
    /// of 4 pixels, 360 MB of scratch under the budget of 384 MiB, and `q` of
    /// 6000 by 4000 pixels under the texture limit: one tile, though a square
    /// of the budget holds only 1295 cells a side.
    #[test]
    fn a_24_megapixel_export_at_step_4_is_one_tile() {
        let plan = plan(0.05, (6000, 4000), (0, 0), (6000, 4000));
        assert_eq!((plan.step, plan.cells), (4, 53));
        assert_eq!(plan.grid(), ((0, 0), (1500, 1000)));
        assert_eq!(1500 * 1000 * bytes_a_cell(plan.step), 360_000_000);
        assert_eq!(budget_side(SCRATCH_BUDGET_BYTES, plan.step), 1295);
        let side = tile_side(&plan, SCRATCH_BUDGET_BYTES);
        assert_eq!(side, 1500);
        let tiles = tiles(&plan, (0, 0, 6000, 4000), side);
        assert_eq!(tiles.len(), 1);
        assert_eq!((tiles[0].first, tiles[0].count), ((0, 0), (1500, 1000)));
        assert!(1500 * plan.step <= TEXTURE_SIDE_LIMIT);
        // A budget under the grid's bytes still cuts it into tiles.
        let under = 1500 * 1000 * bytes_a_cell(plan.step) - 1;
        assert_eq!(tile_side(&plan, under), 1224);
        assert!(self::tiles(&plan, (0, 0, 6000, 4000), 1224).len() > 1);
    }

    /// The texture limit the one-tile rule keeps `q` under is the one a device
    /// of default limits has.
    #[test]
    fn the_texture_side_limit_is_the_default_device_limit() {
        assert_eq!(
            TEXTURE_SIDE_LIMIT,
            wgpu::Limits::default().max_texture_dimension_2d
        );
        assert_eq!(TEXTURE_SIDE_LIMIT, 8192);
        // A grid whose `q` would pass the limit on one side is not one tile,
        // though its cells fit the budget: 2100 cells of 4 pixels is 8400.
        let plan = plan(0.05, (8400, 2000), (0, 0), (8400, 2000));
        assert_eq!(plan.step, 4);
        let (_, grid) = plan.grid();
        assert_eq!(grid, (2100, 500));
        assert!(u64::from(grid.0 * grid.1) * bytes_a_cell(4) <= SCRATCH_BUDGET_BYTES);
        assert_eq!(tile_side(&plan, SCRATCH_BUDGET_BYTES), 1295);
    }

    /// An export of 6000 by 4000 pixels in cells of one pixel is 24 million
    /// cells, far over the budget, and is still refined a tile after another.
    #[test]
    fn a_step_1_export_still_tiles() {
        let plan = plan(0.0012, (6000, 4000), (0, 0), (6000, 4000));
        assert_eq!((plan.step, plan.cells), (1, 5));
        assert_eq!(plan.grid(), ((0, 0), (6000, 4000)));
        let side = tile_side(&plan, SCRATCH_BUDGET_BYTES);
        assert_eq!(side, 1495);
        assert!(u64::from(side * side) * bytes_a_cell(plan.step) <= SCRATCH_BUDGET_BYTES);
        let tiles = tiles(&plan, (0, 0, 6000, 4000), side);
        assert!(tiles.len() > 1);
        assert!(
            tiles
                .iter()
                .all(|tile| tile.count.0 <= side && tile.count.1 <= side)
        );
        println!("a step 1 export: {} tiles of {side} cells", tiles.len());
    }

    /// The `q` target of a tile the budget holds is its side in cells times
    /// the step in pixels, under the largest texture a device of default
    /// limits makes, which is what the headless device asks for.
    #[test]
    fn the_one_tile_side_times_the_step_stays_under_the_texture_limit() {
        let limit = wgpu::Limits::default().max_texture_dimension_2d;
        assert_eq!(limit, 8192);
        for step in 1..=4 {
            let side = budget_side(SCRATCH_BUDGET_BYTES, step);
            assert!(side * step < limit, "step {step}: {side} cells");
        }
        let plan = plan(0.05, (6000, 4000), (960, 0), (4032, 4000));
        assert!(tile_side(&plan, SCRATCH_BUDGET_BYTES) * plan.step < limit);
        assert_eq!(budget_side(SCRATCH_BUDGET_BYTES, 4) * 4, 5180);
    }
}
