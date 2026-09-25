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
//! A tile is drawn in 18 passes. Five take the moments of the source, in one
//! pass of three targets, and their box means; they depend on no mask and no
//! gather, so they are kept while the source, the radius and the tile stay
//! the same, and a tile then costs 13. The moments themselves depend on the
//! side of a cell and the grid alone, never on the radius of the box, and
//! keep targets of their own: a Radius step that keeps the side of a cell
//! draws the four box means of the source over them and not the moments, 17
//! passes. The moments are taken over the cells the gathers work over,
//! joined with the cells already held: a refine over cells they cover takes
//! none, and a refine over cells they partly cover takes them over the strips
//! it adds, at most four rectangles. Each of the three gathers takes four:
//! the moments of `q` over the cells, their box means across and down, and
//! the solve in one pass of four targets. The two later gathers take `q` at
//! each pixel as they sum it, the mask moved by what the gather before
//! solved, so `q` is never stored between gathers. The apply after the last
//! gather moves the mask at full resolution into the r8unorm refined alpha.
//!
//! A device that draws into no more than 32 bytes a sample, such as one made
//! with wgpu's default limits, draws the moments of the source in a pass of
//! two targets and a pass of one, and the solve in two passes of two: the
//! source takes six, a gather five, and a tile 22, or 16 with the moments of
//! the source held and 20 on a Radius step that keeps the side of a cell.
//! The RTX 4080 Laptop GPU on Vulkan and DX12 WARP both offer 128 bytes, so
//! both draw the fused source and the fused solve.
//!
//! A box of [`BLOCK_TAPS`] cells or more (a radius of 12 cells or more)
//! has a block pass before it, which sums [`BLOCK`] cells from every cell on
//! along the axis; the box then adds those sums [`BLOCK`] cells apart and the
//! cells left over, in place of every cell. The ten box passes of a tile gain
//! ten block passes: 28 passes, 19 with the moments of the source held, and
//! 27 on a Radius step that keeps the side of a cell, whose four box means of
//! the source each draw their block pass (32, 22 and 30 with the source and
//! the solve drawn in two passes each). A box keeps its cells, each held at
//! the edges as before; only the order of the sum changes. A smaller box
//! sums every cell in the direct loop, which measured no slower (see
//! [`BLOCK_TAPS`]): a block pass is a pass of its own, and a box of few cells
//! does not win it back.

use bytemuck::{Pod, Zeroable};
use gamut_color::refine::{GATHERS, Plan};

use crate::{FULLSCREEN_VERTICES, fullscreen_primitive};

/// The format of the float targets of the cells.
pub(crate) const MOMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// The bytes a cell costs a target.
const MOMENT_BYTES: u64 = 16;

/// How many Rgba32Float targets of cells a scratch holds of each kind: the
/// source's moments, and apart from them their box means, the far side of
/// every box mean, and what a gather solved. The far side of the mask's box
/// mean is one more.
const SOURCE_TARGETS: usize = 3;
const FAR_TARGETS: usize = 3;
const SOLVED_TARGETS: usize = 4;

/// The most bytes the scratch of a frame takes, its targets of cells, unless
/// one box needs more.
pub(crate) const SCRATCH_BUDGET_BYTES: u64 = 384 * 1024 * 1024;

/// The longest side of a texture a device of default limits makes, which is
/// what the headless device and the app ask for: the targets of cells of one
/// tile over a whole grid stay inside it.
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

/// The bytes one cell of a tile costs the scratch, at every step. Counted
/// are the fourteen Rgba32Float targets of cells, 16 bytes a cell each: the
/// three `r` (the source's moments), the three `s` (their box means), the
/// three `f` (the far side of every box mean), `g` (the far side of the box
/// mean of the mask's moments) and the four `v` (what a gather solved), 224
/// bytes. `q` is never stored, and the refined alpha belongs to its mask and
/// is not counted. The block passes before the boxes add no target: they
/// write `v[0]` and `v[1]`, which no pass reads before the first gather or
/// between the moments of a gather and its solve, and every box lies there
/// (see [`RefinePass::run`]).
pub(crate) const CELL_BYTES: u64 =
    (2 * SOURCE_TARGETS + FAR_TARGETS + 1 + SOLVED_TARGETS) as u64 * MOMENT_BYTES;

/// The most cells a side a square tile holds within `budget` bytes.
pub(crate) fn budget_side(budget: u64) -> u32 {
    u32::try_from((budget / CELL_BYTES).isqrt()).unwrap_or(u32::MAX)
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
    /// The pixels of the render the apply writes.
    pub(crate) out: Rect,
    /// The first cell held and how many, on each axis.
    pub(crate) first: (u32, u32),
    pub(crate) count: (u32, u32),
    /// The cells the gathers are drawn over, as texels of the targets: the
    /// ones the pixels of `out` lie among and the margin around them.
    pub(crate) work: Rect,
}

impl Tile {
    pub(crate) fn uniform(&self, plan: &Plan) -> RefineUniform {
        let (grid_first, grid_count) = plan.grid();
        RefineUniform {
            origin: [plan.origin.0, plan.origin.1],
            size: [plan.size.0, plan.size.1],
            grid_first: [grid_first.0, grid_first.1],
            grid_count: [grid_count.0, grid_count.1],
            tile_first: [self.first.0, self.first.1],
            tile_count: [self.count.0, self.count.1],
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
/// tile when its cells fit `budget` bytes and a texture on each side; the
/// side is then the larger side of the grid. Otherwise as many as
/// a square of `budget` bytes holds and no more than the larger side of the
/// grid. Either way at least what the boxes of the gathers and a few pixels
/// around them need.
pub(crate) fn tile_side(plan: &Plan, budget: u64) -> u32 {
    let (_, grid) = plan.grid();
    let bytes = u64::from(grid.0) * u64::from(grid.1) * CELL_BYTES;
    let whole = bytes <= budget && grid.0 <= TEXTURE_SIDE_LIMIT && grid.1 <= TEXTURE_SIDE_LIMIT;
    let side = if whole {
        grid.0.max(grid.1)
    } else {
        budget_side(budget).min(grid.0.max(grid.1))
    };
    side.max(tile_floor(plan))
}

/// The fewest cells a side a tile of this plan holds: what the boxes of the
/// gathers and a few pixels around them need.
pub(crate) fn tile_floor(plan: &Plan) -> u32 {
    plan.margin() * 2 + 64
}

/// Which part of [`hold`] gave the size of the scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Branch {
    /// No scratch was held: the plan's wanted cells.
    First,
    /// The per-axis maximum of the held size and the plan's wanted cells.
    Fits,
    /// The plan's tiles cut smaller than [`tile_side`], so that the per-axis
    /// maximum fits the budget.
    Cut,
    /// No side from the plan's floor up fits beside the held size: the
    /// scratch is made again at the plan's wanted cells, or kept as it is
    /// when it already holds them, as after a make by this branch.
    Floor,
}

impl Branch {
    /// The name of the branch, as a test reads it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Branch::First => "First",
            Branch::Fits => "Fits",
            Branch::Cut => "Cut",
            Branch::Floor => "Floor",
        }
    }
}

/// The size of the scratch a plan draws with, the side its tiles are cut
/// at, whether the scratch is made for it, and by which branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Hold {
    /// The cells a side the targets of cells hold.
    pub(crate) size: (u32, u32),
    /// The most cells a side a tile of the plan holds.
    pub(crate) side: u32,
    /// Whether the scratch is made, at `size`.
    pub(crate) made: bool,
    pub(crate) branch: Branch,
}

/// The scratch a plan draws with, from the size of the scratch the frame
/// holds, if any, and the budget of the scratch in bytes. Every mask of a
/// frame shares its scratch, and two masks at different steps render one
/// after the other every frame, so the scratch keeps what both want rather
/// than be made again for each: the per-axis maximum of the held size and
/// what the plan wants when that fits the budget. When it does not, the
/// plan's tiles are cut at the largest side from its floor up whose maximum
/// fits, and the scratch keeps its size or grows inside the budget. When no
/// side does, a scratch that already holds what the plan wants at
/// [`tile_side`] on both axes is kept as it is, over the budget, and the
/// tiles are cut at that side: a plan repeated after a floor branch makes
/// nothing again. Only otherwise is the scratch made again at what the plan
/// wants, which may be smaller than what was held.
pub(crate) fn hold(held: Option<(u32, u32)>, plan: &Plan, budget: u64) -> Hold {
    let (_, grid) = plan.grid();
    let most = tile_side(plan, budget);
    let wanted = |side: u32| (grid.0.min(side), grid.1.min(side));
    let Some(held) = held else {
        return Hold {
            size: wanted(most),
            side: most,
            made: true,
            branch: Branch::First,
        };
    };
    // What is held and what the plan wants with tiles of `side` cells.
    let joined = |side: u32| {
        let wanted = wanted(side);
        (held.0.max(wanted.0), held.1.max(wanted.1))
    };
    let fits = |size: (u32, u32)| u64::from(size.0) * u64::from(size.1) * CELL_BYTES <= budget;
    let grown = |side: u32, branch: Branch| {
        let size = joined(side);
        Hold {
            size,
            side,
            made: size != held,
            branch,
        }
    };
    if fits(joined(most)) {
        return grown(most, Branch::Fits);
    }
    if joined(most) == held {
        return Hold {
            size: held,
            side: most,
            made: false,
            branch: Branch::Floor,
        };
    }
    let floor = tile_floor(plan);
    if !fits(joined(floor)) {
        return Hold {
            size: wanted(most),
            side: most,
            made: true,
            branch: Branch::Floor,
        };
    }
    // The maximum grows with the side: the floor fits and `most` does not.
    let (mut fit, mut over) = (floor, most);
    while over - fit > 1 {
        let side = fit + (over - fit) / 2;
        if fits(joined(side)) {
            fit = side;
        } else {
            over = side;
        }
    }
    grown(fit, Branch::Cut)
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

/// The moments of the source the targets `r` hold and their box means the
/// targets `s` hold: the plan they were taken with, but for eps and the
/// amount, the tile, the cells of the tile the moments were taken over, and
/// the radius of the box and the cells of the box means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Held {
    full: (u32, u32),
    origin: (u32, u32),
    size: (u32, u32),
    step: u32,
    /// The radius in cells of the box of the means `s` holds.
    cells: u32,
    first: (u32, u32),
    count: (u32, u32),
    /// The cells `r` holds the moments of.
    area: Rect,
    /// The cells `s` holds the box means of.
    boxed: Rect,
}

impl Held {
    /// Whether these moments were taken with the plan and the tile of
    /// `other`, over whichever cells. The moments of a cell depend on the
    /// side of a cell and the grid, never on the radius of the box, so the
    /// radius is not compared.
    fn takes_as(&self, other: &Held) -> bool {
        Held {
            cells: other.cells,
            area: other.area,
            boxed: other.boxed,
            ..*self
        } == *other
    }

    /// Whether the moments were taken over every cell of `area`. The box
    /// means near the edge of the cells taken read cells not taken, as the
    /// box means of a gather near the edge of its work do, and the margin of
    /// the work keeps both from the pixels a tile writes.
    fn covers(&self, area: Rect) -> bool {
        holds(self.area, area)
    }

    /// Whether the box means were taken with a box of `cells` over every
    /// cell of `area`.
    fn boxes(&self, cells: u32, area: Rect) -> bool {
        self.cells == cells && holds(self.boxed, area)
    }
}

/// Whether `outer` holds every cell of `inner`.
fn holds(outer: Rect, inner: Rect) -> bool {
    let (x, y, width, height) = outer;
    inner.0 >= x
        && inner.1 >= y
        && inner.0 + inner.2 <= x + width
        && inner.1 + inner.3 <= y + height
}

/// The smallest rectangle that holds both.
fn join(a: Rect, b: Rect) -> Rect {
    let (x, y) = (a.0.min(b.0), a.1.min(b.1));
    let right = (a.0 + a.2).max(b.0 + b.2);
    let bottom = (a.1 + a.3).max(b.1 + b.3);
    (x, y, right - x, bottom - y)
}

/// The cells of `outer` outside `inner`, which it holds, as at most four
/// rectangles that do not overlap: the rows above and below `inner` the
/// width of `outer`, then the columns either side of `inner` its height.
fn strips(outer: Rect, inner: Rect) -> Vec<Rect> {
    let (x, y, width, height) = outer;
    let (right, bottom) = (x + width, y + height);
    let (inner_right, inner_bottom) = (inner.0 + inner.2, inner.1 + inner.3);
    [
        (x, y, width, inner.1 - y),
        (x, inner_bottom, width, bottom - inner_bottom),
        (x, inner.1, inner.0 - x, inner.3),
        (inner_right, inner.1, right - inner_right, inner.3),
    ]
    .into_iter()
    .filter(|strip| strip.2 > 0 && strip.3 > 0)
    .collect()
}

/// The float targets of a frame, fourteen of cells.
pub(crate) struct Scratch {
    /// The source's moments: (I, rr), (rg, rb, gg, gb), (bb).
    r: [FloatTarget; SOURCE_TARGETS],
    /// Their box means, in the same layout.
    s: [FloatTarget; SOURCE_TARGETS],
    /// The far side of every box mean. While a mask is gathered `f[0]` holds
    /// the means of (q, q I) and `f[2]` those of (p, p p).
    f: [FloatTarget; FAR_TARGETS],
    /// The far side of the box mean of (p, p p).
    g: FloatTarget,
    /// What a gather solved: 15 numbers a cell.
    v: [FloatTarget; SOLVED_TARGETS],
    /// The cells a side the targets of cells hold.
    size: (u32, u32),
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

    /// Takes the key of the moments out, so the scratch claims none until
    /// [`Scratch::hold`] sets it again: a frame built ahead records a refine
    /// in a slice and sets the key once the slice's commands are submitted.
    pub(crate) fn take_held(&mut self) -> Option<Held> {
        self.held.take()
    }

    /// Sets the key [`Scratch::take_held`] took out.
    pub(crate) fn hold(&mut self, held: Held) {
        self.held = Some(held);
    }

    /// Told apart from every scratch before it, on either frame.
    pub(crate) fn id(&self) -> u64 {
        self.id
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
    /// r0, r1 and r2: the box across of the source's moments.
    source_pair: wgpu::BindGroup,
    source_one: wgpu::BindGroup,
    /// f0, f1 and f2: the box down of the source's moments.
    far_pair: wgpu::BindGroup,
    far_one: wgpu::BindGroup,
    /// f0, f2: the box across of a gather.
    gather_across: wgpu::BindGroup,
    /// s0, s1, s2, f0, f2: the solve.
    solve: wgpu::BindGroup,
    /// v0 to v3: a later gather, which moves the mask as it sums, and the
    /// apply.
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
    /// The moments of the source in one pass of three targets, on a device
    /// that draws into 64 bytes a sample; `None` on one that draws into less.
    source: Option<wgpu::RenderPipeline>,
    /// The moments of the source in a pass of two targets and a pass of
    /// one, when there is no `source`.
    source_a: Option<wgpu::RenderPipeline>,
    source_b: Option<wgpu::RenderPipeline>,
    gather_first: wgpu::RenderPipeline,
    gather_moved: wgpu::RenderPipeline,
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
    apply: wgpu::RenderPipeline,
    /// The distance between two tiles in the uniform buffer.
    stride: u32,
    scratches: u64,
    /// How many times the moments of the source were taken over the whole
    /// work of a tile, in tiles.
    pub(crate) source_builds: u64,
    /// How many strips the moments of the source were taken over, where a
    /// tile works over cells the moments held partly cover.
    pub(crate) source_strips: u64,
    /// The most bytes the scratch takes: [`SCRATCH_BUDGET_BYTES`], unless a
    /// test sets less to draw a render in more tiles.
    scratch_budget: u64,
    /// How many passes and how many tiles this pass family has drawn.
    pub(crate) refine_passes: u32,
    pub(crate) refine_tiles: u32,
    /// Every box sums its cells in the direct loop, whatever its size, so a
    /// test can hold the block sums to it.
    direct_box: bool,
    /// What [`hold`] gave the last plan drawn, so a test can read the size
    /// of the scratch and the side of the tiles.
    pub(crate) last_hold: Option<Hold>,
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
        // The fused solve draws into every solved target at once, and the
        // fused source into every target of the source's moments, fewer
        // bytes a sample: one limit serves both.
        let fused = u64::from(device.limits().max_color_attachment_bytes_per_sample)
            >= SOLVED_TARGETS as u64 * MOMENT_BYTES;
        let split = |fragment: &str, targets: &[wgpu::TextureFormat]| {
            (!fused).then(|| pipeline(fragment, targets))
        };
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        RefinePass {
            source: fused.then(|| pipeline("fs_source", &[MOMENT_FORMAT; SOURCE_TARGETS])),
            source_a: split("fs_source_a", &pair),
            source_b: split("fs_source_b", &one),
            gather_first: pipeline("fs_gather_first", &pair),
            gather_moved: pipeline("fs_gather_moved", &one),
            box_h2: pipeline("fs_box_h2", &pair),
            box_v2: pipeline("fs_box_v2", &pair),
            box_h1: pipeline("fs_box_h1", &one),
            box_v1: pipeline("fs_box_v1", &one),
            block_h2: pipeline("fs_block_h2", &pair),
            block_v2: pipeline("fs_block_v2", &pair),
            block_h1: pipeline("fs_block_h1", &one),
            block_v1: pipeline("fs_block_v1", &one),
            solve: fused.then(|| pipeline("fs_solve", &[MOMENT_FORMAT; SOLVED_TARGETS])),
            solve_a: split("fs_solve_a", &pair),
            solve_b: split("fs_solve_b", &pair),
            apply: pipeline("fs_apply", &[alpha_format]),
            layout,
            stride: (size as u32).div_ceil(alignment) * alignment,
            scratches: 0,
            source_builds: 0,
            source_strips: 0,
            scratch_budget: SCRATCH_BUDGET_BYTES,
            refine_passes: 0,
            refine_tiles: 0,
            direct_box: false,
            last_hold: None,
        }
    }

    /// Whether the moments of the source are drawn in one pass of three
    /// targets and the solve in one pass of four, on a device that draws into
    /// 64 bytes a sample, rather than each in two passes.
    pub(crate) fn fused(&self) -> bool {
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
        let held = scratch.as_ref().map(|held| held.size);
        let hold = hold(held, plan, self.scratch_budget);
        self.last_hold = Some(hold);
        if hold.made {
            let size = hold.size;
            if let (Branch::Floor, Some(held)) = (hold.branch, held) {
                log::warn!(
                    "the refine scratch of {}x{} cells and the {}x{} a plan wants \
                     pass the budget of {} bytes at its floor of {} cells a tile: \
                     made again at {}x{}",
                    held.0,
                    held.1,
                    size.0,
                    size.1,
                    self.scratch_budget,
                    tile_floor(plan),
                    size.0,
                    size.1
                );
            }
            self.scratches += 1;
            let float = |label: &str| FloatTarget {
                view: target(device, label, MOMENT_FORMAT, size.0, size.1).view,
            };
            *scratch = Some(Scratch {
                r: [0; SOURCE_TARGETS].map(|_| float("refine source moments")),
                s: [0; SOURCE_TARGETS].map(|_| float("refine source means")),
                f: [0; FAR_TARGETS].map(|_| float("refine means")),
                g: float("refine mask means"),
                v: [0; SOLVED_TARGETS].map(|_| float("refine solved")),
                size,
                held: None,
                id: self.scratches,
            });
        }
        let scratch = scratch.as_mut().expect("made above");
        let tiles = tiles(plan, over, hold.side);
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
            // Binding 3 is read by no pass; every group binds the alpha there.
            let group = |label: &str, textures: [&FloatTarget; 5]| {
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
                        view(3, alpha),
                        view(4, &textures[0].view),
                        view(5, &textures[1].view),
                        view(6, &textures[2].view),
                        view(7, &textures[3].view),
                        view(8, &textures[4].view),
                    ],
                })
            };
            let (r, s, f, g, v) = (&scratch.r, &scratch.s, &scratch.f, &scratch.g, &scratch.v);
            refined.binds = Some((
                scratch.id,
                Binds {
                    cells: group("refine cells", [&f[1], g, &v[0], &v[1], &v[2]]),
                    source_pair: group("refine source pair", [&r[0], &r[1], &v[0], &v[1], &v[2]]),
                    source_one: group("refine source one", [&r[2], &v[2], &v[0], &v[1], &v[3]]),
                    far_pair: group("refine far pair", [&f[0], &f[1], &v[0], &v[1], &v[2]]),
                    far_one: group("refine far one", [&f[2], &v[2], &v[0], &v[1], &v[3]]),
                    gather_across: group(
                        "refine gather across",
                        [&f[0], &f[2], &v[0], &v[1], &v[2]],
                    ),
                    solve: group("refine solve", [&s[0], &s[1], &s[2], &f[0], &f[2]]),
                    moving: group("refine move", [&v[0], &v[1], &v[2], &v[3], &f[1]]),
                    block_cells: group("refine block cells", [&f[1], g, &v[2], &v[3], &s[2]]),
                    block_source_pair: group(
                        "refine block source pair",
                        [&r[0], &r[1], &v[2], &v[3], g],
                    ),
                    block_source_one: group(
                        "refine block source one",
                        [&r[2], g, &v[2], &v[3], &f[0]],
                    ),
                    block_far_pair: group("refine block far pair", [&f[0], &f[1], &v[2], &v[3], g]),
                    block_far_one: group("refine block far one", [&f[2], g, &v[2], &v[3], &f[0]]),
                    block_gather_across: group(
                        "refine block gather across",
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
            let (f, g, v) = (&scratch.f, &scratch.g, &scratch.v);
            // A box pass of blocks has its block pass first, into v0 (and v1
            // for a pair). Both are free at every box: the solve writes them
            // after the last box of a gather, and the moments of the next
            // gather read them before its first box.
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
            let work = tile.work;
            // The moments of the source over the cells the gathers work
            // over, unless the targets hold them there already: over the
            // whole work when none are held of this plan's grid and tile,
            // and otherwise over the strips the work adds to the cells held.
            // The moments do not depend on the radius of the box; their box
            // means are taken again over the work unless the targets hold
            // them there with this radius.
            let held = Held {
                full: plan.full,
                origin: plan.origin,
                size: plan.size,
                step: plan.step,
                cells: plan.cells,
                first: tile.first,
                count: tile.count,
                area: work,
                boxed: work,
            };
            let had = scratch.held.filter(|had| had.takes_as(&held));
            let (moments, whole) = match had {
                Some(had) if had.covers(work) => (Vec::new(), false),
                Some(had) => (strips(join(had.area, work), had.area), false),
                None => (vec![work], true),
            };
            let (r, s) = (&scratch.r, &scratch.s);
            let cells = &binds.cells;
            for &over in &moments {
                let label = |whole_label: &'static str, strip_label: &'static str| {
                    if whole { whole_label } else { strip_label }
                };
                if let Some(source) = &self.source {
                    pass(
                        label("refine source", "refine source strip"),
                        source,
                        cells,
                        &[&r[0].view, &r[1].view, &r[2].view],
                        over,
                        None,
                    );
                } else {
                    pass(
                        label("refine source a", "refine source a strip"),
                        self.source_a.as_ref().expect("made without a fused source"),
                        cells,
                        &pair(&r[0], &r[1]),
                        over,
                        None,
                    );
                    pass(
                        label("refine source b", "refine source b strip"),
                        self.source_b.as_ref().expect("made without a fused source"),
                        cells,
                        &[&r[2].view],
                        over,
                        None,
                    );
                }
            }
            if whole {
                self.source_builds += 1;
            } else {
                self.source_strips += moments.len() as u64;
            }
            let area = had.map_or(work, |had| join(had.area, work));
            let boxed = had.filter(|had| had.boxes(plan.cells, work));
            scratch.held = Some(match boxed {
                Some(had) => Held { area, ..had },
                None => Held { area, ..held },
            });
            if boxed.is_none() {
                pass(
                    "refine source h a",
                    &self.box_h2,
                    &binds.source_pair,
                    &pair(&f[0], &f[1]),
                    work,
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
                    work,
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
                    work,
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
                    work,
                    Block::down(
                        "refine source v b block",
                        &self.block_v1,
                        &binds.block_far_one,
                    ),
                );
            }
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
                    // q moved from what the gather before solved, pixel by
                    // pixel, as the moments are summed.
                    pass(
                        "refine gather",
                        &self.gather_moved,
                        &binds.moving,
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
                if gather + 1 == GATHERS {
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
    /// taken over, with the same grid, side of a cell and tile, whatever the
    /// radius of the box; a refine over other cells takes them over the
    /// strips it adds. Their box means serve a refine with the same radius
    /// over cells they were taken over.
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
            boxed: (110, 60, 580, 680),
        };
        assert!(held.covers((100, 50, 600, 700)));
        assert!(held.covers((150, 60, 20, 30)));
        assert!(!held.covers((99, 50, 10, 10)));
        assert!(!held.covers((650, 700, 51, 10)));
        assert!(held.takes_as(&Held {
            area: (0, 0, 1008, 1000),
            ..held
        }));
        // A Radius step that keeps the side of a cell keeps the moments.
        let fewer = Held { cells: 52, ..held };
        assert!(fewer.takes_as(&held));
        assert!(held.takes_as(&fewer));
        assert!(held.takes_as(&Held {
            boxed: (0, 0, 1, 1),
            ..held
        }));
        // A step, a grid or a tile of another size does not.
        assert!(!held.takes_as(&Held { step: 3, ..held }));
        assert!(!fewer.takes_as(&Held { step: 3, ..held }));
        assert!(!held.takes_as(&Held {
            size: (4036, 4000),
            ..held
        }));
        assert!(!held.takes_as(&Held {
            first: (0, 0),
            ..held
        }));
        // The box means serve their radius over the cells they were taken
        // over.
        assert!(held.boxes(53, (110, 60, 580, 680)));
        assert!(held.boxes(53, (200, 100, 10, 10)));
        assert!(!held.boxes(52, (200, 100, 10, 10)));
        assert!(!held.boxes(53, (100, 50, 600, 700)));
        assert_eq!(
            join((100, 50, 600, 700), (650, 700, 51, 10)),
            (100, 50, 601, 700)
        );
        assert_eq!(join((10, 20, 5, 5), (0, 0, 3, 3)), (0, 0, 15, 25));
    }

    /// The strips of a rectangle outside one it holds cover every cell of
    /// it outside that one exactly once, in at most four rectangles.
    #[test]
    fn the_strips_cover_what_the_join_adds_once() {
        let cases = [
            // Grown by 3 cells on every side.
            ((97, 47, 606, 706), (100, 50, 600, 700), 4),
            // Grown to the right alone.
            ((100, 50, 601, 700), (100, 50, 600, 700), 1),
            // Grown up and to the left.
            ((90, 40, 610, 710), (100, 50, 600, 700), 2),
            // The same.
            ((100, 50, 600, 700), (100, 50, 600, 700), 0),
        ];
        for (outer, inner, count) in cases {
            let strips = strips(outer, inner);
            assert_eq!(strips.len(), count, "{outer:?} {inner:?}");
            for y in outer.1..outer.1 + outer.3 {
                for x in outer.0..outer.0 + outer.2 {
                    let cell = (x, y, 1, 1);
                    let times = strips.iter().filter(|s| holds(**s, cell)).count();
                    let wanted = usize::from(!holds(inner, cell));
                    assert_eq!(times, wanted, "{outer:?} {inner:?} at {x}, {y}");
                }
            }
        }
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
        assert_eq!(offset("step"), offset_of!(RefineUniform, step));
        assert_eq!(offset("cells"), offset_of!(RefineUniform, cells));
        assert_eq!(offset("eps"), offset_of!(RefineUniform, eps));
        assert_eq!(offset("amount"), offset_of!(RefineUniform, amount));
        assert_eq!(offset("blocks"), offset_of!(RefineUniform, blocks));
        // The pad fills the shader struct to its 8 byte alignment.
        assert_eq!(offset_of!(RefineUniform, pad), 68);
        assert_eq!(members.len(), 11);
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
        assert_eq!(MOMENT_BYTES, 16);
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
        let small = 512 * 512 * CELL_BYTES;
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
        assert!(side > budget_side(SCRATCH_BUDGET_BYTES));
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
    fn a_cell_costs_the_scratch_its_fourteen_float_targets() {
        // 14 targets of 16 bytes, at every step: `q` is never stored.
        assert_eq!(CELL_BYTES, 224);
        assert_eq!(SCRATCH_BUDGET_BYTES, 402_653_184);
        assert_eq!(budget_side(SCRATCH_BUDGET_BYTES), 1340);
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
        const { assert!(1008 * 1000 * CELL_BYTES <= SCRATCH_BUDGET_BYTES) };
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
    /// of 4 pixels, 336 MB of scratch under the budget of 384 MiB, and
    /// targets of 1500 by 1000 cells under the texture limit: one tile.
    #[test]
    fn a_24_megapixel_export_at_step_4_is_one_tile() {
        let plan = plan(0.05, (6000, 4000), (0, 0), (6000, 4000));
        assert_eq!((plan.step, plan.cells), (4, 53));
        assert_eq!(plan.grid(), ((0, 0), (1500, 1000)));
        assert_eq!(1500 * 1000 * CELL_BYTES, 336_000_000);
        assert_eq!(budget_side(SCRATCH_BUDGET_BYTES), 1340);
        let side = tile_side(&plan, SCRATCH_BUDGET_BYTES);
        assert_eq!(side, 1500);
        let tiles = tiles(&plan, (0, 0, 6000, 4000), side);
        assert_eq!(tiles.len(), 1);
        assert_eq!((tiles[0].first, tiles[0].count), ((0, 0), (1500, 1000)));
        const { assert!(1500 <= TEXTURE_SIDE_LIMIT) };
        // A budget under the grid's bytes still cuts it into tiles.
        let under = 1500 * 1000 * CELL_BYTES - 1;
        assert_eq!(tile_side(&plan, under), 1224);
        assert!(self::tiles(&plan, (0, 0, 6000, 4000), 1224).len() > 1);
    }

    /// The texture limit the one-tile rule keeps the targets of cells under
    /// is the one a device of default limits has.
    #[test]
    fn the_texture_side_limit_is_the_default_device_limit() {
        assert_eq!(
            TEXTURE_SIDE_LIMIT,
            wgpu::Limits::default().max_texture_dimension_2d
        );
        assert_eq!(TEXTURE_SIDE_LIMIT, 8192);
        // A grid of 2100 by 500 cells of 4 pixels, 8400 pixels wide, fits
        // the budget and the limit: one tile of the whole grid, though a
        // square of the budget holds only 1340 cells a side.
        let plan = plan(0.05, (8400, 2000), (0, 0), (8400, 2000));
        assert_eq!(plan.step, 4);
        let (_, grid) = plan.grid();
        assert_eq!(grid, (2100, 500));
        assert!(u64::from(grid.0 * grid.1) * CELL_BYTES <= SCRATCH_BUDGET_BYTES);
        assert_eq!(tile_side(&plan, SCRATCH_BUDGET_BYTES), 2100);
        // A grid whose targets would pass the limit on one side is not one
        // tile, though its cells fit the budget: 9000 cells of one pixel.
        let plan = self::plan(0.0008, (9000, 150), (0, 0), (9000, 150));
        assert_eq!(plan.step, 1);
        let (_, grid) = plan.grid();
        assert_eq!(grid, (9000, 150));
        assert!(u64::from(grid.0 * grid.1) * CELL_BYTES <= SCRATCH_BUDGET_BYTES);
        assert_eq!(tile_side(&plan, SCRATCH_BUDGET_BYTES), 1340);
    }

    /// An export of 6000 by 4000 pixels in cells of one pixel is 24 million
    /// cells, far over the budget, and is still refined a tile after another.
    #[test]
    fn a_step_1_export_still_tiles() {
        let plan = plan(0.0012, (6000, 4000), (0, 0), (6000, 4000));
        assert_eq!((plan.step, plan.cells), (1, 5));
        assert_eq!(plan.grid(), ((0, 0), (6000, 4000)));
        let side = tile_side(&plan, SCRATCH_BUDGET_BYTES);
        assert_eq!(side, 1340);
        assert!(u64::from(side * side) * CELL_BYTES <= SCRATCH_BUDGET_BYTES);
        let tiles = tiles(&plan, (0, 0, 6000, 4000), side);
        assert!(tiles.len() > 1);
        assert!(
            tiles
                .iter()
                .all(|tile| tile.count.0 <= side && tile.count.1 <= side)
        );
        println!("a step 1 export: {} tiles of {side} cells", tiles.len());
    }

    /// The targets of cells of a tile the budget holds are its side in cells,
    /// under the largest texture a device of default limits makes, which is
    /// what the headless device asks for.
    #[test]
    fn the_side_of_a_tile_the_budget_holds_stays_under_the_texture_limit() {
        let limit = wgpu::Limits::default().max_texture_dimension_2d;
        assert_eq!(limit, 8192);
        assert!(budget_side(SCRATCH_BUDGET_BYTES) < limit);
        let plan = plan(0.05, (6000, 4000), (960, 0), (4032, 4000));
        assert!(tile_side(&plan, SCRATCH_BUDGET_BYTES) < limit);
    }

    /// What a walk of plans through [`hold`] held, from no scratch, in one
    /// frame whose plans all take the photo whole.
    struct Walk {
        /// What [`hold`] gave after each plan.
        steps: Vec<Hold>,
        /// How many times the scratch was made.
        made: u32,
        /// How many plans left the scratch over the budget.
        over: u32,
        /// How many plans the floor branch made the scratch for.
        floors: u32,
        /// Every rule a plan broke, one line each.
        broken: Vec<String>,
    }

    /// Walks `radii` in order on a photo of `photo` pixels, rendered whole,
    /// through [`hold`] under `budget`, and checks after each plan that the
    /// scratch is inside the budget (or, after a floor branch, made at what
    /// the plan wants or kept as held at [`tile_side`]), that it holds what
    /// the plan wants at the side its tiles are cut at, that the side lies
    /// from the plan's floor to [`tile_side`], and that the scratch shrinks
    /// only when the floor branch makes it again.
    fn walk(name: &str, photo: (u32, u32), radii: &[f32], budget: u64, print: bool) -> Walk {
        let mut out = Walk {
            steps: Vec::new(),
            made: 0,
            over: 0,
            floors: 0,
            broken: Vec::new(),
        };
        let mut held = None;
        for (index, &radius) in radii.iter().enumerate() {
            let plan = plan(radius, photo, (0, 0), photo);
            let hold = hold(held, &plan, budget);
            let (_, grid) = plan.grid();
            let wanted = (grid.0.min(hold.side), grid.1.min(hold.side));
            let bytes = u64::from(hold.size.0) * u64::from(hold.size.1) * CELL_BYTES;
            let (floor, most) = (tile_floor(&plan), tile_side(&plan, budget));
            let at = format!(
                "{name}, plan {} (radius {radius}, step {}, grid {}x{})",
                index + 1,
                plan.step,
                grid.0,
                grid.1
            );
            let mut broken = Vec::new();
            if hold.branch == Branch::Floor {
                out.floors += 1;
                if hold.made && hold.size != wanted {
                    broken.push(format!(
                        "the floor branch holds {:?}, not a scratch made at {wanted:?}",
                        hold.size
                    ));
                }
                if !hold.made && (Some(hold.size) != held || hold.side != most) {
                    broken.push(format!(
                        "the floor branch keeps {:?} at side {}, not {held:?} at {most}",
                        hold.size, hold.side
                    ));
                }
            } else if bytes > budget {
                broken.push(format!(
                    "held {}x{} is {bytes} bytes, over {budget}",
                    hold.size.0, hold.size.1
                ));
            }
            if bytes > budget {
                out.over += 1;
            }
            if hold.size.0 < wanted.0 || hold.size.1 < wanted.1 {
                broken.push(format!(
                    "held {:?} does not cover {wanted:?} at side {}",
                    hold.size, hold.side
                ));
            }
            if !(floor..=most).contains(&hold.side) {
                broken.push(format!("side {} outside {floor} to {most}", hold.side));
            }
            let shrank =
                held.is_some_and(|size: (u32, u32)| hold.size.0 < size.0 || hold.size.1 < size.1);
            if shrank && hold.branch != Branch::Floor {
                broken.push(format!(
                    "the scratch shrank from {held:?} to {:?}",
                    hold.size
                ));
            }
            if !hold.made && Some(hold.size) != held {
                broken.push(format!("{:?} held without a make over {held:?}", hold.size));
            }
            if hold.made {
                out.made += 1;
            }
            if print {
                println!(
                    "{at}: side {} of {floor} to {most}, held {}x{} = {bytes} bytes, {} ({:?})",
                    hold.side,
                    hold.size.0,
                    hold.size.1,
                    if hold.made { "made" } else { "kept" },
                    hold.branch
                );
            }
            out.broken
                .extend(broken.into_iter().map(|line| format!("{at}: {line}")));
            out.steps.push(hold);
            held = Some(hold.size);
        }
        out
    }

    /// Two masks of one frame at different steps render alternately every
    /// frame and share its scratch: whatever the order of their plans, the
    /// scratch stays inside the budget, and a step that does not fit beside
    /// what is held cuts its tiles smaller rather than grow the scratch.
    #[test]
    fn a_frame_of_plans_at_steps_1_to_4_keeps_its_scratch_inside_the_budget() {
        use std::collections::BTreeSet;

        // The name, the photo, the radii in order, the budget, and after each
        // plan the size held and the side of its tiles, and the makes.
        struct Case {
            name: &'static str,
            photo: (u32, u32),
            radii: Vec<f32>,
            budget: u64,
            held: Vec<((u32, u32), u32)>,
            made: u32,
        }
        let budget = SCRATCH_BUDGET_BYTES;
        let small = 16 * 1024 * 1024;
        let cases = [
            // The 24 megapixel photo: step 4 whole, then step 1 cut at the
            // rows 1500 columns leave the budget.
            Case {
                name: "A 0.05 then 0.0012",
                photo: (6000, 4000),
                radii: vec![0.05, 0.0012],
                budget,
                held: vec![((1500, 1000), 1500), ((1500, 1198), 1198)],
                made: 2,
            },
            Case {
                name: "A 0.0012 then 0.05",
                photo: (6000, 4000),
                radii: vec![0.0012, 0.05],
                budget,
                held: vec![((1340, 1340), 1340), ((1341, 1340), 1341)],
                made: 2,
            },
            // The worst found: a step 3 grid of 2731 by 658 cells, whole.
            Case {
                name: "B 0.0035 then 0.001",
                photo: (8192, 1974),
                radii: vec![0.0035, 0.001],
                budget,
                held: vec![((2731, 658), 2731), ((2731, 658), 658)],
                made: 1,
            },
            Case {
                name: "B 0.0035, 0.001, 0.05, 0.0025",
                photo: (8192, 1974),
                radii: vec![0.0035, 0.001, 0.05, 0.0025],
                budget,
                held: vec![
                    ((2731, 658), 2731),
                    ((2731, 658), 658),
                    ((2731, 658), 2048),
                    ((2731, 658), 658),
                ],
                made: 1,
            },
            Case {
                name: "B 0.001 then 0.0035",
                photo: (8192, 1974),
                radii: vec![0.001, 0.0035],
                budget,
                held: vec![((1340, 1340), 1340), ((1341, 1340), 1341)],
                made: 2,
            },
            // The control: no grid of the largest square photo is whole.
            Case {
                name: "C 0.05, 0.001, 0.0025, 0.0035",
                photo: (8192, 8192),
                radii: vec![0.05, 0.001, 0.0025, 0.0035],
                budget,
                held: vec![((1340, 1340), 1340); 4],
                made: 1,
            },
            // Two masks on the 24 megapixel photo, rendered alternately ten
            // times: made twice, then held.
            Case {
                name: "D 0.05 and 0.0012 ten times",
                photo: (6000, 4000),
                radii: [0.05, 0.0012].repeat(10),
                budget,
                held: [((1500, 1000), 1500), ((1500, 1198), 1198)]
                    .into_iter()
                    .chain([((1500, 1198), 1500), ((1500, 1198), 1198)].repeat(9))
                    .collect(),
                made: 2,
            },
            // The floor branch: at 16 MiB the box of step 4 at 0.049 needs
            // 278 cells a side, and 278 by 278 passes the budget alone. The
            // same plan three more times keeps what the floor branch made.
            Case {
                name: "E 0.03 then 0.049 at 16 MiB",
                photo: (1280, 4000),
                radii: vec![0.03, 0.049, 0.049, 0.049, 0.049],
                budget: small,
                held: vec![
                    ((273, 273), 273),
                    ((278, 278), 278),
                    ((278, 278), 278),
                    ((278, 278), 278),
                    ((278, 278), 278),
                ],
                made: 2,
            },
        ];
        let mut broken = Vec::new();
        for case in &cases {
            let walk = walk(case.name, case.photo, &case.radii, case.budget, true);
            broken.extend(walk.broken);
            let read: Vec<_> = walk
                .steps
                .iter()
                .map(|hold| (hold.size, hold.side))
                .collect();
            if read != case.held {
                broken.push(format!(
                    "{}: held and sides {read:?}, expected {:?}",
                    case.name, case.held
                ));
            }
            if walk.made != case.made {
                broken.push(format!(
                    "{}: made {} times, expected {}",
                    case.name, walk.made, case.made
                ));
            }
            println!("{}: made {} times", case.name, walk.made);
        }
        // Case D's two masks: made at most twice over the ten renders.
        let alternate = walk("D", (6000, 4000), &[0.05, 0.0012].repeat(10), budget, false);
        if alternate.made > 2 {
            broken.push(format!("D: made {} times over ten renders", alternate.made));
        }
        // Case E's second plan is the floor branch, made again at what it
        // wants and over the small budget, as one box needs. The same plan
        // after it keeps that scratch as it is.
        let floor = walk(
            "E",
            (1280, 4000),
            &[0.03, 0.049, 0.049, 0.049, 0.049],
            small,
            false,
        );
        let last = floor.steps[1];
        if (last.branch, last.made) != (Branch::Floor, true) {
            broken.push(format!("E: plan 2 is {last:?}, not the floor branch"));
        }
        for (index, kept) in floor.steps.iter().enumerate().skip(2) {
            if kept.made || kept.size != last.size {
                broken.push(format!(
                    "E: plan {} is {kept:?}, not {:?} kept",
                    index + 1,
                    last.size
                ));
            }
        }

        // The sweep: every order of four radii, 12, 20 and 28 pixels and 0.05
        // of the longer side (steps 1 to 4), each order walked twice, on
        // eight photos.
        let orders: Vec<[usize; 4]> = (0..256usize)
            .map(|code| [code & 3, (code >> 2) & 3, (code >> 4) & 3, (code >> 6) & 3])
            .filter(|order| order.iter().fold(0, |seen, index| seen | 1 << index) == 15)
            .collect();
        assert_eq!(orders.len(), 24);
        let photos = [
            (6000, 4000),
            (8192, 1974),
            (8192, 8192),
            (8192, 1000),
            (4032, 3024),
            (1280, 1600),
            (8192, 200),
            (1974, 8192),
        ];
        let (mut plans, mut over, mut floors) = (0, 0, 0);
        for photo in photos {
            let longer = photo.0.max(photo.1) as f32;
            let radii = [12.0 / longer, 20.0 / longer, 28.0 / longer, 0.05];
            let mut sizes = BTreeSet::new();
            let (mut photo_over, mut photo_floors) = (0, 0);
            for order in &orders {
                let twice: Vec<f32> = order.iter().chain(order).map(|&k| radii[k]).collect();
                let name = format!("sweep {}x{} order {order:?}", photo.0, photo.1);
                let walk = walk(&name, photo, &twice, budget, false);
                plans += walk.steps.len();
                photo_over += walk.over;
                photo_floors += walk.floors;
                sizes.extend(walk.steps.iter().map(|hold| hold.size));
                broken.extend(walk.broken);
            }
            let sizes: Vec<String> = sizes
                .iter()
                .map(|size| {
                    let bytes = u64::from(size.0) * u64::from(size.1) * CELL_BYTES;
                    format!("{}x{} = {bytes} bytes", size.0, size.1)
                })
                .collect();
            println!(
                "sweep {}x{}: {photo_over} over the budget, {photo_floors} floor branch, held {}",
                photo.0,
                photo.1,
                sizes.join(", ")
            );
            over += photo_over;
            floors += photo_floors;
        }
        println!("sweep: {plans} plans, {over} over the budget, {floors} floor branch");
        if over != 0 || floors != 0 {
            broken.push(format!(
                "sweep: {over} plans over the budget and {floors} floor branch, expected 0 and 0"
            ));
        }
        for line in &broken {
            println!("broken: {line}");
        }
        assert!(broken.is_empty(), "{} broken", broken.len());
    }
}
