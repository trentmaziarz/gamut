//! Refine edges on the CPU: the finished alpha of a mask moved onto the edges
//! of the picture under it. refine.wgsl in gamut-gpu mirrors this module; the
//! golden tests hold the two together.
//!
//! The input `p` is the alpha of the mask as the GPU stores it. The guide `I`
//! is the SOURCE pixel, encoded to ACEScct, never the developed one, so no
//! slider of the develop chain moves a refined mask.
//!
//! The filter gathers the colours of the guide around each place twice: once
//! for what lies inside the mask and once for what lies outside it, each pixel
//! counted by how far it is in or out. It then places every pixel between the
//! two mean colours, along the one direction of colour that best tells the
//! classes apart. A pixel at the outside's end leaves the mask, one at the
//! inside's end joins it, and one between keeps the alpha it had. Over a box
//! around each place, with `q` the weight of the inside class (`wq` below),
//!
//! ```text
//! mu1 = E[q I] / E[q]            mu0 = (E[I] - E[q I]) / (1 - E[q])
//! Sw  = E[I I] - E[q] mu1 mu1' - (1 - E[q]) mu0 mu0' + eps Id
//! a   = Sw^-1 (mu1 - mu0)        d2  = (mu1 - mu0) . a
//! P   = Sw^-1 - a a' / d2        mid = (mu1 + mu0) / 2
//! ```
//!
//! and at a pixel of colour `I`
//!
//! ```text
//! est = 0.5 + (a . I - a . mid) / d2, held inside 0 to 1
//! m   = (I - mid)' P (I - mid)
//! ```
//!
//! `est` is where the colour sits along the line between the classes and `m`
//! how far it lies off that line; a colour far off the line keeps the alpha
//! it was given. There are [`GATHERS`] gathers. The first weighs the classes
//! by the mask as drawn and each later one by the result of the gather
//! before, so spill that has left stops colouring the inside class. Every
//! gather moves the mask AS DRAWN with its own class colours: the mask as
//! moved so far is only ever a weight. The refined alpha is
//! `p + (q - p) amount / 100` with `q` the result of the last gather.
//!
//! Two gates hold the move. Where the box lies wholly inside or wholly
//! outside the mask nothing moves, so the mask never grows a halo; and a
//! mask that is soft at the scale of the box is soft on purpose and comes
//! back as drawn. Where the guide is flat the classes do not separate and
//! the mask comes back bit for bit.
//!
//! A reached field `rf` says how far each place is joined to the outside of
//! the mask through like colour: seeded on the cell grid by the outside as
//! drawn and spread in a bounded number of passes, [`reached`]. A pixel
//! counts in the inside class only as far as the outside does not reach it,
//!
//! ```text
//! keep = 1 - smoothstep(0.7, 1.0, rf)
//! wq   = q keep + max(q - p, 0) (1 - keep)
//! ```
//!
//! so alpha a move added above the mask as drawn counts inside whether
//! reached or not. The move comes in only where some of the inside in the
//! box is unreached, [`gate`], and a pixel leaves the mask only where the
//! outside reaches it. A stroke drawn much wider than its object gives up
//! the spill the outside reaches through like colour, and a mask drawn over
//! one colour on both of its sides holds still.
//!
//! The moments are taken on a grid of cells: a cell is [`Plan::step`] pixels
//! square and lies on the pixel grid of the WHOLE picture at the scale of the
//! render, not on the grid of the window that is being rendered, so a zoomed
//! window and the full render hold the same cells. What is solved a cell
//! comes back up bilinearly and the move reads the guide at full resolution.
//! Everything is measured from the size of the whole picture, so the fitted
//! view, 100 percent and the export show the same edge at their own scales.
//!
//! Arithmetic is f32 in the order the shader sums in, and every smoothstep is
//! written out as its polynomial. `store` is applied wherever the GPU writes
//! a texture of the filter; those are 32 bit float targets, so a golden test
//! passes the identity.

use gamut_core::mask::Refine;

use crate::acescct;
use crate::mask::Geometry;

/// The guide's eps at a sensitivity of 0, before squaring.
pub const EPS_LOOSE: f32 = 0.1;

/// The guide's eps at a sensitivity of 100, before squaring.
pub const EPS_STRICT: f32 = 0.005;

/// The radius in render pixels one step of the cell grid is taken for.
pub const PIXELS_A_STEP: f32 = 8.0;

/// The widest cell, in render pixels.
pub const MAX_STEP: u32 = 4;

/// The half side of the box as a share of the radius: 1 over the root of 2,
/// so the corner of the box lies one radius out and nothing moves further
/// than the radius.
pub const BOX_OF_RADIUS: f32 = 0.71;

/// How many times the classes are gathered.
pub const GATHERS: usize = 3;

/// The moments of the source in one cell: I (3), then I I as rr, rg, rb, gg,
/// gb, bb. No mask and no gather changes them.
pub const SOURCE_MOMENTS: usize = 9;

/// The moments of the mask as drawn in one cell: p and p p.
pub const MASK_MOMENTS: usize = 2;

/// The moments of one gather in one cell: the weight of the inside class
/// `wq` and `wq I` (3).
pub const GATHER_MOMENTS: usize = 4;

/// What one gather solves a cell: a (3), s, d2, c, P as rr, rg, rb, gg, gb,
/// bb, and mid (3).
pub const SOLVED: usize = 15;

/// The share of the smaller class in the box over which the move comes in.
pub const COVER_LOW: f32 = 0.0;
pub const COVER_HIGH: f32 = 0.1;

/// The hardness of the mask in the box over which the move comes in: 0 is
/// one flat grey over the box and 1 only 0 and 1.
pub const HARD_LOW: f32 = 0.3;
pub const HARD_HIGH: f32 = 0.6;

/// The separation of the classes over which the move comes in.
pub const SEPARATE_LOW: f32 = 0.15;
pub const SEPARATE_HIGH: f32 = 0.8;

/// The squared distance off the colour line over which the move goes out.
pub const LINE_LOW: f32 = 0.6;
pub const LINE_HIGH: f32 = 1.2;

/// The place along the colour line under which a pixel leaves the mask.
pub const OUT_LOW: f32 = 0.15;
pub const OUT_HIGH: f32 = 0.4;

/// The place along the colour line over which a pixel joins the mask.
pub const IN_LOW: f32 = 0.6;
pub const IN_HIGH: f32 = 0.85;

/// The least share of a class its mean colour is divided by.
pub const SHARE_FLOOR: f32 = 1e-6;

/// The least separation anything is divided by.
pub const SEPARATION_FLOOR: f32 = 1e-9;

/// The likeness of two cells the reached field spreads between is
/// `exp(-|Ec[I](x) - Ec[I](n)|^2 / (LIKENESS eps))`.
pub const LIKENESS: f32 = 2.0;

/// The reached field over which a pixel leaves the inside class.
pub const KEEP_LOW: f32 = 0.7;
pub const KEEP_HIGH: f32 = 1.0;

/// The mean of the unreached inside, `E[p keep]` over the box, over which
/// the move comes in.
pub const UNREACHED_LOW: f32 = 0.0;
pub const UNREACHED_HIGH: f32 = 0.02;

/// The reached field over which a pixel may leave the mask.
pub const LEAVE_LOW: f32 = 0.2;
pub const LEAVE_HIGH: f32 = 0.5;

/// The eps of the filter at a sensitivity of 0 to 100: a log scale from
/// [`EPS_LOOSE`] squared to [`EPS_STRICT`] squared.
pub fn eps(sensitivity: f32) -> f32 {
    let root = EPS_LOOSE * (EPS_STRICT / EPS_LOOSE).powf(sensitivity / 100.0);
    root * root
}

/// The guide of a source pixel in linear Rec.2020.
pub fn guide(px: [f32; 3]) -> [f32; 3] {
    acescct::encode_pixel(px)
}

/// 0 under `low`, 1 over `high` and the smoothstep polynomial between. The
/// shader writes the same polynomial out and never calls its builtin.
pub fn smooth(x: f32, low: f32, high: f32) -> f32 {
    let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// What the filter of one mask works with on one render. The GPU builds its
/// uniforms from the same plan, so the two never differ in a number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    /// The size the whole picture has at the scale of the render.
    pub full: (u32, u32),
    /// Where the render begins on the whole picture, in its pixels.
    pub origin: (u32, u32),
    /// The size of the render.
    pub size: (u32, u32),
    /// The side of a cell in render pixels.
    pub step: u32,
    /// The radius of the box in cells.
    pub cells: u32,
    /// How far the reached field spreads at most, in cells: the radius in
    /// cells, `floor(r / step)`.
    pub flood: u32,
    pub eps: f32,
    /// How much of the refined alpha is taken, 0 to 1.
    pub amount: f32,
}

impl Plan {
    /// `refine` is sanitised. `full`, `origin` and `size` are in pixels of
    /// the whole picture at the scale of the render.
    pub fn new(refine: &Refine, full: (u32, u32), origin: (u32, u32), size: (u32, u32)) -> Self {
        let radius = refine.radius * full.0.max(full.1) as f32;
        let step = ((radius / PIXELS_A_STEP).floor() as u32).clamp(1, MAX_STEP);
        let cells = ((radius * BOX_OF_RADIUS / step as f32).round() as u32).max(1);
        let flood = (radius / step as f32).floor() as u32;
        Plan {
            full,
            origin,
            size,
            step,
            cells,
            flood,
            eps: eps(refine.sensitivity),
            amount: refine.amount / 100.0,
        }
    }

    /// The plan of a render with this geometry.
    pub fn for_geometry(refine: &Refine, geometry: &Geometry) -> Self {
        let window = geometry.window;
        let (w, h) = (geometry.size.0 as f32, geometry.size.1 as f32);
        let full = (
            (w / window.width.max(1e-6)).round().max(1.0),
            (h / window.height.max(1e-6)).round().max(1.0),
        );
        let origin = ((window.x * full.0).round(), (window.y * full.1).round());
        Plan::new(
            refine,
            (full.0 as u32, full.1 as u32),
            (origin.0.max(0.0) as u32, origin.1.max(0.0) as u32),
            geometry.size,
        )
    }

    /// How many cells around the two a pixel lies among one tile of the GPU
    /// holds, and the twin reads: a box for each gather, the cell beside for
    /// the bilinear step of the two gathers before the last, and the reached
    /// field gather 1 weighs its inside by: the cell beside for its bilinear
    /// read at the pixel and the flood's cells, [`Plan::flood`], which bound
    /// the steps of its passes. The flood's share is about one Radius.
    pub fn margin(&self) -> u32 {
        GATHERS as u32 * self.cells + (GATHERS as u32 - 1) + 1 + self.flood
    }

    /// How far around a pixel the filter reads, in render pixels: the margin,
    /// the cell beside for the bilinear step of the last gather, and the cell
    /// a window may cut at its border. Three boxes of 0.71 Radius and the
    /// flood's Radius make it about 3.13 Radius, and the cells beside add
    /// five cells.
    pub fn reach(&self) -> u32 {
        (self.margin() + 2) * self.step
    }

    /// The first cell of the render on each axis and how many it touches.
    pub fn grid(&self) -> ((u32, u32), (u32, u32)) {
        let axis = |origin: u32, size: u32| {
            let first = origin / self.step;
            let last = (origin + size.max(1) - 1) / self.step;
            (first, last - first + 1)
        };
        let (x, columns) = axis(self.origin.0, self.size.0);
        let (y, rows) = axis(self.origin.1, self.size.1);
        ((x, y), (columns, rows))
    }
}

/// [`Plan::reach`] of a sanitised refine on a render whose whole picture is
/// `full` pixels, or 0 while it is off: what a window is padded by. It holds
/// the flood's cells of the reached field, so it is about 3.13 Radius.
pub fn reach(refine: &Refine, full: (u32, u32)) -> u32 {
    if refine.is_off() {
        0
    } else {
        Plan::new(refine, full, (0, 0), full).reach()
    }
}

/// The mean of `sample` over the pixels of every cell of the render's grid
/// that the render holds. `sample` takes the index of a pixel of the render.
pub fn cell_moments<const N: usize>(
    plan: &Plan,
    sample: &dyn Fn(usize) -> [f32; N],
    store: &dyn Fn(f32) -> f32,
) -> Vec<[f32; N]> {
    let ((first_x, first_y), (columns, rows)) = plan.grid();
    let (width, height) = plan.size;
    let mut out = Vec::with_capacity((columns * rows) as usize);
    for cy in 0..rows {
        for cx in 0..columns {
            // The pixels of the render inside this cell.
            let span = |first: u32, c: u32, origin: u32, size: u32| {
                let low = ((first + c) * plan.step).max(origin) - origin;
                let high = ((first + c + 1) * plan.step).min(origin + size) - origin;
                (low, high)
            };
            let (x0, x1) = span(first_x, cx, plan.origin.0, width);
            let (y0, y1) = span(first_y, cy, plan.origin.1, height);
            let mut sum = [0.0f32; N];
            for y in y0..y1 {
                for x in x0..x1 {
                    let values = sample((y * width + x) as usize);
                    for (total, value) in sum.iter_mut().zip(values) {
                        *total += value;
                    }
                }
            }
            let count = ((x1 - x0) * (y1 - y0)).max(1) as f32;
            out.push(sum.map(|total| store(total / count)));
        }
    }
    out
}

/// The mean of every channel over a box of `radius` cells, clamped at the
/// edges of the grid, as two passes: across, then down.
pub fn box_mean<const N: usize>(
    source: &[[f32; N]],
    size: (u32, u32),
    radius: u32,
    store: &dyn Fn(f32) -> f32,
) -> Vec<[f32; N]> {
    let (w, h) = (size.0 as i32, size.1 as i32);
    let radius = radius as i32;
    let count = (2 * radius + 1) as f32;
    let pass = |source: &[[f32; N]], dx: i32, dy: i32| -> Vec<[f32; N]> {
        let mut out = vec![[0.0; N]; source.len()];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; N];
                for i in -radius..=radius {
                    let sx = (x + i * dx).clamp(0, w - 1);
                    let sy = (y + i * dy).clamp(0, h - 1);
                    let sample = &source[(sy * w + sx) as usize];
                    for (total, value) in sum.iter_mut().zip(sample) {
                        *total += value;
                    }
                }
                out[(y * w + x) as usize] = sum.map(|total| store(total / count));
            }
        }
        out
    };
    let across = pass(source, 1, 0);
    pass(&across, 0, 1)
}

/// The means of the source's moments over the box, a cell. They depend on no
/// mask and no gather, so the GPU holds them once a source and radius.
pub fn source_means(
    guides: &[[f32; 3]],
    plan: &Plan,
    store: &dyn Fn(f32) -> f32,
) -> Vec<[f32; SOURCE_MOMENTS]> {
    let (_, grid) = plan.grid();
    box_mean(&source_cells(guides, plan, store), grid, plan.cells, store)
}

/// The means of the source's moments over each cell, before the box. The
/// first three are the cell mean of the guide, `Ec[I]`, which the reached
/// field compares.
pub fn source_cells(
    guides: &[[f32; 3]],
    plan: &Plan,
    store: &dyn Fn(f32) -> f32,
) -> Vec<[f32; SOURCE_MOMENTS]> {
    cell_moments(
        plan,
        &|i| {
            let g = guides[i];
            [
                g[0],
                g[1],
                g[2],
                g[0] * g[0],
                g[0] * g[1],
                g[0] * g[2],
                g[1] * g[1],
                g[1] * g[2],
                g[2] * g[2],
            ]
        },
        store,
    )
}

/// The reached field on the cell grid, 0 to 1: how far each cell is joined
/// to the outside of the mask through like colour. `drawn` holds the cell
/// mean of the mask as drawn, `Ec[p]`, and `colours` the source's cell
/// moments, whose first three are `Ec[I]`.
///
/// The seed is the outside as drawn, `1 - Ec[p]`. Passes of doubling steps
/// 1, 2, 4, ... cells then spread it while the steps sum to at most
/// [`Plan::flood`]: a pass keeps at each cell the most of its own value and
/// of the 8 cells a step away across, down and on the diagonals, each of
/// those weighed by its likeness `exp(-|Ec[I](x) - Ec[I](n)|^2 / (2 eps))`.
/// Reads clamp to the edge of the grid and each pass reads the one before,
/// as two targets of the GPU take turns.
pub fn reached(
    drawn: &[[f32; 1]],
    colours: &[[f32; SOURCE_MOMENTS]],
    plan: &Plan,
    store: &dyn Fn(f32) -> f32,
) -> Vec<f32> {
    let (_, grid) = plan.grid();
    let (w, h) = (grid.0 as i32, grid.1 as i32);
    let scale = LIKENESS * plan.eps;
    let mut field: Vec<f32> = drawn.iter().map(|cell| store(1.0 - cell[0])).collect();
    let (mut total, mut step) = (0, 1);
    while total + step <= plan.flood {
        let s = step as i32;
        let mut next = Vec::with_capacity(field.len());
        for y in 0..h {
            for x in 0..w {
                let here = &colours[(y * w + x) as usize];
                let mut most = field[(y * w + x) as usize];
                for (dx, dy) in [
                    (s, 0),
                    (-s, 0),
                    (0, s),
                    (0, -s),
                    (s, s),
                    (s, -s),
                    (-s, s),
                    (-s, -s),
                ] {
                    let n = ((y + dy).clamp(0, h - 1) * w + (x + dx).clamp(0, w - 1)) as usize;
                    let there = &colours[n];
                    let (r, g, b) = (here[0] - there[0], here[1] - there[1], here[2] - there[2]);
                    let like = (-(r * r + g * g + b * b) / scale).exp();
                    most = most.max(field[n] * like);
                }
                next.push(store(most));
            }
        }
        field = next;
        total += step;
        step *= 2;
    }
    field
}

/// The means of the moments of the mask as drawn over the box, a cell.
pub fn mask_means(
    alpha: &[f32],
    plan: &Plan,
    store: &dyn Fn(f32) -> f32,
) -> Vec<[f32; MASK_MOMENTS]> {
    let (_, grid) = plan.grid();
    let moments = cell_moments(plan, &|i| [alpha[i], alpha[i] * alpha[i]], store);
    box_mean(&moments, grid, plan.cells, store)
}

/// The means of the moments of one gather over the box, a cell: the weight
/// `q` of the inside class and the guide weighed by it. [`gathered`] passes
/// the weight `wq` the reached field leaves.
pub fn gather_means(
    q: &[f32],
    guides: &[[f32; 3]],
    plan: &Plan,
    store: &dyn Fn(f32) -> f32,
) -> Vec<[f32; GATHER_MOMENTS]> {
    let (_, grid) = plan.grid();
    let moments = cell_moments(
        plan,
        &|i| {
            let (q, g) = (q[i], guides[i]);
            [q, g[0] * q, g[1] * q, g[2] * q]
        },
        store,
    );
    box_mean(&moments, grid, plan.cells, store)
}

/// How far the box holds both classes and a mask that means an edge, 0 to 1,
/// from the means of the mask's moments: 0 where the box lies wholly inside
/// or wholly outside the mask, which holds the halo at 0, and 0 where the
/// mask is soft at the scale of the box.
pub fn both(mask: &[f32; MASK_MOMENTS]) -> f32 {
    let (p, pp) = (mask[0], mask[1]);
    let hardness = (pp - p * p).max(0.0) / (p * (1.0 - p)).max(SHARE_FLOOR);
    smooth(p.min(1.0 - p), COVER_LOW, COVER_HIGH) * smooth(hardness, HARD_LOW, HARD_HIGH)
}

/// How far the move comes in over a box, 0 to 1: [`both`] of the mask's
/// moments, times how much of the inside the outside does not reach,
/// `E[p keep]`, which is gather 1's own mean weight. Taken once, in gather 1,
/// and served to all three: 0 where the whole inside is joined to the
/// outside, so a mask drawn over like colour on both sides holds still.
pub fn gate(mask: &[f32; MASK_MOMENTS], unreached: f32) -> f32 {
    both(mask) * smooth(unreached, UNREACHED_LOW, UNREACHED_HIGH)
}

/// What one gather solves in one cell from the means of the source's and its
/// own moments and the [`gate`] of gather 1, in the order of [`SOLVED`].
pub fn solve(
    source: &[f32; SOURCE_MOMENTS],
    gather: &[f32; GATHER_MOMENTS],
    gate: f32,
    eps: f32,
) -> [f32; SOLVED] {
    let n1 = gather[0];
    let n0 = 1.0 - n1;
    let (inside, outside) = (n1.max(SHARE_FLOOR), n0.max(SHARE_FLOOR));
    let mu1 = [gather[1] / inside, gather[2] / inside, gather[3] / inside];
    let mu0 = [
        (source[0] - gather[1]) / outside,
        (source[1] - gather[2]) / outside,
        (source[2] - gather[3]) / outside,
    ];
    // The scatter inside the classes, symmetric: rr, rg, rb, gg, gb, bb.
    let within = |mean: f32, i: usize, j: usize| mean - n1 * mu1[i] * mu1[j] - n0 * mu0[i] * mu0[j];
    let rr = within(source[3], 0, 0).max(0.0) + eps;
    let rg = within(source[4], 0, 1);
    let rb = within(source[5], 0, 2);
    let gg = within(source[6], 1, 1).max(0.0) + eps;
    let gb = within(source[7], 1, 2);
    let bb = within(source[8], 2, 2).max(0.0) + eps;
    // Its inverse by cofactors.
    let c_rr = gg * bb - gb * gb;
    let c_rg = rb * gb - rg * bb;
    let c_rb = rg * gb - rb * gg;
    let c_gg = rr * bb - rb * rb;
    let c_gb = rg * rb - rr * gb;
    let c_bb = rr * gg - rg * rg;
    let det = rr * c_rr + rg * c_rg + rb * c_rb;
    // The matrix is positive definite, so its determinant is at least eps
    // cubed; the floor only guards a rounding that went under it.
    let scale = 1.0 / det.max(eps * eps * eps);
    let inv = [
        c_rr * scale,
        c_rg * scale,
        c_rb * scale,
        c_gg * scale,
        c_gb * scale,
        c_bb * scale,
    ];
    let delta = [mu1[0] - mu0[0], mu1[1] - mu0[1], mu1[2] - mu0[2]];
    let a = [
        inv[0] * delta[0] + inv[1] * delta[1] + inv[2] * delta[2],
        inv[1] * delta[0] + inv[3] * delta[1] + inv[4] * delta[2],
        inv[2] * delta[0] + inv[4] * delta[1] + inv[5] * delta[2],
    ];
    let d2 = (delta[0] * a[0] + delta[1] * a[1] + delta[2] * a[2]).max(0.0);
    let c = smooth(d2, SEPARATE_LOW, SEPARATE_HIGH) * gate;
    let over = d2.max(SEPARATION_FLOOR);
    let mid = [
        (mu1[0] + mu0[0]) / 2.0,
        (mu1[1] + mu0[1]) / 2.0,
        (mu1[2] + mu0[2]) / 2.0,
    ];
    let s = a[0] * mid[0] + a[1] * mid[1] + a[2] * mid[2];
    [
        a[0],
        a[1],
        a[2],
        s,
        d2,
        c,
        inv[0] - a[0] * a[0] / over,
        inv[1] - a[0] * a[1] / over,
        inv[2] - a[0] * a[2] / over,
        inv[3] - a[1] * a[1] / over,
        inv[4] - a[1] * a[2] / over,
        inv[5] - a[2] * a[2] / over,
        mid[0],
        mid[1],
        mid[2],
    ]
}

/// The mask as drawn `p` moved at one pixel of guide `g` by what a gather
/// solved, mixed from the four cells around the pixel. `reached` is the
/// reached field at the pixel: a pixel leaves the mask only where the outside
/// reaches it.
pub fn moved(p: f32, g: [f32; 3], v: &[f32; SOLVED], reached: f32) -> f32 {
    let along = (v[0] * g[0] + v[1] * g[1] + v[2] * g[2]) - v[3];
    let est = (0.5 + along / v[4].max(SEPARATION_FLOOR)).clamp(0.0, 1.0);
    let e = [g[0] - v[12], g[1] - v[13], g[2] - v[14]];
    let m = (v[6] * e[0] * e[0]
        + v[9] * e[1] * e[1]
        + v[11] * e[2] * e[2]
        + 2.0 * (v[7] * e[0] * e[1] + v[8] * e[0] * e[2] + v[10] * e[1] * e[2]))
        .max(0.0);
    let on_line = 1.0 - smooth(m, LINE_LOW, LINE_HIGH);
    let out =
        on_line * (1.0 - smooth(est, OUT_LOW, OUT_HIGH)) * smooth(reached, LEAVE_LOW, LEAVE_HIGH);
    let into = on_line * smooth(est, IN_LOW, IN_HIGH);
    p + v[5] * (into * (1.0 - p) - out * p)
}

/// The result of the last gather over every pixel of a render: the mask as
/// drawn moved onto the edges of the guide, before the amount mixes it in.
pub fn gathered(
    alpha: &[f32],
    guides: &[[f32; 3]],
    plan: &Plan,
    store: &dyn Fn(f32) -> f32,
) -> Vec<f32> {
    let ((first_x, first_y), grid) = plan.grid();
    let colours = source_cells(guides, plan, store);
    let source = box_mean(&colours, grid, plan.cells, store);
    let mask = mask_means(alpha, plan, store);
    let drawn = cell_moments(plan, &|i| [alpha[i]], store);
    let field = reached(&drawn, &colours, plan, store);
    let step = plan.step as f32;
    // Where a pixel lies among the centres of the cells on one axis: the
    // cell before it, the one after, and the share of the second.
    let among = |pixel: u32, origin: u32, first: u32, count: u32| {
        let u = ((origin + pixel) as f32 + 0.5) / step - 0.5 - first as f32;
        let low = u.floor();
        let clamp = |i: f32| (i.max(0.0) as u32).min(count - 1);
        (clamp(low), clamp(low + 1.0), u - low)
    };
    let width = plan.size.0;
    // The four cells around a pixel, top left, top right, bottom left and
    // bottom right, and the shares of the second across and down.
    let around = |i: usize| {
        let (x, y) = (i as u32 % width, i as u32 / width);
        let (x0, x1, fx) = among(x, plan.origin.0, first_x, grid.0);
        let (y0, y1, fy) = among(y, plan.origin.1, first_y, grid.1);
        let at = |cx: u32, cy: u32| (cy * grid.0 + cx) as usize;
        ([at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1)], fx, fy)
    };
    // The reached field at each pixel, mixed from the four cells around it,
    // and how much of the pixel the inside class keeps.
    let reach: Vec<f32> = (0..alpha.len())
        .map(|i| {
            let ([tl, tr, bl, br], fx, fy) = around(i);
            let top = field[tl] + (field[tr] - field[tl]) * fx;
            let bottom = field[bl] + (field[br] - field[bl]) * fx;
            top + (bottom - top) * fy
        })
        .collect();
    let keep: Vec<f32> = reach
        .iter()
        .map(|rf| 1.0 - smooth(*rf, KEEP_LOW, KEEP_HIGH))
        .collect();
    let mut q = alpha.to_vec();
    let mut gates = Vec::new();
    for gather in 0..GATHERS {
        // The weight of the inside class: the mask so far where the outside
        // does not reach it, and alpha a move added above the mask as drawn
        // whether reached or not. In gather 1 q is p, so this is p keep.
        let weight: Vec<f32> = q
            .iter()
            .zip(alpha)
            .zip(&keep)
            .map(|((q, p), keep)| q * keep + (q - p).max(0.0) * (1.0 - keep))
            .collect();
        let means = gather_means(&weight, guides, plan, store);
        if gather == 0 {
            // Gather 1's own mean weight is E[p keep].
            gates = means
                .iter()
                .zip(&mask)
                .map(|(own, mask)| store(gate(mask, own[0])))
                .collect();
        }
        let solved: Vec<[f32; SOLVED]> = means
            .iter()
            .enumerate()
            .map(|(cell, gather)| solve(&source[cell], gather, gates[cell], plan.eps).map(store))
            .collect();
        q = alpha
            .iter()
            .zip(guides)
            .zip(&reach)
            .enumerate()
            .map(|(i, ((p, g), rf))| {
                let ([tl, tr, bl, br], fx, fy) = around(i);
                let (tl, tr, bl, br) = (&solved[tl], &solved[tr], &solved[bl], &solved[br]);
                let mut mixed = [0.0f32; SOLVED];
                for (c, value) in mixed.iter_mut().enumerate() {
                    let top = tl[c] + (tr[c] - tl[c]) * fx;
                    let bottom = bl[c] + (br[c] - bl[c]) * fx;
                    *value = top + (bottom - top) * fy;
                }
                store(moved(*p, *g, &mixed, *rf))
            })
            .collect();
    }
    q
}

/// The refined alpha of a sanitised `refine` over every pixel of a render,
/// before the r8unorm store rounds it. `alpha` is the finished alpha of the
/// mask as the GPU stores it and `pixels` the source in linear Rec.2020. An
/// amount of 0 gives `alpha` back bit for bit and computes nothing.
pub fn refined(
    alpha: &[f32],
    pixels: &[[f32; 3]],
    geometry: &Geometry,
    refine: &Refine,
    store: &dyn Fn(f32) -> f32,
) -> Vec<f32> {
    if refine.is_off() {
        return alpha.to_vec();
    }
    let plan = Plan::for_geometry(&refine.sanitised(), geometry);
    let guides: Vec<[f32; 3]> = pixels.iter().map(|px| guide(*px)).collect();
    gathered(alpha, &guides, &plan, store)
        .iter()
        .zip(alpha)
        .map(|(q, p)| (p + (q - p) * plan.amount).clamp(0.0, 1.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::stored_alpha;
    use gamut_core::CropRect;

    const IDENTITY: &dyn Fn(f32) -> f32 = &|v| v;

    fn on(amount: f32, radius: f32, sensitivity: f32) -> Refine {
        Refine {
            amount,
            radius,
            sensitivity,
        }
    }

    /// A picture whose left half is one grey and whose right half another.
    fn step_guide(size: (u32, u32), left: f32, right: f32) -> Vec<[f32; 3]> {
        (0..size.0 * size.1)
            .map(|i| {
                if i % size.0 < size.0 / 2 {
                    [left; 3]
                } else {
                    [right; 3]
                }
            })
            .collect()
    }

    /// An alpha that ramps from 1 on the left to 0 on the right across
    /// `width` pixels about the column `centre`, through 0.5 there.
    fn ramp_alpha(size: (u32, u32), centre: f32, width: f32) -> Vec<f32> {
        (0..size.0 * size.1)
            .map(|i| {
                let x = (i % size.0) as f32 + 0.5 - centre;
                stored_alpha((0.5 - x / width).clamp(0.0, 1.0))
            })
            .collect()
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
    }

    #[test]
    fn an_amount_of_0_is_the_identity_bit_for_bit() {
        let size = (40, 30);
        let geometry = Geometry::full(size, size);
        let pixels = step_guide(size, 0.05, 0.6);
        let alpha = ramp_alpha(size, 20.0, 17.0);
        for radius in [0.001, 0.02, 0.05] {
            let out = refined(&alpha, &pixels, &geometry, &on(0.0, radius, 80.0), IDENTITY);
            assert!(same_bits(&out, &alpha), "radius {radius}");
        }
    }

    #[test]
    fn the_eps_runs_on_a_log_scale_between_its_two_ends() {
        assert!((eps(0.0) - 0.01).abs() < 1e-8);
        assert!((eps(100.0) - 0.000_025).abs() < 1e-10);
        let middle = (EPS_LOOSE * EPS_STRICT).sqrt();
        assert!((eps(50.0) - middle * middle).abs() < 1e-8);
    }

    #[test]
    fn the_smoothstep_is_its_polynomial_between_its_ends() {
        assert_eq!(smooth(0.1, 0.15, 0.8), 0.0);
        assert_eq!(smooth(0.9, 0.15, 0.8), 1.0);
        assert!((smooth(0.475, 0.15, 0.8) - 0.5).abs() < 1e-6);
        assert!((smooth(0.025, 0.0, 0.1) - 0.15625).abs() < 1e-6);
    }

    #[test]
    fn the_plan_takes_its_cells_from_the_whole_picture() {
        // A cell step for every 8 pixels of radius, a box of 0.71 radius.
        let plan = Plan::new(&on(100.0, 0.01, 50.0), (6000, 4000), (0, 0), (6000, 4000));
        assert_eq!((plan.step, plan.cells), (4, 11), "60 pixels");
        // The flood spreads 15 cells, the radius in cells: three boxes, the
        // two cells beside of the gathers, the cell beside of the reached
        // field and its 15.
        assert_eq!(plan.flood, 15);
        assert_eq!(plan.margin(), 51);
        assert_eq!(plan.reach(), 212);
        let plan = Plan::new(&on(100.0, 0.01, 50.0), (1800, 1200), (0, 0), (1800, 1200));
        assert_eq!((plan.step, plan.cells), (2, 6), "18 pixels");
        let plan = Plan::new(&on(100.0, 0.01, 50.0), (1100, 700), (0, 0), (1100, 700));
        assert_eq!((plan.step, plan.cells), (1, 8), "11 pixels");
        let plan = Plan::new(&on(100.0, 0.05, 50.0), (64, 64), (0, 0), (64, 64));
        assert_eq!((plan.step, plan.cells), (1, 2), "3.2 pixels");
        let plan = Plan::new(&on(100.0, 0.001, 50.0), (64, 64), (0, 0), (64, 64));
        assert_eq!((plan.step, plan.cells), (1, 1), "never under one cell");
        // A window of the same picture has the same cells, counted from the
        // cell its first pixel lies in.
        let window = Plan::new(
            &on(100.0, 0.01, 50.0),
            (6000, 4000),
            (1022, 513),
            (900, 700),
        );
        assert_eq!((window.step, window.cells), (4, 11));
        assert_eq!(window.grid(), ((255, 128), (226, 176)));
        assert_eq!(reach(&on(0.0, 0.05, 50.0), (6000, 4000)), 0, "off");
        assert_eq!(reach(&on(100.0, 0.01, 50.0), (6000, 4000)), 212);
    }

    #[test]
    fn the_plan_of_a_geometry_finds_the_whole_picture_and_the_origin() {
        let geometry = Geometry {
            window: CropRect {
                x: 1024.0 / 6000.0,
                y: 512.0 / 4000.0,
                width: 3104.0 / 6000.0,
                height: 2000.0 / 4000.0,
            },
            size: (3104, 2000),
            photo: (6000, 4000),
        };
        let plan = Plan::for_geometry(&on(100.0, 0.01, 50.0), &geometry);
        assert_eq!(plan.full, (6000, 4000));
        assert_eq!(plan.origin, (1024, 512));
        assert_eq!(plan.size, (3104, 2000));
    }

    #[test]
    fn a_flat_guide_gives_the_mask_back_exactly() {
        let size = (48, 24);
        let geometry = Geometry::full(size, size);
        let alpha = ramp_alpha(size, 24.0, 3.0);
        for grey in [0.0, 0.18, 0.7] {
            let pixels = vec![[grey, grey * 0.9, grey * 1.1]; (size.0 * size.1) as usize];
            for sensitivity in [0.0, 50.0, 100.0] {
                let refine = on(100.0, 0.05, sensitivity);
                let out = refined(&alpha, &pixels, &geometry, &refine, IDENTITY);
                // The classes do not separate, so nothing is allowed to move.
                assert!(same_bits(&out, &alpha), "grey {grey} at {sensitivity}");
            }
        }
    }

    /// Diagonal stripes of many colours over the whole picture, one colour
    /// along each diagonal x + y, under a hard band of mask drawn across
    /// them: every pixel of the band has its own colour on the outside a few
    /// pixels up or down its diagonal.
    #[test]
    fn a_mask_the_outside_reaches_everywhere_comes_back_as_drawn() {
        let size = (160, 60);
        let geometry = Geometry::full(size, size);
        let wave = |d: f32, rate: f32, phase: f32| 0.5 + 0.5 * (d * rate + phase).sin();
        let pixels: Vec<[f32; 3]> = (0..size.0 * size.1)
            .map(|i| {
                let d = (i % size.0 + i / size.0) as f32;
                [
                    0.05 + 0.4 * wave(d, 0.45, 0.0),
                    0.05 + 0.3 * wave(d, 0.31, 1.0),
                    0.05 + 0.5 * wave(d, 0.23, 2.0),
                ]
            })
            .collect();
        // A band 10 rows high that stops 20 columns short of either side:
        // every pixel of it lies at most 5 rows from the outside, and the
        // diagonal to its own colour there stays inside the picture.
        let (rows, columns) = (25..35, 20..140);
        let alpha: Vec<f32> = (0..size.0 * size.1)
            .map(|i| {
                let inside = rows.contains(&(i / size.0)) && columns.contains(&(i % size.0));
                if inside { 1.0 } else { 0.0 }
            })
            .collect();
        let refine = on(100.0, 0.05, 50.0);
        let out = refined(&alpha, &pixels, &geometry, &refine, IDENTITY);
        let moved = out.iter().zip(&alpha).filter(|(a, b)| a != b).count();
        assert!(same_bits(&out, &alpha), "{moved} pixels moved");
    }

    // The growth into the unreached gap is partial under the connectivity prior, ruled 2026-09-24 (a2_grow_rt7, case 8's gap 0.26/0.21); the figures come from lead_a2.py.
    #[test]
    fn a_loose_alpha_short_of_a_step_grows_as_the_ruled_design_does() {
        let size = (200, 16);
        let geometry = Geometry::full(size, size);
        let pixels = step_guide(size, 0.03, 0.5);
        let refine = on(100.0, 0.05, 50.0);
        let plan = Plan::for_geometry(&refine, &geometry);
        assert_eq!((plan.step, plan.cells), (1, 7), "a radius of 10 pixels");
        // The mask is 1 on the dark side and spills 4 pixels over the edge
        // of the guide, a blurred step 3 pixels wide.
        let edge = size.0 / 2;
        let alpha = ramp_alpha(size, edge as f32 + 4.0, 3.0);
        let out = refined(&alpha, &pixels, &geometry, &refine, IDENTITY);
        let row = |x: u32| out[(8 * size.0 + x) as usize];
        let given = |x: u32| alpha[(8 * size.0 + x) as usize];
        assert!(given(edge) > 0.9 && given(edge + 2) > 0.9, "the spill");
        assert!(
            row(edge - 10) > 0.9,
            "one radius inside: {}",
            row(edge - 10)
        );
        assert!(row(edge + 9) < 0.1, "one radius outside: {}", row(edge + 9));
        for x in edge..edge + 6 {
            assert!(row(x) < 0.1, "the spill at {x} is left at {}", row(x));
        }
        for x in edge - 6..edge {
            assert!(row(x) > 0.9, "the kept side at {x} is left at {}", row(x));
        }
        // A mask drawn short of the edge grows into the gap as far as the
        // ruled design grows it: row 8 from column 94 to 100.
        let short = ramp_alpha(size, edge as f32 - 4.0, 3.0);
        let grown = refined(&short, &pixels, &geometry, &refine, IDENTITY);
        let ruled = [1.0, 0.7503, 0.6278, 0.0, 0.0, 0.0, 0.0];
        for (x, want) in (edge - 6..=edge).zip(ruled) {
            let got = grown[(8 * size.0 + x) as usize];
            assert!(
                (got - want).abs() < 1e-3,
                "short of the edge at {x}: {got}, not {want}"
            );
        }
        assert!(grown[(8 * size.0 + edge) as usize] < 0.1);
        // Half the amount goes half the way.
        let half = refined(&alpha, &pixels, &geometry, &on(50.0, 0.05, 50.0), IDENTITY);
        let i = (8 * size.0 + edge + 1) as usize;
        assert!((half[i] - (alpha[i] + out[i]) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_colour_far_off_the_line_between_the_classes_keeps_its_mask() {
        // A dark object on the left of a blue field under a mask that spills
        // over the edge: what one gather solves at the edge.
        let size = (200, 16);
        let geometry = Geometry::full(size, size);
        let (dark, blue) = ([0.02, 0.03, 0.05], [0.25, 0.5, 0.9]);
        let edge = size.0 / 2;
        let pixels: Vec<[f32; 3]> = (0..size.0 * size.1)
            .map(|i| if i % size.0 < edge { dark } else { blue })
            .collect();
        let refine = on(100.0, 0.05, 50.0);
        let alpha = ramp_alpha(size, edge as f32 + 4.0, 3.0);
        let guides: Vec<[f32; 3]> = pixels.iter().map(|px| guide(*px)).collect();
        let plan = Plan::for_geometry(&refine, &geometry);
        let (_, grid) = plan.grid();
        let source = source_means(&guides, &plan, IDENTITY);
        let mask = mask_means(&alpha, &plan, IDENTITY);
        let gather = gather_means(&alpha, &guides, &plan, IDENTITY);
        let cell = (8 * grid.0 + edge) as usize;
        let solved = solve(&source[cell], &gather[cell], both(&mask[cell]), plan.eps);
        assert!(solved[5] > 0.99, "the move is allowed here: {}", solved[5]);
        let place = |g: [f32; 3]| {
            let along = (solved[0] * g[0] + solved[1] * g[1] + solved[2] * g[2]) - solved[3];
            let e = [g[0] - solved[12], g[1] - solved[13], g[2] - solved[14]];
            let m = solved[6] * e[0] * e[0]
                + solved[9] * e[1] * e[1]
                + solved[11] * e[2] * e[2]
                + 2.0
                    * (solved[7] * e[0] * e[1]
                        + solved[8] * e[0] * e[2]
                        + solved[10] * e[1] * e[2]);
            (0.5 + along / solved[4], m)
        };
        // The field lies at the outside's end of the line and on it: it
        // leaves the mask. The object stays.
        let (est, m) = place(guide(blue));
        assert!(est < OUT_LOW && m < LINE_LOW, "the field: {est}, {m}");
        assert!(moved(1.0, guide(blue), &solved, 1.0) < 0.01);
        assert!(moved(1.0, guide(dark), &solved, 1.0) > 0.99);
        // A bright warm colour, a sunlit face of the object: along the line
        // it lies at the outside's end too, and off the line it is far from
        // both classes. It keeps the alpha it was given, whatever that was.
        let warm = guide([0.9, 0.5, 0.25]);
        let (est, m) = place(warm);
        assert!(est < OUT_LOW, "along the line the warm colour reads {est}");
        assert!(m > LINE_HIGH, "off the line the warm colour reads {m}");
        for p in [1.0, 0.4, 0.0] {
            assert_eq!(moved(p, warm, &solved, 1.0), p);
        }
    }

    #[test]
    fn a_mask_that_is_a_ramp_wider_than_twice_the_box_comes_back() {
        let size = (240, 16);
        let geometry = Geometry::full(size, size);
        let pixels = step_guide(size, 0.03, 0.5);
        let refine = on(100.0, 0.05, 50.0);
        let plan = Plan::for_geometry(&refine, &geometry);
        let side = (2 * plan.cells + 1) * plan.step;
        for width in [2.0 * side as f32 + 1.0, 4.0 * side as f32] {
            let alpha = ramp_alpha(size, 120.0, width);
            let out = refined(&alpha, &pixels, &geometry, &refine, IDENTITY);
            let most = out
                .iter()
                .zip(&alpha)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(most <= 0.02, "a ramp of {width} pixels moved by {most}");
        }
    }

    /// A dark disc with a grain on a bright field with the same, under a mask
    /// that is a disc with a hard rim a few pixels off the first: it spills
    /// over the edge on one side and stops short of it on the other.
    fn busy(full: (u32, u32)) -> (Vec<[f32; 3]>, Vec<f32>) {
        let mut pixels = Vec::new();
        let mut alpha = Vec::new();
        let (cx, cy) = (full.0 as f32 * 0.5, full.1 as f32 * 0.5);
        let rim = full.1 as f32 * 0.3;
        for y in 0..full.1 {
            for x in 0..full.0 {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let from = |cx: f32, cy: f32| ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
                // The edge of the disc is 8 pixels wide, so many colours lie
                // between the classes and a changed class mean shows.
                let out = ((from(cx, cy) - rim) / 8.0 + 0.5).clamp(0.0, 1.0);
                let side = 0.04 * 10.0f32.powf(out);
                let grain = 0.3 * side * ((fx * 0.9).sin() * (fy * 1.3).cos() + 1.0);
                pixels.push([side + grain, side + 0.5 * grain, side * 0.8 + grain]);
                let d = from(cx + 2.5, cy - 1.5);
                alpha.push(stored_alpha(((rim - d) / 2.0 + 0.5).clamp(0.0, 1.0)));
            }
        }
        (pixels, alpha)
    }

    #[test]
    fn nothing_moves_further_than_one_box_from_the_transition_of_the_mask() {
        let full = (240, 160);
        let (pixels, alpha) = busy(full);
        let guides: Vec<[f32; 3]> = pixels.iter().map(|px| guide(*px)).collect();
        // 0.08 is past what a sanitised refine holds: cells of 2 pixels.
        for radius in [0.02, 0.05, 0.08] {
            let plan = Plan::new(&on(100.0, radius, 50.0), full, (0, 0), full);
            let out = gathered(&alpha, &guides, &plan, IDENTITY);
            let moved = out.iter().zip(&alpha).filter(|(a, b)| a != b).count();
            assert!(
                moved > 100,
                "radius {radius}: the filter moves {moved} pixels"
            );
            // A pixel is far when every pixel within the box, and the cell
            // beside for the bilinear step, holds the same 0 or the same 1.
            let far = ((plan.cells + 2) * plan.step) as i32;
            let (w, h) = (full.0 as i32, full.1 as i32);
            for y in 0..h {
                for x in 0..w {
                    let here = alpha[(y * w + x) as usize];
                    if here != 0.0 && here != 1.0 {
                        continue;
                    }
                    let flat = (-far..=far).all(|dy| {
                        (-far..=far).all(|dx| {
                            let (sx, sy) = ((x + dx).clamp(0, w - 1), (y + dy).clamp(0, h - 1));
                            alpha[(sy * w + sx) as usize] == here
                        })
                    });
                    if flat {
                        let got = out[(y * w + x) as usize];
                        assert!(
                            got.to_bits() == here.to_bits(),
                            "radius {radius}, pixel ({x}, {y}): {here} became {got}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nothing_divides_by_zero_on_a_flat_black_guide_or_a_mask_of_all_1() {
        let size = (32, 32);
        let geometry = Geometry::full(size, size);
        let black = vec![[0.0; 3]; (size.0 * size.1) as usize];
        let (busy_pixels, _) = busy(size);
        let ramp = ramp_alpha(size, 16.0, 6.0);
        let full = vec![1.0; (size.0 * size.1) as usize];
        let none = vec![0.0; (size.0 * size.1) as usize];
        for sensitivity in [0.0, 100.0] {
            let refine = on(100.0, 0.05, sensitivity);
            for (pixels, alpha) in [
                (&black, &ramp),
                (&black, &full),
                (&busy_pixels, &full),
                (&busy_pixels, &none),
            ] {
                let out = refined(alpha, pixels, &geometry, &refine, IDENTITY);
                assert!(out.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
            }
            let out = refined(&full, &busy_pixels, &geometry, &refine, IDENTITY);
            assert!(same_bits(&out, &full), "a mask of all 1 comes back");
        }
        let solved = solve(
            &[0.0; SOURCE_MOMENTS],
            &[0.0; 4],
            gate(&[0.0; 2], 0.0),
            eps(100.0),
        );
        assert!(solved.iter().all(|v| v.is_finite()));
        assert!(moved(0.5, [0.0; 3], &solved, 1.0).is_finite());
    }

    /// The refined alpha of one rectangle of the busy picture, from the whole
    /// render and from a window that holds the rectangle and `pad` pixels
    /// around it.
    fn whole_and_window(radius: f32, pad: u32) -> (Vec<f32>, Vec<f32>) {
        let full = (640, 480);
        let (pixels, alpha) = busy(full);
        whole_and_window_of(&pixels, &alpha, full, radius, pad)
    }

    /// [`whole_and_window`] of any picture of 640 x 480.
    fn whole_and_window_of(
        pixels: &[[f32; 3]],
        alpha: &[f32],
        full: (u32, u32),
        radius: f32,
        pad: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let refine = on(100.0, radius, 60.0);
        let whole = refined(
            alpha,
            pixels,
            &Geometry::full(full, full),
            &refine,
            IDENTITY,
        );
        // What is wanted, off the cell grid on purpose, across the rim.
        let wanted = (401u32, 187u32, 60u32, 50u32);
        let (x0, y0) = (wanted.0 - pad.min(wanted.0), wanted.1 - pad.min(wanted.1));
        let x1 = (wanted.0 + wanted.2 + pad).min(full.0);
        let y1 = (wanted.1 + wanted.3 + pad).min(full.1);
        let size = (x1 - x0, y1 - y0);
        let at = |i: u32| ((y0 + i / size.0) * full.0 + x0 + i % size.0) as usize;
        let window_alpha: Vec<f32> = (0..size.0 * size.1).map(|i| alpha[at(i)]).collect();
        let window_pixels: Vec<[f32; 3]> = (0..size.0 * size.1).map(|i| pixels[at(i)]).collect();
        let geometry = Geometry {
            window: CropRect {
                x: x0 as f32 / full.0 as f32,
                y: y0 as f32 / full.1 as f32,
                width: size.0 as f32 / full.0 as f32,
                height: size.1 as f32 / full.1 as f32,
            },
            size,
            photo: full,
        };
        let plan = Plan::for_geometry(&refine, &geometry);
        assert_eq!(
            (plan.full, plan.origin),
            (full, (x0, y0)),
            "radius {radius}"
        );
        let window = refined(&window_alpha, &window_pixels, &geometry, &refine, IDENTITY);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for y in wanted.1..wanted.1 + wanted.3 {
            for x in wanted.0..wanted.0 + wanted.2 {
                a.push(whole[(y * full.0 + x) as usize]);
                b.push(window[((y - y0) * size.0 + x - x0) as usize]);
            }
        }
        (a, b)
    }

    #[test]
    fn a_window_padded_by_the_reach_gives_what_the_full_render_gives() {
        for radius in [0.012, 0.03, 0.05] {
            let refine = on(100.0, radius, 60.0);
            let plan = Plan::new(&refine, (640, 480), (0, 0), (640, 480));
            let (whole, window) = whole_and_window(radius, plan.reach());
            assert!(
                whole.iter().any(|v| *v != 0.0 && *v != 1.0),
                "the mask has an edge in what is wanted"
            );
            assert!(same_bits(&whole, &window), "radius {radius}");
            // The same on a picture whose reached field carries its whole
            // flood toward the window's edge.
            let (pixels, alpha) = block_at_the_end(full_size());
            let (whole, window) =
                whole_and_window_of(&pixels, &alpha, full_size(), radius, plan.reach());
            assert!(same_bits(&whole, &window), "the block at radius {radius}");
        }
    }

    fn full_size() -> (u32, u32) {
        (640, 480)
    }

    /// Stripes along the rows, one colour a row, under a hard rectangle of
    /// mask from column 100 to 481 whose top edge is a ramp 3 pixels wide
    /// about row 210. From row 210 down, the rectangle's last 8 columns and
    /// the outside to its right are one flat colour, so the outside reaches
    /// that part of the mask along its rows from the right and from nowhere
    /// else.
    fn block_at_the_end(full: (u32, u32)) -> (Vec<[f32; 3]>, Vec<f32>) {
        let wave = |d: f32, rate: f32, phase: f32| 0.5 + 0.5 * (d * rate + phase).sin();
        let end = 482;
        let mut pixels = Vec::new();
        let mut alpha = Vec::new();
        for y in 0..full.1 {
            for x in 0..full.0 {
                let d = y as f32;
                pixels.push(if x + 8 >= end && (210..330).contains(&y) {
                    [0.3, 0.35, 0.6]
                } else {
                    [
                        0.05 + 0.4 * wave(d, 1.7, 0.0),
                        0.05 + 0.3 * wave(d, 2.3, 1.0),
                        0.05 + 0.5 * wave(d, 2.9, 2.0),
                    ]
                });
                let inside = (100..end).contains(&x) && (150..330).contains(&y);
                let top = ((y as f32 + 0.5 - 210.0) / 3.0 + 0.5).clamp(0.0, 1.0);
                alpha.push(if inside { top } else { 0.0 });
            }
        }
        (pixels, alpha)
    }

    #[test]
    fn a_window_short_by_the_flood_differs_from_the_full_render() {
        // At Radius 0.012 the cells are 1 pixel, the box 5 cells and the
        // flood 7 cells, steps 1, 2 and 4. The reach less the flood's cells
        // ends the window at column 481, one short of the outside that
        // reaches the block's part of the mask, and the three gathers carry
        // the difference into what is wanted.
        let radius = 0.012;
        let plan = Plan::new(&on(100.0, radius, 60.0), full_size(), (0, 0), full_size());
        assert_eq!((plan.step, plan.cells, plan.flood), (1, 5, 7));
        let short = plan.reach() - plan.flood * plan.step;
        assert_eq!(401 + 60 + short, 481);
        let (pixels, alpha) = block_at_the_end(full_size());
        let (whole, window) = whole_and_window_of(&pixels, &alpha, full_size(), radius, short);
        let most = whole
            .iter()
            .zip(&window)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            most > 1e-4,
            "the window short by {short} pixels differs by {most}"
        );
    }

    #[test]
    fn the_cells_at_the_far_edge_hold_the_pixels_there_are() {
        // 10 pixels in cells of 4: the last cell holds two.
        let plan = Plan {
            full: (10, 4),
            origin: (0, 0),
            size: (10, 4),
            step: 4,
            cells: 1,
            flood: 1,
            eps: eps(50.0),
            amount: 1.0,
        };
        assert_eq!(plan.grid(), ((0, 0), (3, 1)));
        let alpha: Vec<f32> = (0..40)
            .map(|i| if i % 10 >= 8 { 1.0 } else { 0.0 })
            .collect();
        let moments = cell_moments(&plan, &|i| [alpha[i]], IDENTITY);
        assert_eq!(moments.len(), 3);
        assert_eq!(moments[2][0], 1.0, "the mean of the two columns it holds");
        assert_eq!(moments[1][0], 0.0);
    }
}
