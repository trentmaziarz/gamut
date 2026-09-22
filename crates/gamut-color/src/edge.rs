//! The edge controls on the CPU: Shift edge, Feather and Contrast, applied in
//! that order to the alpha Refine edges hands to the blend. edge.wgsl in
//! gamut-gpu mirrors this module; the golden tests hold the two together.
//!
//! The input is the alpha as the GPU holds it in r8unorm: the stored alpha of
//! the components when Refine edges is off, the stored refined alpha when it
//! is on. The controls read that alpha and nothing else, never the source and
//! never the developed pixel, so no slider of the develop chain moves them.
//!
//! Shift edge is grey-level morphology over a regular octagon: a maximum over
//! it grows the mask and a minimum shrinks it. The octagon is four runs, one
//! after another: half-width `axis` along x, then along y, then `diagonal`
//! steps along (1, 1), then along (1, -1). Each run is built in doubling
//! passes, a run forward and a run backward, and the last backward pass also
//! takes the forward run in. A sample past the edge of the render takes the
//! edge pixel. A maximum or a minimum makes no new value, so every pass keeps
//! the 8-bit codes it read and no store rounds.
//!
//! Feather is a Gaussian on a grid of cells laid on the pixel grid of the
//! WHOLE picture at the scale of the render, as Refine edges lays its cells:
//! a cell is the mean of its pixels, the cells are blurred across and down
//! with the kernel arithmetic of [`basic::gaussian_stored`], and each pixel
//! reads the four cells around it bilinearly. At cells of one pixel the result
//! is `gaussian_stored`'s, bit for bit.
//!
//! Contrast is pointwise: `q = 0.5 + (p - 0.5) k`, held inside 0 to 1, with
//! `k = 1 / (1 - contrast / 100)`; at 100 it is 1 from 0.5 up and 0 under.
//!
//! Every length is measured from the size of the whole picture, so the fitted
//! view, 100 percent and the export show the same edge at their own scales.

use gamut_core::mask::{Edge, MAX_EDGE_CONTRAST};

use crate::basic;
use crate::mask::Geometry;

/// The half-width of each axis run of the octagon as a share of the shift.
pub const OCTAGON_AXIS: f32 = 0.398;

/// The steps of each diagonal run of the octagon as a share of the shift.
pub const OCTAGON_DIAGONAL: f32 = 0.281;

/// The four runs of the octagon, in the order they are taken.
pub const RUNS: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];

/// The sigma in render pixels one step of the feather's cell grid is taken
/// for.
pub const FEATHER_PIXELS_A_STEP: f32 = 4.0;

/// The widest cell of the feather, in render pixels.
pub const FEATHER_MAX_STEP: u32 = 16;

/// How many sigmas the feather's kernel reaches.
pub const FEATHER_SIGMAS: f32 = 3.0;

/// The alpha about which Contrast turns.
pub const CONTRAST_MIDDLE: f32 = 0.5;

/// One run of the octagon: its direction and how many steps it reaches on
/// each side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub direction: (i32, i32),
    pub half: u32,
}

/// What the edge controls of one mask do on one render. The GPU builds its
/// uniforms from the same plan, so the two never differ in a number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    /// The size the whole picture has at the scale of the render.
    pub full: (u32, u32),
    /// Where the render begins on the whole picture, in its pixels.
    pub origin: (u32, u32),
    /// The size of the render.
    pub size: (u32, u32),
    /// Shift edge grows the mask (a maximum), or shrinks it (a minimum).
    pub grow: bool,
    /// The half-width of each axis run in pixels, and the steps of each
    /// diagonal run. Both 0: Shift edge is at rest.
    pub axis: u32,
    pub diagonal: u32,
    /// Feather's sigma in render pixels; 0 when Feather is at rest.
    pub sigma: f32,
    /// The side of a feather cell in render pixels.
    pub step: u32,
    /// Sigma in cells, and the radius of the kernel in cells.
    pub sigma_cells: f32,
    pub radius_cells: u32,
    /// Contrast, 0 (at rest) to 100.
    pub contrast: f32,
    /// The gain of Contrast below 100.
    pub gain: f32,
}

impl Plan {
    /// `full`, `origin` and `size` are in pixels of the whole picture at the
    /// scale of the render.
    pub fn new(edge: &Edge, full: (u32, u32), origin: (u32, u32), size: (u32, u32)) -> Self {
        let edge = edge.sanitised();
        let longer = full.0.max(full.1) as f32;
        let r = edge.shift.abs() * longer;
        let axis = (r * OCTAGON_AXIS).round() as u32;
        let diagonal = (r * OCTAGON_DIAGONAL).round() as u32;
        let wanted = edge.feather * longer;
        let step = ((wanted / FEATHER_PIXELS_A_STEP).floor() as u32).clamp(1, FEATHER_MAX_STEP);
        let sigma_cells = wanted / step as f32;
        // A sigma whose square is no number the kernel can divide by is a
        // feather too small to be one.
        let feathers = 2.0 * sigma_cells * sigma_cells > 0.0;
        let contrast = edge.contrast;
        Plan {
            full,
            origin,
            size,
            grow: edge.shift > 0.0,
            axis,
            diagonal,
            sigma: if feathers { wanted } else { 0.0 },
            step: if feathers { step } else { 1 },
            sigma_cells: if feathers { sigma_cells } else { 0.0 },
            radius_cells: if feathers {
                (FEATHER_SIGMAS * sigma_cells).ceil() as u32
            } else {
                0
            },
            contrast,
            gain: if contrast < MAX_EDGE_CONTRAST {
                1.0 / (1.0 - contrast / 100.0)
            } else {
                1.0
            },
        }
    }

    /// The plan of a render with this geometry: the whole picture and the
    /// origin found as Refine edges finds them.
    pub fn for_geometry(edge: &Edge, geometry: &Geometry) -> Self {
        let window = geometry.window;
        let (w, h) = (geometry.size.0 as f32, geometry.size.1 as f32);
        let full = (
            (w / window.width.max(1e-6)).round().max(1.0),
            (h / window.height.max(1e-6)).round().max(1.0),
        );
        let origin = ((window.x * full.0).round(), (window.y * full.1).round());
        Plan::new(
            edge,
            (full.0 as u32, full.1 as u32),
            (origin.0.max(0.0) as u32, origin.1.max(0.0) as u32),
            geometry.size,
        )
    }

    /// Whether Shift edge moves anything: a run of at least one step.
    pub fn shifts(&self) -> bool {
        self.axis > 0 || self.diagonal > 0
    }

    /// Whether Feather is on.
    pub fn feathers(&self) -> bool {
        self.sigma > 0.0
    }

    /// Whether Contrast is on.
    pub fn contrasts(&self) -> bool {
        self.contrast > 0.0
    }

    /// Whether all three are at rest: no pass runs.
    pub fn is_off(&self) -> bool {
        !self.shifts() && !self.feathers() && !self.contrasts()
    }

    /// The runs of the octagon that take a step, in order.
    pub fn runs(&self) -> Vec<Run> {
        RUNS.iter()
            .enumerate()
            .map(|(i, direction)| Run {
                direction: *direction,
                half: if i < 2 { self.axis } else { self.diagonal },
            })
            .filter(|run| run.half > 0)
            .collect()
    }

    /// How far Shift edge reads around a pixel: an axis run and two diagonal
    /// runs.
    pub fn shift_reach(&self) -> u32 {
        self.axis + 2 * self.diagonal
    }

    /// How far Feather reads around a pixel, in render pixels: the kernel,
    /// the cell beside for the bilinear read, and the cell a window may cut.
    pub fn feather_reach(&self) -> u32 {
        if self.feathers() {
            (self.radius_cells + 2) * self.step
        } else {
            0
        }
    }

    /// How far the edge controls read around a pixel. Contrast reads the
    /// pixel alone.
    pub fn reach(&self) -> u32 {
        self.shift_reach() + self.feather_reach()
    }

    /// The first feather cell of the render on each axis and how many it
    /// touches, on the grid of the whole picture.
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

/// [`Plan::reach`] of the edge controls on a render whose whole picture is
/// `full` pixels, 0 while all three are at rest: what a window is padded by
/// beyond the reach of Refine edges.
pub fn reach(edge: &Edge, full: (u32, u32)) -> u32 {
    Plan::new(edge, full, (0, 0), full).reach()
}

/// The offsets of the doubling passes of a one-sided run that reaches `half`
/// steps: each pass takes the run so far at the pixel and at the pixel that
/// many steps on. `ceil(log2(half + 1))` passes.
pub fn doubling(half: u32) -> Vec<u32> {
    let mut offsets = Vec::new();
    let mut length = 1;
    while length < half + 1 {
        let add = length.min(half + 1 - length);
        offsets.push(add);
        length += add;
    }
    offsets
}

fn at(x: i32, y: i32, size: (u32, u32)) -> usize {
    let x = x.clamp(0, size.0 as i32 - 1);
    let y = y.clamp(0, size.1 as i32 - 1);
    (y as u32 * size.0 + x as u32) as usize
}

/// The maximum (`grow`) or minimum of two alphas.
fn pick(grow: bool, a: f32, b: f32) -> f32 {
    if grow { a.max(b) } else { a.min(b) }
}

/// One doubling pass: the run so far at each pixel with the run so far
/// `offset` steps along `direction`, whose place is held inside the render.
fn doubled(
    source: &[f32],
    size: (u32, u32),
    direction: (i32, i32),
    offset: i32,
    grow: bool,
) -> Vec<f32> {
    let mut out = vec![0.0; source.len()];
    for y in 0..size.1 as i32 {
        for x in 0..size.0 as i32 {
            let far = source[at(x + offset * direction.0, y + offset * direction.1, size)];
            out[at(x, y, size)] = pick(grow, source[at(x, y, size)], far);
        }
    }
    out
}

/// One run of the octagon over the whole render, as the GPU draws it: the
/// forward run by doubling, then the backward run by doubling, whose last
/// pass takes the forward run in.
pub fn run(alpha: &[f32], size: (u32, u32), run: Run, grow: bool) -> Vec<f32> {
    let offsets = doubling(run.half);
    let (dx, dy) = run.direction;
    let mut forward = alpha.to_vec();
    for offset in &offsets {
        forward = doubled(&forward, size, (dx, dy), *offset as i32, grow);
    }
    let mut backward = alpha.to_vec();
    for offset in &offsets {
        backward = doubled(&backward, size, (-dx, -dy), *offset as i32, grow);
    }
    forward
        .iter()
        .zip(&backward)
        .map(|(f, b)| pick(grow, *f, *b))
        .collect()
}

/// Shift edge over every pixel of a render: the four runs of the octagon in
/// order. `alpha` holds 8-bit codes and so does the result. At rest it gives
/// `alpha` back bit for bit and computes nothing.
pub fn shift(alpha: &[f32], plan: &Plan) -> Vec<f32> {
    let mut out = alpha.to_vec();
    for step in plan.runs() {
        out = run(&out, plan.size, step, plan.grow);
    }
    out
}

/// The mean of `alpha` over the pixels of every cell of the render's grid
/// that the render holds.
pub fn cells(alpha: &[f32], plan: &Plan, store: &dyn Fn(f32) -> f32) -> Vec<f32> {
    let ((first_x, first_y), (columns, rows)) = plan.grid();
    let (width, height) = plan.size;
    let mut out = Vec::with_capacity((columns * rows) as usize);
    for cy in 0..rows {
        for cx in 0..columns {
            let span = |first: u32, c: u32, origin: u32, size: u32| {
                let low = ((first + c) * plan.step).max(origin) - origin;
                let high = ((first + c + 1) * plan.step).min(origin + size) - origin;
                (low, high)
            };
            let (x0, x1) = span(first_x, cx, plan.origin.0, width);
            let (y0, y1) = span(first_y, cy, plan.origin.1, height);
            let mut sum = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += alpha[(y * width + x) as usize];
                }
            }
            let count = ((x1 - x0) * (y1 - y0)).max(1) as f32;
            out.push(store(sum / count));
        }
    }
    out
}

/// Feather over every pixel of a render: the cells, blurred across and down,
/// read back bilinearly from the four cells around each pixel. `store` is
/// applied where the GPU writes a cell target (32 bit floats, so a golden
/// test passes the identity).
pub fn feather(alpha: &[f32], plan: &Plan, store: &dyn Fn(f32) -> f32) -> Vec<f32> {
    if !plan.feathers() {
        return alpha.to_vec();
    }
    let ((first_x, first_y), grid) = plan.grid();
    let means = cells(alpha, plan, store);
    let blurred = basic::gaussian_stored_radius(
        &means,
        grid.0,
        grid.1,
        plan.sigma_cells,
        plan.radius_cells as i32,
        store,
    );
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
    (0..alpha.len())
        .map(|i| {
            let (x, y) = (i as u32 % width, i as u32 / width);
            let (x0, x1, fx) = among(x, plan.origin.0, first_x, grid.0);
            let (y0, y1, fy) = among(y, plan.origin.1, first_y, grid.1);
            let cell = |cx: u32, cy: u32| blurred[(cy * grid.0 + cx) as usize];
            let (tl, tr, bl, br) = (cell(x0, y0), cell(x1, y0), cell(x0, y1), cell(x1, y1));
            let top = tl + (tr - tl) * fx;
            let bottom = bl + (br - bl) * fx;
            top + (bottom - top) * fy
        })
        .collect()
}

/// Contrast at one alpha. At rest it gives `p` back bit for bit.
pub fn contrast(p: f32, plan: &Plan) -> f32 {
    if plan.contrast >= MAX_EDGE_CONTRAST {
        if p >= CONTRAST_MIDDLE { 1.0 } else { 0.0 }
    } else if plan.contrasts() {
        (CONTRAST_MIDDLE + (p - CONTRAST_MIDDLE) * plan.gain).clamp(0.0, 1.0)
    } else {
        p
    }
}

/// The finished alpha of a render before the r8unorm store rounds it: Shift
/// edge, then Feather, then Contrast, each skipped while it is at rest.
/// `alpha` is the alpha as the GPU holds it (8-bit codes).
pub fn finished_with(alpha: &[f32], plan: &Plan, store: &dyn Fn(f32) -> f32) -> Vec<f32> {
    let shifted = if plan.shifts() {
        shift(alpha, plan)
    } else {
        alpha.to_vec()
    };
    let feathered = feather(&shifted, plan, store);
    if plan.contrasts() {
        feathered.into_iter().map(|p| contrast(p, plan)).collect()
    } else {
        feathered
    }
}

/// [`finished_with`] on a render of this geometry.
pub fn finished(
    alpha: &[f32],
    geometry: &Geometry,
    edge: &Edge,
    store: &dyn Fn(f32) -> f32,
) -> Vec<f32> {
    finished_with(alpha, &Plan::for_geometry(edge, geometry), store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::stored_alpha;

    const IDENTITY: &dyn Fn(f32) -> f32 = &|v| v;

    fn edge(shift: f32, feather: f32, contrast: f32) -> Edge {
        Edge {
            shift,
            feather,
            contrast,
        }
    }

    fn plan(edge: Edge, size: (u32, u32)) -> Plan {
        Plan::new(&edge, size, (0, 0), size)
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
    }

    /// Stored alphas with every code, in a pattern that holds edges both ways.
    fn busy(size: (u32, u32)) -> Vec<f32> {
        (0..size.0 * size.1)
            .map(|i| {
                let (x, y) = ((i % size.0) as f32, (i / size.0) as f32);
                let v = 0.5 + 0.5 * ((x * 0.37).sin() * (y * 0.23).cos());
                let hard = if (x - 20.0).abs() < 9.0 && (y - 14.0).abs() < 6.0 {
                    1.0
                } else {
                    v * 0.6
                };
                stored_alpha(hard)
            })
            .collect()
    }

    #[test]
    fn the_plan_takes_its_numbers_from_the_whole_picture() {
        // Shift 1 percent of 6000 pixels: r is 60, the axis runs 24 steps a
        // side and the diagonal runs 17.
        let p = plan(edge(0.01, 0.0, 0.0), (6000, 4000));
        assert!(p.grow && p.shifts() && !p.feathers() && !p.contrasts());
        assert_eq!((p.axis, p.diagonal), (24, 17));
        assert_eq!(p.shift_reach(), 58);
        let p = plan(edge(-0.01, 0.0, 0.0), (4000, 6000));
        assert!(!p.grow);
        assert_eq!((p.axis, p.diagonal, p.reach()), (24, 17, 58));
        // Feather 1 percent of 6000 pixels: sigma 60, cells of 15, 4 sigmas
        // of a cell and 12 cells of kernel.
        let p = plan(edge(0.0, 0.01, 0.0), (6000, 4000));
        assert!(p.feathers() && !p.shifts());
        assert_eq!(p.step, 15);
        assert!((p.sigma - 60.0).abs() < 1e-3 && (p.sigma_cells - 4.0).abs() < 1e-4);
        assert_eq!(p.radius_cells, 12);
        assert_eq!(p.feather_reach(), 14 * 15);
        // The widest cell is 16 pixels, the narrowest 1.
        let p = plan(edge(0.0, 0.05, 0.0), (6000, 4000));
        assert_eq!((p.step, p.radius_cells), (16, 57));
        // 0.001 of 6000 is 6.0000005 in f32: three sigmas round up to 19.
        let p = plan(edge(0.0, 0.001, 0.0), (6000, 4000));
        assert_eq!((p.step, p.radius_cells), (1, 19));
        // Contrast reads the pixel alone.
        let p = plan(edge(0.0, 0.0, 80.0), (6000, 4000));
        assert!(p.contrasts() && p.reach() == 0);
        assert!((p.gain - 5.0).abs() < 1e-5);
        // All three at once add their reaches.
        let p = plan(edge(0.01, 0.01, 50.0), (6000, 4000));
        assert_eq!(p.reach(), 58 + 210);
        assert_eq!(reach(&edge(0.01, 0.01, 50.0), (6000, 4000)), 268);
        assert_eq!(reach(&Edge::default(), (6000, 4000)), 0);
        // A shift under about 1.3 pixels takes no step and runs no pass.
        let p = plan(edge(0.0002, 0.0, 0.0), (6000, 4000));
        assert!(!p.shifts() && p.is_off() && p.runs().is_empty());
    }

    #[test]
    fn the_plan_of_a_geometry_finds_the_whole_picture_and_the_origin() {
        let geometry = Geometry {
            window: gamut_core::CropRect {
                x: 100.0 / 640.0,
                y: 60.0 / 480.0,
                width: 200.0 / 640.0,
                height: 120.0 / 480.0,
            },
            size: (200, 120),
            photo: (3000, 2250),
        };
        let p = Plan::for_geometry(&edge(0.02, 0.02, 10.0), &geometry);
        assert_eq!(
            (p.full, p.origin, p.size),
            ((640, 480), (100, 60), (200, 120))
        );
        assert_eq!(
            p,
            Plan::new(&edge(0.02, 0.02, 10.0), (640, 480), (100, 60), (200, 120))
        );
    }

    #[test]
    fn a_run_takes_ceil_log2_of_its_length_in_doubling_passes() {
        assert!(doubling(0).is_empty());
        assert_eq!(doubling(1), vec![1]);
        assert_eq!(doubling(2), vec![1, 1]);
        assert_eq!(doubling(3), vec![1, 2]);
        assert_eq!(doubling(24), vec![1, 2, 4, 8, 9]);
        for half in 1..300u32 {
            let offsets = doubling(half);
            assert_eq!(1 + offsets.iter().sum::<u32>(), half + 1, "{half}");
            assert_eq!(
                offsets.len() as u32,
                (half + 1).next_power_of_two().ilog2(),
                "{half}"
            );
        }
    }

    /// The maximum (or minimum) over the run's samples, each held inside the
    /// render, the long way.
    fn direct_run(alpha: &[f32], size: (u32, u32), run: Run, grow: bool) -> Vec<f32> {
        let mut out = vec![0.0; alpha.len()];
        for y in 0..size.1 as i32 {
            for x in 0..size.0 as i32 {
                let mut v = alpha[at(x, y, size)];
                for i in -(run.half as i32)..=run.half as i32 {
                    let s = alpha[at(x + i * run.direction.0, y + i * run.direction.1, size)];
                    v = pick(grow, v, s);
                }
                out[at(x, y, size)] = v;
            }
        }
        out
    }

    #[test]
    fn the_doubling_runs_equal_the_direct_runs_edges_included() {
        let size = (37, 29);
        let alpha = busy(size);
        for grow in [true, false] {
            for direction in RUNS {
                for half in [1, 2, 3, 5, 8, 13, 40] {
                    let step = Run { direction, half };
                    assert!(
                        same_bits(
                            &run(&alpha, size, step, grow),
                            &direct_run(&alpha, size, step, grow)
                        ),
                        "{direction:?} {half} {grow}"
                    );
                }
            }
        }
    }

    #[test]
    fn each_control_at_rest_is_the_identity_bit_for_bit_alone_and_together() {
        let size = (48, 40);
        let alpha = busy(size);
        let rest = plan(Edge::default(), size);
        assert!(rest.is_off());
        assert!(same_bits(&finished_with(&alpha, &rest, IDENTITY), &alpha));
        assert!(same_bits(&shift(&alpha, &rest), &alpha));
        assert!(same_bits(&feather(&alpha, &rest, IDENTITY), &alpha));
        assert!(
            alpha
                .iter()
                .all(|p| contrast(*p, &rest).to_bits() == p.to_bits())
        );
        // One control at rest beside the other two leaves the chain of the
        // other two.
        let on = plan(edge(0.1, 0.05, 60.0), size);
        let no_shift = plan(edge(0.0, 0.05, 60.0), size);
        let want: Vec<f32> = feather(&alpha, &on, IDENTITY)
            .into_iter()
            .map(|p| contrast(p, &on))
            .collect();
        assert!(same_bits(
            &finished_with(&alpha, &no_shift, IDENTITY),
            &want
        ));
        let no_feather = plan(edge(0.1, 0.0, 60.0), size);
        let want: Vec<f32> = shift(&alpha, &on)
            .into_iter()
            .map(|p| contrast(p, &on))
            .collect();
        assert!(same_bits(
            &finished_with(&alpha, &no_feather, IDENTITY),
            &want
        ));
        let no_contrast = plan(edge(0.1, 0.05, 0.0), size);
        let want = feather(&shift(&alpha, &on), &on, IDENTITY);
        assert!(same_bits(
            &finished_with(&alpha, &no_contrast, IDENTITY),
            &want
        ));
        // Each alone, with the others at rest.
        let shifted = plan(edge(0.1, 0.0, 0.0), size);
        assert!(same_bits(
            &finished_with(&alpha, &shifted, IDENTITY),
            &shift(&alpha, &on)
        ));
    }

    /// How far a grown (or shrunk) disc reaches past (or inside) its rim along
    /// 360 directions: the least and the most. The render is a window of 520
    /// pixels at the corner of a picture of 2000, so r of 100 is inside the
    /// range of Shift edge.
    fn disc_extent(r_pixels: f32, grow: bool) -> (f32, f32) {
        let side = 520u32;
        let whole = 2000u32;
        let size = (side, side);
        let centre = side as f32 / 2.0;
        let rim = if grow { 60.0 } else { 200.0 };
        let disc: Vec<f32> = (0..side * side)
            .map(|i| {
                let (x, y) = (
                    (i % side) as f32 + 0.5 - centre,
                    (i / side) as f32 + 0.5 - centre,
                );
                if x * x + y * y <= rim * rim { 1.0 } else { 0.0 }
            })
            .collect();
        let shift = r_pixels / whole as f32;
        let p = Plan::new(
            &edge(if grow { shift } else { -shift }, 0.0, 0.0),
            (whole, whole),
            (0, 0),
            size,
        );
        let out = super::shift(&disc, &p);
        let value = |d: f32, t: f32| {
            let x = (centre + d * t.cos()).floor() as i32;
            let y = (centre + d * t.sin()).floor() as i32;
            out[at(x, y, size)]
        };
        let mut least = f32::MAX;
        let mut most = 0.0f32;
        for k in 0..360 {
            let t = (k as f32).to_radians();
            let steps = (0..1000).map(|j| j as f32 * 0.25);
            let extent = if grow {
                // The farthest place still inside, past the rim.
                steps.filter(|d| value(*d, t) > 0.5).fold(0.0f32, f32::max) - rim
            } else {
                // The first place outside, inside the rim.
                rim - steps
                    .take_while(|d| value(*d, t) > 0.5)
                    .last()
                    .unwrap_or(0.0)
            };
            least = least.min(extent);
            most = most.max(extent);
        }
        (least, most)
    }

    #[test]
    fn a_hard_disc_grown_or_shrunk_by_r_moves_between_0_96_r_and_1_04_r_in_every_direction() {
        // The octagon lies between 0.96 r and 1.04 r; a pixel either way is
        // the grid of the picture and of the sampling.
        for r in [60.0f32, 100.0] {
            for grow in [true, false] {
                let (least, most) = disc_extent(r, grow);
                assert!(
                    least >= 0.96 * r - 1.0 && most <= 1.04 * r + 1.0,
                    "r {r} grow {grow}: {least} to {most}"
                );
            }
        }
    }

    /// Where the 0.5 contour of the finished alpha lands on row 0, between
    /// two pixel centres.
    fn contour_after(alpha: &[f32], size: (u32, u32), p: &Plan) -> f32 {
        let out = finished_with(alpha, p, IDENTITY);
        // The first column under 0.5, as a place between two pixel centres.
        let row = &out[..size.0 as usize];
        let first = row.iter().position(|v| *v < 0.5).expect("an edge") as f32;
        if first == 0.0 {
            return 0.0;
        }
        let (a, b) = (row[first as usize - 1], row[first as usize]);
        first - 0.5 + (a - 0.5) / (a - b)
    }

    #[test]
    fn a_straight_edge_moves_round_0_96_r_within_1_pixel_each_way() {
        // A render of 1200 pixels at the corner of a picture of 2400.
        let size = (1200u32, 3u32);
        let whole = (2400u32, 2400u32);
        let edge_at = 600.0;
        let step: Vec<f32> = (0..size.0 * size.1)
            .map(|i| {
                if ((i % size.0) as f32) < edge_at {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let base = contour_after(&step, size, &plan(Edge::default(), size));
        assert!((base - edge_at).abs() < 1e-3);
        for r in [16.0f32, 60.0, 100.0] {
            let want = (0.96 * r).round();
            for grow in [true, false] {
                let s = r / whole.0 as f32;
                let p = Plan::new(
                    &edge(if grow { s } else { -s }, 0.0, 0.0),
                    whole,
                    (0, 0),
                    size,
                );
                let moved = contour_after(&step, size, &p) - base;
                let moved = if grow { moved } else { -moved };
                assert!(
                    (moved - want).abs() <= 1.0,
                    "r {r} grow {grow}: moved {moved}, want {want}"
                );
            }
        }
    }

    #[test]
    fn a_ramp_shifted_keeps_its_slope() {
        let size = (800u32, 2u32);
        let ramp: Vec<f32> = (0..size.0 * size.1)
            .map(|i| stored_alpha(((500.0 - (i % size.0) as f32) / 100.0).clamp(0.0, 1.0)))
            .collect();
        for grow in [true, false] {
            let s = 60.0 / size.0 as f32;
            let p = plan(edge(if grow { s } else { -s }, 0.0, 0.0), size);
            let moved = p.shift_reach() as i32 * if grow { 1 } else { -1 };
            let out = shift(&ramp, &p);
            // Every pixel of the ramp is the one `moved` pixels back.
            for x in 300..700i32 {
                let want = ramp[(x - moved).clamp(0, size.0 as i32 - 1) as usize];
                assert_eq!(out[x as usize].to_bits(), want.to_bits(), "{x} {grow}");
            }
        }
    }

    #[test]
    fn a_mask_of_all_1_or_all_0_shifted_is_itself() {
        let size = (64, 48);
        for value in [0.0f32, 1.0] {
            let flat = vec![value; 64 * 48];
            for s in [0.2f32, -0.2] {
                let out = shift(&flat, &plan(edge(s, 0.0, 0.0), size));
                assert!(same_bits(&out, &flat), "{value} {s}");
            }
        }
    }

    /// The 0.16 to 0.84 band of a row, in pixels.
    fn band(row: &[f32]) -> usize {
        row.iter().filter(|v| **v > 0.16 && **v < 0.84).count()
    }

    fn hard_step(size: (u32, u32), at_column: u32) -> Vec<f32> {
        (0..size.0 * size.1)
            .map(|i| if i % size.0 < at_column { 1.0 } else { 0.0 })
            .collect()
    }

    #[test]
    fn a_feathered_hard_step_is_symmetric_about_the_edge_and_its_band_is_2_sigma() {
        // Edges on the cell grid: cells of 4 and of 15 pixels.
        for (width, sigma) in [(512u32, 16.0f32), (1200, 60.0)] {
            let size = (width, 3);
            let step = hard_step(size, width / 2);
            let p = plan(edge(0.0, sigma / width as f32, 0.0), size);
            let out = feather(&step, &p, IDENTITY);
            let row = &out[..width as usize];
            let wide = band(row) as f32;
            assert!(
                (wide - 2.0 * sigma).abs() <= 0.1 * 2.0 * sigma,
                "sigma {sigma}: band {wide}"
            );
            // The same shape either side of the edge, turned over.
            let e = (width / 2) as usize;
            for k in 0..(3.0 * sigma) as usize {
                let (inside, outside) = (row[e - 1 - k], row[e + k]);
                assert!(
                    (inside - (1.0 - outside)).abs() * 255.0 < 1.0,
                    "sigma {sigma}, {k}: {inside} {outside}"
                );
            }
        }
    }

    #[test]
    fn a_flat_mask_feathered_is_itself() {
        let size = (300, 200);
        let flat = vec![stored_alpha(0.6); 300 * 200];
        for feather_share in [0.004f32, 0.02, 0.05] {
            let p = plan(edge(0.0, feather_share, 0.0), size);
            let out = feather(&flat, &p, IDENTITY);
            assert!(
                out.iter().all(|v| stored_alpha(*v) == flat[0]),
                "{feather_share}"
            );
        }
    }

    #[test]
    fn at_cells_of_one_pixel_the_feather_equals_gaussian_stored_bit_for_bit() {
        let size = (90, 70);
        let alpha = busy(size);
        // Sigmas of 1.8, 4.5 and 7.9 pixels, the last on a render that is a
        // window of a wider picture.
        for (full, share) in [((90, 70), 0.02f32), ((90, 70), 0.05), ((158, 70), 0.05)] {
            let p = Plan::new(&edge(0.0, share, 0.0), full, (0, 0), size);
            assert_eq!(p.step, 1, "{full:?} {share}");
            let want = basic::gaussian_stored(&alpha, size.0, size.1, p.sigma, IDENTITY);
            assert!(
                same_bits(&feather(&alpha, &p, IDENTITY), &want),
                "{full:?} {share}"
            );
        }
    }

    #[test]
    fn the_cell_feather_is_within_1_code_of_a_full_resolution_gaussian() {
        let size = (6000u32, 2u32);
        let step = hard_step(size, 3000);
        for sigma in [16.0f32, 60.0, 300.0] {
            let p = plan(edge(0.0, sigma / size.0 as f32, 0.0), size);
            assert!(p.step > 1, "sigma {sigma}");
            let out = feather(&step, &p, IDENTITY);
            let full = basic::gaussian_stored_radius(
                &step,
                size.0,
                size.1,
                p.sigma,
                (FEATHER_SIGMAS * p.sigma).ceil() as i32,
                IDENTITY,
            );
            let worst = out
                .iter()
                .zip(&full)
                .map(|(a, b)| (a - b).abs() * 255.0)
                .fold(0.0f32, f32::max);
            println!("sigma {sigma}, cells of {}: {worst:.3} codes", p.step);
            assert!(worst <= 1.0, "sigma {sigma}: {worst} codes");
        }
    }

    #[test]
    fn contrast_0_is_the_identity_and_100_is_a_step_at_a_half() {
        let size = (256, 1);
        let ramp: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let rest = plan(edge(0.0, 0.0, 0.0), size);
        assert!(
            ramp.iter()
                .all(|p| contrast(*p, &rest).to_bits() == p.to_bits())
        );
        let hard = plan(edge(0.0, 0.0, 100.0), size);
        for p in &ramp {
            assert_eq!(contrast(*p, &hard), if *p >= 0.5 { 1.0 } else { 0.0 });
        }
        assert_eq!(contrast(0.5, &hard), 1.0);
        // Contrast turns about a half and never leaves 0 to 1.
        let half = plan(edge(0.0, 0.0, 50.0), size);
        assert_eq!(contrast(0.5, &half), 0.5);
        assert_eq!(contrast(0.75, &half), 1.0);
        assert!((contrast(0.4, &half) - 0.3).abs() < 1e-6);
        assert_eq!(contrast(0.0, &half), 0.0);
    }

    #[test]
    fn contrast_90_shrinks_a_2_sigma_band_to_a_tenth_within_2_pixels() {
        let size = (1200u32, 2u32);
        let step = hard_step(size, 600);
        let sigma = 60.0;
        let soft = plan(edge(0.0, sigma / 1200.0, 0.0), size);
        let firm = plan(edge(0.0, sigma / 1200.0, 90.0), size);
        let before = band(&finished_with(&step, &soft, IDENTITY)[..1200]) as f32;
        let after = band(&finished_with(&step, &firm, IDENTITY)[..1200]) as f32;
        assert!(
            (before - 2.0 * sigma).abs() <= 0.1 * 2.0 * sigma,
            "{before}"
        );
        assert!((after - before / 10.0).abs() <= 2.0, "{before} to {after}");
    }

    #[test]
    fn the_chain_runs_shift_then_feather_then_contrast() {
        // A line one pixel wide: grown first and then feathered it keeps a
        // peak near 1, feathered first and then grown it does not.
        let size = (200u32, 60u32);
        let line: Vec<f32> = (0..size.0 * size.1)
            .map(|i| if i % size.0 == 100 { 1.0 } else { 0.0 })
            .collect();
        let p = plan(edge(0.05, 0.02, 60.0), size);
        let out = finished_with(&line, &p, IDENTITY);
        let ruled: Vec<f32> = feather(&shift(&line, &p), &p, IDENTITY)
            .into_iter()
            .map(|v| contrast(v, &p))
            .collect();
        assert!(same_bits(&out, &ruled));
        let feather_first: Vec<f32> = shift(&feather(&line, &p, IDENTITY), &p)
            .into_iter()
            .map(|v| contrast(v, &p))
            .collect();
        let contrast_first = feather(
            &shift(
                &line.iter().map(|v| contrast(*v, &p)).collect::<Vec<_>>(),
                &p,
            ),
            &p,
            IDENTITY,
        );
        let peak = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
        assert!(peak(&out) > 0.9, "{}", peak(&out));
        assert!(peak(&feather_first) < 0.5, "{}", peak(&feather_first));
        assert!(!same_bits(&out, &contrast_first));
    }

    /// The finished alpha of one rectangle from the whole render and from a
    /// window that holds it and `pad` pixels around it. With Feather on the
    /// alpha is a ramp, so a cell the window cuts holds another mean. With
    /// Shift edge alone it is one column, 1 on 0 when it grows and 0 on 1 when
    /// it shrinks, exactly the reach past the right of what is wanted: the
    /// farthest sample the octagon reads.
    fn edge_whole_and_window(edge: Edge, pad: u32) -> (Vec<f32>, Vec<f32>) {
        let full = (640u32, 480u32);
        let p = plan(edge, full);
        let line = 460 + p.shift_reach();
        let alpha: Vec<f32> = (0..full.0 * full.1)
            .map(|i| {
                let (x, y) = (i % full.0, i / full.0);
                if p.feathers() {
                    stored_alpha((0.8 * x as f32 + 0.5 * y as f32 - 300.0) / 255.0)
                } else if (x == line) == p.grow {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let whole = finished_with(&alpha, &Plan::new(&edge, full, (0, 0), full), IDENTITY);
        let wanted = (401u32, 187u32, 60u32, 50u32);
        let (x0, y0) = (wanted.0 - pad.min(wanted.0), wanted.1 - pad.min(wanted.1));
        let x1 = (wanted.0 + wanted.2 + pad).min(full.0);
        let y1 = (wanted.1 + wanted.3 + pad).min(full.1);
        let size = (x1 - x0, y1 - y0);
        let seen: Vec<f32> = (0..size.0 * size.1)
            .map(|i| alpha[((y0 + i / size.0) * full.0 + x0 + i % size.0) as usize])
            .collect();
        let window = finished_with(&seen, &Plan::new(&edge, full, (x0, y0), size), IDENTITY);
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
        let full = (640, 480);
        // Shift edge either way, with one pixel less; Feather on cells of 4,
        // with one cell less; all three, whose octagon on a ramp reads its
        // farthest sample on one side only, so no step less is tried there.
        for (e, less) in [
            (edge(0.02, 0.0, 0.0), 1),
            (edge(-0.03, 0.0, 40.0), 1),
            (edge(0.0, 0.025, 0.0), 4),
            (edge(0.015, 0.03, 60.0), 0),
        ] {
            let p = plan(e, full);
            let (whole, window) = edge_whole_and_window(e, p.reach());
            assert!(same_bits(&whole, &window), "{e:?}, pad {}", p.reach());
            if less == 0 {
                continue;
            }
            let (whole, window) = edge_whole_and_window(e, p.reach() - less);
            assert!(
                !same_bits(&whole, &window),
                "{e:?}, pad {}",
                p.reach() - less
            );
        }
    }

    #[test]
    fn nothing_divides_by_zero_on_a_mask_of_all_1_all_0_and_a_single_pixel() {
        for size in [(1u32, 1u32), (1, 7), (9, 1), (33, 17)] {
            let n = (size.0 * size.1) as usize;
            for value in [0.0f32, 1.0, stored_alpha(0.5)] {
                let alpha = vec![value; n];
                for e in [
                    edge(0.05, 0.05, 100.0),
                    edge(-0.05, 0.05, 99.99),
                    edge(0.05, 0.0001, 50.0),
                    edge(0.0, f32::MIN_POSITIVE, 0.0),
                ] {
                    let p = plan(e, size);
                    let out = finished_with(&alpha, &p, IDENTITY);
                    assert!(
                        out.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                        "{size:?} {value} {e:?}"
                    );
                    if value == 0.0 || value == 1.0 {
                        assert!(same_bits(&out, &alpha), "{size:?} {value} {e:?}");
                    }
                }
            }
        }
    }
}
