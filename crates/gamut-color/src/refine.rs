//! Refine edges on the CPU: the finished alpha of a mask moved onto the edges
//! of the picture under it. refine.wgsl in gamut-gpu mirrors this module; the
//! golden tests hold the two together.
//!
//! It is the colour guided filter of He, Sun and Tang. The input `p` is the
//! alpha of the mask as the GPU stores it. The guide `I` is the SOURCE pixel,
//! encoded to ACEScct, never the developed one, so no slider of the develop
//! chain moves a refined mask. Over a box around each place,
//!
//! ```text
//! a = (var(I) + eps Id)^-1 cov(I, p)      b = mean(p) - a . mean(I)
//! q = mean(a) . I + mean(b), held inside 0 to 1
//! ```
//!
//! and the refined alpha is `p + (q - p) amount / 100`. Where the guide is
//! flat `a` is 0 and `q` is a box blur of `p`; where the guide has an edge `q`
//! steps with it.
//!
//! The moments are taken on a grid of cells (the fast guided filter): a cell
//! is [`Plan::step`] pixels square and lies on the pixel grid of the WHOLE
//! picture at the scale of the render, not on the grid of the window that is
//! being rendered, so a zoomed window and the full render hold the same cells.
//! `a` and `b` come back up bilinearly and `q` reads the guide at full
//! resolution. Everything is measured from the size of the whole picture, so
//! the fitted view, 100 percent and the export show the same edge at their
//! own scales.
//!
//! Arithmetic is f32 in the order the shader sums in. `store` is applied
//! wherever the GPU writes a texture of the filter; those are 32 bit float
//! targets, so a golden test passes the identity.

use gamut_core::mask::Refine;

use crate::acescct;
use crate::mask::Geometry;

/// The guide's eps at a sensitivity of 0, before squaring: only an edge of
/// about two stops holds the mask.
pub const EPS_LOOSE: f32 = 0.1;

/// The guide's eps at a sensitivity of 100, before squaring.
pub const EPS_STRICT: f32 = 0.005;

/// The radius in render pixels one step of the cell grid is taken for.
pub const PIXELS_A_STEP: f32 = 4.0;

/// The widest cell, in render pixels.
pub const MAX_STEP: u32 = 4;

/// The channels of the moments of one cell: p, I p (3), I (3), I I (6).
pub const MOMENTS: usize = 13;

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
        let cells = ((radius / step as f32).round() as u32).max(1);
        Plan {
            full,
            origin,
            size,
            step,
            cells,
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

    /// How far around a pixel the filter reads, in render pixels: the box of
    /// the moments, the box of `a` and `b`, the cell beside for the bilinear
    /// step, and the cell a window may cut at its border.
    pub fn reach(&self) -> u32 {
        (2 * self.cells + 2) * self.step
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
/// `full` pixels, or 0 while it is off: what a window is padded by.
pub fn reach(refine: &Refine, full: (u32, u32)) -> u32 {
    if refine.is_off() {
        0
    } else {
        Plan::new(refine, full, (0, 0), full).reach()
    }
}

/// The moments of every cell of the render's grid: each the mean of p, I p,
/// I and I I over the pixels of the cell that the render holds.
pub fn cell_moments(
    alpha: &[f32],
    guides: &[[f32; 3]],
    plan: &Plan,
    store: &dyn Fn(f32) -> f32,
) -> Vec<[f32; MOMENTS]> {
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
            let mut sum = [0.0f32; MOMENTS];
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * width + x) as usize;
                    let (p, g) = (alpha[i], guides[i]);
                    let sample = [
                        p,
                        g[0] * p,
                        g[1] * p,
                        g[2] * p,
                        g[0],
                        g[1],
                        g[2],
                        g[0] * g[0],
                        g[0] * g[1],
                        g[0] * g[2],
                        g[1] * g[1],
                        g[1] * g[2],
                        g[2] * g[2],
                    ];
                    for (total, value) in sum.iter_mut().zip(sample) {
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

/// `a` (three channels) and `b` of one cell from the means of its moments.
pub fn solve(mean: &[f32; MOMENTS], eps: f32) -> [f32; 4] {
    let p = mean[0];
    let mu = [mean[4], mean[5], mean[6]];
    let cov = [
        mean[1] - mu[0] * p,
        mean[2] - mu[1] * p,
        mean[3] - mu[2] * p,
    ];
    // var(I) + eps Id, symmetric: rr, rg, rb, gg, gb, bb.
    let rr = mean[7] - mu[0] * mu[0] + eps;
    let rg = mean[8] - mu[0] * mu[1];
    let rb = mean[9] - mu[0] * mu[2];
    let gg = mean[10] - mu[1] * mu[1] + eps;
    let gb = mean[11] - mu[1] * mu[2];
    let bb = mean[12] - mu[2] * mu[2] + eps;
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
    let a = [
        (c_rr * cov[0] + c_rg * cov[1] + c_rb * cov[2]) * scale,
        (c_rg * cov[0] + c_gg * cov[1] + c_gb * cov[2]) * scale,
        (c_rb * cov[0] + c_gb * cov[1] + c_bb * cov[2]) * scale,
    ];
    let b = p - (a[0] * mu[0] + a[1] * mu[1] + a[2] * mu[2]);
    [a[0], a[1], a[2], b]
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
    let ((first_x, first_y), grid) = plan.grid();
    let moments = cell_moments(alpha, &guides, &plan, store);
    let means = box_mean(&moments, grid, plan.cells, store);
    let ab: Vec<[f32; 4]> = means
        .iter()
        .map(|mean| solve(mean, plan.eps).map(store))
        .collect();
    let ab = box_mean(&ab, grid, plan.cells, store);
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
    alpha
        .iter()
        .zip(&guides)
        .enumerate()
        .map(|(i, (p, g))| {
            let (x, y) = (i as u32 % width, i as u32 / width);
            let (x0, x1, fx) = among(x, plan.origin.0, first_x, grid.0);
            let (y0, y1, fy) = among(y, plan.origin.1, first_y, grid.1);
            let at = |cx: u32, cy: u32| ab[(cy * grid.0 + cx) as usize];
            let (tl, tr, bl, br) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            let mut q = 0.0;
            for c in 0..4 {
                let top = tl[c] + (tr[c] - tl[c]) * fx;
                let bottom = bl[c] + (br[c] - bl[c]) * fx;
                let value = top + (bottom - top) * fy;
                q += if c < 3 { value * g[c] } else { value };
            }
            let q = q.clamp(0.0, 1.0);
            p + (q - p) * plan.amount
        })
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
    /// `width` pixels about the middle column, through 0.5 there.
    fn ramp_alpha(size: (u32, u32), width: f32) -> Vec<f32> {
        (0..size.0 * size.1)
            .map(|i| {
                let x = (i % size.0) as f32 + 0.5 - size.0 as f32 / 2.0;
                stored_alpha((0.5 - x / width).clamp(0.0, 1.0))
            })
            .collect()
    }

    #[test]
    fn an_amount_of_0_is_the_identity_bit_for_bit() {
        let size = (40, 30);
        let geometry = Geometry::full(size, size);
        let pixels = step_guide(size, 0.05, 0.6);
        let alpha = ramp_alpha(size, 17.0);
        for radius in [0.001, 0.02, 0.05] {
            let out = refined(&alpha, &pixels, &geometry, &on(0.0, radius, 80.0), IDENTITY);
            assert!(
                out.iter()
                    .zip(&alpha)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "radius {radius}"
            );
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
    fn the_plan_takes_its_cells_from_the_whole_picture() {
        let plan = Plan::new(&on(100.0, 0.01, 50.0), (6000, 4000), (0, 0), (6000, 4000));
        assert_eq!((plan.step, plan.cells), (4, 15), "60 pixels");
        assert_eq!(plan.reach(), 128);
        let plan = Plan::new(&on(100.0, 0.01, 50.0), (1100, 700), (0, 0), (1100, 700));
        assert_eq!((plan.step, plan.cells), (2, 6), "11 pixels");
        let plan = Plan::new(&on(100.0, 0.05, 50.0), (64, 64), (0, 0), (64, 64));
        assert_eq!((plan.step, plan.cells), (1, 3), "3.2 pixels");
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
        assert_eq!((window.step, window.cells), (4, 15));
        assert_eq!(window.grid(), ((255, 128), (226, 176)));
        assert_eq!(reach(&on(0.0, 0.05, 50.0), (6000, 4000)), 0, "off");
        assert_eq!(reach(&on(100.0, 0.01, 50.0), (6000, 4000)), 128);
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
    fn a_flat_guide_gives_a_plain_box_blur_of_the_alpha() {
        let size = (48, 24);
        let geometry = Geometry::full(size, size);
        let pixels = vec![[0.18; 3]; (size.0 * size.1) as usize];
        let alpha = ramp_alpha(size, 9.0);
        let refine = on(100.0, 0.05, 50.0);
        let plan = Plan::for_geometry(&refine, &geometry);
        assert_eq!((plan.step, plan.cells), (1, 2));
        let out = refined(&alpha, &pixels, &geometry, &refine, IDENTITY);
        // With a flat guide a is 0 and b the mean of p, so q is the box mean
        // of the box mean.
        let cells: Vec<[f32; 1]> = alpha.iter().map(|p| [*p]).collect();
        let once = box_mean(&cells, size, plan.cells, IDENTITY);
        let twice = box_mean(&once, size, plan.cells, IDENTITY);
        for (i, (got, want)) in out.iter().zip(&twice).enumerate() {
            assert!(
                (got - want[0]).abs() < 2e-4,
                "pixel {i}: {got} is not {}",
                want[0]
            );
        }
    }

    #[test]
    fn a_blurred_alpha_snaps_onto_a_step_of_the_guide() {
        let size = (96, 16);
        let geometry = Geometry::full(size, size);
        let pixels = step_guide(size, 0.03, 0.5);
        // A soft edge of the mask: a ramp one radius wide through 0.5 at the
        // edge of the guide. The filter pulls each side toward the mean of
        // the alpha on that side of the edge inside its box, so a ramp far
        // wider than the box stays a ramp; this one snaps.
        let refine = on(100.0, 0.05, 50.0);
        let plan = Plan::for_geometry(&refine, &geometry);
        let radius = (plan.cells * plan.step) as f32;
        let alpha = ramp_alpha(size, radius);
        let out = refined(&alpha, &pixels, &geometry, &refine, IDENTITY);
        let row = |x: u32| out[(8 * size.0 + x) as usize];
        let given = |x: u32| alpha[(8 * size.0 + x) as usize];
        let edge = size.0 / 2;
        let (inside, outside) = (edge - radius as u32, edge + radius as u32 - 1);
        assert!(given(edge - 1) - given(edge) < 0.25, "the input is a ramp");
        assert!(row(inside) > 0.9, "one radius inside: {}", row(inside));
        assert!(row(outside) < 0.1, "one radius outside: {}", row(outside));
        let step = row(edge - 1) - row(edge);
        let given_step = given(edge - 1) - given(edge);
        assert!(
            step > 0.6 && step > 3.0 * given_step,
            "it steps with the guide: {step} where the input steps {given_step}"
        );
        // Half the amount goes half the way.
        let half = refined(&alpha, &pixels, &geometry, &on(50.0, 0.05, 50.0), IDENTITY);
        let i = (8 * size.0 + inside) as usize;
        assert!((half[i] - (alpha[i] + out[i]) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_sensitivity_of_0_smooths_across_a_weak_edge_that_100_keeps() {
        let size = (96, 16);
        let geometry = Geometry::full(size, size);
        // A third of a stop: 0.02 of the ACEScct code between the two sides.
        let pixels = step_guide(size, 0.18, 0.18 * 1.26);
        let alpha = ramp_alpha(size, 5.0);
        let at = |sensitivity: f32| {
            let out = refined(
                &alpha,
                &pixels,
                &geometry,
                &on(100.0, 0.05, sensitivity),
                IDENTITY,
            );
            let edge = size.0 / 2;
            out[(8 * size.0 + edge - 1) as usize] - out[(8 * size.0 + edge) as usize]
        };
        let (loose, strict) = (at(0.0), at(100.0));
        assert!(loose < 0.15, "smoothed across: a step of {loose}");
        assert!(strict > 0.5, "kept: a step of {strict}");
    }

    #[test]
    fn nothing_divides_by_zero_on_a_flat_black_guide() {
        let size = (32, 32);
        let geometry = Geometry::full(size, size);
        let pixels = vec![[0.0; 3]; (size.0 * size.1) as usize];
        let alpha = ramp_alpha(size, 6.0);
        for sensitivity in [0.0, 100.0] {
            let out = refined(
                &alpha,
                &pixels,
                &geometry,
                &on(100.0, 0.05, sensitivity),
                IDENTITY,
            );
            assert!(out.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        }
        assert!(
            solve(&[0.0; MOMENTS], eps(100.0))
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// A picture with a texture and an edge, so nothing about it is regular.
    fn busy(full: (u32, u32)) -> (Vec<[f32; 3]>, Vec<f32>) {
        let mut pixels = Vec::new();
        let mut alpha = Vec::new();
        for y in 0..full.1 {
            for x in 0..full.0 {
                let (fx, fy) = (x as f32, y as f32);
                let side = if fx + 0.3 * fy < full.0 as f32 * 0.55 {
                    0.04
                } else {
                    0.4
                };
                let grain = 0.02 * ((fx * 0.9).sin() * (fy * 1.3).cos() + 1.0);
                pixels.push([side + grain, side + 0.5 * grain, side * 0.8 + grain]);
                let d = ((fx - full.0 as f32 * 0.5).powi(2) + (fy - full.1 as f32 * 0.5).powi(2))
                    .sqrt();
                alpha.push(stored_alpha(
                    (1.4 - d / (full.1 as f32 * 0.4)).clamp(0.0, 1.0),
                ));
            }
        }
        (pixels, alpha)
    }

    #[test]
    fn a_window_padded_by_the_reach_gives_what_the_full_render_gives() {
        let full = (200, 120);
        let (pixels, alpha) = busy(full);
        for radius in [0.012, 0.03, 0.05] {
            let refine = on(100.0, radius, 60.0);
            let whole = refined(
                &alpha,
                &pixels,
                &Geometry::full(full, full),
                &refine,
                IDENTITY,
            );
            let reach = reach(&refine, full);
            // What is wanted, off the cell grid on purpose, and the window
            // that holds it and the reach.
            let wanted = (71u32, 33u32, 50u32, 40u32);
            let (x0, y0) = (
                wanted.0 - reach.min(wanted.0),
                wanted.1 - reach.min(wanted.1),
            );
            let x1 = (wanted.0 + wanted.2 + reach).min(full.0);
            let y1 = (wanted.1 + wanted.3 + reach).min(full.1);
            let size = (x1 - x0, y1 - y0);
            let cut = |source: &[f32]| -> Vec<f32> {
                (0..size.0 * size.1)
                    .map(|i| source[((y0 + i / size.0) * full.0 + x0 + i % size.0) as usize])
                    .collect()
            };
            let window_pixels: Vec<[f32; 3]> = (0..size.0 * size.1)
                .map(|i| pixels[((y0 + i / size.0) * full.0 + x0 + i % size.0) as usize])
                .collect();
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
            let window = refined(&cut(&alpha), &window_pixels, &geometry, &refine, IDENTITY);
            for y in wanted.1..wanted.1 + wanted.3 {
                for x in wanted.0..wanted.0 + wanted.2 {
                    let a = whole[(y * full.0 + x) as usize];
                    let b = window[((y - y0) * size.0 + x - x0) as usize];
                    assert!(
                        a.to_bits() == b.to_bits(),
                        "radius {radius}, pixel ({x}, {y}): {a} in the whole, {b} in the window"
                    );
                }
            }
        }
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
            eps: eps(50.0),
            amount: 1.0,
        };
        assert_eq!(plan.grid(), ((0, 0), (3, 1)));
        let alpha: Vec<f32> = (0..40)
            .map(|i| if i % 10 >= 8 { 1.0 } else { 0.0 })
            .collect();
        let guides = vec![[0.3; 3]; 40];
        let moments = cell_moments(&alpha, &guides, &plan, IDENTITY);
        assert_eq!(moments.len(), 3);
        assert_eq!(moments[2][0], 1.0, "the mean of the two columns it holds");
        assert_eq!(moments[1][0], 0.0);
    }
}
