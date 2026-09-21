//! The painted mask source on the CPU: where the dabs of a stroke fall, the
//! shape of one dab and how dabs build an alpha. The brush pass of gamut-gpu
//! stamps the same dabs, which it takes from [`dab_centres`], through two
//! blend states that are [`Dab::apply`]; the golden tests hold the two
//! together.
//!
//! A stroke is stamped along its path at a quarter of its radius, starting
//! on its first point; a path of one point is one dab. A dab is a disc
//! measured after the aspect scaling every mask distance uses, so it is
//! round on a photo of any shape: full inside its radius less the feather,
//! falling to 0 at the radius by the smoothstep of the radial gradient.
//! Dabs build in order. A paint dab of strength s takes the alpha a to
//! a + s (1 - a), an erase dab to a (1 - s), so an erase removes only what
//! was painted before it.
//!
//! An auto stroke paints only the colour under each dab. A dab of it has a
//! reference colour, the source under its centre, and a pixel takes the dab
//! at its strength times a [`Gate`]: 1 for a source pixel close to the
//! reference, falling to 0 by a smoothstep. Both sides are the source in
//! linear Rec.2020 before any operator, so no slider moves the mask. The
//! reference is read from a [`Proxy`] of the whole source and never from the
//! render, so the fitted view, a zoomed window that leaves the centre outside
//! and the export read the same colour.
//!
//! A stroke painted with a pen holds a pressure at each point. The pressure
//! of a dab is the blend of the two points it lies between. It scales the
//! flow, the radius or both, as the flags of the stroke say; where it scales
//! the radius the next dab falls a quarter of the radius of the last, so the
//! dabs of a path are still the first dabs of the path continued.

use gamut_core::brush::{Brush, Stroke};

use crate::mask::RADIAL_INNER_CEILING;
use crate::{acescct, hue, wheels};

/// The distance between two dabs as a share of the radius.
pub const DAB_SPACING: f32 = 0.25;

/// The most dabs one stroke stamps: a line drawn across the whole reach of
/// a mask position with the smallest brush stays a bounded draw.
pub const MAX_STROKE_DABS: usize = 1 << 16;

/// The share of the radius a pen at no pressure keeps.
pub const PRESSURE_SIZE_FLOOR: f32 = 0.2;

/// The share of the flow a pen at no pressure keeps.
pub const PRESSURE_FLOW_FLOOR: f32 = 0.05;

/// The colour distance an auto dab lets through at a sensitivity of 0.
pub const GATE_PASS_LOOSE: f32 = 0.30;

/// The same at a sensitivity of 100.
pub const GATE_PASS_STRICT: f32 = 0.03;

/// How far past the pass distance the gate falls to 0, as a share of it.
pub const GATE_FALL: f32 = 0.5;

/// What a step on the chroma plane counts against a step of tone.
pub const GATE_CHROMA_WEIGHT: f32 = 2.0;

/// The longer side of the reference proxy in pixels.
pub const PROXY_LONG_SIDE: u32 = 1024;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much of a dab is on at a distance from its centre, given as a share
/// of its radius. A feather of 0 keeps the floor of the radial gradient, so
/// nothing divides by zero.
pub fn dab_shape(distance: f32, feather: f32) -> f32 {
    let inner = (1.0 - feather / 100.0).min(RADIAL_INNER_CEILING);
    1.0 - smoothstep(inner, 1.0, distance)
}

/// The share of the radius a dab keeps at a pressure.
pub fn pressure_size(pressure: f32) -> f32 {
    PRESSURE_SIZE_FLOOR + (1.0 - PRESSURE_SIZE_FLOOR) * pressure
}

/// The share of the flow a dab keeps at a pressure.
pub fn pressure_flow(pressure: f32) -> f32 {
    pressure.max(PRESSURE_FLOW_FLOOR)
}

/// Where one dab falls, normalised to the photo, and the pressure there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub centre: [f32; 2],
    pub pressure: f32,
}

/// Where the dabs of a stroke fall: one on the first point and then one every
/// [`DAB_SPACING`] of the radius along the path, the distance measured after
/// scaling by `aspect`. The radius is the stroke's, or where the pressure
/// scales it the radius of the dab before. The dabs of a path are the first
/// dabs of any path that continues it, which is what lets a growing stroke
/// be painted by its new dabs alone.
pub fn placed_dabs(stroke: &Stroke, aspect: [f32; 2]) -> Vec<Placed> {
    let Some(first) = stroke.points.first() else {
        return Vec::new();
    };
    let sized = stroke.pressure_size && !stroke.pressure.is_empty();
    let spacing_at = |pressure: f32| {
        if sized {
            stroke.size * pressure_size(pressure) * DAB_SPACING
        } else {
            stroke.size * DAB_SPACING
        }
    };
    let mut placed = vec![Placed {
        centre: *first,
        pressure: stroke.pressure_at(0),
    }];
    let mut until = spacing_at(stroke.pressure_at(0));
    for (i, pair) in stroke.points.windows(2).enumerate() {
        let (p, q) = (pair[0], pair[1]);
        let (from, to) = (stroke.pressure_at(i), stroke.pressure_at(i + 1));
        let along = [(q[0] - p[0]) * aspect[0], (q[1] - p[1]) * aspect[1]];
        let length = (along[0] * along[0] + along[1] * along[1]).sqrt();
        let mut walked = 0.0;
        while walked + until <= length {
            if placed.len() >= MAX_STROKE_DABS {
                return placed;
            }
            walked += until;
            let t = walked / length;
            let pressure = from + (to - from) * t;
            placed.push(Placed {
                centre: [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t],
                pressure,
            });
            until = spacing_at(pressure);
        }
        until -= length - walked;
    }
    placed
}

/// The centres of [`placed_dabs`].
pub fn dab_centres(stroke: &Stroke, aspect: [f32; 2]) -> Vec<[f32; 2]> {
    placed_dabs(stroke, aspect)
        .into_iter()
        .map(|placed| placed.centre)
        .collect()
}

/// The radius and the flow (as a share, 0 to 1) of a dab of a stroke at a
/// pressure. A stroke that holds no pressure, or has both flags off, gives
/// its own size and flow untouched.
pub fn dab_size_and_flow(stroke: &Stroke, pressure: f32) -> (f32, f32) {
    let held = !stroke.pressure.is_empty();
    let radius = if held && stroke.pressure_size {
        stroke.size * pressure_size(pressure)
    } else {
        stroke.size
    };
    let flow = if held && stroke.pressure_flow {
        stroke.flow / 100.0 * pressure_flow(pressure)
    } else {
        stroke.flow / 100.0
    };
    (radius, flow)
}

/// The pass distance of the gate at a sensitivity of 0 to 100.
pub fn gate_pass(sensitivity: f32) -> f32 {
    GATE_PASS_LOOSE + (GATE_PASS_STRICT - GATE_PASS_LOOSE) * (sensitivity / 100.0)
}

/// Where a source pixel in linear Rec.2020 lies for the gate: its tone, then
/// its chroma plane weighted by [`GATE_CHROMA_WEIGHT`], both of its ACEScct
/// encoding.
pub fn gate_place(px: [f32; 3]) -> [f32; 3] {
    let v = acescct::encode_pixel(px);
    let plane = hue::chroma_plane(v);
    [
        wheels::tone(v),
        plane[0] * GATE_CHROMA_WEIGHT,
        plane[1] * GATE_CHROMA_WEIGHT,
    ]
}

/// The colour gate of one auto dab.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gate {
    /// [`gate_place`] of the reference colour.
    pub place: [f32; 3],
    /// The distance that passes whole.
    pub pass: f32,
}

impl Gate {
    pub fn new(reference: [f32; 3], sensitivity: f32) -> Self {
        Gate {
            place: gate_place(reference),
            pass: gate_pass(sensitivity),
        }
    }

    /// How much of the dab a source pixel takes: 1 within the pass distance
    /// of the reference, 0 from one and a half of it.
    pub fn open(&self, px: [f32; 3]) -> f32 {
        let place = gate_place(px);
        let d = [0, 1, 2].map(|c| place[c] - self.place[c]);
        let distance = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        1.0 - smoothstep(self.pass, self.pass * (1.0 + GATE_FALL), distance)
    }
}

/// The whole source reduced by a box filter so its longer side is
/// [`PROXY_LONG_SIDE`] pixels: what an auto dab reads its reference from. It
/// is a product of the source and of nothing else, so every render of a photo
/// reads the same references.
#[derive(Clone, Debug, PartialEq)]
pub struct Proxy {
    pub size: (u32, u32),
    pub pixels: Vec<[f32; 3]>,
}

impl Proxy {
    /// The size of the proxy of a source. A source that is smaller than the
    /// proxy keeps its size.
    pub fn size_for(source: (u32, u32)) -> (u32, u32) {
        let long = source.0.max(source.1);
        if long <= PROXY_LONG_SIDE {
            return (source.0.max(1), source.1.max(1));
        }
        let side = |s: u32| {
            let scaled =
                (u64::from(s) * u64::from(PROXY_LONG_SIDE) + u64::from(long) / 2) / u64::from(long);
            (scaled as u32).max(1)
        };
        (side(source.0), side(source.1))
    }

    /// The span of source pixels along one axis that a proxy pixel covers,
    /// as (first pixel, weight) pairs. The weights are the covered lengths.
    fn taps(index: u32, source: u32, proxy: u32) -> Vec<(u32, f32)> {
        let span = source as f32 / proxy as f32;
        let from = index as f32 * span;
        let to = from + span;
        let first = from.floor() as u32;
        let last = (to.ceil() as u32).min(source);
        (first..last)
            .map(|s| (s, to.min(s as f32 + 1.0) - from.max(s as f32)))
            .filter(|(_, weight)| *weight > 0.0)
            .collect()
    }

    /// The proxy of a source in linear Rec.2020. `store` is the rounding of
    /// the texture the GPU keeps the proxy in.
    pub fn from_source(
        pixels: &[[f32; 3]],
        source: (u32, u32),
        store: &dyn Fn(f32) -> f32,
    ) -> Self {
        let size = Proxy::size_for(source);
        let columns: Vec<Vec<(u32, f32)>> = (0..size.0)
            .map(|x| Proxy::taps(x, source.0, size.0))
            .collect();
        let mut out = Vec::with_capacity(size.0 as usize * size.1 as usize);
        for y in 0..size.1 {
            let rows = Proxy::taps(y, source.1, size.1);
            for column in &columns {
                let (mut sum, mut total) = ([0.0_f32; 3], 0.0_f32);
                for (sy, wy) in &rows {
                    for (sx, wx) in column {
                        let px = pixels[(*sy * source.0 + *sx) as usize];
                        let weight = wx * wy;
                        for c in 0..3 {
                            sum[c] += px[c] * weight;
                        }
                        total += weight;
                    }
                }
                out.push(sum.map(|s| store(s / total)));
            }
        }
        Proxy { size, pixels: out }
    }

    /// One bilinear sample at a position normalised to the photo, clamped to
    /// the photo.
    pub fn sample(&self, at: [f32; 2]) -> [f32; 3] {
        let (w, h) = self.size;
        let x = at[0].clamp(0.0, 1.0) * w as f32 - 0.5;
        let y = at[1].clamp(0.0, 1.0) * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let texel = |dx: f32, dy: f32| {
            let tx = (x0 + dx).clamp(0.0, w as f32 - 1.0) as u32;
            let ty = (y0 + dy).clamp(0.0, h as f32 - 1.0) as u32;
            self.pixels[(ty * w + tx) as usize]
        };
        let (a, b, c, d) = (
            texel(0.0, 0.0),
            texel(1.0, 0.0),
            texel(0.0, 1.0),
            texel(1.0, 1.0),
        );
        [0, 1, 2].map(|i| {
            let top = a[i] + (b[i] - a[i]) * fx;
            let bottom = c[i] + (d[i] - c[i]) * fx;
            top + (bottom - top) * fy
        })
    }
}

/// One stamped dab, its centre already scaled by the aspect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dab {
    pub centre: [f32; 2],
    pub radius: f32,
    pub feather: f32,
    /// The flow as a share, 0 to 1.
    pub flow: f32,
    pub erase: bool,
    /// The colour gate of a dab of an auto stroke.
    pub gate: Option<Gate>,
}

impl Dab {
    /// How strongly the dab acts at an aspect-scaled point, before its gate.
    pub fn strength(&self, point: [f32; 2]) -> f32 {
        let (dx, dy) = (point[0] - self.centre[0], point[1] - self.centre[1]);
        if dx.abs() >= self.radius || dy.abs() >= self.radius {
            return 0.0;
        }
        dab_shape((dx * dx + dy * dy).sqrt() / self.radius, self.feather) * self.flow
    }

    /// How strongly the dab acts at an aspect-scaled point over a source
    /// pixel: the shape times the flow times the gate.
    pub fn strength_over(&self, point: [f32; 2], px: [f32; 3]) -> f32 {
        let s = self.strength(point);
        match &self.gate {
            Some(gate) if s > 0.0 => s * gate.open(px),
            _ => s,
        }
    }

    /// The alpha after the dab acted with strength `s`.
    pub fn apply(&self, alpha: f32, s: f32) -> f32 {
        if self.erase {
            alpha * (1.0 - s)
        } else {
            alpha + s * (1.0 - alpha)
        }
    }
}

/// Every dab of a brush in the order it is stamped. An auto stroke reads the
/// reference of each dab from `proxy`.
///
/// # Panics
/// When the brush holds an auto stroke and `proxy` is `None`: a twin that
/// stamped such a stroke with no gate would be a wrong reference.
pub fn dabs(brush: &Brush, aspect: [f32; 2], proxy: Option<&Proxy>) -> Vec<Dab> {
    brush
        .strokes
        .iter()
        .flat_map(|stroke| {
            placed_dabs(stroke, aspect).into_iter().map(move |placed| {
                let (radius, flow) = dab_size_and_flow(stroke, placed.pressure);
                Dab {
                    centre: [placed.centre[0] * aspect[0], placed.centre[1] * aspect[1]],
                    radius,
                    feather: stroke.feather,
                    flow,
                    erase: stroke.erase,
                    gate: stroke.auto.then(|| {
                        let proxy = proxy.expect("an auto stroke reads the proxy of its source");
                        Gate::new(proxy.sample(placed.centre), stroke.sensitivity)
                    }),
                }
            })
        })
        .collect()
}

/// The alpha stamped dabs build at a normalised position over a source pixel.
/// `store` is applied after every dab that touches the position, where the
/// GPU writes the layer it blends into; a golden test passes the rounding of
/// that layer.
pub fn dabs_alpha(
    dabs: &[Dab],
    at: [f32; 2],
    aspect: [f32; 2],
    px: [f32; 3],
    store: &dyn Fn(f32) -> f32,
) -> f32 {
    let point = [at[0] * aspect[0], at[1] * aspect[1]];
    dabs.iter().fold(0.0, |alpha, dab| {
        let s = dab.strength(point);
        if s > 0.0 {
            let s = match &dab.gate {
                Some(gate) => s * gate.open(px),
                None => s,
            };
            store(dab.apply(alpha, s))
        } else {
            alpha
        }
    })
}

/// The alpha of a brush at a normalised position over a source pixel. It
/// stamps the strokes again on every call; a whole image prepares the
/// [`dabs`] once.
pub fn brush_alpha(
    brush: &Brush,
    at: [f32; 2],
    aspect: [f32; 2],
    px: [f32; 3],
    proxy: Option<&Proxy>,
) -> f32 {
    dabs_alpha(&dabs(brush, aspect, proxy), at, aspect, px, &|alpha| alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamut_core::brush::SharedStroke;

    const SQUARE: [f32; 2] = [1.0, 1.0];
    const GREY: [f32; 3] = [0.18; 3];

    /// The alpha of a brush with no auto stroke: it reads no pixel.
    fn plain_alpha(brush: &Brush, at: [f32; 2], aspect: [f32; 2]) -> f32 {
        brush_alpha(brush, at, aspect, GREY, None)
    }

    fn stroke(points: &[[f32; 2]], size: f32, feather: f32, flow: f32) -> Stroke {
        Stroke {
            points: points.to_vec(),
            size,
            feather,
            flow,
            ..Stroke::default()
        }
    }

    fn brush(strokes: &[Stroke]) -> Brush {
        Brush {
            strokes: strokes.iter().map(SharedStroke::new).collect(),
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn one_dab_is_1_at_its_centre_and_0_past_its_radius() {
        let dab = brush(&[stroke(&[[0.5, 0.5]], 0.1, 50.0, 100.0)]);
        assert_eq!(plain_alpha(&dab, [0.5, 0.5], SQUARE), 1.0);
        assert_eq!(
            plain_alpha(&dab, [0.54, 0.5], SQUARE),
            1.0,
            "inside the feather"
        );
        // Halfway down the feather band a smoothstep is a half.
        assert!(close(plain_alpha(&dab, [0.575, 0.5], SQUARE), 0.5));
        assert_eq!(plain_alpha(&dab, [0.6, 0.5], SQUARE), 0.0, "on the rim");
        assert_eq!(plain_alpha(&dab, [0.7, 0.7], SQUARE), 0.0);
        assert_eq!(plain_alpha(&Brush::default(), [0.5, 0.5], SQUARE), 0.0);
        assert_eq!(
            plain_alpha(&brush(&[stroke(&[], 0.1, 50.0, 100.0)]), [0.5, 0.5], SQUARE),
            0.0,
            "a stroke of no points paints nothing"
        );
    }

    #[test]
    fn a_dab_is_round_on_a_3_to_2_photo() {
        // The radius is a share of the longer side: on a photo 3 wide and 2
        // high it reaches 0.1 across and 0.15 of the height down.
        let aspect = [1.0, 2.0 / 3.0];
        let dab = brush(&[stroke(&[[0.5, 0.5]], 0.1, 100.0, 100.0)]);
        let across = plain_alpha(&dab, [0.55, 0.5], aspect);
        let down = plain_alpha(&dab, [0.5, 0.575], aspect);
        assert!(close(across, 0.5) && close(down, 0.5), "{across} {down}");
        let diagonal = 0.05 / 2f32.sqrt();
        let between = plain_alpha(&dab, [0.5 + diagonal, 0.5 + diagonal * 1.5], aspect);
        assert!(close(between, 0.5), "{between}");
        assert!(plain_alpha(&dab, [0.5, 0.64], aspect) > 0.0);
        assert_eq!(plain_alpha(&dab, [0.5, 0.66], aspect), 0.0);
    }

    #[test]
    fn a_feather_of_0_keeps_a_floor_and_divides_nothing_by_zero() {
        let dab = brush(&[stroke(&[[0.5, 0.5]], 0.1, 0.0, 100.0)]);
        for x in [0.5, 0.55, 0.5995, 0.59995, 0.6, 0.61] {
            let alpha = plain_alpha(&dab, [x, 0.5], SQUARE);
            assert!(
                alpha.is_finite() && (0.0..=1.0).contains(&alpha),
                "{x}: {alpha}"
            );
        }
        assert_eq!(plain_alpha(&dab, [0.5995, 0.5], SQUARE), 1.0);
        assert_eq!(plain_alpha(&dab, [0.6, 0.5], SQUARE), 0.0);
        let rim = plain_alpha(&dab, [0.59995, 0.5], SQUARE);
        assert!(
            rim > 0.0 && rim < 1.0,
            "the last thousandth is a ramp: {rim}"
        );
    }

    #[test]
    fn a_flow_of_50_twice_reaches_three_quarters() {
        let once = stroke(&[[0.5, 0.5]], 0.1, 0.0, 50.0);
        assert!(close(
            plain_alpha(&brush(std::slice::from_ref(&once)), [0.5, 0.5], SQUARE),
            0.5
        ));
        let twice = brush(&[once.clone(), once]);
        assert!(close(plain_alpha(&twice, [0.5, 0.5], SQUARE), 0.75));
    }

    #[test]
    fn an_erase_removes_what_was_painted_before_it_and_nothing_after() {
        let paint = stroke(&[[0.5, 0.5]], 0.1, 0.0, 100.0);
        let erase = Stroke {
            erase: true,
            ..stroke(&[[0.5, 0.5]], 0.05, 0.0, 100.0)
        };
        let painted_then_erased = brush(&[paint.clone(), erase.clone()]);
        assert_eq!(plain_alpha(&painted_then_erased, [0.5, 0.5], SQUARE), 0.0);
        assert_eq!(
            plain_alpha(&painted_then_erased, [0.57, 0.5], SQUARE),
            1.0,
            "outside the eraser"
        );
        let erased_then_painted = brush(&[erase.clone(), paint]);
        assert_eq!(plain_alpha(&erased_then_painted, [0.5, 0.5], SQUARE), 1.0);
        assert_eq!(
            plain_alpha(&brush(std::slice::from_ref(&erase)), [0.5, 0.5], SQUARE),
            0.0
        );

        let half = Stroke {
            flow: 50.0,
            ..erase
        };
        let softened = brush(&[stroke(&[[0.5, 0.5]], 0.1, 0.0, 100.0), half]);
        assert!(close(plain_alpha(&softened, [0.5, 0.5], SQUARE), 0.5));
    }

    #[test]
    fn two_paint_strokes_agree_in_either_order_and_paint_and_erase_do_not() {
        let a = stroke(&[[0.4, 0.5], [0.6, 0.5]], 0.08, 60.0, 40.0);
        let b = stroke(&[[0.5, 0.4], [0.5, 0.6]], 0.05, 20.0, 70.0);
        for at in [[0.5, 0.5], [0.47, 0.52], [0.55, 0.46]] {
            let one_way = plain_alpha(&brush(&[a.clone(), b.clone()]), at, SQUARE);
            let other_way = plain_alpha(&brush(&[b.clone(), a.clone()]), at, SQUARE);
            assert!(
                one_way > 0.0 && close(one_way, other_way),
                "{one_way} {other_way}"
            );
        }
        let eraser = Stroke {
            erase: true,
            ..b.clone()
        };
        let paint_then_erase =
            plain_alpha(&brush(&[a.clone(), eraser.clone()]), [0.5, 0.5], SQUARE);
        let erase_then_paint = plain_alpha(&brush(&[eraser, a]), [0.5, 0.5], SQUARE);
        assert!(
            erase_then_paint - paint_then_erase > 0.2,
            "{paint_then_erase} {erase_then_paint}"
        );
    }

    #[test]
    fn dabs_fall_every_quarter_radius_from_the_first_point_and_carry_over_corners() {
        let line = stroke(&[[0.2, 0.5], [0.405, 0.5]], 0.04, 50.0, 100.0);
        let centres = dab_centres(&line, SQUARE);
        assert_eq!(
            centres.len(),
            21,
            "0.205 long at 0.01 apart, the first on the start"
        );
        assert_eq!(centres[0], [0.2, 0.5]);
        assert!(close(centres[7][0], 0.27) && centres[7][1] == 0.5);

        // Around a corner the spacing is kept along the path.
        let bent = stroke(&[[0.2, 0.5], [0.215, 0.5], [0.215, 0.6]], 0.04, 50.0, 100.0);
        let centres = dab_centres(&bent, SQUARE);
        assert!(close(centres[1][0], 0.21) && centres[1][1] == 0.5);
        assert!(close(centres[2][0], 0.215) && close(centres[2][1], 0.505));

        // Points closer than the spacing still stamp at the spacing.
        let dense: Vec<[f32; 2]> = (0..=205).map(|i| [0.2 + i as f32 * 0.001, 0.5]).collect();
        assert_eq!(
            dab_centres(&stroke(&dense, 0.04, 50.0, 100.0), SQUARE).len(),
            21
        );

        // On a 3:2 photo a path down the photo is shorter than it looks.
        let down = stroke(&[[0.5, 0.2], [0.5, 0.5075]], 0.04, 50.0, 100.0);
        assert_eq!(dab_centres(&down, [1.0, 2.0 / 3.0]).len(), 21);

        let repeated = stroke(&[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], 0.04, 50.0, 100.0);
        assert_eq!(dab_centres(&repeated, SQUARE), [[0.5, 0.5]]);
    }

    #[test]
    fn the_dabs_of_a_path_begin_the_dabs_of_the_path_continued() {
        let points: Vec<[f32; 2]> = (0..40)
            .map(|i| {
                let t = i as f32 / 39.0;
                [0.2 + t * 0.6, 0.5 + (t * 9.0).sin() * 0.1]
            })
            .collect();
        let whole = dab_centres(&stroke(&points, 0.03, 50.0, 100.0), [1.0, 0.75]);
        for cut in [1, 2, 7, 23, 39] {
            let part = dab_centres(&stroke(&points[..cut], 0.03, 50.0, 100.0), [1.0, 0.75]);
            assert_eq!(part[..], whole[..part.len()], "cut at {cut}");
        }
    }

    #[test]
    fn a_long_line_with_the_smallest_brush_stops_at_the_dab_limit() {
        let line = stroke(&[[-4.0, -4.0], [5.0, 5.0]], 0.0002, 0.0, 100.0);
        assert_eq!(dab_centres(&line, SQUARE).len(), MAX_STROKE_DABS);
    }

    #[test]
    fn a_straight_stroke_is_even_along_its_length() {
        for (feather, flow) in [(100.0, 20.0), (50.0, 10.0), (0.0, 35.0), (100.0, 100.0)] {
            let line = brush(&[stroke(&[[0.2, 0.5], [0.8, 0.5]], 0.05, feather, flow)]);
            let prepared = dabs(&line, SQUARE, None);
            let along: Vec<f32> = (0..=400)
                .map(|i| 0.3 + i as f32 * 0.001)
                .map(|x| dabs_alpha(&prepared, [x, 0.5], SQUARE, GREY, &|a| a))
                .collect();
            let (least, most) = along
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), a| (lo.min(*a), hi.max(*a)));
            assert!(
                least > 0.0 && (most - least) / most < 0.02,
                "{feather} {flow}: {least} {most}"
            );
        }
    }

    #[test]
    fn a_store_is_applied_after_every_dab_that_touches() {
        let twice = brush(&[
            stroke(&[[0.5, 0.5]], 0.1, 0.0, 50.0),
            stroke(&[[0.9, 0.9]], 0.01, 0.0, 50.0),
            stroke(&[[0.5, 0.5]], 0.1, 0.0, 50.0),
        ]);
        let stored = std::cell::Cell::new(0);
        let alpha = dabs_alpha(
            &dabs(&twice, SQUARE, None),
            [0.5, 0.5],
            SQUARE,
            GREY,
            &|a| {
                stored.set(stored.get() + 1);
                (a * 2.0).round() / 2.0
            },
        );
        assert_eq!(stored.get(), 2, "the far dab stores nothing here");
        assert_eq!(alpha, 1.0, "0.5 stored, then 0.75 stored as 1");
    }

    /// `dab_centres` as it stood at d2ab79d, before pressure, kept word for
    /// word: a stroke with no pressure in play must still fall on it.
    fn dab_centres_before_pressure(stroke: &Stroke, aspect: [f32; 2]) -> Vec<[f32; 2]> {
        let Some(first) = stroke.points.first() else {
            return Vec::new();
        };
        let spacing = stroke.size * DAB_SPACING;
        let mut centres = vec![*first];
        let mut until = spacing;
        for pair in stroke.points.windows(2) {
            let (p, q) = (pair[0], pair[1]);
            let along = [(q[0] - p[0]) * aspect[0], (q[1] - p[1]) * aspect[1]];
            let length = (along[0] * along[0] + along[1] * along[1]).sqrt();
            let mut walked = 0.0;
            while walked + until <= length {
                walked += until;
                let t = walked / length;
                centres.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
                until = spacing;
            }
            until -= length - walked;
        }
        centres
    }

    fn wandering(points: usize) -> Vec<[f32; 2]> {
        (0..points)
            .map(|i| {
                let t = i as f32 / (points - 1) as f32;
                [0.1 + 0.8 * t, 0.5 + 0.3 * (t * 9.0).sin()]
            })
            .collect()
    }

    fn rising(points: usize) -> Vec<f32> {
        (0..points)
            .map(|i| i as f32 / (points - 1) as f32)
            .collect()
    }

    #[test]
    fn with_no_pressure_in_play_the_dabs_are_those_of_before() {
        let aspect = [1.0, 2.0 / 3.0];
        let plain = [
            stroke(&wandering(40), 0.031, 35.0, 60.0),
            stroke(&[[0.2, 0.2], [0.9, 0.85], [0.3, 0.7]], 0.0123, 0.0, 7.0),
            stroke(&[[0.5, 0.5]], 0.2, 100.0, 100.0),
        ];
        // The flags with no pressure, and a pressure with no flag, are out
        // of play too.
        let flagged = Stroke {
            pressure_size: true,
            pressure_flow: true,
            ..plain[0].clone()
        };
        let unflagged = Stroke {
            pressure: rising(40),
            ..plain[0].clone()
        };
        for s in plain.iter().chain([&flagged, &unflagged]) {
            // As a brush holds it: the points on their stored steps.
            let s = &s.sanitised();
            let before = dab_centres_before_pressure(s, aspect);
            assert!(!before.is_empty());
            assert!(dab_centres(s, aspect) == before, "the centres moved");
            let now = dabs(&brush(std::slice::from_ref(s)), aspect, None);
            let then: Vec<Dab> = before
                .iter()
                .map(|centre| Dab {
                    centre: [centre[0] * aspect[0], centre[1] * aspect[1]],
                    radius: s.size,
                    feather: s.feather,
                    flow: s.flow / 100.0,
                    erase: s.erase,
                    gate: None,
                })
                .collect();
            assert_eq!(now.len(), then.len());
            let moved = now.iter().zip(&then).position(|(a, b)| a != b);
            assert_eq!(moved, None, "the first dab that differs");
        }
    }

    const SKY: [f32; 3] = [0.2, 0.4, 0.8];
    const ROOF: [f32; 3] = [0.4, 0.2, 0.1];

    /// A photo of `side` pixels square, sky on the left half and roof on the
    /// right.
    fn two_colours(side: u32) -> Vec<[f32; 3]> {
        (0..side * side)
            .map(|i| if i % side < side / 2 { SKY } else { ROOF })
            .collect()
    }

    fn proxy_of_two_colours() -> Proxy {
        Proxy::from_source(&two_colours(64), (64, 64), &|v| v)
    }

    fn auto(stroke: Stroke, sensitivity: f32) -> Stroke {
        Stroke {
            auto: true,
            sensitivity,
            ..stroke
        }
    }

    #[test]
    fn an_auto_dab_paints_the_colour_under_it_and_not_the_other() {
        let proxy = proxy_of_two_colours();
        let on_sky = brush(&[auto(stroke(&[[0.4, 0.5]], 0.3, 0.0, 100.0), 50.0)]);
        let alpha = |at: [f32; 2], px: [f32; 3]| brush_alpha(&on_sky, at, SQUARE, px, Some(&proxy));
        assert_eq!(alpha([0.45, 0.5], SKY), 1.0);
        assert_eq!(alpha([0.55, 0.5], ROOF), 0.0);
        assert_eq!(
            alpha([0.9, 0.5], SKY),
            0.0,
            "outside the dab the colour counts for nothing"
        );
        let plain = brush(&[stroke(&[[0.4, 0.5]], 0.3, 0.0, 100.0)]);
        assert_eq!(brush_alpha(&plain, [0.55, 0.5], SQUARE, ROOF, None), 1.0);
    }

    #[test]
    fn a_sensitivity_of_0_lets_a_near_colour_through_and_100_does_not() {
        let proxy = proxy_of_two_colours();
        let near = SKY.map(|c| c * 1.5);
        let at = |sensitivity: f32, px: [f32; 3]| {
            let b = brush(&[auto(stroke(&[[0.25, 0.5]], 0.2, 0.0, 100.0), sensitivity)]);
            brush_alpha(&b, [0.25, 0.5], SQUARE, px, Some(&proxy))
        };
        assert_eq!(at(0.0, near), 1.0);
        assert_eq!(at(100.0, near), 0.0);
        assert_eq!(at(100.0, SKY), 1.0, "the colour itself always passes");
        assert_eq!(at(50.0, ROOF), 0.0);
        assert_eq!(
            at(0.0, [0.01, 0.005, 0.002]),
            0.0,
            "and a far colour never does"
        );
    }

    #[test]
    fn the_gate_is_a_smooth_ramp_and_divides_nothing_by_zero() {
        for sensitivity in [0.0, 50.0, 100.0] {
            let gate = Gate::new(SKY, sensitivity);
            assert!(close(gate.pass, gate_pass(sensitivity)));
            let mut last = 1.0;
            let mut between = 0;
            // Darker by a two hundredth of a stop a step, 16 stops down. The
            // tone stops at diffuse white, so brighter would not go far.
            for i in 0..=3200 {
                let px = SKY.map(|c| c * (-(i as f32) / 200.0).exp2());
                let open = gate.open(px);
                assert!(open.is_finite() && (0.0..=1.0).contains(&open));
                assert!(open <= last, "never opens again further away");
                assert!(
                    last - open < 0.12,
                    "no step at {sensitivity}: {last} to {open}"
                );
                between += usize::from(open > 0.0 && open < 1.0);
                last = open;
            }
            assert_eq!(last, 0.0);
            assert!(between >= 2, "a ramp, not a cut, at {sensitivity}");
        }
        assert!(
            close(gate_pass(0.0), GATE_PASS_LOOSE) && close(gate_pass(100.0), GATE_PASS_STRICT)
        );
        let black = Gate::new([0.0; 3], 100.0);
        assert_eq!(black.open([0.0; 3]), 1.0);
        assert!(black.open([-1.0, 0.0, f32::MIN_POSITIVE]).is_finite());
    }

    #[test]
    fn the_gate_weighs_the_chroma_plane_twice_the_tone() {
        let v = acescct::encode_pixel(SKY);
        let plane = hue::chroma_plane(v);
        let place = gate_place(SKY);
        assert!(close(place[0], wheels::tone(v)));
        assert!(close(place[1], plane[0] * 2.0) && close(place[2], plane[1] * 2.0));
    }

    #[test]
    fn an_auto_erase_erases_only_the_colour_under_it() {
        let proxy = proxy_of_two_colours();
        let paint = stroke(&[[0.5, 0.5]], 0.4, 0.0, 100.0);
        let erase = Stroke {
            erase: true,
            ..auto(stroke(&[[0.4, 0.5]], 0.4, 0.0, 100.0), 50.0)
        };
        let b = brush(&[paint, erase]);
        assert_eq!(brush_alpha(&b, [0.45, 0.5], SQUARE, SKY, Some(&proxy)), 0.0);
        assert_eq!(
            brush_alpha(&b, [0.55, 0.5], SQUARE, ROOF, Some(&proxy)),
            1.0
        );
    }

    #[test]
    fn the_proxy_is_a_box_filter_with_a_longer_side_of_1024() {
        assert_eq!(Proxy::size_for((6000, 4000)), (1024, 683));
        assert_eq!(Proxy::size_for((4000, 6000)), (683, 1024));
        assert_eq!(Proxy::size_for((3840, 2160)), (1024, 576));
        assert_eq!(
            Proxy::size_for((800, 600)),
            (800, 600),
            "a small source keeps its size"
        );
        assert_eq!(Proxy::size_for((5000, 1)), (1024, 1));

        // 2048 by 3 to 1024 by 3: every proxy pixel is the mean of a pair.
        let source: Vec<[f32; 3]> = (0..2048 * 3).map(|i| [(i % 2048) as f32; 3]).collect();
        let proxy = Proxy::from_source(&source, (2048, 3), &|v| v);
        assert_eq!(proxy.size, (1024, 2));
        assert_eq!(proxy.pixels[0][0], 0.5);
        assert_eq!(proxy.pixels[1023][0], 2046.5);

        // 1536 to 1024 is a span of one and a half pixels: the fractions
        // count.
        let source: Vec<[f32; 3]> = (0..1536).map(|i| [i as f32; 3]).collect();
        let proxy = Proxy::from_source(&source, (1536, 1), &|v| v);
        assert!(close(proxy.pixels[0][0], (0.0 + 0.5 * 1.0) / 1.5));
        assert!(close(proxy.pixels[1][0], (0.5 * 1.0 + 2.0) / 1.5));

        let flat = Proxy::from_source(&vec![[0.25; 3]; 3000 * 7], (3000, 7), &|v| v);
        assert!(flat.pixels.iter().all(|px| close(px[0], 0.25)));
        let stored = Proxy::from_source(&[[0.3; 3]], (1, 1), &|v| v * 2.0);
        assert_eq!(stored.pixels, [[0.6; 3]], "the store is applied");
    }

    #[test]
    fn a_proxy_sample_is_bilinear_and_clamped_to_the_photo() {
        let proxy = Proxy {
            size: (2, 2),
            pixels: vec![[0.0; 3], [1.0; 3], [2.0; 3], [3.0; 3]],
        };
        assert_eq!(proxy.sample([0.25, 0.25]), [0.0; 3], "a pixel centre");
        assert_eq!(proxy.sample([0.75, 0.75]), [3.0; 3]);
        assert_eq!(proxy.sample([0.5, 0.5]), [1.5; 3]);
        assert_eq!(proxy.sample([0.5, 0.25]), [0.5; 3]);
        assert_eq!(proxy.sample([-3.0, -3.0]), [0.0; 3], "clamped");
        assert_eq!(proxy.sample([4.0, 0.0]), [1.0; 3]);
        assert_eq!(proxy.sample([1.0, 1.0]), [3.0; 3]);
    }

    fn pen(points: Vec<[f32; 2]>, pressure: Vec<f32>, size: bool, flow: bool) -> Stroke {
        Stroke {
            pressure,
            pressure_size: size,
            pressure_flow: flow,
            ..stroke(&points, 0.05, 0.0, 80.0)
        }
    }

    #[test]
    fn pressure_on_the_flow_scales_it_and_keeps_a_floor() {
        let line = pen(vec![[0.1, 0.5], [0.9, 0.5]], vec![0.0, 1.0], false, true);
        let stamped = dabs(&brush(std::slice::from_ref(&line)), SQUARE, None);
        assert_eq!(
            stamped.len(),
            dab_centres_before_pressure(&line, SQUARE).len(),
            "the flow moves no dab"
        );
        assert!(
            close(stamped[0].flow, 0.8 * PRESSURE_FLOW_FLOOR),
            "{}",
            stamped[0].flow
        );
        assert!(close(stamped.last().expect("dabs").flow, 0.8));
        let middle = stamped[stamped.len() / 2];
        assert!((middle.flow - 0.8 * 0.5).abs() < 1e-3, "{}", middle.flow);
        assert!(stamped.iter().all(|dab| dab.radius == 0.05));
        assert!(stamped.windows(2).all(|pair| pair[1].flow >= pair[0].flow));
    }

    #[test]
    fn pressure_on_the_size_scales_the_radius_and_the_spacing_with_it() {
        let line = pen(vec![[0.1, 0.5], [0.9, 0.5]], vec![0.0, 1.0], true, false);
        let stamped = dabs(&brush(std::slice::from_ref(&line)), SQUARE, None);
        assert!(close(stamped[0].radius, 0.05 * PRESSURE_SIZE_FLOOR));
        assert!(stamped.iter().all(|dab| dab.flow == 0.8));
        for pair in stamped.windows(2) {
            let step = pair[1].centre[0] - pair[0].centre[0];
            assert!(
                (step - pair[0].radius * DAB_SPACING).abs() < 1e-6,
                "a quarter of the radius where the dab before fell"
            );
            assert!(pair[1].radius > pair[0].radius);
        }
        let last = stamped.last().expect("dabs");
        assert!(last.radius <= 0.05 && last.radius > 0.045);
        assert!(close(pressure_size(0.0), 0.2) && close(pressure_size(1.0), 1.0));
    }

    #[test]
    fn the_dabs_of_a_pen_path_begin_the_dabs_of_the_path_continued() {
        let points = wandering(30);
        let pressure: Vec<f32> = (0..30)
            .map(|i| 0.5 + 0.5 * (i as f32 * 0.7).sin())
            .collect();
        for (size, flow) in [(true, false), (false, true), (true, true)] {
            let whole = placed_dabs(&pen(points.clone(), pressure.clone(), size, flow), SQUARE);
            for cut in [1, 2, 9, 17, 29] {
                let part = pen(points[..cut].to_vec(), pressure[..cut].to_vec(), size, flow);
                let placed = placed_dabs(&part, SQUARE);
                assert!(!placed.is_empty());
                assert_eq!(placed[..], whole[..placed.len()], "{cut} points");
            }
        }
    }

    #[test]
    fn a_pressure_that_rises_paints_a_line_that_widens_or_darkens_evenly() {
        let (points, pressure) = (vec![[0.1, 0.5], [0.9, 0.5]], vec![0.1, 1.0]);
        // Darkens: a low flow so the build-up does not reach 1.
        let darker = Stroke {
            flow: 5.0,
            feather: 100.0,
            ..pen(points.clone(), pressure.clone(), false, true)
        };
        let prepared = dabs(&brush(&[darker]), SQUARE, None);
        let along: Vec<f32> = (0..=60)
            .map(|i| {
                dabs_alpha(
                    &prepared,
                    [0.2 + 0.01 * i as f32, 0.5],
                    SQUARE,
                    GREY,
                    &|a| a,
                )
            })
            .collect();
        assert!(along[0] > 0.0 && along[60] < 1.0);
        for pair in along.windows(2) {
            assert!(pair[1] > pair[0], "darker all the way");
            assert!(pair[1] - pair[0] < 0.03, "and by small steps");
        }
        // Widens: the half width, read off the line, grows with x.
        let wider = pen(points, pressure, true, false);
        let prepared = dabs(&brush(&[wider]), SQUARE, None);
        let half_width = |x: f32| {
            (0..200)
                .map(|i| i as f32 * 0.0005)
                .take_while(|dy| dabs_alpha(&prepared, [x, 0.5 + dy], SQUARE, GREY, &|a| a) > 0.0)
                .count()
        };
        let widths: Vec<usize> = (0..=6).map(|i| half_width(0.2 + 0.1 * i as f32)).collect();
        assert!(
            widths.windows(2).all(|pair| pair[1] > pair[0]),
            "{widths:?}"
        );
        let steps: Vec<usize> = widths.windows(2).map(|pair| pair[1] - pair[0]).collect();
        let (least, most) = (
            steps.iter().min().expect("steps"),
            steps.iter().max().expect("steps"),
        );
        assert!(most - least <= 2, "evenly: {widths:?}");
    }
}
