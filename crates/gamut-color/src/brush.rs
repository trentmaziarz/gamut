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

use gamut_core::brush::{Brush, Stroke};

use crate::mask::RADIAL_INNER_CEILING;

/// The distance between two dabs as a share of the radius.
pub const DAB_SPACING: f32 = 0.25;

/// The most dabs one stroke stamps: a line drawn across the whole reach of
/// a mask position with the smallest brush stays a bounded draw.
pub const MAX_STROKE_DABS: usize = 1 << 16;

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

/// Where the dabs of a stroke fall, normalised to the photo: one on the
/// first point and then one every [`DAB_SPACING`] of the radius along the
/// path, the distance measured after scaling by `aspect`. The dabs of a path
/// are the first dabs of any path that continues it, which is what lets a
/// growing stroke be painted by its new dabs alone.
pub fn dab_centres(stroke: &Stroke, aspect: [f32; 2]) -> Vec<[f32; 2]> {
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
            if centres.len() >= MAX_STROKE_DABS {
                return centres;
            }
            walked += until;
            let t = walked / length;
            centres.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
            until = spacing;
        }
        until -= length - walked;
    }
    centres
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
}

impl Dab {
    /// How strongly the dab acts at an aspect-scaled point.
    pub fn strength(&self, point: [f32; 2]) -> f32 {
        let (dx, dy) = (point[0] - self.centre[0], point[1] - self.centre[1]);
        if dx.abs() >= self.radius || dy.abs() >= self.radius {
            return 0.0;
        }
        dab_shape((dx * dx + dy * dy).sqrt() / self.radius, self.feather) * self.flow
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

/// Every dab of a brush in the order it is stamped.
pub fn dabs(brush: &Brush, aspect: [f32; 2]) -> Vec<Dab> {
    brush
        .strokes
        .iter()
        .flat_map(|stroke| {
            dab_centres(stroke, aspect).into_iter().map(|centre| Dab {
                centre: [centre[0] * aspect[0], centre[1] * aspect[1]],
                radius: stroke.size,
                feather: stroke.feather,
                flow: stroke.flow / 100.0,
                erase: stroke.erase,
            })
        })
        .collect()
}

/// The alpha stamped dabs build at a normalised position. `store` is applied
/// after every dab that touches the position, where the GPU writes the
/// layer it blends into; a golden test passes the rounding of that layer.
pub fn dabs_alpha(dabs: &[Dab], at: [f32; 2], aspect: [f32; 2], store: &dyn Fn(f32) -> f32) -> f32 {
    let point = [at[0] * aspect[0], at[1] * aspect[1]];
    dabs.iter().fold(0.0, |alpha, dab| {
        let s = dab.strength(point);
        if s > 0.0 {
            store(dab.apply(alpha, s))
        } else {
            alpha
        }
    })
}

/// The alpha of a brush at a normalised position. It stamps the strokes
/// again on every call; a whole image prepares the [`dabs`] once.
pub fn brush_alpha(brush: &Brush, at: [f32; 2], aspect: [f32; 2]) -> f32 {
    dabs_alpha(&dabs(brush, aspect), at, aspect, &|alpha| alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamut_core::brush::SharedStroke;

    const SQUARE: [f32; 2] = [1.0, 1.0];

    fn stroke(points: &[[f32; 2]], size: f32, feather: f32, flow: f32) -> Stroke {
        Stroke {
            points: points.to_vec(),
            size,
            feather,
            flow,
            erase: false,
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
        assert_eq!(brush_alpha(&dab, [0.5, 0.5], SQUARE), 1.0);
        assert_eq!(
            brush_alpha(&dab, [0.54, 0.5], SQUARE),
            1.0,
            "inside the feather"
        );
        // Halfway down the feather band a smoothstep is a half.
        assert!(close(brush_alpha(&dab, [0.575, 0.5], SQUARE), 0.5));
        assert_eq!(brush_alpha(&dab, [0.6, 0.5], SQUARE), 0.0, "on the rim");
        assert_eq!(brush_alpha(&dab, [0.7, 0.7], SQUARE), 0.0);
        assert_eq!(brush_alpha(&Brush::default(), [0.5, 0.5], SQUARE), 0.0);
        assert_eq!(
            brush_alpha(&brush(&[stroke(&[], 0.1, 50.0, 100.0)]), [0.5, 0.5], SQUARE),
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
        let across = brush_alpha(&dab, [0.55, 0.5], aspect);
        let down = brush_alpha(&dab, [0.5, 0.575], aspect);
        assert!(close(across, 0.5) && close(down, 0.5), "{across} {down}");
        let diagonal = 0.05 / 2f32.sqrt();
        let between = brush_alpha(&dab, [0.5 + diagonal, 0.5 + diagonal * 1.5], aspect);
        assert!(close(between, 0.5), "{between}");
        assert!(brush_alpha(&dab, [0.5, 0.64], aspect) > 0.0);
        assert_eq!(brush_alpha(&dab, [0.5, 0.66], aspect), 0.0);
    }

    #[test]
    fn a_feather_of_0_keeps_a_floor_and_divides_nothing_by_zero() {
        let dab = brush(&[stroke(&[[0.5, 0.5]], 0.1, 0.0, 100.0)]);
        for x in [0.5, 0.55, 0.5995, 0.59995, 0.6, 0.61] {
            let alpha = brush_alpha(&dab, [x, 0.5], SQUARE);
            assert!(
                alpha.is_finite() && (0.0..=1.0).contains(&alpha),
                "{x}: {alpha}"
            );
        }
        assert_eq!(brush_alpha(&dab, [0.5995, 0.5], SQUARE), 1.0);
        assert_eq!(brush_alpha(&dab, [0.6, 0.5], SQUARE), 0.0);
        let rim = brush_alpha(&dab, [0.59995, 0.5], SQUARE);
        assert!(
            rim > 0.0 && rim < 1.0,
            "the last thousandth is a ramp: {rim}"
        );
    }

    #[test]
    fn a_flow_of_50_twice_reaches_three_quarters() {
        let once = stroke(&[[0.5, 0.5]], 0.1, 0.0, 50.0);
        assert!(close(
            brush_alpha(&brush(std::slice::from_ref(&once)), [0.5, 0.5], SQUARE),
            0.5
        ));
        let twice = brush(&[once.clone(), once]);
        assert!(close(brush_alpha(&twice, [0.5, 0.5], SQUARE), 0.75));
    }

    #[test]
    fn an_erase_removes_what_was_painted_before_it_and_nothing_after() {
        let paint = stroke(&[[0.5, 0.5]], 0.1, 0.0, 100.0);
        let erase = Stroke {
            erase: true,
            ..stroke(&[[0.5, 0.5]], 0.05, 0.0, 100.0)
        };
        let painted_then_erased = brush(&[paint.clone(), erase.clone()]);
        assert_eq!(brush_alpha(&painted_then_erased, [0.5, 0.5], SQUARE), 0.0);
        assert_eq!(
            brush_alpha(&painted_then_erased, [0.57, 0.5], SQUARE),
            1.0,
            "outside the eraser"
        );
        let erased_then_painted = brush(&[erase.clone(), paint]);
        assert_eq!(brush_alpha(&erased_then_painted, [0.5, 0.5], SQUARE), 1.0);
        assert_eq!(
            brush_alpha(&brush(std::slice::from_ref(&erase)), [0.5, 0.5], SQUARE),
            0.0
        );

        let half = Stroke {
            flow: 50.0,
            ..erase
        };
        let softened = brush(&[stroke(&[[0.5, 0.5]], 0.1, 0.0, 100.0), half]);
        assert!(close(brush_alpha(&softened, [0.5, 0.5], SQUARE), 0.5));
    }

    #[test]
    fn two_paint_strokes_agree_in_either_order_and_paint_and_erase_do_not() {
        let a = stroke(&[[0.4, 0.5], [0.6, 0.5]], 0.08, 60.0, 40.0);
        let b = stroke(&[[0.5, 0.4], [0.5, 0.6]], 0.05, 20.0, 70.0);
        for at in [[0.5, 0.5], [0.47, 0.52], [0.55, 0.46]] {
            let one_way = brush_alpha(&brush(&[a.clone(), b.clone()]), at, SQUARE);
            let other_way = brush_alpha(&brush(&[b.clone(), a.clone()]), at, SQUARE);
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
            brush_alpha(&brush(&[a.clone(), eraser.clone()]), [0.5, 0.5], SQUARE);
        let erase_then_paint = brush_alpha(&brush(&[eraser, a]), [0.5, 0.5], SQUARE);
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
            let prepared = dabs(&line, SQUARE);
            let along: Vec<f32> = (0..=400)
                .map(|i| 0.3 + i as f32 * 0.001)
                .map(|x| dabs_alpha(&prepared, [x, 0.5], SQUARE, &|a| a))
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
        let alpha = dabs_alpha(&dabs(&twice, SQUARE), [0.5, 0.5], SQUARE, &|a| {
            stored.set(stored.get() + 1);
            (a * 2.0).round() / 2.0
        });
        assert_eq!(stored.get(), 2, "the far dab stores nothing here");
        assert_eq!(alpha, 1.0, "0.5 stored, then 0.75 stored as 1");
    }
}
