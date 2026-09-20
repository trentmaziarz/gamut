//! The look parameters of M3: tone curves, the HSL mixer and the colour
//! wheels. Plain data; gamut-color holds the arithmetic. Every type takes its
//! neutral value by default, so a sidecar written before M3 still parses.

use serde::{Deserialize, Serialize};

/// The fewest control points a curve holds: the two end points.
pub const MIN_POINTS: usize = 2;

/// The most control points a curve holds.
pub const MAX_POINTS: usize = 16;

/// The number of hue ranges in the HSL mixer.
pub const HSL_RANGES: usize = 8;

/// The names of the hue ranges, in the order of `Look::hsl`.
pub const HSL_NAMES: [&str; HSL_RANGES] = [
    "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta",
];

/// One tone curve: control points `[x, y]` on the normalised ACEScct axis,
/// where 0 is black and 1 is diffuse white, sorted by x.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Curve {
    pub points: Vec<[f32; 2]>,
}

impl Default for Curve {
    fn default() -> Self {
        Curve {
            points: vec![[0.0, 0.0], [1.0, 1.0]],
        }
    }
}

impl Curve {
    /// True for the default curve, the two end points and nothing else, so
    /// the lookup can be skipped. A curve with more points runs the lookup
    /// even when every point sits on the diagonal.
    pub fn is_identity(&self) -> bool {
        self.points == [[0.0, 0.0], [1.0, 1.0]]
    }

    /// The curve made safe to interpolate: points that are not finite are
    /// dropped, the rest are clamped to 0 to 1 and sorted by x, a repeated x
    /// keeps its first point, and the count is held between [`MIN_POINTS`]
    /// and [`MAX_POINTS`]. A list left with fewer than two points becomes the
    /// identity; a list with too many keeps its first fifteen and its last.
    pub fn sanitised(&self) -> Curve {
        let mut points: Vec<[f32; 2]> = self
            .points
            .iter()
            .filter(|p| p[0].is_finite() && p[1].is_finite())
            .map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)])
            .collect();
        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        points.dedup_by(|b, a| a[0] == b[0]);
        if points.len() < MIN_POINTS {
            return Curve::default();
        }
        if points.len() > MAX_POINTS {
            let last = points[points.len() - 1];
            points.truncate(MAX_POINTS - 1);
            points.push(last);
        }
        Curve { points }
    }
}

/// The four tone curves. A channel curve runs first, then the master.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToneCurves {
    pub master: Curve,
    pub red: Curve,
    pub green: Curve,
    pub blue: Curve,
}

impl ToneCurves {
    pub fn is_identity(&self) -> bool {
        self.master.is_identity()
            && self.red.is_identity()
            && self.green.is_identity()
            && self.blue.is_identity()
    }
}

/// One hue range of the mixer. Each field runs from -100 to 100.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HslRange {
    pub hue: f32,
    pub saturation: f32,
    pub luminance: f32,
}

impl HslRange {
    pub fn is_identity(&self) -> bool {
        *self == HslRange::default()
    }
}

/// One colour wheel: a point on the unit disc and a luminance slider from
/// -100 to 100.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Wheel {
    pub x: f32,
    pub y: f32,
    pub luminance: f32,
}

impl Wheel {
    pub fn is_identity(&self) -> bool {
        *self == Wheel::default()
    }
}

/// The three wheels: shadows as lift, midtones as gamma, highlights as gain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Wheels {
    pub shadows: Wheel,
    pub midtones: Wheel,
    pub highlights: Wheel,
}

impl Wheels {
    pub fn is_identity(&self) -> bool {
        *self == Wheels::default()
    }
}

/// Everything that runs in the ACEScct domain.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Look {
    pub curves: ToneCurves,
    pub hsl: [HslRange; HSL_RANGES],
    pub wheels: Wheels,
}

impl Look {
    pub fn hsl_is_identity(&self) -> bool {
        self.hsl.iter().all(HslRange::is_identity)
    }

    pub fn is_identity(&self) -> bool {
        self.curves.is_identity() && self.hsl_is_identity() && self.wheels.is_identity()
    }
}

#[cfg(test)]
mod tests {
    use super::{Curve, Look, MAX_POINTS};

    #[test]
    fn the_default_look_is_the_identity() {
        assert!(Look::default().is_identity());
        assert_eq!(Curve::default().points, vec![[0.0, 0.0], [1.0, 1.0]]);
    }

    #[test]
    fn sanitising_sorts_clamps_and_drops_repeats() {
        let curve = Curve {
            points: vec![
                [1.0, 1.2],
                [0.5, 0.6],
                [f32::NAN, 0.3],
                [0.5, 0.1],
                [-0.2, 0.0],
            ],
        };
        assert_eq!(
            curve.sanitised().points,
            vec![[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]]
        );
    }

    #[test]
    fn sanitising_keeps_between_two_and_sixteen_points() {
        let short = Curve {
            points: vec![[0.3, 0.9]],
        };
        assert_eq!(short.sanitised(), Curve::default());

        let long = Curve {
            points: (0..40)
                .map(|i| [i as f32 / 39.0, i as f32 / 39.0])
                .collect(),
        };
        let kept = long.sanitised().points;
        assert_eq!(kept.len(), MAX_POINTS);
        assert_eq!(kept[0], [0.0, 0.0]);
        assert_eq!(kept[MAX_POINTS - 1], [1.0, 1.0]);
    }

    #[test]
    fn a_look_round_trips_through_json_and_fills_missing_fields() {
        let mut look = Look::default();
        look.curves.red.points.insert(1, [0.5, 0.6]);
        look.hsl[1].saturation = -60.0;
        look.wheels.shadows.x = 0.3;
        let text = serde_json::to_string(&look).expect("serialize");
        let back: Look = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, look);

        let partial: Look = serde_json::from_str(r#"{"wheels": {"midtones": {"luminance": 5}}}"#)
            .expect("deserialize");
        assert_eq!(partial.wheels.midtones.luminance, 5.0);
        assert!(partial.curves.is_identity());
    }
}
