//! A painted mask source: an ordered list of strokes. Plain data; gamut-color
//! holds the dab arithmetic and gamut-gpu rasterises the strokes.
//!
//! A stroke is its points on the uncropped photo, normalised as every mask
//! position is, with the brush it was painted with. It is stored as points
//! and never as pixels, so it is exact at any zoom and at the export.
//!
//! A finished stroke never changes and is shared by reference count: a clone
//! of a brush, of a sidecar or of an undo step copies pointers, and two
//! clones compare equal by pointer before any point is read. The stroke being
//! painted is the only one that grows. A [`SharedStroke`] is sanitised when it
//! is made and by every change, so a brush is sanitised without reading its
//! points.
//!
//! In a file the points of a stroke are one string, "x,y x,y", the way an SVG
//! polyline writes them: a pretty-printed array would put every number on its
//! own line. Every other field has a default, so a later build can add fields.

use std::fmt::Write as _;
use std::ops::Deref;
use std::sync::Arc;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::mask::POSITION_REACH;

/// The most strokes one brush holds.
pub const MAX_STROKES: usize = 2000;

/// The most points one stroke holds.
pub const MAX_STROKE_POINTS: usize = 4096;

/// The smallest brush radius, as a fraction of the longer side of the photo.
pub const MIN_BRUSH_SIZE: f32 = 0.0002;

/// The largest brush radius: half the longer side.
pub const MAX_BRUSH_SIZE: f32 = 0.5;

/// The least flow: a dab always paints something.
pub const MIN_FLOW: f32 = 1.0;

/// The steps a stored coordinate is rounded to: a millionth of the photo,
/// under a hundredth of a pixel on 8,000 pixels, and short in the file.
const POINT_STEPS: f64 = 1e6;

/// The steps a stored pressure is rounded to: three decimals, finer than the
/// 1,024 levels Windows reports for a pen.
const PRESSURE_STEPS: f32 = 1000.0;

/// The sensitivity of an auto stroke that names none.
pub const DEFAULT_SENSITIVITY: f32 = 50.0;

fn is_false(flag: &bool) -> bool {
    !*flag
}

fn is_default_sensitivity(sensitivity: &f32) -> bool {
    *sensitivity == DEFAULT_SENSITIVITY
}

/// One stroke: where the brush went and what it was.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Stroke {
    /// The path, normalised to the uncropped photo. A single point is a dab.
    #[serde(with = "points_text")]
    pub points: Vec<[f32; 2]>,
    /// The pen pressure at each point, 0 to 1. Empty when the stroke was
    /// painted with no pen, and every point is then at 1. In a file it is
    /// one string, "p p p", and the key is left out when it is empty.
    #[serde(with = "pressure_text", skip_serializing_if = "Vec::is_empty")]
    pub pressure: Vec<f32>,
    /// The radius as a fraction of the longer side of the photo.
    pub size: f32,
    /// How much of the radius the fall takes, 0 (a hard rim) to 100 (from
    /// the centre).
    pub feather: f32,
    /// How much one dab paints, 1 to 100.
    pub flow: f32,
    /// Whether the stroke takes away what was painted before it.
    pub erase: bool,
    /// Whether each dab paints only the colour under its centre.
    #[serde(skip_serializing_if = "is_false")]
    pub auto: bool,
    /// How close to that colour a pixel must be, 0 (loose) to 100 (strict).
    /// It means nothing on a stroke that is not auto and is the default
    /// there.
    #[serde(skip_serializing_if = "is_default_sensitivity")]
    pub sensitivity: f32,
    /// Whether the pressure scales the radius: the toggle at the press.
    #[serde(skip_serializing_if = "is_false")]
    pub pressure_size: bool,
    /// Whether the pressure scales the flow: the toggle at the press.
    #[serde(skip_serializing_if = "is_false")]
    pub pressure_flow: bool,
}

impl Default for Stroke {
    fn default() -> Self {
        Stroke {
            points: Vec::new(),
            pressure: Vec::new(),
            size: 0.03,
            feather: 50.0,
            flow: 100.0,
            erase: false,
            auto: false,
            sensitivity: DEFAULT_SENSITIVITY,
            pressure_size: false,
            pressure_flow: false,
        }
    }
}

impl Stroke {
    /// The stroke with every number finite and inside its range, at most
    /// [`MAX_STROKE_POINTS`] points, each point on the stored steps, and a
    /// pressure list that is empty or as long as the points: a longer one is
    /// cut and a shorter one goes on at its last value.
    pub fn sanitised(&self) -> Stroke {
        let default = Stroke::default();
        let number = |value: f32, default: f32| if value.is_finite() { value } else { default };
        // A point that is dropped takes its pressure with it.
        let kept = || {
            self.points
                .iter()
                .enumerate()
                .filter_map(|(i, point)| stored_point(*point).map(|point| (i, point)))
                .take(MAX_STROKE_POINTS)
        };
        let pressure = if self.pressure.is_empty() {
            Vec::new()
        } else {
            let last = self.pressure.len() - 1;
            kept()
                .map(|(i, _)| stored_pressure(self.pressure[i.min(last)]))
                .collect()
        };
        Stroke {
            points: kept().map(|(_, point)| point).collect(),
            pressure,
            size: number(self.size, default.size).clamp(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE),
            feather: number(self.feather, default.feather).clamp(0.0, 100.0),
            flow: number(self.flow, default.flow).clamp(MIN_FLOW, 100.0),
            erase: self.erase,
            auto: self.auto,
            sensitivity: if self.auto {
                number(self.sensitivity, default.sensitivity).clamp(0.0, 100.0)
            } else {
                default.sensitivity
            },
            pressure_size: self.pressure_size,
            pressure_flow: self.pressure_flow,
        }
    }

    /// The pressure at a point: 1 where the stroke holds none.
    pub fn pressure_at(&self, index: usize) -> f32 {
        self.pressure.get(index).copied().unwrap_or(1.0)
    }

    /// Whether the pressure changes any dab: the stroke holds pressures and
    /// one of the two flags is on.
    pub fn uses_pressure(&self) -> bool {
        !self.pressure.is_empty() && (self.pressure_size || self.pressure_flow)
    }
}

/// A pressure as a stroke stores it: 0 to 1 on the stored steps. A pressure
/// that is not finite is a full one.
pub fn stored_pressure(pressure: f32) -> f32 {
    if pressure.is_finite() {
        (pressure.clamp(0.0, 1.0) * PRESSURE_STEPS).round() / PRESSURE_STEPS
    } else {
        1.0
    }
}

/// A point as a stroke stores it: inside the reach of a mask position and on
/// the stored steps. A point that is not finite is no point.
pub fn stored_point(point: [f32; 2]) -> Option<[f32; 2]> {
    (point[0].is_finite() && point[1].is_finite()).then(|| {
        point.map(|c| {
            let held = f64::from(c.clamp(-POSITION_REACH, 1.0 + POSITION_REACH));
            ((held * POINT_STEPS).round() / POINT_STEPS) as f32
        })
    })
}

/// A sanitised stroke behind a reference count. Equal pointers are equal
/// strokes, so comparing two clones of a brush reads no point.
#[derive(Clone, Debug)]
pub struct SharedStroke(Arc<Stroke>);

impl SharedStroke {
    pub fn new(stroke: &Stroke) -> Self {
        SharedStroke(Arc::new(stroke.sanitised()))
    }

    /// Adds a point to the path. Only the stroke being painted grows; when
    /// the stroke is shared (an undo step holds it) the other holders keep
    /// the path they had. False when the point is not finite or the stroke
    /// is full.
    ///
    /// `pressure` is what a pen reports and `None` from a mouse or a finger.
    /// A stroke that has met no pen holds no pressures. The first pressure
    /// puts the points before it at 1, and from then on a point with none
    /// takes the pressure of the point before it.
    pub fn push(&mut self, point: [f32; 2], pressure: Option<f32>) -> bool {
        let Some(point) = stored_point(point) else {
            return false;
        };
        if self.0.points.len() >= MAX_STROKE_POINTS {
            return false;
        }
        let stroke = Arc::make_mut(&mut self.0);
        match (pressure, stroke.pressure.last().copied()) {
            (Some(pressure), _) => {
                stroke.pressure.resize(stroke.points.len(), 1.0);
                stroke.pressure.push(stored_pressure(pressure));
            }
            (None, Some(last)) => stroke.pressure.push(last),
            (None, None) => {}
        }
        stroke.points.push(point);
        true
    }

    /// Whether the two are one allocation.
    pub fn shares_with(&self, other: &SharedStroke) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Whether `self` is `earlier` with points added: the same brush, and
    /// the path of `earlier` at the start of this one. How many points were
    /// added.
    pub fn grown_from(&self, earlier: &SharedStroke) -> Option<usize> {
        if self.shares_with(earlier) {
            return Some(0);
        }
        let (now, then) = (&*self.0, &*earlier.0);
        let same_brush = now.size == then.size
            && now.feather == then.feather
            && now.flow == then.flow
            && now.erase == then.erase
            && now.auto == then.auto
            && now.sensitivity == then.sensitivity
            && now.pressure_size == then.pressure_size
            && now.pressure_flow == then.pressure_flow;
        // A stroke that met its pen late holds 1 where it held nothing.
        let same_pressure = || {
            (now.pressure.is_empty() && then.pressure.is_empty())
                || (0..then.points.len()).all(|i| now.pressure_at(i) == then.pressure_at(i))
        };
        (same_brush && now.points.starts_with(&then.points) && same_pressure())
            .then(|| now.points.len() - then.points.len())
    }
}

impl Deref for SharedStroke {
    type Target = Stroke;

    fn deref(&self) -> &Stroke {
        &self.0
    }
}

impl PartialEq for SharedStroke {
    fn eq(&self, other: &Self) -> bool {
        self.shares_with(other) || *self.0 == *other.0
    }
}

impl Serialize for SharedStroke {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SharedStroke {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Stroke::deserialize(deserializer).map(|stroke| SharedStroke::new(&stroke))
    }
}

/// A painted source: strokes applied in order, so an erase stroke removes
/// only what was painted before it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Brush {
    pub strokes: Vec<SharedStroke>,
}

impl Brush {
    /// The brush with at most [`MAX_STROKES`] strokes. The strokes are
    /// sanitised already and are shared, not copied.
    pub fn sanitised(&self) -> Brush {
        Brush {
            strokes: self.strokes.iter().take(MAX_STROKES).cloned().collect(),
        }
    }

    /// Whether another stroke fits.
    pub fn has_room(&self) -> bool {
        self.strokes.len() < MAX_STROKES
    }

    /// Where the last stroke ended: where a straight line continues from.
    pub fn last_point(&self) -> Option<[f32; 2]> {
        self.strokes
            .iter()
            .rev()
            .find_map(|s| s.points.last().copied())
    }
}

/// The points of a stroke as one string, "x,y x,y".
mod points_text {
    use super::*;

    pub fn serialize<S: Serializer>(points: &[[f32; 2]], serializer: S) -> Result<S::Ok, S::Error> {
        let mut text = String::with_capacity(points.len() * 18);
        for (i, [x, y]) in points.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            write!(text, "{x},{y}").expect("a string takes any write");
        }
        serializer.serialize_str(&text)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<[f32; 2]>, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.split_ascii_whitespace()
            .map(|pair| {
                let (x, y) = pair.split_once(',')?;
                Some([x.parse().ok()?, y.parse().ok()?])
            })
            .collect::<Option<Vec<[f32; 2]>>>()
            .ok_or_else(|| {
                D::Error::custom("the points of a stroke are pairs \"x,y\" apart by spaces")
            })
    }
}

/// The pressures of a stroke as one string, "p p p".
mod pressure_text {
    use super::*;

    pub fn serialize<S: Serializer>(pressure: &[f32], serializer: S) -> Result<S::Ok, S::Error> {
        let mut text = String::with_capacity(pressure.len() * 6);
        for (i, p) in pressure.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            write!(text, "{p}").expect("a string takes any write");
        }
        serializer.serialize_str(&text)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f32>, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.split_ascii_whitespace()
            .map(|p| p.parse().ok())
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| {
                D::Error::custom("the pressures of a stroke are numbers apart by spaces")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::{Component, Mask, MaskOp, MaskSource};

    fn stroke(points: &[[f32; 2]]) -> Stroke {
        Stroke {
            points: points.to_vec(),
            size: 0.02,
            feather: 35.0,
            flow: 60.0,
            ..Stroke::default()
        }
    }

    fn painted() -> Brush {
        Brush {
            strokes: vec![
                SharedStroke::new(&stroke(&[[0.1, 0.2], [0.15, 0.25], [0.3, 0.123457]])),
                SharedStroke::new(&Stroke {
                    erase: true,
                    ..stroke(&[[0.5, 0.5]])
                }),
            ],
        }
    }

    #[test]
    fn a_brush_round_trips_through_json_and_names_its_type() {
        let mut mask = Mask::new("Hair", MaskSource::Brush(painted()));
        mask.components.push(Component {
            op: MaskOp::Subtract,
            source: MaskSource::Brush(Brush::default()),
            invert: false,
        });
        let text = serde_json::to_string_pretty(&mask).expect("serialize");
        assert!(text.contains("\"type\": \"Brush\""), "{text}");
        assert!(
            text.contains("\"points\": \"0.1,0.2 0.15,0.25 0.3,0.123457\""),
            "{text}"
        );
        let back: Mask = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, mask);
    }

    #[test]
    fn missing_fields_of_a_brush_and_of_a_stroke_take_their_defaults() {
        let source: MaskSource = serde_json::from_str(r#"{"type": "Brush"}"#).expect("parse");
        assert_eq!(source, MaskSource::Brush(Brush::default()));
        let source: MaskSource = serde_json::from_str(
            r#"{"type": "Brush", "strokes": [{"points": "0.5,0.5"}, {}], "later": 1}"#,
        )
        .expect("parse");
        let MaskSource::Brush(brush) = source else {
            panic!("a brush");
        };
        let default = Stroke::default();
        assert_eq!(brush.strokes[0].points, [[0.5, 0.5]]);
        assert_eq!(brush.strokes[0].size, default.size);
        assert_eq!(brush.strokes[0].feather, default.feather);
        assert_eq!(brush.strokes[0].flow, 100.0);
        assert!(!brush.strokes[0].erase);
        assert!(brush.strokes[1].points.is_empty());
    }

    #[test]
    fn points_that_are_not_pairs_are_an_error() {
        for bad in ["0.5", "0.5,0.5,0.5 1", "a,b", "0.5;0.5"] {
            let text = format!(r#"{{"points": "{bad}"}}"#);
            assert!(serde_json::from_str::<Stroke>(&text).is_err(), "{bad}");
        }
        let empty: Stroke = serde_json::from_str(r#"{"points": ""}"#).expect("no points");
        assert!(empty.points.is_empty());
    }

    #[test]
    fn sanitising_clamps_a_stroke_and_the_limits() {
        let wild = Stroke {
            points: vec![[f32::NAN, 0.5], [40.0, -40.0], [0.123_456_8, 0.5]],
            size: 9.0,
            feather: -5.0,
            flow: 0.0,
            erase: true,
            ..Stroke::default()
        }
        .sanitised();
        assert_eq!(
            wild.points,
            [[1.0 + POSITION_REACH, -POSITION_REACH], [0.123_457, 0.5]]
        );
        assert_eq!(wild.size, MAX_BRUSH_SIZE);
        assert_eq!(wild.feather, 0.0);
        assert_eq!(wild.flow, MIN_FLOW);
        assert!(wild.erase);
        let unreal = Stroke {
            size: f32::NAN,
            feather: f32::INFINITY,
            flow: f32::NAN,
            ..Stroke::default()
        }
        .sanitised();
        assert_eq!(unreal, Stroke::default());

        let long = SharedStroke::new(&stroke(&vec![[0.5, 0.5]; MAX_STROKE_POINTS + 10]));
        assert_eq!(long.points.len(), MAX_STROKE_POINTS);
        let mut full = long.clone();
        assert!(!full.push([0.1, 0.1], None), "a full stroke takes no point");
        assert!(full.shares_with(&long), "and is not copied for it");

        let many = Brush {
            strokes: vec![SharedStroke::new(&stroke(&[[0.5, 0.5]])); MAX_STROKES + 5],
        };
        assert!(!many.has_room());
        assert_eq!(many.sanitised().strokes.len(), MAX_STROKES);
    }

    #[test]
    fn a_stored_point_survives_the_file_exactly() {
        let mut growing = SharedStroke::new(&stroke(&[]));
        for i in 0..500 {
            let t = i as f32 / 499.0;
            assert!(growing.push([t * 1.3 - 0.1, (t * 37.0).sin() * 0.5 + 0.5], None));
        }
        let text = serde_json::to_string(&growing).expect("serialize");
        let back: SharedStroke = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.points, growing.points);
        assert_eq!(serde_json::to_string(&back).expect("serialize"), text);
    }

    #[test]
    fn a_cloned_brush_shares_its_strokes() {
        let brush = painted();
        let clone = brush.clone();
        let sanitised = brush.sanitised();
        for (i, stroke) in brush.strokes.iter().enumerate() {
            assert!(stroke.shares_with(&clone.strokes[i]));
            assert!(stroke.shares_with(&sanitised.strokes[i]));
        }
    }

    #[test]
    fn only_the_growing_stroke_is_copied_when_a_clone_holds_it() {
        let mut brush = painted();
        let held = brush.clone();
        assert!(brush.strokes[0].push([0.4, 0.4], None));
        assert!(!brush.strokes[0].shares_with(&held.strokes[0]));
        assert_eq!(held.strokes[0].points.len(), 3, "the clone keeps its path");
        assert!(brush.strokes[1].shares_with(&held.strokes[1]));
        assert_ne!(brush, held);
        assert_eq!(brush.strokes[0].grown_from(&held.strokes[0]), Some(1));
        assert_eq!(held.strokes[0].grown_from(&brush.strokes[0]), None);
        assert_eq!(brush.strokes[1].grown_from(&held.strokes[0]), None);
    }

    #[test]
    fn equal_clones_compare_without_reading_a_point() {
        // NaN is not equal to itself, so a comparison that read the points
        // would call these two different. SharedStroke::new would sanitise
        // the NaN away, so the stroke is built around it here.
        let poisoned = SharedStroke(Arc::new(stroke(&[[f32::NAN, 0.5]])));
        let brush = Brush {
            strokes: vec![poisoned],
        };
        let clone = brush.clone();
        assert_eq!(brush, clone);
        let rebuilt = Brush {
            strokes: vec![SharedStroke(Arc::new(stroke(&[[f32::NAN, 0.5]])))],
        };
        assert_ne!(
            brush, rebuilt,
            "content is compared when the pointers differ"
        );
    }

    fn pen_stroke() -> Stroke {
        Stroke {
            pressure: vec![0.2, 0.5, 1.0],
            auto: true,
            sensitivity: 80.0,
            pressure_size: true,
            pressure_flow: true,
            ..stroke(&[[0.1, 0.2], [0.15, 0.25], [0.3, 0.4]])
        }
    }

    #[test]
    fn a_stroke_with_every_new_field_round_trips_and_its_pressure_is_one_string() {
        let brush = Brush {
            strokes: vec![SharedStroke::new(&pen_stroke())],
        };
        let text = serde_json::to_string_pretty(&brush).expect("serialize");
        assert!(text.contains("\"pressure\": \"0.2 0.5 1\""), "{text}");
        assert!(text.contains("\"auto\": true"), "{text}");
        assert!(text.contains("\"sensitivity\": 80.0"), "{text}");
        assert!(text.contains("\"pressure_size\": true"), "{text}");
        assert!(text.contains("\"pressure_flow\": true"), "{text}");
        let back: Brush = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, brush);
        assert_eq!(
            serde_json::to_string_pretty(&back).expect("serialize"),
            text
        );
    }

    #[test]
    fn a_stroke_with_no_new_field_writes_none_of_their_keys() {
        let text = serde_json::to_string(&SharedStroke::new(&stroke(&[[0.5, 0.5]]))).expect("text");
        for key in ["pressure", "auto", "sensitivity"] {
            assert!(!text.contains(key), "{key} in {text}");
        }
    }

    #[test]
    fn missing_new_fields_take_their_defaults() {
        let old: Stroke = serde_json::from_str(r#"{"points": "0.5,0.5 0.6,0.6"}"#).expect("parse");
        assert!(old.pressure.is_empty());
        assert!(!old.auto && !old.pressure_size && !old.pressure_flow);
        assert_eq!(old.sensitivity, DEFAULT_SENSITIVITY);
        assert_eq!(old.pressure_at(1), 1.0, "no pressure is a full one");
        assert!(!old.uses_pressure());
        let bad = r#"{"points": "0.5,0.5", "pressure": "0.5 x"}"#;
        assert!(serde_json::from_str::<Stroke>(bad).is_err());
    }

    #[test]
    fn sanitising_clamps_the_sensitivity_and_brings_the_pressure_to_the_points() {
        let wild = Stroke {
            sensitivity: 400.0,
            pressure: vec![-1.0, f32::NAN, 0.123_46, 2.0, 0.7],
            ..pen_stroke()
        }
        .sanitised();
        assert_eq!(wild.sensitivity, 100.0);
        assert_eq!(wild.pressure, [0.0, 1.0, 0.123], "cut to the three points");

        let short = Stroke {
            pressure: vec![0.4],
            sensitivity: f32::NAN,
            ..pen_stroke()
        }
        .sanitised();
        assert_eq!(
            short.pressure,
            [0.4, 0.4, 0.4],
            "filled with its last value"
        );
        assert_eq!(short.sensitivity, DEFAULT_SENSITIVITY);

        let dropped = Stroke {
            points: vec![[0.1, 0.1], [f32::NAN, 0.5], [0.3, 0.3]],
            pressure: vec![0.1, 0.2, 0.3],
            ..pen_stroke()
        }
        .sanitised();
        assert_eq!(
            dropped.pressure,
            [0.1, 0.3],
            "a dropped point takes its pressure"
        );

        let plain = Stroke {
            auto: false,
            sensitivity: 80.0,
            ..pen_stroke()
        }
        .sanitised();
        assert_eq!(
            plain.sensitivity, DEFAULT_SENSITIVITY,
            "a stroke that is not auto holds no sensitivity"
        );
    }

    #[test]
    fn a_stroke_grown_from_another_has_the_same_new_fields() {
        let earlier = SharedStroke::new(&pen_stroke());
        let mut grown = earlier.clone();
        assert!(grown.push([0.4, 0.5], Some(0.9)));
        assert_eq!(grown.grown_from(&earlier), Some(1));
        let changes: [fn(&mut Stroke); 5] = [
            |s| s.auto = false,
            |s| s.sensitivity = 20.0,
            |s| s.pressure_size = false,
            |s| s.pressure_flow = false,
            |s| s.pressure[1] = 0.6,
        ];
        for change in changes {
            let mut other = (*grown).clone();
            change(&mut other);
            assert_eq!(SharedStroke::new(&other).grown_from(&earlier), None);
        }
    }

    #[test]
    fn push_takes_a_pressure_with_a_point() {
        let mut mouse = SharedStroke::new(&stroke(&[[0.1, 0.1]]));
        assert!(mouse.push([0.2, 0.2], None));
        assert!(mouse.pressure.is_empty(), "a mouse stores no pressure");

        let mut pen = SharedStroke::new(&stroke(&[]));
        assert!(pen.push([0.1, 0.1], Some(0.25)));
        assert!(pen.push([0.2, 0.2], Some(0.123_46)));
        assert!(pen.push([0.3, 0.3], None));
        assert_eq!(
            pen.pressure,
            [0.25, 0.123, 0.123],
            "none goes on at the last"
        );
        assert!(!pen.push([f32::NAN, 0.3], Some(0.5)));
        assert_eq!(pen.pressure.len(), pen.points.len());

        let earlier = mouse.clone();
        assert!(mouse.push([0.3, 0.3], Some(0.5)));
        assert_eq!(
            mouse.pressure,
            [1.0, 1.0, 0.5],
            "the points before a pen are at 1"
        );
        assert_eq!(mouse.grown_from(&earlier), Some(1));

        let text = serde_json::to_string(&pen).expect("serialize");
        let back: SharedStroke = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.pressure, pen.pressure);
        assert_eq!(serde_json::to_string(&back).expect("serialize"), text);
    }

    #[test]
    fn a_line_continues_from_the_end_of_the_last_stroke() {
        assert_eq!(Brush::default().last_point(), None);
        assert_eq!(painted().last_point(), Some([0.5, 0.5]));
        let mut brush = painted();
        brush.strokes.push(SharedStroke::new(&stroke(&[])));
        assert_eq!(
            brush.last_point(),
            Some([0.5, 0.5]),
            "an empty stroke ends nowhere"
        );
    }
}
