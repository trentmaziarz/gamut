//! Masks: where a local adjustment applies and what it adjusts. Plain data;
//! slate-color holds the arithmetic and slate-gpu rasterises the alpha.
//!
//! A mask is a list of components combined in order into one alpha, plus an
//! [`Adjustments`] that is applied on top of the global edit where the alpha
//! is. Positions are normalised to the uncropped photo, 0 to 1 on each axis,
//! so a mask stays where it was put when the crop moves. Lengths (the radii
//! of a radial gradient) are fractions of the longer side of the photo, so a
//! radial gradient with equal radii is a circle on any photo.
//!
//! [`MaskSource`] is a tagged enum: a file names the type of each source, and
//! a build that does not know a type refuses the file by that name instead of
//! dropping the mask.

use serde::{Deserialize, Serialize};

use crate::Adjustments;

/// The most masks one edit holds.
pub const MAX_MASKS: usize = 8;

/// The most components one mask holds.
pub const MAX_COMPONENTS: usize = 8;

/// How a component joins the alpha built so far, `a`, with its own, `b`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskOp {
    /// a + b - a b.
    #[default]
    Add,
    /// a (1 - b).
    Subtract,
    /// a b.
    Intersect,
}

impl MaskOp {
    pub const ALL: [(MaskOp, &str); 3] = [
        (MaskOp::Add, "Add"),
        (MaskOp::Subtract, "Subtract"),
        (MaskOp::Intersect, "Intersect"),
    ];
}

/// A linear gradient: alpha 0 at `start`, 1 at `end`, a smoothstep between,
/// constant along the perpendicular.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LinearGradient {
    pub start: [f32; 2],
    pub end: [f32; 2],
}

impl Default for LinearGradient {
    /// From the middle of the photo up to the top fifth: a sky.
    fn default() -> Self {
        LinearGradient {
            start: [0.5, 0.6],
            end: [0.5, 0.2],
        }
    }
}

/// A radial gradient: alpha 1 inside the ellipse, falling to 0 at its rim
/// across the feather band.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RadialGradient {
    pub centre: [f32; 2],
    /// The two radii as fractions of the longer side of the photo.
    pub radius: [f32; 2],
    /// The turn of the ellipse in degrees, -180 to 180.
    pub rotation: f32,
    /// How much of the radius the fall takes, 0 (a hard rim) to 100 (from
    /// the centre).
    pub feather: f32,
}

impl Default for RadialGradient {
    fn default() -> Self {
        RadialGradient {
            centre: [0.5, 0.5],
            radius: [0.25, 0.25],
            rotation: 0.0,
            feather: 50.0,
        }
    }
}

/// A luminance range of the source pixel on the normalised ACEScct axis:
/// alpha 1 from `low` to `high`, falling to 0 across `falloff` either side.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LuminanceRange {
    pub low: f32,
    pub high: f32,
    pub falloff: f32,
}

impl Default for LuminanceRange {
    /// The brighter half.
    fn default() -> Self {
        LuminanceRange {
            low: 0.5,
            high: 1.0,
            falloff: 0.1,
        }
    }
}

/// A colour range of the source pixel in the hue plane of ACEScct: alpha 1
/// within half of `hue_width` of `hue`, falling to 0 across `falloff`, and 0
/// for a pixel whose chroma is under `chroma_low`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColourRange {
    /// The centre hue in degrees, 0 to 360.
    pub hue: f32,
    /// The full width of the range in degrees.
    pub hue_width: f32,
    /// The chroma under which a pixel has no colour to speak of.
    pub chroma_low: f32,
    /// The fall at each edge of the range in degrees.
    pub falloff: f32,
}

impl Default for ColourRange {
    fn default() -> Self {
        ColourRange {
            hue: 0.0,
            hue_width: 60.0,
            chroma_low: 0.02,
            falloff: 15.0,
        }
    }
}

/// Where a component's alpha comes from.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MaskSource {
    Linear(LinearGradient),
    Radial(RadialGradient),
    Luminance(LuminanceRange),
    Colour(ColourRange),
}

impl Default for MaskSource {
    fn default() -> Self {
        MaskSource::Linear(LinearGradient::default())
    }
}

impl MaskSource {
    /// The name of the source for a list row.
    pub fn label(&self) -> &'static str {
        match self {
            MaskSource::Linear(_) => "Linear gradient",
            MaskSource::Radial(_) => "Radial gradient",
            MaskSource::Luminance(_) => "Luminance range",
            MaskSource::Colour(_) => "Colour range",
        }
    }

    /// Whether the alpha depends on the pixel and not only on where it is.
    pub fn reads_the_pixel(&self) -> bool {
        matches!(self, MaskSource::Luminance(_) | MaskSource::Colour(_))
    }

    /// The source with every number finite and inside its range.
    pub fn sanitised(&self) -> MaskSource {
        match *self {
            MaskSource::Linear(linear) => {
                let default = LinearGradient::default();
                MaskSource::Linear(LinearGradient {
                    start: point_or(linear.start, default.start),
                    end: point_or(linear.end, default.end),
                })
            }
            MaskSource::Radial(radial) => {
                let default = RadialGradient::default();
                let radius = point_or(radial.radius, default.radius);
                MaskSource::Radial(RadialGradient {
                    centre: point_or(radial.centre, default.centre),
                    radius: radius.map(|r| r.clamp(MIN_RADIUS, MAX_RADIUS)),
                    rotation: wrap_half_turn(finite_or(radial.rotation, default.rotation)),
                    feather: finite_or(radial.feather, default.feather).clamp(0.0, 100.0),
                })
            }
            MaskSource::Luminance(range) => {
                let default = LuminanceRange::default();
                let low = finite_or(range.low, default.low).clamp(0.0, 1.0);
                let high = finite_or(range.high, default.high).clamp(0.0, 1.0);
                MaskSource::Luminance(LuminanceRange {
                    low: low.min(high),
                    high: low.max(high),
                    falloff: finite_or(range.falloff, default.falloff).clamp(0.0, 1.0),
                })
            }
            MaskSource::Colour(range) => {
                let default = ColourRange::default();
                MaskSource::Colour(ColourRange {
                    hue: finite_or(range.hue, default.hue).rem_euclid(360.0),
                    hue_width: finite_or(range.hue_width, default.hue_width).clamp(0.0, 360.0),
                    chroma_low: finite_or(range.chroma_low, default.chroma_low).clamp(0.0, 1.0),
                    falloff: finite_or(range.falloff, default.falloff).clamp(0.0, 180.0),
                })
            }
        }
    }
}

/// The smallest radius of a radial gradient: it never collapses to a point.
pub const MIN_RADIUS: f32 = 0.001;

/// The largest radius of a radial gradient: four photos across.
pub const MAX_RADIUS: f32 = 4.0;

/// How far outside the photo a position may sit, in photos.
pub const POSITION_REACH: f32 = 4.0;

fn finite_or(value: f32, default: f32) -> f32 {
    if value.is_finite() { value } else { default }
}

fn point_or(point: [f32; 2], default: [f32; 2]) -> [f32; 2] {
    if point[0].is_finite() && point[1].is_finite() {
        point.map(|c| c.clamp(-POSITION_REACH, 1.0 + POSITION_REACH))
    } else {
        default
    }
}

/// An angle in degrees brought into -180 to 180.
fn wrap_half_turn(degrees: f32) -> f32 {
    let wrapped = (degrees + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 && degrees > 0.0 {
        180.0
    } else {
        wrapped
    }
}

/// One part of a mask: a source, whether it is inverted, and how it joins
/// the alpha built by the components before it. The alpha starts at 0, so a
/// first component that adds gives its own alpha.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Component {
    pub op: MaskOp,
    pub source: MaskSource,
    pub invert: bool,
}

impl Component {
    pub fn new(source: MaskSource) -> Self {
        Component {
            op: MaskOp::Add,
            source,
            invert: false,
        }
    }
}

/// One mask: where, and what it adjusts there. The adjustments are added to
/// the global ones, so a mask whose `adjust` is the default changes nothing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Mask {
    pub name: String,
    pub enabled: bool,
    pub invert: bool,
    /// How much of the mask applies, 0 to 100.
    pub opacity: f32,
    pub components: Vec<Component>,
    pub adjust: Adjustments,
}

impl Default for Mask {
    fn default() -> Self {
        Mask {
            name: String::new(),
            enabled: true,
            invert: false,
            opacity: 100.0,
            components: Vec::new(),
            adjust: Adjustments::default(),
        }
    }
}

impl Mask {
    /// An enabled mask of one added component.
    pub fn new(name: &str, source: MaskSource) -> Self {
        Mask {
            name: name.to_string(),
            components: vec![Component::new(source)],
            ..Mask::default()
        }
    }

    /// Whether the mask changes the picture: it is enabled, some of it
    /// applies, and it adjusts something.
    pub fn is_active(&self) -> bool {
        self.enabled && self.opacity > 0.0 && self.adjust != Adjustments::default()
    }

    /// What decides the alpha of the mask, and nothing of what it adjusts:
    /// the key an alpha that was rasterised once is kept under.
    pub fn shape(&self) -> MaskShape {
        let mask = self.sanitised();
        MaskShape {
            components: mask.components,
            invert: mask.invert,
        }
    }

    /// The mask with at most [`MAX_COMPONENTS`] components, every source
    /// sanitised and the opacity inside 0 to 100.
    pub fn sanitised(&self) -> Mask {
        let opacity = if self.opacity.is_finite() {
            self.opacity.clamp(0.0, 100.0)
        } else {
            100.0
        };
        Mask {
            name: self.name.clone(),
            enabled: self.enabled,
            invert: self.invert,
            opacity,
            components: self
                .components
                .iter()
                .take(MAX_COMPONENTS)
                .map(|component| Component {
                    source: component.source.sanitised(),
                    ..*component
                })
                .collect(),
            adjust: self.adjust.clone(),
        }
    }
}

/// The part of a mask its alpha depends on.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskShape {
    pub components: Vec<Component>,
    pub invert: bool,
}

/// The first [`MAX_MASKS`] masks of a list, each sanitised.
pub fn sanitised(masks: &[Mask]) -> Vec<Mask> {
    masks.iter().take(MAX_MASKS).map(Mask::sanitised).collect()
}

/// A name no mask of the list has yet: the base, or the base and a number.
pub fn free_name(masks: &[Mask], base: &str) -> String {
    let taken = |name: &str| masks.iter().any(|m| m.name.eq_ignore_ascii_case(name));
    if !taken(base) {
        return base.to_string();
    }
    (2..)
        .map(|n| format!("{base} {n}"))
        .find(|name| !taken(name))
        .expect("a free number exists")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_of_each() -> [MaskSource; 4] {
        [
            MaskSource::Linear(LinearGradient {
                start: [0.1, 0.9],
                end: [0.8, 0.2],
            }),
            MaskSource::Radial(RadialGradient {
                centre: [0.4, 0.6],
                radius: [0.3, 0.15],
                rotation: 30.0,
                feather: 25.0,
            }),
            MaskSource::Luminance(LuminanceRange {
                low: 0.2,
                high: 0.7,
                falloff: 0.05,
            }),
            MaskSource::Colour(ColourRange {
                hue: 210.0,
                hue_width: 40.0,
                chroma_low: 0.03,
                falloff: 10.0,
            }),
        ]
    }

    #[test]
    fn a_mask_round_trips_through_json_with_each_source() {
        for source in one_of_each() {
            let mut mask = Mask::new("Sky", source);
            mask.invert = true;
            mask.opacity = 60.0;
            mask.components.push(Component {
                op: MaskOp::Subtract,
                source,
                invert: true,
            });
            mask.adjust.exposure = -0.7;
            mask.adjust.look.wheels.shadows.x = 0.2;
            let text = serde_json::to_string_pretty(&mask).expect("serialize");
            let back: Mask = serde_json::from_str(&text).expect("deserialize");
            assert_eq!(back, mask, "{text}");
        }
    }

    #[test]
    fn a_source_names_its_type_in_the_file() {
        let names = one_of_each().map(|source| {
            let text = serde_json::to_string(&source).expect("serialize");
            let value: serde_json::Value = serde_json::from_str(&text).expect("parse");
            value["type"].as_str().expect("a type").to_string()
        });
        assert_eq!(names, ["Linear", "Radial", "Luminance", "Colour"]);
    }

    #[test]
    fn an_unknown_source_type_is_an_error_that_names_it() {
        let text = r#"{"name": "Hair", "components": [{"op": "Add", "source": {"type": "Brush", "strokes": []}}]}"#;
        let error = serde_json::from_str::<Mask>(text).expect_err("a later format");
        let message = error.to_string();
        assert!(message.contains("Brush"), "{message}");
        assert!(message.contains("Linear"), "{message}");
    }

    #[test]
    fn missing_fields_of_a_mask_take_defaults() {
        let mask: Mask = serde_json::from_str(
            r#"{"name": "Face", "components": [{"source": {"type": "Radial"}}]}"#,
        )
        .expect("parse");
        assert!(mask.enabled && !mask.invert);
        assert_eq!(mask.opacity, 100.0);
        assert_eq!(mask.components[0].op, MaskOp::Add);
        assert_eq!(
            mask.components[0].source,
            MaskSource::Radial(RadialGradient::default())
        );
        assert_eq!(mask.adjust, Adjustments::default());
        assert!(!mask.is_active(), "it adjusts nothing yet");
    }

    #[test]
    fn sanitising_holds_eight_masks_and_eight_components() {
        let mut mask = Mask::new("Many", MaskSource::default());
        mask.components = vec![Component::default(); 20];
        let masks = vec![mask; 12];
        let held = sanitised(&masks);
        assert_eq!(held.len(), MAX_MASKS);
        assert!(held.iter().all(|m| m.components.len() == MAX_COMPONENTS));
        assert_eq!(sanitised(&masks[..3]).len(), 3);
    }

    #[test]
    fn sanitising_brings_every_number_into_its_range() {
        let radial = MaskSource::Radial(RadialGradient {
            centre: [f32::NAN, 0.5],
            radius: [0.0, 9.0],
            rotation: 270.0,
            feather: 140.0,
        })
        .sanitised();
        assert_eq!(
            radial,
            MaskSource::Radial(RadialGradient {
                centre: [0.5, 0.5],
                radius: [MIN_RADIUS, MAX_RADIUS],
                rotation: -90.0,
                feather: 100.0,
            })
        );
        let luminance = MaskSource::Luminance(LuminanceRange {
            low: 0.9,
            high: 0.2,
            falloff: f32::INFINITY,
        })
        .sanitised();
        assert_eq!(
            luminance,
            MaskSource::Luminance(LuminanceRange {
                low: 0.2,
                high: 0.9,
                falloff: 0.1,
            })
        );
        let colour = MaskSource::Colour(ColourRange {
            hue: -30.0,
            hue_width: 500.0,
            chroma_low: -1.0,
            falloff: 200.0,
        })
        .sanitised();
        assert_eq!(
            colour,
            MaskSource::Colour(ColourRange {
                hue: 330.0,
                hue_width: 360.0,
                chroma_low: 0.0,
                falloff: 180.0,
            })
        );
        let linear = MaskSource::Linear(LinearGradient {
            start: [0.2, f32::NEG_INFINITY],
            end: [40.0, 0.5],
        })
        .sanitised();
        assert_eq!(
            linear,
            MaskSource::Linear(LinearGradient {
                start: LinearGradient::default().start,
                end: [1.0 + POSITION_REACH, 0.5],
            })
        );
        let mut mask = Mask::new("Loud", MaskSource::default());
        mask.opacity = 250.0;
        assert_eq!(mask.sanitised().opacity, 100.0);
        mask.opacity = f32::NAN;
        assert_eq!(mask.sanitised().opacity, 100.0);
    }

    #[test]
    fn the_shape_of_a_mask_ignores_what_it_adjusts() {
        let mut mask = Mask::new("Sky", MaskSource::default());
        let shape = mask.shape();
        mask.adjust.exposure = 1.0;
        mask.opacity = 40.0;
        mask.name = "Renamed".to_string();
        mask.enabled = false;
        assert_eq!(mask.shape(), shape);
        mask.invert = true;
        assert_ne!(mask.shape(), shape);
        mask.invert = false;
        mask.components[0].invert = true;
        assert_ne!(mask.shape(), shape);
    }

    #[test]
    fn a_mask_is_active_only_when_it_can_change_the_picture() {
        let mut mask = Mask::new("Sky", MaskSource::default());
        assert!(!mask.is_active());
        mask.adjust.exposure = -1.0;
        assert!(mask.is_active());
        mask.opacity = 0.0;
        assert!(!mask.is_active());
        mask.opacity = 50.0;
        mask.enabled = false;
        assert!(!mask.is_active());
    }

    #[test]
    fn a_free_name_counts_up_past_the_taken_ones() {
        let masks = vec![
            Mask::new("Linear", MaskSource::default()),
            Mask::new("linear 2", MaskSource::default()),
        ];
        assert_eq!(free_name(&masks, "Radial"), "Radial");
        assert_eq!(free_name(&masks, "Linear"), "Linear 3");
    }
}
