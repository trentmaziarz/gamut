//! gamut-core holds the project document. It carries the edit parameters for
//! a photo and the timeline model for a video. The timeline model covers
//! tracks, clips, in and out points, speed curves, keyframes, transitions and
//! overlays. Presets and undo history live here too. It is plain data with
//! serde, no GPU and no I/O. A photo edit is stored as a sidecar JSON next to
//! the original, a video project is a .slate JSON file with media paths
//! relative to it, and presets are JSON subsets of the photo parameters. All
//! three are text, so they diff in git.

use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};

pub mod crop;
pub mod export;
pub mod look;
pub mod mask;
pub mod preset;
pub mod project;
pub mod sidecar;
pub mod timeline;

pub use crop::{Crop, CropAspect, CropRect};
pub use export::ExportPreset;
pub use look::{Curve, HslRange, Look, ToneCurves, Wheel, Wheels};
pub use mask::{Component, Mask, MaskOp, MaskSource};
pub use preset::{LookPreset, PartialEdit};
pub use project::{MediaRef, Project};
pub use sidecar::{NamedVersion, Sidecar, VersionError};
pub use timeline::{Clip, Track};

/// The exposure slider runs from minus to plus this many stops.
pub const EXPOSURE_LIMIT: f32 = 5.0;

/// Every other slider runs from minus to plus this.
pub const SLIDER_LIMIT: f32 = 100.0;

/// Everything an edit adjusts: the Basic panel of M1, the presence sliders
/// and the look of M3. The global edit holds one and every mask holds one of
/// its own, which is why it is a type apart from [`PhotoEdit`].
///
/// Every scalar field is a slider. Zero is the neutral position for each of
/// them, so `Adjustments::default()` leaves the photo as it was shot.
/// Temperature and tint are offsets from the white balance read from the
/// file, exposure is in stops, and the rest run from -100 to 100 in the
/// Lightroom manner. The tone curves hold point lists, so the type is
/// `Clone` and not `Copy`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Adjustments {
    pub white_balance_temperature: f32,
    pub white_balance_tint: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub vibrance: f32,
    pub saturation: f32,
    pub texture: f32,
    pub clarity: f32,
    pub dehaze: f32,
    pub look: Look,
}

/// The edit parameters of one photo: the global [`Adjustments`] and the
/// masks that adjust parts of the picture on top of them, in list order.
///
/// The adjustments are flattened into the edit in a file, so an edit written
/// before masks existed reads unchanged, and the edit dereferences to them,
/// so `edit.exposure` is the global exposure.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoEdit {
    #[serde(flatten)]
    pub adjust: Adjustments,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub masks: Vec<Mask>,
}

impl Deref for PhotoEdit {
    type Target = Adjustments;

    fn deref(&self) -> &Adjustments {
        &self.adjust
    }
}

impl DerefMut for PhotoEdit {
    fn deref_mut(&mut self) -> &mut Adjustments {
        &mut self.adjust
    }
}

impl From<Adjustments> for PhotoEdit {
    /// An edit of these adjustments and no masks.
    fn from(adjust: Adjustments) -> Self {
        PhotoEdit {
            adjust,
            masks: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Adjustments, PhotoEdit};

    #[test]
    fn photo_edit_round_trips_through_json() {
        let edit = PhotoEdit::from(Adjustments {
            white_balance_temperature: -12.5,
            white_balance_tint: 3.0,
            exposure: 0.35,
            contrast: 10.0,
            highlights: -40.0,
            shadows: 25.0,
            whites: 5.0,
            blacks: -8.0,
            vibrance: 15.0,
            saturation: -3.0,
            texture: 20.0,
            clarity: -15.0,
            dehaze: 30.0,
            look: Default::default(),
        });
        let text = serde_json::to_string(&edit).expect("serialize");
        let back: PhotoEdit = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(edit, back);
    }

    #[test]
    fn missing_fields_take_the_neutral_value() {
        let back: PhotoEdit = serde_json::from_str(r#"{"exposure": 1.0}"#).expect("deserialize");
        assert_eq!(back.exposure, 1.0);
        assert_eq!(back.contrast, 0.0);
        assert_eq!(
            back,
            PhotoEdit::from(Adjustments {
                exposure: 1.0,
                ..Adjustments::default()
            })
        );
    }
}
