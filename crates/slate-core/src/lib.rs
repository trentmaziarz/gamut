//! slate-core holds the project document. It carries the edit parameters for
//! a photo and the timeline model for a video. The timeline model covers
//! tracks, clips, in and out points, speed curves, keyframes, transitions and
//! overlays. Presets and undo history live here too. It is plain data with
//! serde, no GPU and no I/O. A photo edit is stored as a sidecar JSON next to
//! the original, a video project is a .slate JSON file with media paths
//! relative to it, and presets are JSON subsets of the photo parameters. All
//! three are text, so they diff in git.

use serde::{Deserialize, Serialize};

pub mod crop;
pub mod export;
pub mod sidecar;

pub use crop::{Crop, CropAspect, CropRect};
pub use export::ExportPreset;
pub use sidecar::Sidecar;

/// The edit parameters of one photo: the Basic panel of M1.
///
/// Every field is a slider. Zero is the neutral position for each of them,
/// so `PhotoEdit::default()` leaves the photo as it was shot. Temperature
/// and tint are offsets from the white balance read from the file, exposure
/// is in stops, and the rest run from -100 to 100 in the Lightroom manner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotoEdit {
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
}

#[cfg(test)]
mod tests {
    use super::PhotoEdit;

    #[test]
    fn photo_edit_round_trips_through_json() {
        let edit = PhotoEdit {
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
        };
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
            PhotoEdit {
                exposure: 1.0,
                ..PhotoEdit::default()
            }
        );
    }
}
