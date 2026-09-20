//! The sidecar: the edit of one photo as JSON next to the original. For
//! IMG_0001.HEIC the sidecar is IMG_0001.HEIC.slate.json.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Crop, PhotoEdit};

/// The sidecar format version this build writes. Version 2 added the
/// presence sliders and the look to the edit; every new field has a default,
/// so a version 1 file reads as a version 2 file with a neutral look.
pub const VERSION: u32 = 2;

/// What is appended to the photo's file name.
pub const SUFFIX: &str = ".slate.json";

/// The saved state of one photo.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sidecar {
    pub version: u32,
    pub edit: PhotoEdit,
    pub crop: Crop,
}

impl Default for Sidecar {
    fn default() -> Self {
        Sidecar {
            version: VERSION,
            edit: PhotoEdit::default(),
            crop: Crop::default(),
        }
    }
}

impl Sidecar {
    pub fn new(edit: PhotoEdit, crop: Crop) -> Self {
        Sidecar {
            version: VERSION,
            edit,
            crop,
        }
    }

    /// The sidecar path for a photo: its full file name plus [`SUFFIX`].
    pub fn path_for(photo: &Path) -> PathBuf {
        let mut name = photo.file_name().map(OsString::from).unwrap_or_default();
        name.push(SUFFIX);
        photo.with_file_name(name)
    }

    /// Pretty JSON, one field per line, so the file diffs in git.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a sidecar always serializes")
    }

    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

#[cfg(test)]
mod tests {
    use super::{SUFFIX, Sidecar, VERSION};
    use crate::{Crop, CropAspect, CropRect, PhotoEdit};
    use std::path::Path;

    #[test]
    fn a_sidecar_round_trips_through_json() {
        let sidecar = Sidecar::new(
            PhotoEdit {
                exposure: 0.7,
                highlights: -30.0,
                ..PhotoEdit::default()
            },
            Crop {
                aspect: CropAspect::Grid3x4,
                rect: CropRect {
                    x: 0.125,
                    y: 0.0,
                    width: 0.75,
                    height: 1.0,
                },
            },
        );
        let back = Sidecar::from_json(&sidecar.to_json()).expect("parse");
        assert_eq!(back, sidecar);
        assert_eq!(back.version, VERSION);
    }

    #[test]
    fn missing_fields_take_defaults() {
        let back = Sidecar::from_json(r#"{"edit": {"contrast": 12}}"#).expect("parse");
        assert_eq!(back.version, VERSION);
        assert_eq!(back.edit.contrast, 12.0);
        assert_eq!(back.crop, Crop::default());
    }

    /// A sidecar exactly as version 1 of the format wrote it.
    const VERSION_1: &str = r#"{
  "version": 1,
  "edit": {
    "white_balance_temperature": -4.0,
    "white_balance_tint": 2.0,
    "exposure": 0.5,
    "contrast": 10.0,
    "highlights": -20.0,
    "shadows": 15.0,
    "whites": 5.0,
    "blacks": -5.0,
    "vibrance": 12.0,
    "saturation": -3.0
  },
  "crop": {
    "aspect": "Feed4x5",
    "rect": {
      "x": 0.25,
      "y": 0.0,
      "width": 0.5,
      "height": 1.0
    }
  }
}"#;

    #[test]
    fn a_version_1_sidecar_reads_with_a_neutral_look() {
        let back = Sidecar::from_json(VERSION_1).expect("parse");
        assert_eq!(back.version, 1);
        assert_eq!(
            back.edit,
            PhotoEdit {
                white_balance_temperature: -4.0,
                white_balance_tint: 2.0,
                exposure: 0.5,
                contrast: 10.0,
                highlights: -20.0,
                shadows: 15.0,
                whites: 5.0,
                blacks: -5.0,
                vibrance: 12.0,
                saturation: -3.0,
                ..PhotoEdit::default()
            }
        );
        assert!(back.edit.look.is_identity());
        assert_eq!(back.crop.aspect, CropAspect::Feed4x5);
        assert_eq!(back.crop.rect.width, 0.5);
        // Saved again, it is a version 2 file.
        let saved = Sidecar::new(back.edit, back.crop);
        assert_eq!(saved.version, 2);
        assert_eq!(VERSION, 2);
    }

    #[test]
    fn the_sidecar_sits_next_to_the_photo_with_its_full_name() {
        let path = Sidecar::path_for(Path::new("C:/photos/IMG_0001.HEIC"));
        assert_eq!(path, Path::new("C:/photos/IMG_0001.HEIC.slate.json"));
        assert!(path.to_string_lossy().ends_with(SUFFIX));
        assert_eq!(
            Sidecar::path_for(Path::new("Portrait_8.jpg")),
            Path::new("Portrait_8.jpg.slate.json")
        );
    }
}
