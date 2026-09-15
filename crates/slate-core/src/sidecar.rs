//! The sidecar: the edit of one photo as JSON next to the original. For
//! IMG_0001.HEIC the sidecar is IMG_0001.HEIC.slate.json.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Crop, PhotoEdit};

/// The sidecar format version this build writes.
pub const VERSION: u32 = 1;

/// What is appended to the photo's file name.
pub const SUFFIX: &str = ".slate.json";

/// The saved state of one photo.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
