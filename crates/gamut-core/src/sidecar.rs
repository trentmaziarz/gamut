//! The sidecar: the edit of one photo as JSON next to the original. For
//! IMG_0001.HEIC the sidecar is IMG_0001.HEIC.gamut.json.

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Crop, PhotoEdit};

/// The sidecar format version this build writes. Version 2 added the
/// presence sliders and the look to the edit, version 3 the masks; every new
/// field has a default, so a version 1 file reads as a version 3 file with a
/// neutral look and a version 2 file as one with no masks. A later version
/// that names a mask source this build does not know is refused by that
/// name, not read with the mask dropped.
pub const VERSION: u32 = 3;

/// What is appended to the photo's file name.
pub const SUFFIX: &str = ".gamut.json";

/// One named version of a photo: an edit and a crop kept under a name.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NamedVersion {
    pub name: String,
    pub edit: PhotoEdit,
    pub crop: Crop,
}

/// Why a version could not be saved, renamed or found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VersionError {
    /// The name is empty.
    EmptyName,
    /// Another version already has this name; case does not tell names
    /// apart.
    NameTaken(String),
    /// No version has this name. Holds the names that exist.
    NotFound { name: String, existing: Vec<String> },
}

impl fmt::Display for VersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VersionError::EmptyName => f.write_str("a version needs a name"),
            VersionError::NameTaken(name) => write!(f, "a version named {name} already exists"),
            VersionError::NotFound { name, existing } if existing.is_empty() => {
                write!(f, "no version named {name}: this photo has no versions")
            }
            VersionError::NotFound { name, existing } => write!(
                f,
                "no version named {name}: the versions are {}",
                existing.join(", ")
            ),
        }
    }
}

impl std::error::Error for VersionError {}

/// The saved state of one photo. The top-level edit and crop are the working
/// state; `versions` holds the named ones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sidecar {
    pub version: u32,
    pub edit: PhotoEdit,
    pub crop: Crop,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub versions: Vec<NamedVersion>,
    /// The version the working state was last switched to, saved as or
    /// updated into: the one [`Sidecar::is_dirty`] compares against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_version: Option<String>,
}

impl Default for Sidecar {
    fn default() -> Self {
        Sidecar::new(PhotoEdit::default(), Crop::default())
    }
}

impl Sidecar {
    pub fn new(edit: PhotoEdit, crop: Crop) -> Self {
        Sidecar {
            version: VERSION,
            edit,
            crop,
            versions: Vec::new(),
            active_version: None,
        }
    }

    /// The index of the version with this name, compared without case.
    pub fn find_version(&self, name: &str) -> Option<usize> {
        let wanted = name.trim().to_lowercase();
        self.versions
            .iter()
            .position(|v| v.name.to_lowercase() == wanted)
    }

    fn not_found(&self, name: &str) -> VersionError {
        VersionError::NotFound {
            name: name.trim().to_string(),
            existing: self.versions.iter().map(|v| v.name.clone()).collect(),
        }
    }

    /// The version with this name, or the names that exist.
    pub fn version(&self, name: &str) -> Result<&NamedVersion, VersionError> {
        match self.find_version(name) {
            Some(index) => Ok(&self.versions[index]),
            None => Err(self.not_found(name)),
        }
    }

    /// Keeps the working state under a new name and makes it the active
    /// version.
    pub fn save_version(&mut self, name: &str) -> Result<(), VersionError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(VersionError::EmptyName);
        }
        if self.find_version(name).is_some() {
            return Err(VersionError::NameTaken(name.to_string()));
        }
        self.versions.push(NamedVersion {
            name: name.to_string(),
            edit: self.edit.clone(),
            crop: self.crop,
        });
        self.active_version = Some(name.to_string());
        Ok(())
    }

    /// Copies a version into the working state and makes it the active
    /// version. A switch never writes into a version: what the working state
    /// held is gone unless [`Sidecar::update_version`] or
    /// [`Sidecar::save_version`] kept it first, which is what
    /// [`Sidecar::is_dirty`] is there to ask about.
    pub fn switch_to(&mut self, name: &str) -> Result<(), VersionError> {
        let target = self
            .find_version(name)
            .ok_or_else(|| self.not_found(name))?;
        let version = self.versions[target].clone();
        self.edit = version.edit;
        self.crop = version.crop;
        self.active_version = Some(version.name);
        Ok(())
    }

    /// Copies the working state into the version with this name and makes it
    /// the active version. The only way a version changes after it is saved.
    pub fn update_version(&mut self, name: &str) -> Result<(), VersionError> {
        let index = self
            .find_version(name)
            .ok_or_else(|| self.not_found(name))?;
        self.versions[index].edit = self.edit.clone();
        self.versions[index].crop = self.crop;
        self.active_version = Some(self.versions[index].name.clone());
        Ok(())
    }

    /// Whether the working state holds work its active version does not:
    /// true when an active version exists and the working edit or crop
    /// differs from it. With no active version there is nothing to be behind.
    pub fn is_dirty(&self) -> bool {
        self.active_version
            .as_deref()
            .and_then(|active| self.find_version(active))
            .is_some_and(|index| {
                let version = &self.versions[index];
                version.edit != self.edit || version.crop != self.crop
            })
    }

    pub fn rename_version(&mut self, name: &str, new_name: &str) -> Result<(), VersionError> {
        let new_name = new_name.trim();
        if new_name.is_empty() {
            return Err(VersionError::EmptyName);
        }
        let index = self
            .find_version(name)
            .ok_or_else(|| self.not_found(name))?;
        if self
            .find_version(new_name)
            .is_some_and(|other| other != index)
        {
            return Err(VersionError::NameTaken(new_name.to_string()));
        }
        if self.active_version.as_deref() == Some(self.versions[index].name.as_str()) {
            self.active_version = Some(new_name.to_string());
        }
        self.versions[index].name = new_name.to_string();
        Ok(())
    }

    pub fn delete_version(&mut self, name: &str) -> Result<(), VersionError> {
        let index = self
            .find_version(name)
            .ok_or_else(|| self.not_found(name))?;
        let removed = self.versions.remove(index);
        if self.active_version.as_deref() == Some(removed.name.as_str()) {
            self.active_version = None;
        }
        Ok(())
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
    use super::{SUFFIX, Sidecar, VERSION, VersionError};
    use crate::{Adjustments, Crop, CropAspect, CropRect, Mask, MaskSource, PhotoEdit};
    use std::path::Path;

    #[test]
    fn a_sidecar_round_trips_through_json() {
        let sidecar = Sidecar::new(
            PhotoEdit::from(Adjustments {
                exposure: 0.7,
                highlights: -30.0,
                ..Adjustments::default()
            }),
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
    fn version_names_collide_without_regard_to_case() {
        let mut sidecar = Sidecar::default();
        sidecar.save_version("Warm").expect("the first");
        assert_eq!(
            sidecar.save_version(" warm "),
            Err(VersionError::NameTaken("warm".to_string()))
        );
        assert_eq!(sidecar.save_version("  "), Err(VersionError::EmptyName));
        sidecar.save_version("Cold").expect("another name");
        assert_eq!(
            sidecar.rename_version("cold", "WARM"),
            Err(VersionError::NameTaken("WARM".to_string()))
        );
        sidecar
            .rename_version("cold", "Cold")
            .expect("its own name");
        sidecar.rename_version("COLD", "Cool").expect("a free name");
        assert_eq!(sidecar.active_version.as_deref(), Some("Cool"));
        assert!(sidecar.find_version("cool").is_some());
        let error = sidecar.version("nope").expect_err("unknown");
        assert_eq!(
            error.to_string(),
            "no version named nope: the versions are Warm, Cool"
        );
    }

    /// Two versions, Bright (exposure 1) and Dark (exposure -1), with Dark
    /// the active one.
    fn bright_and_dark() -> Sidecar {
        let mut sidecar = Sidecar::default();
        sidecar.edit.exposure = 1.0;
        sidecar.save_version("Bright").expect("save");
        sidecar.edit.exposure = -1.0;
        sidecar.save_version("Dark").expect("save");
        sidecar
    }

    #[test]
    fn switching_on_a_clean_state_brings_the_version_in() {
        let mut sidecar = bright_and_dark();
        assert!(!sidecar.is_dirty());
        sidecar.switch_to("bright").expect("switch");
        assert_eq!(sidecar.edit.exposure, 1.0);
        assert_eq!(sidecar.active_version.as_deref(), Some("Bright"));
        assert!(!sidecar.is_dirty());
        assert_eq!(sidecar.version("Dark").expect("found").edit.exposure, -1.0);
    }

    /// The rule that replaced the save-back of version 2: a switch never
    /// writes into a version, so work that was not kept is gone.
    #[test]
    fn switching_on_a_dirty_state_leaves_the_version_it_came_from_untouched() {
        let mut sidecar = bright_and_dark();
        let before = sidecar.versions.clone();

        // Working on Dark: a change, then a switch away and back.
        sidecar.edit.contrast = 30.0;
        assert!(sidecar.is_dirty());
        sidecar.switch_to("bright").expect("switch");
        assert_eq!(sidecar.versions, before, "no version was written");
        assert_eq!(sidecar.edit.exposure, 1.0);
        assert_eq!(sidecar.edit.contrast, 0.0);
        sidecar.switch_to("Dark").expect("switch back");
        assert_eq!(sidecar.edit.exposure, -1.0);
        assert_eq!(sidecar.edit.contrast, 0.0, "Dark is as it was saved");
        assert_eq!(sidecar.versions, before);

        let back = Sidecar::from_json(&sidecar.to_json()).expect("parse");
        assert_eq!(back, sidecar);
    }

    #[test]
    fn update_version_writes_the_working_state_into_the_named_version() {
        let mut sidecar = bright_and_dark();
        sidecar.edit.contrast = 30.0;
        sidecar.crop.rect.width = 0.5;
        sidecar.update_version("dark").expect("update");
        assert!(!sidecar.is_dirty());
        let dark = sidecar.version("Dark").expect("found");
        assert_eq!(dark.edit.contrast, 30.0);
        assert_eq!(dark.crop.rect.width, 0.5);
        assert_eq!(sidecar.version("Bright").expect("found").edit.contrast, 0.0);

        // Updating another version makes that one the active version.
        sidecar.edit.contrast = 45.0;
        sidecar.update_version("Bright").expect("update");
        assert_eq!(sidecar.active_version.as_deref(), Some("Bright"));
        assert_eq!(
            sidecar.version("Bright").expect("found").edit.contrast,
            45.0
        );
        assert_eq!(sidecar.version("Dark").expect("found").edit.contrast, 30.0);
        assert!(!sidecar.is_dirty());

        let error = sidecar.update_version("nope").expect_err("unknown");
        assert_eq!(
            error.to_string(),
            "no version named nope: the versions are Bright, Dark"
        );
    }

    #[test]
    fn is_dirty_follows_the_edit_and_the_crop() {
        let mut sidecar = Sidecar::default();
        sidecar.edit.exposure = 0.5;
        assert!(
            !sidecar.is_dirty(),
            "no active version, nothing to be behind"
        );
        sidecar.save_version("One").expect("save");
        assert!(!sidecar.is_dirty());

        sidecar.edit.exposure = 0.75;
        assert!(sidecar.is_dirty());
        sidecar.edit.exposure = 0.5;
        assert!(!sidecar.is_dirty(), "back where the version is");

        sidecar.crop.rect.x = 0.1;
        assert!(sidecar.is_dirty());
        sidecar.switch_to("One").expect("switch");
        assert!(!sidecar.is_dirty());
        assert_eq!(sidecar.crop.rect.x, 0.0);

        sidecar.edit.look.wheels.shadows.x = 0.2;
        assert!(sidecar.is_dirty());
        sidecar.delete_version("One").expect("delete");
        assert!(!sidecar.is_dirty(), "the active version is gone");
    }

    #[test]
    fn rename_and_delete_keep_working_around_the_active_version() {
        let mut sidecar = bright_and_dark();
        sidecar.edit.contrast = 30.0;
        sidecar.rename_version("dark", "Night").expect("rename");
        assert_eq!(sidecar.active_version.as_deref(), Some("Night"));
        assert!(sidecar.is_dirty(), "a rename keeps the comparison");
        sidecar.update_version("Night").expect("update");
        assert!(!sidecar.is_dirty());

        sidecar.delete_version("night").expect("delete");
        assert_eq!(sidecar.active_version, None);
        assert_eq!(sidecar.versions.len(), 1);
        assert_eq!(sidecar.edit.contrast, 30.0, "the working state stays");
        sidecar.delete_version("Bright").expect("delete");
        assert!(sidecar.versions.is_empty());
    }

    #[test]
    fn versions_and_the_active_version_round_trip_through_json() {
        let mut sidecar = bright_and_dark();
        sidecar.edit.contrast = 30.0;
        let back = Sidecar::from_json(&sidecar.to_json()).expect("parse");
        assert_eq!(back, sidecar);
        assert!(
            back.is_dirty(),
            "the unsaved work survives a reload as unsaved"
        );
        assert_eq!(back.active_version.as_deref(), Some("Dark"));
    }

    /// Trent at the window, 2026-09-20: save a version, change the look,
    /// switch back to the version. The version must come back as it was
    /// saved, and a second version saved after a reset must not replace it.
    #[test]
    fn switching_to_the_version_in_use_brings_it_back_as_saved() {
        let mut sidecar = Sidecar::default();
        sidecar.edit.exposure = 1.0;
        sidecar.save_version("Sunset Version").expect("save");
        sidecar.edit = PhotoEdit::default();
        sidecar.switch_to("Sunset Version").expect("switch");
        assert_eq!(sidecar.edit.exposure, 1.0, "the look comes back");
        assert_eq!(
            sidecar.versions[0].edit.exposure, 1.0,
            "the version is kept"
        );

        sidecar.edit = PhotoEdit::default();
        sidecar.save_version("Normal Version").expect("save");
        sidecar.switch_to("Sunset Version").expect("switch");
        assert_eq!(sidecar.edit.exposure, 1.0);
        sidecar.switch_to("Normal Version").expect("switch");
        assert_eq!(sidecar.edit.exposure, 0.0);
        let back = Sidecar::from_json(&sidecar.to_json()).expect("parse");
        assert_eq!(
            back.version("Sunset Version").expect("found").edit.exposure,
            1.0
        );
        assert_eq!(
            back.version("Normal Version").expect("found").edit.exposure,
            0.0
        );
    }

    #[test]
    fn a_sidecar_without_versions_writes_no_versions_key() {
        let text = Sidecar::default().to_json();
        assert!(!text.contains("versions") && !text.contains("active_version"));
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
            PhotoEdit::from(Adjustments {
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
                ..Adjustments::default()
            })
        );
        assert!(back.edit.look.is_identity());
        assert_eq!(back.crop.aspect, CropAspect::Feed4x5);
        assert_eq!(back.crop.rect.width, 0.5);
        assert!(back.edit.masks.is_empty());
        // Saved again, it is a version 3 file.
        let saved = Sidecar::new(back.edit, back.crop);
        assert_eq!(saved.version, 3);
        assert_eq!(VERSION, 3);
    }

    /// A sidecar exactly as version 2 of the format wrote it, with a look,
    /// two versions and an active version.
    const VERSION_2: &str = include_str!("testdata/sidecar_version_2.json");

    #[test]
    fn a_version_2_sidecar_reads_with_no_masks_and_saves_as_version_3() {
        let back = Sidecar::from_json(VERSION_2).expect("parse");
        assert_eq!(back.version, 2);
        assert!(back.edit.masks.is_empty());
        assert_eq!(back.edit.exposure, 0.5);
        assert_eq!(back.edit.clarity, 35.0);
        assert_eq!(back.edit.look.curves.master.points.len(), 4);
        assert_eq!(back.edit.look.hsl[4].saturation, -40.0);
        assert_eq!(back.edit.look.wheels.shadows.x, -0.5);
        assert_eq!(back.crop.aspect, CropAspect::Feed4x5);
        assert_eq!(back.versions.len(), 2);
        assert_eq!(back.active_version.as_deref(), Some("Teal"));
        assert!(!back.is_dirty());
        assert!(back.versions.iter().all(|v| v.edit.masks.is_empty()));

        let saved = Sidecar {
            versions: back.versions.clone(),
            active_version: back.active_version.clone(),
            ..Sidecar::new(back.edit.clone(), back.crop)
        };
        let text = saved.to_json();
        assert!(text.contains("\"version\": 3"), "{text}");
        assert!(
            !text.contains("masks"),
            "an edit without masks writes no key"
        );
        // Apart from the version number the file is the one version 2 wrote.
        assert_eq!(
            text.replace("\"version\": 3", "\"version\": 2").trim(),
            VERSION_2.replace("\r\n", "\n").trim()
        );
    }

    /// A sidecar exactly as version 3 wrote it before the brush: two masks
    /// of two components, one of each source, and a version holding them.
    const VERSION_3_BRUSH: &str = include_str!("testdata/sidecar_version_3_brush_4c.json");

    #[test]
    fn a_sidecar_with_a_brush_from_before_the_auto_mask_saves_back_unchanged() {
        let back = Sidecar::from_json(VERSION_3_BRUSH).expect("parse");
        assert_eq!(back.version, 3);
        let MaskSource::Brush(brush) = &back.edit.masks[0].components[0].source else {
            panic!("a brush");
        };
        assert_eq!(brush.strokes.len(), 3);
        assert!(brush.strokes.iter().all(|s| !s.auto && !s.uses_pressure()));
        assert!(brush.strokes[2].erase);
        assert_eq!(
            back.to_json().trim(),
            VERSION_3_BRUSH.replace("\r\n", "\n").trim()
        );
    }

    const VERSION_3: &str = include_str!("testdata/sidecar_version_3.json");

    #[test]
    fn a_version_3_sidecar_from_before_the_brush_saves_back_unchanged() {
        let back = Sidecar::from_json(VERSION_3).expect("parse");
        assert_eq!(back.version, 3);
        assert_eq!(back.edit.masks.len(), 2);
        assert_eq!(back.edit.masks[1].opacity, 60.0);
        assert_eq!(back.versions[0].edit.masks[1].opacity, 80.0);
        assert!(back.is_dirty());
        assert_eq!(
            back.to_json().trim(),
            VERSION_3.replace("\r\n", "\n").trim()
        );
    }

    #[test]
    fn masks_round_trip_in_the_working_edit_and_in_a_version() {
        let mut sidecar = Sidecar::default();
        let mut mask = Mask::new("Sky", MaskSource::default());
        mask.adjust.exposure = -0.8;
        sidecar.edit.masks.push(mask);
        sidecar.save_version("With sky").expect("save");
        sidecar.edit.masks[0].adjust.exposure = -1.2;
        assert!(sidecar.is_dirty(), "a mask change is a change");
        let back = Sidecar::from_json(&sidecar.to_json()).expect("parse");
        assert_eq!(back, sidecar);
        assert_eq!(back.versions[0].edit.masks[0].adjust.exposure, -0.8);
    }

    #[test]
    fn refine_round_trips_in_the_working_edit_and_a_version_carries_its_own() {
        use crate::mask::Refine;
        let mut sidecar = Sidecar::default();
        let mut mask = Mask::new("Roofs", MaskSource::default());
        mask.adjust.exposure = 0.7;
        mask.refine = Refine {
            amount: 100.0,
            radius: 0.02,
            sensitivity: 70.0,
        };
        sidecar.edit.masks.push(mask);
        sidecar.save_version("Refined").expect("save");
        sidecar.edit.masks[0].refine.amount = 40.0;
        assert!(sidecar.is_dirty(), "a refine change is a change");
        let text = sidecar.to_json();
        let back = Sidecar::from_json(&text).expect("parse");
        assert_eq!(back, sidecar);
        assert_eq!(back.version, 3);
        assert_eq!(back.edit.masks[0].refine.amount, 40.0);
        assert_eq!(back.versions[0].edit.masks[0].refine.amount, 100.0);
        assert_eq!(back.versions[0].edit.masks[0].refine.sensitivity, 70.0);
        assert!(
            !text.contains("history") && !text.contains("undo"),
            "{text}"
        );
    }

    #[test]
    fn a_sidecar_with_an_unknown_mask_source_is_refused_by_name() {
        let text = r#"{"version": 4, "edit": {"masks": [{"name": "Hair",
            "components": [{"source": {"type": "Depth"}}]}]}}"#;
        let message = Sidecar::from_json(text)
            .expect_err("a later format")
            .to_string();
        assert!(message.contains("Depth"), "{message}");
    }

    #[test]
    fn the_sidecar_sits_next_to_the_photo_with_its_full_name() {
        let path = Sidecar::path_for(Path::new("C:/photos/IMG_0001.HEIC"));
        assert_eq!(path, Path::new("C:/photos/IMG_0001.HEIC.gamut.json"));
        assert!(path.to_string_lossy().ends_with(SUFFIX));
        assert_eq!(
            Sidecar::path_for(Path::new("Portrait_8.jpg")),
            Path::new("Portrait_8.jpg.gamut.json")
        );
    }
}
