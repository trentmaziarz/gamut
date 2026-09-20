//! Look presets: JSON subsets of the photo parameters. A preset carries only
//! the groups it was saved with, and applying it overwrites only those, so a
//! grading preset leaves the exposure of the photo alone. A preset never
//! holds a crop. The export sizes are [`crate::ExportPreset`], another thing.

use serde::{Deserialize, Serialize};

use crate::PhotoEdit;
use crate::look::{HSL_RANGES, HslRange, ToneCurves, Wheels};

/// The look preset format version this build writes.
pub const VERSION: u32 = 1;

/// The groups of the save dialog: one checkbox each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Groups {
    /// The ten Basic sliders.
    pub basic: bool,
    /// Texture, clarity and dehaze.
    pub presence: bool,
    /// The four tone curves.
    pub curve: bool,
    /// The HSL mixer.
    pub mixer: bool,
    /// The three colour wheels.
    pub grading: bool,
}

impl Groups {
    pub const ALL: Groups = Groups {
        basic: true,
        presence: true,
        curve: true,
        mixer: true,
        grading: true,
    };

    pub const NONE: Groups = Groups {
        basic: false,
        presence: false,
        curve: false,
        mixer: false,
        grading: false,
    };

    pub fn any(&self) -> bool {
        *self != Groups::NONE
    }
}

impl Default for Groups {
    fn default() -> Self {
        Groups::ALL
    }
}

/// `PhotoEdit` with every field optional. The curves, the mixer and the
/// wheels are present or absent as whole groups. An absent field is left out
/// of the JSON.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PartialEdit {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub white_balance_temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub white_balance_tint: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contrast: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highlights: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadows: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub whites: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blacks: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vibrance: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saturation: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub texture: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clarity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dehaze: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curves: Option<ToneCurves>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hsl: Option<[HslRange; HSL_RANGES]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wheels: Option<Wheels>,
}

impl PartialEdit {
    /// The part of `edit` that `groups` names.
    pub fn from_edit(edit: &PhotoEdit, groups: Groups) -> Self {
        let mut partial = PartialEdit::default();
        if groups.basic {
            partial.white_balance_temperature = Some(edit.white_balance_temperature);
            partial.white_balance_tint = Some(edit.white_balance_tint);
            partial.exposure = Some(edit.exposure);
            partial.contrast = Some(edit.contrast);
            partial.highlights = Some(edit.highlights);
            partial.shadows = Some(edit.shadows);
            partial.whites = Some(edit.whites);
            partial.blacks = Some(edit.blacks);
            partial.vibrance = Some(edit.vibrance);
            partial.saturation = Some(edit.saturation);
        }
        if groups.presence {
            partial.texture = Some(edit.texture);
            partial.clarity = Some(edit.clarity);
            partial.dehaze = Some(edit.dehaze);
        }
        if groups.curve {
            partial.curves = Some(edit.look.curves.clone());
        }
        if groups.mixer {
            partial.hsl = Some(edit.look.hsl);
        }
        if groups.grading {
            partial.wheels = Some(edit.look.wheels);
        }
        partial
    }

    /// Overwrites the fields of `edit` that are present here and leaves the
    /// rest as they are.
    pub fn apply(&self, edit: &mut PhotoEdit) {
        let scalars = [
            (
                self.white_balance_temperature,
                &mut edit.white_balance_temperature,
            ),
            (self.white_balance_tint, &mut edit.white_balance_tint),
            (self.exposure, &mut edit.exposure),
            (self.contrast, &mut edit.contrast),
            (self.highlights, &mut edit.highlights),
            (self.shadows, &mut edit.shadows),
            (self.whites, &mut edit.whites),
            (self.blacks, &mut edit.blacks),
            (self.vibrance, &mut edit.vibrance),
            (self.saturation, &mut edit.saturation),
            (self.texture, &mut edit.texture),
            (self.clarity, &mut edit.clarity),
            (self.dehaze, &mut edit.dehaze),
        ];
        for (value, slot) in scalars {
            if let Some(value) = value {
                *slot = value;
            }
        }
        if let Some(curves) = &self.curves {
            edit.look.curves = curves.clone();
        }
        if let Some(hsl) = self.hsl {
            edit.look.hsl = hsl;
        }
        if let Some(wheels) = self.wheels {
            edit.look.wheels = wheels;
        }
    }

    /// The groups this holds anything of.
    pub fn groups(&self) -> Groups {
        Groups {
            basic: self.white_balance_temperature.is_some()
                || self.white_balance_tint.is_some()
                || self.exposure.is_some()
                || self.contrast.is_some()
                || self.highlights.is_some()
                || self.shadows.is_some()
                || self.whites.is_some()
                || self.blacks.is_some()
                || self.vibrance.is_some()
                || self.saturation.is_some(),
            presence: self.texture.is_some() || self.clarity.is_some() || self.dehaze.is_some(),
            curve: self.curves.is_some(),
            mixer: self.hsl.is_some(),
            grading: self.wheels.is_some(),
        }
    }
}

/// A saved look: a name and a subset of the photo parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LookPreset {
    pub version: u32,
    pub name: String,
    pub edit: PartialEdit,
}

impl Default for LookPreset {
    fn default() -> Self {
        LookPreset {
            version: VERSION,
            name: String::new(),
            edit: PartialEdit::default(),
        }
    }
}

impl LookPreset {
    /// A preset of the named groups of `edit`.
    pub fn from_edit(name: &str, edit: &PhotoEdit, groups: Groups) -> Self {
        LookPreset {
            version: VERSION,
            name: name.trim().to_string(),
            edit: PartialEdit::from_edit(edit, groups),
        }
    }

    /// Overwrites the present fields of `edit`. There is no crop to touch.
    pub fn apply(&self, edit: &mut PhotoEdit) {
        self.edit.apply(edit);
    }

    /// The file name of this preset: its name in lower case with every run
    /// of other characters as one hyphen, plus `.json`.
    pub fn file_name(&self) -> String {
        format!("{}.json", slug(&self.name))
    }

    /// Pretty JSON, one field per line, so the file diffs in git.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a preset always serializes")
    }

    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

/// A name as a file stem: letters and digits in lower case, every run of
/// anything else as one hyphen, none at the ends. An empty result is
/// `preset`.
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_matches('-');
    if out.is_empty() {
        "preset".to_string()
    } else {
        out.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{Groups, LookPreset, PartialEdit, slug};
    use crate::PhotoEdit;
    use crate::look::{Curve, Wheel};

    fn busy_edit() -> PhotoEdit {
        let mut edit = PhotoEdit {
            white_balance_temperature: 8.0,
            exposure: 1.0,
            contrast: 20.0,
            saturation: -5.0,
            texture: 10.0,
            clarity: 30.0,
            dehaze: -15.0,
            ..PhotoEdit::default()
        };
        edit.look.curves.master = Curve {
            points: vec![[0.0, 0.0], [0.3, 0.2], [1.0, 1.0]],
        };
        edit.look.hsl[1].saturation = -60.0;
        edit.look.hsl[5].hue = 25.0;
        edit.look.wheels.shadows = Wheel {
            x: -0.3,
            y: -0.2,
            luminance: 5.0,
        };
        edit
    }

    #[test]
    fn a_mixer_preset_leaves_every_other_field_untouched() {
        let mut source = PhotoEdit::default();
        source.look.hsl[3].luminance = 40.0;
        source.exposure = -2.0;
        let preset = LookPreset::from_edit(
            "Greens up",
            &source,
            Groups {
                mixer: true,
                ..Groups::NONE
            },
        );
        let mut edit = busy_edit();
        preset.apply(&mut edit);
        let mut expected = busy_edit();
        expected.look.hsl = source.look.hsl;
        assert_eq!(edit, expected);
        assert_eq!(edit.exposure, 1.0);
    }

    #[test]
    fn a_preset_round_trips_and_omits_absent_groups_from_the_text() {
        let preset = LookPreset::from_edit(
            "Teal and orange",
            &busy_edit(),
            Groups {
                curve: true,
                grading: true,
                ..Groups::NONE
            },
        );
        let text = preset.to_json();
        assert_eq!(LookPreset::from_json(&text).expect("parse"), preset);
        assert!(text.contains("\"curves\"") && text.contains("\"wheels\""));
        for absent in ["exposure", "contrast", "texture", "clarity", "hsl", "crop"] {
            assert!(!text.contains(absent), "{absent} is in {text}");
        }
        assert_eq!(
            preset.edit.groups(),
            Groups {
                curve: true,
                grading: true,
                ..Groups::NONE
            }
        );
    }

    #[test]
    fn every_group_carries_the_whole_edit() {
        let edit = busy_edit();
        let mut applied = PhotoEdit::default();
        PartialEdit::from_edit(&edit, Groups::ALL).apply(&mut applied);
        assert_eq!(applied, edit);
        let mut untouched = busy_edit();
        PartialEdit::from_edit(&edit, Groups::NONE).apply(&mut untouched);
        assert_eq!(untouched, busy_edit());
        assert!(!Groups::NONE.any() && Groups::ALL.any());
    }

    #[test]
    fn a_hand_written_preset_with_one_field_parses() {
        let preset =
            LookPreset::from_json(r#"{"name": "Warm", "edit": {"white_balance_temperature": 25}}"#)
                .expect("parse");
        let mut edit = busy_edit();
        preset.apply(&mut edit);
        assert_eq!(edit.white_balance_temperature, 25.0);
        assert_eq!(edit.exposure, 1.0);
    }

    #[test]
    fn names_become_file_names() {
        assert_eq!(slug("Teal & Orange"), "teal-orange");
        assert_eq!(slug("  Film 400  "), "film-400");
        assert_eq!(slug("../../etc"), "etc");
        assert_eq!(slug("???"), "preset");
        let preset = LookPreset::from_edit("My Look", &PhotoEdit::default(), Groups::ALL);
        assert_eq!(preset.file_name(), "my-look.json");
    }
}
