//! The export presets: the four Instagram sizes a photo leaves slate at.

use serde::{Deserialize, Serialize};

use crate::CropAspect;

/// One of the four export sizes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportPreset {
    /// The 4:5 feed post, 1080 by 1350.
    #[default]
    Feed4x5,
    /// The square post, 1080 by 1080.
    Square,
    /// The 3:4 profile grid, 1080 by 1440.
    Grid3x4,
    /// The 9:16 story and reel, 1080 by 1920.
    Story9x16,
}

impl ExportPreset {
    pub const ALL: [ExportPreset; 4] = [
        ExportPreset::Feed4x5,
        ExportPreset::Square,
        ExportPreset::Grid3x4,
        ExportPreset::Story9x16,
    ];

    /// Width and height in pixels.
    pub fn size(self) -> (u32, u32) {
        match self {
            ExportPreset::Feed4x5 => (1080, 1350),
            ExportPreset::Square => (1080, 1080),
            ExportPreset::Grid3x4 => (1080, 1440),
            ExportPreset::Story9x16 => (1080, 1920),
        }
    }

    /// The crop aspect this preset exports.
    pub fn aspect(self) -> CropAspect {
        match self {
            ExportPreset::Feed4x5 => CropAspect::Feed4x5,
            ExportPreset::Square => CropAspect::Square,
            ExportPreset::Grid3x4 => CropAspect::Grid3x4,
            ExportPreset::Story9x16 => CropAspect::Story9x16,
        }
    }

    /// The preset that matches a crop aspect.
    pub fn for_aspect(aspect: CropAspect) -> ExportPreset {
        match aspect {
            CropAspect::Feed4x5 => ExportPreset::Feed4x5,
            CropAspect::Square => ExportPreset::Square,
            CropAspect::Grid3x4 => ExportPreset::Grid3x4,
            CropAspect::Story9x16 => ExportPreset::Story9x16,
        }
    }

    /// The short name on the command line and in file names.
    pub fn name(self) -> &'static str {
        match self {
            ExportPreset::Feed4x5 => "4x5",
            ExportPreset::Square => "1x1",
            ExportPreset::Grid3x4 => "3x4",
            ExportPreset::Story9x16 => "9x16",
        }
    }

    /// The name in the export dialog.
    pub fn label(self) -> &'static str {
        match self {
            ExportPreset::Feed4x5 => "Feed post 4:5, 1080 by 1350",
            ExportPreset::Square => "Square 1:1, 1080 by 1080",
            ExportPreset::Grid3x4 => "Profile grid 3:4, 1080 by 1440",
            ExportPreset::Story9x16 => "Story and reel 9:16, 1080 by 1920",
        }
    }

    /// Reads a preset from its name, its ratio or its use, in any case.
    pub fn parse(text: &str) -> Option<ExportPreset> {
        match text.to_ascii_lowercase().as_str() {
            "4x5" | "4:5" | "feed" => Some(ExportPreset::Feed4x5),
            "1x1" | "1:1" | "square" => Some(ExportPreset::Square),
            "3x4" | "3:4" | "grid" => Some(ExportPreset::Grid3x4),
            "9x16" | "9:16" | "story" | "reel" => Some(ExportPreset::Story9x16),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExportPreset;
    use crate::CropAspect;

    #[test]
    fn the_four_sizes() {
        assert_eq!(ExportPreset::Feed4x5.size(), (1080, 1350));
        assert_eq!(ExportPreset::Square.size(), (1080, 1080));
        assert_eq!(ExportPreset::Grid3x4.size(), (1080, 1440));
        assert_eq!(ExportPreset::Story9x16.size(), (1080, 1920));
    }

    #[test]
    fn every_size_has_the_aspect_of_its_crop() {
        for preset in ExportPreset::ALL {
            let (w, h) = preset.size();
            let ratio = w as f32 / h as f32;
            assert!((ratio - preset.aspect().ratio()).abs() < 1e-6, "{preset:?}");
            assert_eq!(ExportPreset::for_aspect(preset.aspect()), preset);
        }
        assert_eq!(
            ExportPreset::for_aspect(CropAspect::Story9x16),
            ExportPreset::Story9x16
        );
    }

    #[test]
    fn names_parse_back() {
        for preset in ExportPreset::ALL {
            assert_eq!(ExportPreset::parse(preset.name()), Some(preset));
        }
        assert_eq!(ExportPreset::parse("STORY"), Some(ExportPreset::Story9x16));
        assert_eq!(ExportPreset::parse("4:5"), Some(ExportPreset::Feed4x5));
        assert_eq!(ExportPreset::parse("16x9"), None);
    }
}
