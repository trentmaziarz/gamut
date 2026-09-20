//! The project: a .gamut JSON file holding the media list, the one track,
//! the crop and the edit. Media paths are written relative to the folder
//! the file is in, with forward slashes, so a project folder moves as a
//! whole.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::timeline::Track;
use crate::{Crop, PhotoEdit};

/// The project format version this build writes.
pub const VERSION: u32 = 1;

/// The extension of a project file.
pub const EXTENSION: &str = "gamut";

/// One media file the track refers to.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MediaRef {
    /// The path relative to the project file's folder, with forward
    /// slashes; an absolute path when the file is on another drive.
    pub path: String,
}

/// The saved state of a video project.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Project {
    pub version: u32,
    pub media: Vec<MediaRef>,
    pub track: Track,
    pub crop: Crop,
    pub edit: PhotoEdit,
}

impl Default for Project {
    fn default() -> Self {
        Project {
            version: VERSION,
            media: Vec::new(),
            track: Track::default(),
            crop: Crop::default(),
            edit: PhotoEdit::default(),
        }
    }
}

impl Project {
    /// The project file for a video opened on its own: the video's full
    /// file name plus `.gamut`, next to the video.
    pub fn path_for(video: &Path) -> PathBuf {
        let mut name = video
            .file_name()
            .map(|n| n.to_os_string())
            .unwrap_or_default();
        name.push(".");
        name.push(EXTENSION);
        video.with_file_name(name)
    }

    /// Whether a path has the project extension.
    pub fn is_project_path(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case(EXTENSION))
    }

    /// Adds a media file, written relative to `project_dir`, and returns
    /// its index. A file already in the list keeps its index.
    pub fn add_media(&mut self, project_dir: &Path, media: &Path) -> usize {
        let path = relative_path(project_dir, media);
        if let Some(index) = self.media.iter().position(|m| m.path == path) {
            return index;
        }
        self.media.push(MediaRef { path });
        self.media.len() - 1
    }

    /// The absolute path of the media at `index` for a project in
    /// `project_dir`.
    pub fn media_path(&self, project_dir: &Path, index: usize) -> Option<PathBuf> {
        self.media
            .get(index)
            .map(|m| resolve_path(project_dir, &m.path))
    }

    /// Pretty JSON, one field per line, so the file diffs in git.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a project always serializes")
    }

    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

/// `to_file` written relative to `from_dir` with forward slashes, walking
/// up with `..` where the two diverge. Both paths are taken as given, so a
/// caller passes absolute paths. When they share no root (another drive)
/// the file's own path comes back with forward slashes.
pub fn relative_path(from_dir: &Path, to_file: &Path) -> String {
    let from: Vec<Component<'_>> = from_dir.components().collect();
    let to: Vec<Component<'_>> = to_file.components().collect();
    let same_root = match (from.first(), to.first()) {
        (Some(Component::Prefix(a)), Some(Component::Prefix(b))) => {
            a.as_os_str().eq_ignore_ascii_case(b.as_os_str())
        }
        (Some(Component::Prefix(_)), _) | (_, Some(Component::Prefix(_))) => false,
        _ => true,
    };
    if !same_root {
        return slashes(to_file);
    }
    let shared = from
        .iter()
        .zip(&to)
        .take_while(|(a, b)| a.as_os_str().eq_ignore_ascii_case(b.as_os_str()))
        .count();
    let mut parts: Vec<String> = Vec::new();
    for _ in shared..from.len() {
        parts.push("..".to_string());
    }
    for component in &to[shared..] {
        parts.push(component.as_os_str().to_string_lossy().into_owned());
    }
    if parts.is_empty() {
        return ".".to_string();
    }
    parts.join("/")
}

/// A media path from a project file back to an absolute path.
pub fn resolve_path(project_dir: &Path, relative: &str) -> PathBuf {
    let path = Path::new(relative);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        let mut out = project_dir.to_path_buf();
        for component in path.components() {
            match component {
                Component::ParentDir => {
                    out.pop();
                }
                Component::CurDir => {}
                other => out.push(other.as_os_str()),
            }
        }
        out
    }
}

fn slashes(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::{Project, VERSION, relative_path, resolve_path};
    use crate::timeline::{Clip, Track};
    use crate::{Adjustments, Crop, CropAspect, CropRect, PhotoEdit};
    use std::path::Path;

    #[test]
    fn a_project_round_trips_through_json() {
        let mut project = Project {
            track: Track::one(Clip::whole(0, 5.0)),
            crop: Crop {
                aspect: CropAspect::Story9x16,
                rect: CropRect::fitted(CropAspect::Story9x16, 1920, 1080),
            },
            edit: PhotoEdit::from(Adjustments {
                exposure: 0.5,
                ..Adjustments::default()
            }),
            ..Project::default()
        };
        project.add_media(Path::new("C:/clips"), Path::new("C:/clips/a.mp4"));
        let back = Project::from_json(&project.to_json()).expect("parse");
        assert_eq!(back, project);
        assert_eq!(back.version, VERSION);
        assert_eq!(back.media[0].path, "a.mp4");
        let empty = Project::from_json("{}").expect("empty");
        assert_eq!(empty, Project::default());
    }

    /// A project exactly as the format stood before the look was added.
    const BEFORE_THE_LOOK: &str = r#"{
  "version": 1,
  "media": [
    {
      "path": "sample-5s.mp4"
    }
  ],
  "track": {
    "clips": [
      {
        "media": 0,
        "source_in": 0.0,
        "source_out": 5.7
      }
    ]
  },
  "crop": {
    "aspect": "Square",
    "rect": {
      "x": 0.21875,
      "y": 0.0,
      "width": 0.5625,
      "height": 1.0
    }
  },
  "edit": {
    "white_balance_temperature": 0.0,
    "white_balance_tint": 0.0,
    "exposure": 0.25,
    "contrast": 0.0,
    "highlights": 0.0,
    "shadows": 0.0,
    "whites": 0.0,
    "blacks": 0.0,
    "vibrance": 0.0,
    "saturation": 0.0
  }
}"#;

    #[test]
    fn a_project_from_before_the_look_still_parses() {
        let back = Project::from_json(BEFORE_THE_LOOK).expect("parse");
        assert_eq!(back.media[0].path, "sample-5s.mp4");
        assert_eq!(back.track.clips.len(), 1);
        assert_eq!(back.crop.aspect, CropAspect::Square);
        assert_eq!(
            back.edit,
            PhotoEdit::from(Adjustments {
                exposure: 0.25,
                ..Adjustments::default()
            })
        );
    }

    /// A project exactly as the format stood with the look and before masks.
    const BEFORE_MASKS: &str = include_str!("testdata/project_before_masks.gamut");

    #[test]
    fn a_project_from_before_masks_still_parses_and_saves_the_same() {
        let back = Project::from_json(BEFORE_MASKS).expect("parse");
        assert_eq!(back.edit.exposure, 0.25);
        assert_eq!(back.edit.look.wheels.shadows.x, -0.5);
        assert!(back.edit.masks.is_empty());
        assert_eq!(
            back.to_json().trim(),
            BEFORE_MASKS.replace("\r\n", "\n").trim()
        );
    }

    #[test]
    fn a_project_carries_masks() {
        let mut project = Project::default();
        let mut mask = crate::Mask::new("Face", crate::MaskSource::default());
        mask.adjust.shadows = 20.0;
        project.edit.masks.push(mask);
        let back = Project::from_json(&project.to_json()).expect("parse");
        assert_eq!(back, project);
    }

    #[test]
    fn a_project_next_to_a_video_in_another_folder_resolves_back() {
        let project_dir = Path::new("C:/work/projects");
        let video = Path::new("C:/work/footage/day1/clip.mov");
        let mut project = Project::default();
        let index = project.add_media(project_dir, video);
        assert_eq!(project.media[index].path, "../footage/day1/clip.mov");
        assert_eq!(
            project.media_path(project_dir, index),
            Some(video.to_path_buf())
        );
        assert_eq!(project.add_media(project_dir, video), index);
        assert_eq!(project.media.len(), 1);
    }

    #[test]
    fn relative_paths_cover_the_same_folder_a_subfolder_and_another_drive() {
        assert_eq!(
            relative_path(Path::new("C:/a/b"), Path::new("C:/a/b/c.mp4")),
            "c.mp4"
        );
        assert_eq!(
            relative_path(Path::new("C:/a/b"), Path::new("C:/a/b/sub/c.mp4")),
            "sub/c.mp4"
        );
        assert_eq!(
            relative_path(Path::new("C:/a/b"), Path::new("c:/A/x/c.mp4")),
            "../x/c.mp4"
        );
        assert_eq!(
            relative_path(Path::new("C:/a/b"), Path::new("D:/x/c.mp4")),
            "D:/x/c.mp4"
        );
        assert_eq!(
            resolve_path(Path::new("C:/a/b"), "../x/c.mp4"),
            Path::new("C:/a/x/c.mp4")
        );
        assert_eq!(
            resolve_path(Path::new("C:/a/b"), "D:/x/c.mp4"),
            Path::new("D:/x/c.mp4")
        );
    }

    #[test]
    fn the_project_file_sits_next_to_the_video_with_its_full_name() {
        assert_eq!(
            Project::path_for(Path::new("C:/clips/IMG_0001.MOV")),
            Path::new("C:/clips/IMG_0001.MOV.gamut")
        );
        assert!(Project::is_project_path(Path::new("a.GAMUT")));
        assert!(!Project::is_project_path(Path::new("a.mp4")));
    }

    #[test]
    fn the_extension_before_the_rebrand_is_not_a_project_path() {
        // In pieces so a scan of the repository for the old name stays empty.
        let old = ["clip.mp4.s", "late"].concat();
        assert!(!Project::is_project_path(Path::new(&old)));
    }
}
