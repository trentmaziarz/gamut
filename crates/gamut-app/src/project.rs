//! Reading and writing the .gamut project file, and opening its media.

use std::path::{Path, PathBuf};

use gamut_core::{Clip, Project, Track};

use crate::player::MediaInfo;

/// A project file read from disk, with the folder its media paths are
/// relative to.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedProject {
    pub path: PathBuf,
    pub dir: PathBuf,
    pub project: Project,
}

/// Reads a project from `path`.
pub fn load(path: &Path) -> Result<LoadedProject, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let project = Project::from_json(&text)
        .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
    Ok(LoadedProject {
        path: path.to_path_buf(),
        dir: dir_of(path),
        project,
    })
}

/// Writes `project` to `path`.
pub fn save(path: &Path, project: &Project) -> std::io::Result<()> {
    std::fs::write(path, project.to_json())?;
    log::info!("saved {}", path.display());
    Ok(())
}

/// The folder a project file is in, absolute when the path can be made so.
pub fn dir_of(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    absolute
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// A new project for one video: the whole clip on the track, saved next to
/// the video as `<name>.gamut`.
pub fn for_video(video: &Path, info: &MediaInfo) -> LoadedProject {
    let video = std::path::absolute(video).unwrap_or_else(|_| video.to_path_buf());
    let path = Project::path_for(&video);
    let dir = dir_of(&path);
    let mut project = Project::default();
    let index = project.add_media(&dir, &video);
    project.track = Track::one(Clip::whole(index, info.duration));
    project.crop =
        gamut_core::Crop::fitted(gamut_core::CropAspect::Story9x16, info.width, info.height);
    LoadedProject { path, dir, project }
}

/// Probes every media file of a loaded project, in order.
pub fn probe_media(loaded: &LoadedProject) -> Result<Vec<MediaInfo>, String> {
    (0..loaded.project.media.len())
        .map(|index| {
            let path = loaded
                .project
                .media_path(&loaded.dir, index)
                .expect("index in range");
            MediaInfo::probe(&path).map_err(|error| error.to_string())
        })
        .collect()
}

/// Rewrites the media paths of `project` for a file in `new_dir`.
pub fn rebase(project: &mut Project, old_dir: &Path, new_dir: &Path) {
    for media in &mut project.media {
        let absolute = gamut_core::project::resolve_path(old_dir, &media.path);
        media.path = gamut_core::project::relative_path(new_dir, &absolute);
    }
}
