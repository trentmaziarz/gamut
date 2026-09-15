//! Reading and writing the sidecar next to a photo.

use std::path::{Path, PathBuf};

use slate_core::Sidecar;

/// The sidecar of `photo`, when one exists and parses. A sidecar that does
/// not parse is reported in the log and treated as absent, so a damaged
/// file never blocks the photo from opening.
pub fn load(photo: &Path) -> Option<Sidecar> {
    let path = Sidecar::path_for(photo);
    if !path.is_file() {
        return None;
    }
    match load_from(&path) {
        Ok(sidecar) => Some(sidecar),
        Err(error) => {
            log::error!("{error}");
            None
        }
    }
}

/// Reads a sidecar from an explicit path.
pub fn load_from(path: &Path) -> Result<Sidecar, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    Sidecar::from_json(&text)
        .map_err(|error| format!("could not parse {}: {error}", path.display()))
}

/// Writes the sidecar next to `photo` and returns its path.
pub fn save(photo: &Path, sidecar: &Sidecar) -> std::io::Result<PathBuf> {
    let path = Sidecar::path_for(photo);
    std::fs::write(&path, sidecar.to_json())?;
    log::info!("saved {}", path.display());
    Ok(path)
}
