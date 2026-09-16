//! The Reel export: runs the timeline through the video pass and the
//! develop graph at 1080x1920 and hands each frame to the writer. Lands
//! with the export phase; until then the entry points report that.

use std::path::Path;

use slate_core::Project;

use crate::headless::HeadlessError;

/// Exports `project`, whose media paths are relative to `dir`, to `out`.
/// Returns the wall time in seconds.
pub fn export(_project: &Project, _dir: &Path, _out: &Path) -> Result<f64, HeadlessError> {
    Err(HeadlessError::Edit(
        "the Reel export is not built yet".to_string(),
    ))
}

/// Exports a video or a project file to `out`.
pub fn export_file(_input: &Path, _out: &Path) -> Result<f64, HeadlessError> {
    Err(HeadlessError::Edit(
        "the Reel export is not built yet".to_string(),
    ))
}
