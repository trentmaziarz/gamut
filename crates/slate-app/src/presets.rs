//! Look presets on disk: one JSON file per preset in a presets folder under
//! the user's config directory. On Windows that is %APPDATA%\slate\presets;
//! elsewhere $XDG_CONFIG_HOME/slate/presets or ~/.config/slate/presets. The
//! folder is read from the environment, so no crate is needed for it.

use std::path::{Path, PathBuf};

use slate_core::LookPreset;

/// Overrides the presets folder, for tests and for a portable setup.
pub const FOLDER_VARIABLE: &str = "SLATE_PRESETS_DIR";

/// The presets folder, or `None` when the environment names no config
/// directory.
pub fn folder() -> Option<PathBuf> {
    let variable = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
    if let Some(folder) = variable(FOLDER_VARIABLE) {
        return Some(PathBuf::from(folder));
    }
    let config = variable("APPDATA")
        .map(PathBuf::from)
        .or_else(|| variable("XDG_CONFIG_HOME").map(PathBuf::from))
        .or_else(|| variable("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(config.join("slate").join("presets"))
}

/// Reads one preset file.
pub fn load(path: &Path) -> Result<LookPreset, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    LookPreset::from_json(&text)
        .map_err(|error| format!("could not parse {}: {error}", path.display()))
}

/// Every preset in `folder`, sorted by name. A file that does not parse is
/// reported in the log and left out. A folder that does not exist yet holds
/// no presets.
pub fn list(folder: &Path) -> Vec<(PathBuf, LookPreset)> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut presets: Vec<(PathBuf, LookPreset)> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        })
        .filter_map(|path| match load(&path) {
            Ok(preset) => Some((path, preset)),
            Err(error) => {
                log::error!("{error}");
                None
            }
        })
        .collect();
    presets.sort_by_key(|(_, preset)| preset.name.to_lowercase());
    presets
}

/// Writes `preset` into `folder` under its slugged name, replacing a preset
/// of the same file name, and returns the path.
pub fn save(folder: &Path, preset: &LookPreset) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(folder)?;
    let path = folder.join(preset.file_name());
    std::fs::write(&path, preset.to_json())?;
    log::info!("saved {}", path.display());
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::{list, load, save};
    use slate_core::preset::Groups;
    use slate_core::{LookPreset, PhotoEdit};

    #[test]
    fn presets_save_list_and_load_from_a_folder() {
        let folder = std::env::temp_dir().join("slate-presets-test");
        let _ = std::fs::remove_dir_all(&folder);
        assert!(list(&folder).is_empty(), "no folder yet, no presets");

        let edit = PhotoEdit {
            clarity: 20.0,
            ..PhotoEdit::default()
        };
        let warm = LookPreset::from_edit("Warm Film", &edit, Groups::ALL);
        let cold = LookPreset::from_edit("cold", &edit, Groups::ALL);
        let path = save(&folder, &warm).expect("save");
        save(&folder, &cold).expect("save");
        std::fs::write(folder.join("broken.json"), "{").expect("write");
        assert_eq!(path.file_name().unwrap(), "warm-film.json");
        assert_eq!(load(&path).expect("load"), warm);

        let names: Vec<String> = list(&folder).into_iter().map(|(_, p)| p.name).collect();
        assert_eq!(names, ["cold", "Warm Film"]);
    }
}
