//! Where the test fixtures live. The photos the tests open are not in the
//! repository; they sit in a GitHub release and in a folder on the machine.

use std::path::PathBuf;

/// The environment variable that names the fixtures folder.
pub const ENV: &str = "SLATE_FIXTURES";

/// The release that holds the fixtures.
pub const RELEASE: &str = "fixtures-m1";

const DEFAULT_DIR: &str = "C:/slate/fixtures";

/// The fixtures folder: `SLATE_FIXTURES` when set, else `C:/slate/fixtures`
/// when it exists, else `./fixtures`.
pub fn dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(ENV) {
        return PathBuf::from(dir);
    }
    let default = PathBuf::from(DEFAULT_DIR);
    if default.is_dir() {
        return default;
    }
    PathBuf::from("fixtures")
}

/// The path of the fixture called `name`. Panics, naming the download
/// command, when the file is missing.
pub fn require(name: &str) -> PathBuf {
    let path = dir().join(name);
    assert!(
        path.is_file(),
        "fixture {name} is missing at {}; download the fixtures with: \
         gh release download {RELEASE} --repo trentmaziarz/slate -p '*' -D {}",
        path.display(),
        dir().display()
    );
    path
}

#[cfg(test)]
mod tests {
    use super::{dir, require};

    #[test]
    fn dir_is_a_folder_name() {
        assert!(!dir().as_os_str().is_empty());
    }

    #[test]
    #[should_panic(expected = "gh release download")]
    fn a_missing_fixture_names_the_download_command() {
        require("no-such-fixture.bin");
    }
}
