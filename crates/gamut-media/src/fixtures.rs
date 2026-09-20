//! Where the test fixtures live. The photos and clips the tests open are
//! not in the repository; they sit in two GitHub releases and in a folder
//! on the machine. Two video files are local only: an excerpt of private
//! footage and a generated timing clip, each made by a command this module
//! prints when the file is missing.

use std::path::PathBuf;

/// The environment variable that names the fixtures folder.
pub const ENV: &str = "GAMUT_FIXTURES";

/// The release that holds the M1 photo fixtures.
pub const RELEASE: &str = "fixtures-m1";

/// The release that holds the public M2 video fixtures.
pub const RELEASE_M2: &str = "fixtures-m2";

/// The fixtures that are in no release and are made on the machine.
pub const LOCAL_ONLY: [&str; 2] = ["iphone_hevc_10s.mov", "timing_4k30_60s.mp4"];

const DEFAULT_DIR: &str = "C:/gamut/fixtures";

/// The fixtures folder: `GAMUT_FIXTURES` when set, else `C:/gamut/fixtures`
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
         gh release download {RELEASE} --repo trentmaziarz/slate -p '*' -D {dir} \
         and gh release download {RELEASE_M2} --repo trentmaziarz/slate -p '*' -D {dir}",
        path.display(),
        dir = dir().display()
    );
    path
}

/// The path of a local-only fixture when it exists. `None` means the test
/// should skip with a printed line; a test that must have the file calls
/// [`require_local`] instead.
pub fn local(name: &str) -> Option<PathBuf> {
    let path = dir().join(name);
    path.is_file().then_some(path)
}

/// The path of a local-only fixture. Panics, naming the cut or generation
/// command, when the file is missing.
pub fn require_local(name: &str) -> PathBuf {
    let path = dir().join(name);
    assert!(
        path.is_file(),
        "local fixture {name} is missing at {}; it is in no release; make it with: {}",
        path.display(),
        local_command(name)
    );
    path
}

/// The command that makes a local-only fixture, looked up by the file
/// name part of `name`.
pub fn local_command(name: &str) -> String {
    let file = std::path::Path::new(name)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    match file.as_str() {
        "iphone_hevc_10s.mov" => "ffmpeg -ss 60 -t 10 -i <iphone source.mov> -map_metadata 0 \
             -c copy -movflags +faststart+use_metadata_tags iphone_hevc_10s.mov"
            .to_string(),
        "timing_4k30_60s.mp4" => "ffmpeg -f lavfi -i testsrc2=size=3840x2160:rate=30 \
             -f lavfi -i sine=frequency=440:sample_rate=48000 -t 60 -c:v hevc_nvenc -b:v 20M \
             -pix_fmt yuv420p -c:a aac -b:a 128k timing_4k30_60s.mp4"
            .to_string(),
        other => format!("(no command is known for {other})"),
    }
}

#[cfg(test)]
mod tests {
    use super::{LOCAL_ONLY, dir, local_command, require, require_local};

    #[test]
    fn dir_is_a_folder_name() {
        assert!(!dir().as_os_str().is_empty());
    }

    #[test]
    #[should_panic(expected = "gh release download")]
    fn a_missing_fixture_names_the_download_command() {
        require("no-such-fixture.bin");
    }

    #[test]
    #[should_panic(expected = "hevc_nvenc")]
    fn a_missing_local_fixture_names_the_generation_command() {
        // A folder that does not exist keeps the panic honest on a
        // machine that has the file at its usual place.
        assert!(local_command(LOCAL_ONLY[1]).contains("testsrc2"));
        require_local("missing-dir-that-does-not-exist/timing_4k30_60s.mp4");
    }

    #[test]
    fn every_local_fixture_has_a_command() {
        for name in LOCAL_ONLY {
            assert!(local_command(name).starts_with("ffmpeg "), "{name}");
        }
    }
}
