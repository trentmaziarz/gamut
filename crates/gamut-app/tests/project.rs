//! Builds a project for a copy of a fixture as the app does, cuts it,
//! saves it, loads it back and checks the cut and the relative media path.

use gamut_app::player::MediaInfo;
use gamut_app::project::{for_video, load, probe_media, save};
use gamut_core::Project;
use gamut_media::fixtures;

#[test]
fn a_cut_project_round_trips_next_to_a_copied_clip() {
    let dir = std::env::temp_dir().join("gamut-project-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let video = dir.join("sample-5s.mp4");
    std::fs::copy(fixtures::require("sample-5s.mp4"), &video).expect("copy the fixture");
    let info = MediaInfo::probe(&video).expect("probe the clip");
    assert!((info.duration - 5.76).abs() < 0.1, "{}", info.duration);
    assert_eq!((info.width, info.height), (1920, 1080));

    let mut loaded = for_video(&video, &info);
    assert_eq!(loaded.path, Project::path_for(&video));
    assert_eq!(loaded.path.file_name().unwrap(), "sample-5s.mp4.slate");
    assert_eq!(loaded.project.media[0].path, "sample-5s.mp4");
    assert_eq!(loaded.project.track.clips.len(), 1);

    let second = loaded.project.track.split_at(2.0).expect("split");
    assert_eq!(second, 1);
    loaded
        .project
        .track
        .ripple_delete(0)
        .expect("delete the first");
    let _ = std::fs::remove_file(&loaded.path);
    save(&loaded.path, &loaded.project).expect("save");

    let back = load(&loaded.path).expect("load");
    assert_eq!(back.project, loaded.project);
    assert_eq!(back.project.track.clips.len(), 1);
    let clip = back.project.track.clips[0];
    assert!((clip.source_in - 2.0).abs() < 1e-9, "{clip:?}");
    assert!((clip.source_out - info.duration).abs() < 1e-9, "{clip:?}");
    assert_eq!(back.project.media[0].path, "sample-5s.mp4");
    assert!(!std::path::Path::new(&back.project.media[0].path).is_absolute());
    let media = probe_media(&back).expect("probe the media");
    assert_eq!(media[0].path, video);
    let text = std::fs::read_to_string(&loaded.path).expect("read the file");
    assert!(text.contains("\"path\": \"sample-5s.mp4\""), "{text}");
}
