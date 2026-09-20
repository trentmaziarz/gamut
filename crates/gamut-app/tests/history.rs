//! Undo and redo on the session, walked the way the window walks them: the
//! tabs change the session, and once a frame the history is shown the state
//! with what the window knows about a change being under way. Every step a
//! person takes at the window that is data logic is a test here, so the
//! window is left for feel.

use std::time::{Duration, Instant};

use gamut_app::app::{OpenPhoto, OpenProject, Session, SwitchAnswer};
use gamut_app::player::{MediaInfo, Player};
use gamut_app::project;
use gamut_app::sidecar::load;
use gamut_app::undo::{KEY_SETTLE, Settle, Snapshot};
use gamut_core::look::Wheel;
use gamut_core::mask::{LinearGradient, RadialGradient};
use gamut_core::preset::Groups;
use gamut_core::{Crop, CropAspect, LookPreset, Mask, MaskSource, Sidecar};
use gamut_media::fixtures;

/// A frame in which nothing is held: whatever changed is a step.
fn settle(session: &mut Session) {
    session.observe_history(Settle::default(), Instant::now());
}

/// A frame in the middle of a drag.
fn drag_frame(session: &mut Session) {
    let dragging = Settle {
        pointer_busy: true,
        ..Settle::default()
    };
    session.observe_history(dragging, Instant::now());
}

/// A session as a freshly opened photo leaves it.
fn opened() -> Session {
    let mut session = Session::default();
    session.begin_history();
    session
}

fn undo_steps(session: &Session) -> usize {
    session.history.as_ref().map_or(0, |h| h.undo_steps())
}

/// Makes one change, lets it settle, and holds that an undo gives the state
/// before it back and a redo the state after it, both equal in everything
/// the file saves.
fn round_trip(session: &mut Session, name: &str, change: impl FnOnce(&mut Session)) {
    settle(session);
    let before = session.snapshot();
    let steps = undo_steps(session);
    change(session);
    settle(session);
    let after = session.snapshot();
    assert_ne!(before, after, "{name}: the change changed nothing");
    assert_eq!(undo_steps(session), steps + 1, "{name}: one step");
    assert!(session.undo(), "{name}: undo");
    assert_eq!(session.snapshot(), before, "{name}: undone");
    assert!(session.changed_at.is_some(), "{name}: the file is stale");
    assert!(session.develop_dirty, "{name}: the picture is stale");
    assert!(session.redo(), "{name}: redo");
    assert_eq!(session.snapshot(), after, "{name}: redone");
    // The walk itself is no new step.
    settle(session);
    assert_eq!(undo_steps(session), steps + 1, "{name}: still one step");
}

#[test]
fn a_slider_drag_of_many_frames_is_one_undo() {
    let mut session = opened();
    for frame in 1..=60 {
        session.edit.exposure = frame as f32 / 40.0;
        session.mark_edited();
        drag_frame(&mut session);
        assert_eq!(undo_steps(&session), 0, "no step while the pointer is down");
    }
    // The pointer lets go.
    settle(&mut session);
    assert_eq!(undo_steps(&session), 1);
    assert!(session.undo());
    assert_eq!(session.edit.exposure, 0.0, "the whole drag in one undo");
    assert!(!session.undo(), "and nothing before it");
    assert!(session.redo());
    assert_eq!(session.edit.exposure, 1.5);
}

#[test]
fn a_drag_that_ends_where_it_began_is_no_step() {
    let mut session = opened();
    session.edit.contrast = 30.0;
    drag_frame(&mut session);
    session.edit.contrast = 0.0;
    drag_frame(&mut session);
    settle(&mut session);
    assert_eq!(undo_steps(&session), 0);
    assert!(!session.can_undo());
}

#[test]
fn a_run_of_arrow_keys_on_a_slider_is_one_step_after_it_rests() {
    let mut session = opened();
    let start = Instant::now();
    let key = Settle {
        arrow_key: true,
        ..Settle::default()
    };
    let mut at = start;
    for press in 1..=5 {
        at = start + Duration::from_millis(120 * press);
        session.edit.shadows = press as f32;
        let wait = session.observe_history(key, at);
        assert_eq!(wait, Some(KEY_SETTLE), "the window is asked to come back");
        assert_eq!(undo_steps(&session), 0);
    }
    // The key is up; 399 ms after the last press nothing has settled.
    let wait = session.observe_history(Settle::default(), at + Duration::from_millis(399));
    assert_eq!(wait, Some(Duration::from_millis(1)));
    assert_eq!(undo_steps(&session), 0);
    assert_eq!(
        session.observe_history(Settle::default(), at + KEY_SETTLE),
        None
    );
    assert_eq!(undo_steps(&session), 1);
    assert!(session.undo());
    assert_eq!(session.edit.shadows, 0.0);
}

#[test]
fn every_kind_of_edit_undoes_and_redoes_to_an_equal_state() {
    let mut session = opened();
    round_trip(&mut session, "a slider", |s| s.edit.exposure = 0.7);
    round_trip(&mut session, "a curve point", |s| {
        s.edit.look.curves.master.points = vec![[0.0, 0.0], [0.3, 0.2], [1.0, 1.0]];
    });
    round_trip(&mut session, "a curve point moved", |s| {
        s.edit.look.curves.master.points[1] = [0.35, 0.18];
    });
    round_trip(&mut session, "a wheel move", |s| {
        s.edit.look.wheels.shadows = Wheel {
            x: -0.3,
            y: 0.2,
            luminance: 5.0,
        };
    });
    round_trip(&mut session, "a mixer slider", |s| {
        s.edit.look.hsl[4].saturation = -35.0;
    });
    round_trip(&mut session, "a mask added", |s| {
        s.edit.masks.push(Mask::new("Sky", MaskSource::default()));
    });
    round_trip(&mut session, "a second mask added", |s| {
        let radial = MaskSource::Radial(RadialGradient::default());
        s.edit.masks.push(Mask::new("Face", radial));
    });
    round_trip(&mut session, "a mask handle moved", |s| {
        s.edit.masks[0].components[0].source = MaskSource::Linear(LinearGradient {
            start: [0.2, 0.9],
            end: [0.25, 0.1],
        });
    });
    round_trip(&mut session, "a mask slider", |s| {
        s.edit.masks[1].adjust.exposure = -0.8;
        s.edit.masks[1].opacity = 60.0;
    });
    round_trip(&mut session, "the masks reordered", |s| {
        s.edit.masks.swap(0, 1)
    });
    round_trip(&mut session, "a mask deleted", |s| {
        s.edit.masks.remove(0);
    });
    round_trip(&mut session, "the crop aspect", |s| {
        s.crop = Crop::fitted(CropAspect::Square, 6000, 4000);
    });
    round_trip(&mut session, "the crop moved", |s| {
        s.crop.rect = s.crop.rect.moved(0.07, 0.0);
    });
    round_trip(&mut session, "a preset applied", |s| {
        let mut look = gamut_core::Adjustments {
            contrast: 40.0,
            clarity: 25.0,
            ..Default::default()
        };
        look.look.wheels.highlights.x = 0.2;
        let preset = LookPreset::from_edit(
            "Punch",
            &look,
            Groups {
                basic: true,
                presence: true,
                grading: true,
                ..Groups::default()
            },
        );
        preset.apply(&mut s.edit.adjust);
    });
    round_trip(&mut session, "a version saved", |s| {
        s.with_versions(|sidecar| sidecar.save_version("First"))
            .expect("save");
    });
    round_trip(&mut session, "a second version saved", |s| {
        s.edit.exposure = -1.0;
        s.with_versions(|sidecar| {
            sidecar.active_version = None;
            sidecar.save_version("Second")
        })
        .expect("save");
    });
    round_trip(&mut session, "a version switch", |s| {
        s.request_switch("First").expect("a clean switch");
        assert_eq!(s.adjust.pending_switch, None);
    });
    round_trip(&mut session, "a version update", |s| {
        s.edit.vibrance = 45.0;
        s.update_version("First").expect("update");
    });
    round_trip(&mut session, "a version renamed", |s| {
        s.with_versions(|sidecar| sidecar.rename_version("Second", "Dark"))
            .expect("rename");
    });
    round_trip(&mut session, "a version deleted", |s| {
        s.with_versions(|sidecar| sidecar.delete_version("Dark"))
            .expect("delete");
    });
    round_trip(&mut session, "reset all", |s| s.edit = Default::default());
    // All of it walks back to the file as it was opened.
    while session.undo() {}
    assert_eq!(session.snapshot(), opened().snapshot());
}

#[test]
fn a_switch_answered_with_save_is_one_step_that_undoes_the_version_too() {
    let mut session = opened();
    session
        .with_versions(|sidecar| sidecar.save_version("A"))
        .expect("save");
    session.edit.exposure = 1.0;
    session
        .with_versions(|sidecar| {
            sidecar.active_version = None;
            sidecar.save_version("B")
        })
        .expect("save");
    session.edit.contrast = 22.0;
    settle(&mut session);
    let before = session.snapshot();
    session.request_switch("A").expect("the prompt is raised");
    assert!(session.adjust.pending_switch.is_some());
    session.answer_switch(SwitchAnswer::Save).expect("save");
    settle(&mut session);
    assert_eq!(
        session.sidecar().version("B").expect("B").edit.contrast,
        22.0
    );
    assert!(session.undo());
    assert_eq!(session.snapshot(), before);
    assert_eq!(
        session.sidecar().version("B").expect("B").edit.contrast,
        0.0,
        "what Save wrote into the version is undone with the switch"
    );
    assert_eq!(session.edit.contrast, 22.0);
}

#[test]
fn undo_is_refused_while_the_switch_prompt_is_up() {
    let mut session = opened();
    session
        .with_versions(|sidecar| sidecar.save_version("A"))
        .expect("save");
    session
        .with_versions(|sidecar| {
            sidecar.active_version = None;
            sidecar.save_version("B")
        })
        .expect("save");
    session.edit.exposure = 0.5;
    settle(&mut session);
    assert!(session.can_undo());
    session.request_switch("A").expect("the prompt is raised");
    assert!(session.adjust.pending_switch.is_some());
    let shown = session.snapshot();
    assert!(!session.can_undo() && !session.can_redo());
    assert!(!session.undo(), "refused under the prompt");
    assert!(!session.redo());
    assert_eq!(session.snapshot(), shown);
    assert!(session.adjust.pending_switch.is_some(), "the prompt stays");
    session.answer_switch(SwitchAnswer::Cancel).expect("cancel");
    assert!(session.undo(), "and works again once it is answered");
    assert_eq!(session.edit.exposure, 0.0);
}

#[test]
fn taking_a_state_back_lets_go_of_what_belonged_to_the_one_that_is_gone() {
    let mut session = opened();
    session.edit.exposure = 0.3;
    settle(&mut session);
    session
        .edit
        .masks
        .push(Mask::new("Range", MaskSource::default()));
    settle(&mut session);
    session.adjust.select_mask(Some(0));
    session.adjust.picking = Some(0);
    session.adjust.mask_renaming = Some((0, "Ran".to_string()));
    session.adjust.renaming = Some(("A".to_string(), "B".to_string()));
    session.develop_dirty = false;
    session.changed_at = None;
    assert!(session.undo());
    assert!(session.edit.masks.is_empty());
    assert_eq!(session.adjust.selected_mask, None, "the mask is gone");
    assert_eq!(session.adjust.picking, None);
    assert_eq!(session.adjust.mask_renaming, None);
    assert_eq!(session.adjust.renaming, None);
    assert!(session.develop_dirty && session.changed_at.is_some());
    assert_eq!(
        session.status.as_deref(),
        Some("Undo: 1 more to undo, 1 to redo.")
    );
    // A selection that still exists is kept: the view is not history.
    assert!(session.redo());
    session.adjust.select_mask(Some(0));
    session.edit.masks[0].opacity = 40.0;
    settle(&mut session);
    assert!(session.undo());
    assert_eq!(session.adjust.selected_mask, Some(0));
    assert_eq!(
        session.status.as_deref(),
        Some("Undo: 2 more to undo, 1 to redo.")
    );
}

#[test]
fn an_undo_before_a_change_settled_loses_nothing() {
    let mut session = opened();
    session.edit.exposure = 0.4;
    settle(&mut session);
    // Ctrl+Z inside the 400 ms after an arrow key: the change is pending.
    session.edit.exposure = 0.5;
    let key = Settle {
        arrow_key: true,
        ..Settle::default()
    };
    session.observe_history(key, Instant::now());
    assert!(session.can_undo());
    assert!(session.undo());
    assert_eq!(session.edit.exposure, 0.4);
    assert!(session.redo());
    assert_eq!(session.edit.exposure, 0.5);
}

#[test]
fn a_new_file_starts_with_an_empty_history() {
    let mut session = opened();
    session.edit.exposure = 1.0;
    settle(&mut session);
    session.edit.contrast = 10.0;
    settle(&mut session);
    assert!(session.undo());
    assert!(session.can_undo() && session.can_redo());
    // Another file opens: what it holds is the beginning.
    session.take_sidecar(Sidecar::default());
    session.begin_history();
    assert!(!session.can_undo() && !session.can_redo());
    assert!(!session.undo() && !session.redo());
    assert_eq!(undo_steps(&session), 0);
    // A session nobody began starts at its first frame.
    let mut bare = Session::default();
    assert!(bare.history.is_none());
    settle(&mut bare);
    assert!(!bare.can_undo());
    bare.edit.exposure = 0.2;
    settle(&mut bare);
    assert!(bare.undo());
    assert_eq!(bare.edit.exposure, 0.0);
}

#[test]
fn the_view_and_the_selection_are_not_history() {
    let mut session = opened();
    session.view = gamut_app::view::View {
        zoom: gamut_app::view::Zoom::Scale(2.0),
        centre: [0.2, 0.8],
    };
    session.adjust.mask_overlay = true;
    session.grid_guide = true;
    settle(&mut session);
    assert_eq!(undo_steps(&session), 0);
    assert!(!session.can_undo());
    session.edit.exposure = 0.3;
    settle(&mut session);
    let view = session.view;
    assert!(session.undo());
    assert_eq!(session.view, view, "an undo leaves the view alone");
}

#[test]
fn the_saved_sidecar_equals_what_is_shown_after_an_undo_and_after_a_redo() {
    let dir = std::env::temp_dir().join("gamut-history-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let photo = dir.join("undo.jpg");
    let _ = std::fs::remove_file(Sidecar::path_for(&photo));
    let mut session = Session {
        photo: Some(OpenPhoto {
            path: photo.clone(),
            width: 6000,
            height: 4000,
        }),
        ..Session::default()
    };
    session.begin_history();
    session.edit.exposure = 0.9;
    session
        .edit
        .masks
        .push(Mask::new("Sky", MaskSource::default()));
    session.mark_edited();
    settle(&mut session);
    session
        .with_versions(|sidecar| sidecar.save_version("Kept"))
        .expect("save");
    settle(&mut session);
    session.edit.clarity = 33.0;
    session.mark_edited();
    settle(&mut session);
    session.save_now();
    assert_eq!(load(&photo).expect("saved"), session.sidecar());

    let on_disk = |session: &mut Session| -> Sidecar {
        // The window writes the file half a second after the last change,
        // and only when one is pending.
        assert!(session.changed_at.is_some(), "the file is stale");
        session.save_now();
        let text = std::fs::read_to_string(Sidecar::path_for(&photo)).expect("read");
        assert!(text.contains("\"version\": 3"), "{text}");
        assert!(
            !text.contains("history") && !text.contains("undo"),
            "{text}"
        );
        load(&photo).expect("the sidecar loads")
    };
    assert!(session.undo());
    assert_eq!(session.edit.clarity, 0.0);
    assert_eq!(on_disk(&mut session), session.sidecar());
    assert!(session.undo(), "the version goes");
    assert!(session.versions.is_empty());
    assert_eq!(on_disk(&mut session), session.sidecar());
    assert!(session.redo());
    assert!(session.redo());
    let saved = on_disk(&mut session);
    assert_eq!(saved, session.sidecar());
    assert_eq!(saved.edit.clarity, 33.0);
    assert_eq!(saved.versions.len(), 1);
    assert_eq!(saved.edit.masks.len(), 1);
    // A reload starts from that state with no history.
    let mut fresh = Session::default();
    fresh.take_sidecar(saved);
    fresh.begin_history();
    assert_eq!(fresh.sidecar(), session.sidecar());
    assert!(!fresh.can_undo());
}

/// A project on the five second sample clip, as the window opens it.
fn open_sample() -> Session {
    let path = fixtures::require("sample-5s.mp4");
    let info = MediaInfo::probe(&path).expect("probe the clip");
    let loaded = project::for_video(&path, &info);
    let media = project::probe_media(&loaded).expect("probe the media");
    let track = loaded.project.track.clone();
    let player = Player::new(media.clone(), track.clone());
    let mut session = Session {
        project: Some(OpenProject {
            // Never saved: the tests write nothing next to the fixture.
            path: std::env::temp_dir().join("gamut-history-test.gamut"),
            dir: loaded.dir,
            media,
            project: loaded.project,
            track,
            player,
            selected: None,
            track_dirty: false,
        }),
        ..Session::default()
    };
    session.begin_history();
    session
}

fn opened_project(session: &mut Session) -> &mut OpenProject {
    session.project.as_mut().expect("a project is open")
}

#[test]
fn a_split_a_trim_and_a_delete_undo_and_seat_the_player_again() {
    let mut session = open_sample();
    let whole = opened_project(&mut session).player.duration();
    assert!(whole > 4.0, "{whole}");
    assert!(matches!(session.snapshot(), Snapshot::Project { .. }));

    round_trip(&mut session, "a split", |s| {
        let project = opened_project(s);
        project.player.seek(2.0);
        project.split_at_playhead();
        assert_eq!(project.track.clips.len(), 2);
    });
    // A trim drag of many frames is one step.
    settle(&mut session);
    let steps = undo_steps(&session);
    for _ in 0..20 {
        opened_project(&mut session).trim(1, false, -0.05);
        drag_frame(&mut session);
    }
    settle(&mut session);
    assert_eq!(undo_steps(&session), steps + 1, "the trim drag is one step");
    let trimmed = opened_project(&mut session).player.duration();
    assert!((whole - trimmed - 1.0).abs() < 0.05, "{whole} {trimmed}");

    round_trip(&mut session, "a delete", |s| {
        let project = opened_project(s);
        project.selected = Some(0);
        project.delete_selected();
        assert_eq!(project.track.clips.len(), 1);
    });
    let after_delete = opened_project(&mut session).player.duration();
    assert!((after_delete - (trimmed - 2.0)).abs() < 0.05);

    // Back over the delete: both clips, and the player plays them again.
    assert!(session.undo());
    let project_now = opened_project(&mut session);
    assert_eq!(project_now.track.clips.len(), 2);
    assert!((project_now.player.duration() - trimmed).abs() < 1e-9);
    assert!(
        project_now.track_dirty,
        "the session is told the track moved"
    );
    // Back over the trim, then the split.
    assert!(session.undo());
    assert!((opened_project(&mut session).player.duration() - whole).abs() < 1e-9);
    assert!(session.undo());
    assert_eq!(opened_project(&mut session).track.clips.len(), 1);
    assert!(!session.undo());
    // Forward again to the end.
    while session.redo() {}
    assert_eq!(opened_project(&mut session).track.clips.len(), 1);
    assert!((opened_project(&mut session).player.duration() - after_delete).abs() < 1e-9);
}

#[test]
fn a_selection_past_the_track_is_let_go_by_an_undo() {
    let mut session = open_sample();
    let project_now = opened_project(&mut session);
    project_now.player.seek(2.0);
    project_now.split_at_playhead();
    assert_eq!(project_now.selected, Some(1), "the second half is selected");
    settle(&mut session);
    assert!(session.undo());
    assert_eq!(opened_project(&mut session).selected, None);
    // The grade of a project is history too.
    session.edit.exposure = 0.6;
    settle(&mut session);
    assert!(session.undo());
    assert_eq!(session.edit.exposure, 0.0);
}
