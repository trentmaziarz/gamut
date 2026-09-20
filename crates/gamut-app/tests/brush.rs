//! The brush on the session, walked the way the window walks it: the Paint
//! button arms a component, a press starts a stroke, the pointer grows it
//! frame by frame while the history is shown an unsettled change, and the
//! release lets it settle. Every step of painting that is data logic is a
//! test here, so the window is left for feel.

use std::time::Instant;

use egui::Pos2;
use gamut_app::app::{OpenPhoto, OpenProject, Session};
use gamut_app::brush_tool::{BrushKey, FEATHER_STEP, SIZE_STEP};
use gamut_app::player::{MediaInfo, Player};
use gamut_app::project;
use gamut_app::sidecar::load;
use gamut_app::undo::{Settle, Snapshot};
use gamut_core::brush::{Brush, MAX_STROKE_POINTS, MAX_STROKES, SharedStroke, Stroke};
use gamut_core::mask::{Component, LinearGradient, RadialGradient};
use gamut_core::{Mask, MaskSource, Sidecar};
use gamut_media::fixtures;

/// The picture fills a square of 1000 points from the origin, so a position
/// on the photo is a thousandth of its place on the screen.
const SIDE: f32 = 1000.0;

fn screen(at: [f32; 2]) -> Pos2 {
    Pos2::new(at[0] * SIDE, at[1] * SIDE)
}

fn settle(session: &mut Session) {
    session.observe_history(Settle::default(), Instant::now());
}

/// A frame with the pointer down.
fn drag_frame(session: &mut Session) {
    let dragging = Settle {
        pointer_busy: true,
        ..Settle::default()
    };
    session.observe_history(dragging, Instant::now());
}

fn undo_steps(session: &Session) -> usize {
    session.history.as_ref().map_or(0, |h| h.undo_steps())
}

/// A photo session with one brush mask selected and its brush armed.
fn armed() -> Session {
    let mut session = Session::default();
    session
        .edit
        .masks
        .push(Mask::new("Brush", MaskSource::Brush(Brush::default())));
    session.edit.masks[0].adjust.exposure = 1.0;
    session.begin_history();
    session.adjust.select_mask(Some(0));
    session.toggle_brush(0);
    assert_eq!(session.adjust.brush.armed, Some(0));
    session
}

fn brush(session: &Session) -> &Brush {
    match &session.edit.masks[0].components[0].source {
        MaskSource::Brush(brush) => brush,
        other => panic!("a brush, not {other:?}"),
    }
}

/// Paints one stroke through `path` the way the window does: the press, a
/// frame for every place with the pointer down, the release, a settled frame.
fn paint(session: &mut Session, path: &[[f32; 2]], erase: bool, shift: bool) {
    let radius = session.adjust.brush.size * SIDE;
    assert!(session.begin_stroke(path[0], screen(path[0]), erase, shift));
    drag_frame(session);
    for at in &path[1..] {
        session.extend_stroke(*at, screen(*at), radius, 1.0);
        drag_frame(session);
    }
    session.end_stroke();
    settle(session);
}

fn line(from: [f32; 2], to: [f32; 2], frames: usize) -> Vec<[f32; 2]> {
    (0..=frames)
        .map(|i| {
            let t = i as f32 / frames as f32;
            [
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
            ]
        })
        .collect()
}

#[test]
fn one_stroke_of_many_frames_is_one_undo() {
    let mut session = armed();
    let radius = session.adjust.brush.size * SIDE;
    let path = line([0.2, 0.3], [0.7, 0.6], 80);
    assert!(session.begin_stroke(path[0], screen(path[0]), false, false));
    assert_eq!(
        brush(&session).strokes.len(),
        1,
        "in the edit from its first dab"
    );
    assert!(session.develop_dirty, "and on the picture");
    for at in &path[1..] {
        session.extend_stroke(*at, screen(*at), radius, 1.0);
        drag_frame(&mut session);
        assert_eq!(undo_steps(&session), 0, "no step while the pointer is down");
    }
    session.end_stroke();
    settle(&mut session);
    assert_eq!(undo_steps(&session), 1);
    assert_eq!(brush(&session).strokes.len(), 1);
    let points = brush(&session).strokes[0].points.len();
    assert!(points > 20 && points <= 81, "{points} points");

    paint(
        &mut session,
        &line([0.3, 0.7], [0.6, 0.2], 40),
        false,
        false,
    );
    assert_eq!(undo_steps(&session), 2);
    assert!(session.undo());
    assert_eq!(brush(&session).strokes.len(), 1, "one stroke goes");
    assert!(session.undo());
    assert!(brush(&session).strokes.is_empty());
    assert!(!session.undo(), "and nothing before the first stroke");
    assert!(session.redo() && session.redo());
    assert_eq!(brush(&session).strokes.len(), 2);
    assert_eq!(
        session.adjust.brush.armed,
        Some(0),
        "the brush stays in the hand"
    );
}

#[test]
fn a_stroke_takes_a_point_every_quarter_radius_of_the_pointer() {
    let mut session = armed();
    session.adjust.brush.size = 0.04;
    // 0.4 of the photo in 400 frames of a thousandth: a point every 0.01.
    paint(
        &mut session,
        &line([0.2, 0.5], [0.6, 0.5], 400),
        false,
        false,
    );
    let points = &brush(&session).strokes[0].points;
    assert!((40..=41).contains(&points.len()), "{} points", points.len());
    assert_eq!(points[0], [0.2, 0.5]);
    assert!((points[1][0] - 0.21).abs() < 1e-3, "{:?}", points[1]);
    // A click without a move is one point: one dab.
    paint(&mut session, &[[0.8, 0.8]], false, false);
    assert_eq!(brush(&session).strokes[1].points, [[0.8, 0.8]]);
}

#[test]
fn a_stroke_carries_the_settings_of_the_press_and_they_are_no_history() {
    let mut session = armed();
    session.adjust.brush.size = 0.05;
    session.adjust.brush.feather = 20.0;
    session.adjust.brush.flow = 35.0;
    settle(&mut session);
    assert_eq!(
        undo_steps(&session),
        0,
        "the settings of the tool are no step"
    );
    paint(
        &mut session,
        &line([0.2, 0.2], [0.4, 0.2], 10),
        false,
        false,
    );
    // Changed after the press, the next stroke has them and the last does not.
    session.brush_key(BrushKey::Larger);
    session.brush_key(BrushKey::MoreFeather);
    paint(
        &mut session,
        &line([0.2, 0.4], [0.4, 0.4], 10),
        false,
        false,
    );
    let strokes = &brush(&session).strokes;
    assert_eq!(
        (strokes[0].size, strokes[0].feather, strokes[0].flow),
        (0.05, 20.0, 35.0)
    );
    assert!((strokes[1].size - 0.05 * SIZE_STEP).abs() < 1e-6);
    assert_eq!(strokes[1].feather, 20.0 + FEATHER_STEP);
    assert_eq!(undo_steps(&session), 2, "two strokes, two steps");
    // An undo gives the stroke back and leaves the tool as it is.
    assert!(session.undo());
    assert!((session.adjust.brush.size - 0.05 * SIZE_STEP).abs() < 1e-6);
}

#[test]
fn alt_or_the_toggle_erases_and_shift_draws_a_line_from_the_last_stroke() {
    let mut session = armed();
    paint(
        &mut session,
        &line([0.2, 0.2], [0.4, 0.3], 20),
        false,
        false,
    );
    let end = *brush(&session).strokes[0].points.last().expect("points");
    // Shift with a click: a straight line from where the last stroke ended.
    paint(&mut session, &[[0.8, 0.25]], false, true);
    assert_eq!(brush(&session).strokes[1].points, [end, [0.8, 0.25]]);
    assert!(!brush(&session).strokes[1].erase);
    // Alt held for one stroke.
    paint(&mut session, &line([0.3, 0.1], [0.3, 0.4], 20), true, false);
    assert!(brush(&session).strokes[2].erase);
    // The toggle, as the window passes it: the toggle or Alt.
    session.adjust.brush.erase = true;
    let erase = session.adjust.brush.erase;
    paint(&mut session, &[[0.5, 0.5]], erase, false);
    assert!(brush(&session).strokes[3].erase);
    assert_eq!(undo_steps(&session), 4);
    // With Shift and nothing painted yet a click is a dab.
    let mut fresh = armed();
    paint(&mut fresh, &[[0.5, 0.5]], false, true);
    assert_eq!(brush(&fresh).strokes[0].points, [[0.5, 0.5]]);
}

/// Makes one change, lets it settle, and holds that an undo gives the state
/// before it back and a redo the state after it.
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
    assert!(session.develop_dirty, "{name}: the picture is stale");
    assert!(session.redo(), "{name}: redo");
    assert_eq!(session.snapshot(), after, "{name}: redone");
    settle(session);
    assert_eq!(undo_steps(session), steps + 1, "{name}: still one step");
}

#[test]
fn an_erase_stroke_clear_strokes_and_a_deleted_component_undo_and_redo() {
    let mut session = armed();
    paint(
        &mut session,
        &line([0.2, 0.3], [0.7, 0.6], 40),
        false,
        false,
    );
    paint(
        &mut session,
        &line([0.2, 0.6], [0.7, 0.3], 40),
        false,
        false,
    );
    round_trip(&mut session, "an erase stroke", |s| {
        let path = line([0.4, 0.2], [0.5, 0.7], 30);
        let radius = s.adjust.brush.size * SIDE;
        assert!(s.begin_stroke(path[0], screen(path[0]), true, false));
        for at in &path[1..] {
            s.extend_stroke(*at, screen(*at), radius, 1.0);
        }
        s.end_stroke();
    });
    assert_eq!(brush(&session).strokes.len(), 3);
    round_trip(&mut session, "Clear strokes", |s| {
        s.clear_strokes(0);
        assert!(brush(s).strokes.is_empty());
    });
    // A second component, then the brush component deleted as the panel does.
    session.edit.masks[0]
        .components
        .push(Component::new(
            MaskSource::Linear(LinearGradient::default()),
        ));
    session.mark_edited();
    round_trip(&mut session, "a deleted brush component", |s| {
        s.edit.masks[0].components.remove(0);
        s.adjust.put_brush_down();
        s.mark_edited();
    });
    // Redone, the brush component is gone and so is the brush.
    assert!(matches!(
        session.edit.masks[0].components[0].source,
        MaskSource::Linear(_)
    ));
    assert_eq!(session.adjust.brush.armed, None);
    // Undone, the strokes are back; clearing nothing is no step.
    assert!(session.undo());
    assert!(
        brush(&session).strokes.is_empty(),
        "as Clear strokes left it"
    );
    let steps = undo_steps(&session);
    session.clear_strokes(0);
    settle(&mut session);
    assert_eq!(undo_steps(&session), steps);
}

#[test]
fn an_undo_that_removes_the_armed_component_puts_the_brush_down() {
    let mut session = Session::default();
    session.begin_history();
    // The mask arrives as the New brush button makes it, and is armed.
    session
        .edit
        .masks
        .push(Mask::new("Brush", MaskSource::Brush(Brush::default())));
    session.adjust.select_mask(Some(0));
    session.mark_edited();
    settle(&mut session);
    session.toggle_brush(0);
    paint(
        &mut session,
        &line([0.2, 0.3], [0.7, 0.6], 40),
        false,
        false,
    );
    assert!(session.undo(), "the stroke");
    assert_eq!(
        session.adjust.brush.armed,
        Some(0),
        "the component is still there"
    );
    assert!(session.undo(), "the mask");
    assert!(session.edit.masks.is_empty());
    assert_eq!(session.adjust.brush.armed, None);
    assert_eq!(session.adjust.selected_mask, None);
    assert!(!session.begin_stroke([0.5, 0.5], screen([0.5, 0.5]), false, false));

    // An undo in the middle of a stroke lets the stroke go with its state.
    let mut session = armed();
    paint(
        &mut session,
        &line([0.2, 0.3], [0.7, 0.6], 40),
        false,
        false,
    );
    assert!(session.begin_stroke([0.5, 0.5], screen([0.5, 0.5]), false, false));
    drag_frame(&mut session);
    assert!(session.undo());
    assert!(!session.adjust.brush.is_painting());
    assert_eq!(brush(&session).strokes.len(), 1);
}

#[test]
fn what_puts_the_brush_down_and_gives_the_overlay_back() {
    // Arming turns the overlay on; putting the brush down gives it back.
    let mut session = armed();
    assert!(session.adjust.mask_overlay);
    assert_eq!(session.adjust.overlay(), Some(0));
    session.toggle_brush(0);
    assert_eq!(session.adjust.brush.armed, None, "Paint pressed again");
    assert!(!session.adjust.mask_overlay, "the overlay was off before");

    // An overlay that was on before stays on afterwards.
    session.adjust.mask_overlay = true;
    session.toggle_brush(0);
    session.brush_key(BrushKey::PutDown);
    assert_eq!(session.adjust.brush.armed, None, "Esc");
    assert!(session.adjust.mask_overlay);

    // O and the checkbox still toggle it while armed; down, it is as before.
    session.adjust.mask_overlay = false;
    session.toggle_brush(0);
    session.brush_key(BrushKey::ToggleOverlay);
    assert!(!session.adjust.mask_overlay, "O hides it while painting");
    session.brush_key(BrushKey::ToggleOverlay);
    assert!(session.adjust.mask_overlay);
    session.put_brush_down();
    assert!(!session.adjust.mask_overlay);

    // Done, which selects no mask, and another mask selected.
    session.toggle_brush(0);
    session.adjust.select_mask(None);
    assert_eq!(session.adjust.brush.armed, None, "Done");
    session.edit.masks.push(Mask::new(
        "Radial",
        MaskSource::Radial(RadialGradient::default()),
    ));
    session.adjust.select_mask(Some(0));
    session.toggle_brush(0);
    session.adjust.select_mask(Some(1));
    assert_eq!(session.adjust.brush.armed, None, "another mask");
    // A component that is no brush cannot be armed.
    session.toggle_brush(0);
    assert_eq!(session.adjust.brush.armed, None);
    assert!(
        !session.adjust.mask_overlay,
        "and the overlay is left alone"
    );
}

#[test]
fn the_brush_is_put_down_under_the_save_discard_cancel_prompt() {
    let mut session = armed();
    session
        .with_versions(|sidecar| sidecar.save_version("Kept"))
        .expect("save");
    session
        .with_versions(|sidecar| sidecar.save_version("Other"))
        .expect("save");
    paint(
        &mut session,
        &line([0.2, 0.3], [0.7, 0.6], 40),
        false,
        false,
    );
    assert!(session.version_is_dirty());
    session.request_switch("Kept").expect("a version");
    assert!(session.adjust.pending_switch.is_some(), "the prompt is up");
    // The frame under the prompt.
    session.check_brush();
    assert_eq!(session.adjust.brush.armed, None);
    assert!(!session.begin_stroke([0.5, 0.5], screen([0.5, 0.5]), false, false));
    assert_eq!(
        brush(&session).strokes.len(),
        1,
        "nothing is painted under it"
    );
}

#[test]
fn a_full_stroke_goes_on_in_a_new_one_and_a_full_brush_says_so() {
    let mut session = armed();
    session.adjust.brush.size = 0.0004;
    // More places than one stroke holds, each past a quarter radius on.
    let frames = MAX_STROKE_POINTS + 50;
    let step = 0.00015;
    let radius = session.adjust.brush.size * SIDE;
    assert!(session.begin_stroke([0.1, 0.5], screen([0.1, 0.5]), false, false));
    for i in 1..=frames {
        let at = [0.1 + i as f32 * step, 0.5];
        session.extend_stroke(at, screen(at), radius, 1e6);
        drag_frame(&mut session);
    }
    session.end_stroke();
    settle(&mut session);
    let strokes = &brush(&session).strokes;
    assert_eq!(strokes.len(), 2, "the path goes on in a second stroke");
    assert_eq!(strokes[0].points.len(), MAX_STROKE_POINTS);
    assert_eq!(
        strokes[1].points[0],
        *strokes[0].points.last().expect("points")
    );
    assert_eq!(undo_steps(&session), 1, "and it is still one step");
    assert!(session.undo());
    assert!(brush(&session).strokes.is_empty());

    // A brush that holds all it can refuses the press and says why.
    let dab = SharedStroke::new(&Stroke {
        points: vec![[0.5, 0.5]],
        ..Stroke::default()
    });
    let MaskSource::Brush(full) = &mut session.edit.masks[0].components[0].source else {
        panic!("a brush");
    };
    full.strokes = vec![dab; MAX_STROKES];
    assert!(!session.begin_stroke([0.5, 0.5], screen([0.5, 0.5]), false, false));
    assert!(
        session
            .status
            .as_deref()
            .is_some_and(|s| s.contains("2000 strokes"))
    );
}

#[test]
fn the_saved_sidecar_after_an_undo_carries_the_strokes_that_remain() {
    let dir = std::env::temp_dir().join("gamut-brush-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let photo = dir.join("painted.jpg");
    let _ = std::fs::remove_file(Sidecar::path_for(&photo));
    let mut session = armed();
    session.photo = Some(OpenPhoto {
        path: photo.clone(),
        width: 6000,
        height: 4000,
    });
    paint(
        &mut session,
        &line([0.2, 0.3], [0.7, 0.6], 60),
        false,
        false,
    );
    paint(&mut session, &line([0.3, 0.7], [0.6, 0.2], 60), true, false);
    paint(&mut session, &[[0.5, 0.5]], false, false);

    let on_disk = |session: &mut Session| -> Sidecar {
        assert!(session.changed_at.is_some(), "the file is stale");
        session.save_now();
        let text = std::fs::read_to_string(Sidecar::path_for(&photo)).expect("read");
        assert!(text.contains("\"version\": 3"), "{text}");
        assert!(text.contains("\"type\": \"Brush\""), "{text}");
        assert!(
            !text.contains("history") && !text.contains("undo"),
            "{text}"
        );
        load(&photo).expect("the sidecar loads")
    };
    assert_eq!(on_disk(&mut session), session.sidecar());
    assert!(session.undo());
    let saved = on_disk(&mut session);
    assert_eq!(saved, session.sidecar(), "the file is what is shown");
    let MaskSource::Brush(kept) = &saved.edit.masks[0].components[0].source else {
        panic!("a brush");
    };
    assert_eq!(kept.strokes.len(), 2);
    assert!(kept.strokes[1].erase);
    assert!(session.redo());
    assert_eq!(on_disk(&mut session), session.sidecar());

    // Closed and opened again: the strokes are there and no history is.
    let mut fresh = Session::default();
    fresh.take_sidecar(load(&photo).expect("loads"));
    fresh.begin_history();
    assert_eq!(fresh.sidecar(), session.sidecar());
    assert_eq!(brush(&fresh).strokes.len(), 3);
    assert!(!fresh.can_undo());
    assert_eq!(
        fresh.adjust.brush.armed, None,
        "and the brush is not in the hand"
    );
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
            path: std::env::temp_dir().join("gamut-brush-test.gamut"),
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

#[test]
fn a_project_paints_and_undoes() {
    let mut session = open_sample();
    assert!(matches!(session.snapshot(), Snapshot::Project { .. }));
    session
        .edit
        .masks
        .push(Mask::new("Brush", MaskSource::Brush(Brush::default())));
    session.edit.masks[0].adjust.exposure = 0.8;
    session.mark_edited();
    settle(&mut session);
    session.adjust.select_mask(Some(0));
    session.toggle_brush(0);
    paint(
        &mut session,
        &line([0.2, 0.3], [0.7, 0.6], 40),
        false,
        false,
    );
    paint(
        &mut session,
        &line([0.2, 0.6], [0.7, 0.3], 40),
        false,
        false,
    );
    assert_eq!(undo_steps(&session), 3, "the mask and two strokes");
    let Snapshot::Project { edit, .. } = session.snapshot() else {
        panic!("a project");
    };
    assert_eq!(
        *edit, session.edit,
        "the strokes are in what the project saves"
    );
    assert!(session.undo());
    assert_eq!(brush(&session).strokes.len(), 1);
    assert!(session.redo());
    assert_eq!(brush(&session).strokes.len(), 2);
    assert_eq!(session.adjust.brush.armed, Some(0));
}
