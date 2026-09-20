//! The Timeline tab: the transport, the time ruler, the one track with a
//! rectangle per clip, the playhead, the trim handles, the selection and
//! the shortcuts. Space plays and pauses, S splits at the playhead, Delete
//! ripple-deletes the selected clip, Home and End move the playhead, Left
//! and Right step one frame.

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind};
use gamut_core::Track;

use crate::app::{OpenProject, Session};

/// The height of the ruler in points.
const RULER_HEIGHT: f32 = 22.0;

/// The height of the track in points.
const TRACK_HEIGHT: f32 = 56.0;

/// The width of a trim handle in points.
const HANDLE_WIDTH: f32 = 8.0;

/// `seconds` as minutes, seconds and frames.
pub fn timecode(seconds: f64, frame_rate: f64) -> String {
    let seconds = seconds.max(0.0);
    let whole = seconds.floor();
    let frames = ((seconds - whole) * frame_rate.max(1.0)).round() as u32;
    let minutes = (whole / 60.0) as u32;
    let secs = (whole % 60.0) as u32;
    format!("{minutes:02}:{secs:02}.{frames:02}")
}

/// The spacing of the ruler ticks in seconds for a given scale.
pub fn tick_seconds(pixels_per_second: f32) -> f64 {
    for step in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0] {
        if step * f64::from(pixels_per_second) >= 60.0 {
            return step;
        }
    }
    120.0
}

pub fn ui(ui: &mut egui::Ui, session: &mut Session) {
    let Some(project) = session.project.as_mut() else {
        ui.weak("Open a video: File > Open, drop a file on the window, or gamut-app <path>.");
        return;
    };
    shortcuts(ui, project);
    transport(ui, project);
    track_area(ui, project);
    if project.player.is_playing() {
        ui.ctx().request_repaint();
    }
    if project.track_dirty {
        project.track_dirty = false;
        session.mark_edited();
    }
}

fn shortcuts(ui: &egui::Ui, project: &mut OpenProject) {
    let (space, split, delete, home, end, left, right) = ui.input(|i| {
        (
            i.key_pressed(egui::Key::Space),
            i.key_pressed(egui::Key::S),
            i.key_pressed(egui::Key::Delete),
            i.key_pressed(egui::Key::Home),
            i.key_pressed(egui::Key::End),
            i.key_pressed(egui::Key::ArrowLeft),
            i.key_pressed(egui::Key::ArrowRight),
        )
    });
    if space {
        project.player.toggle();
    }
    if split {
        project.split_at_playhead();
    }
    if delete {
        project.delete_selected();
    }
    if home {
        project.player.pause();
        project.player.seek(0.0);
    }
    if end {
        project.player.pause();
        let end = project.player.duration();
        project.player.seek(end);
    }
    if left {
        project.player.step(-1);
    }
    if right {
        project.player.step(1);
    }
}

fn transport(ui: &mut egui::Ui, project: &mut OpenProject) {
    ui.horizontal(|ui| {
        let label = if project.player.is_playing() {
            "Pause"
        } else {
            "Play"
        };
        if ui.button(label).clicked() {
            project.player.toggle();
        }
        if ui.button("Split").clicked() {
            project.split_at_playhead();
        }
        let has_selection = project.selected.is_some();
        if ui
            .add_enabled(has_selection, egui::Button::new("Delete"))
            .clicked()
        {
            project.delete_selected();
        }
        let rate = project.frame_rate();
        ui.monospace(format!(
            "{} / {}",
            timecode(project.player.position(), rate),
            timecode(project.player.duration(), rate)
        ));
    });
}

fn track_area(ui: &mut egui::Ui, project: &mut OpenProject) {
    let width = ui.available_width().max(1.0);
    let (area, response) = ui.allocate_exact_size(
        egui::vec2(width, RULER_HEIGHT + TRACK_HEIGHT),
        Sense::click_and_drag(),
    );
    let duration = project.player.duration().max(0.001);
    let pixels_per_second = (width / duration as f32).max(0.0001);
    let time_at = |x: f32| f64::from((x - area.min.x) / pixels_per_second).clamp(0.0, duration);
    let x_at = |seconds: f64| area.min.x + seconds as f32 * pixels_per_second;
    let ruler = Rect::from_min_size(area.min, egui::vec2(width, RULER_HEIGHT));
    let track = Rect::from_min_size(
        Pos2::new(area.min.x, area.min.y + RULER_HEIGHT),
        egui::vec2(width, TRACK_HEIGHT),
    );
    let painter = ui.painter().with_clip_rect(area);
    painter.rect_filled(ruler, 0, Color32::from_gray(30));
    painter.rect_filled(track, 0, Color32::from_gray(40));

    // The ruler.
    let step = tick_seconds(pixels_per_second);
    let mut t = 0.0;
    while t <= duration + 1e-9 {
        let x = x_at(t);
        painter.line_segment(
            [Pos2::new(x, ruler.max.y - 6.0), Pos2::new(x, ruler.max.y)],
            Stroke::new(1.0, Color32::from_gray(160)),
        );
        painter.text(
            Pos2::new(x + 3.0, ruler.min.y + 2.0),
            Align2::LEFT_TOP,
            timecode(t, project.frame_rate()),
            FontId::monospace(10.0),
            Color32::from_gray(180),
        );
        t += step;
    }

    // The clips, with their trim handles.
    let clips = project.track.clips.clone();
    let mut start = 0.0;
    let mut trim: Option<(usize, bool, f32)> = None;
    let mut clicked_clip = None;
    for (index, clip) in clips.iter().enumerate() {
        let end = start + clip.duration();
        let rect = Rect::from_min_max(
            Pos2::new(x_at(start), track.min.y + 6.0),
            Pos2::new(x_at(end), track.max.y - 6.0),
        );
        let selected = project.selected == Some(index);
        let fill = if selected {
            Color32::from_rgb(70, 110, 160)
        } else {
            Color32::from_rgb(60, 80, 110)
        };
        painter.rect_filled(rect, 3, fill);
        painter.rect_stroke(
            rect,
            3,
            Stroke::new(1.0, Color32::from_gray(200)),
            StrokeKind::Inside,
        );
        let name = project
            .media
            .get(clip.media)
            .and_then(|m| m.path.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let inner = painter.with_clip_rect(rect.shrink(2.0));
        inner.text(
            Pos2::new(rect.min.x + HANDLE_WIDTH + 4.0, rect.min.y + 4.0),
            Align2::LEFT_TOP,
            name,
            FontId::proportional(12.0),
            Color32::WHITE,
        );
        inner.text(
            Pos2::new(rect.min.x + HANDLE_WIDTH + 4.0, rect.max.y - 4.0),
            Align2::LEFT_BOTTOM,
            timecode(clip.duration(), project.frame_rate()),
            FontId::monospace(10.0),
            Color32::from_gray(220),
        );
        let handle_width = HANDLE_WIDTH.min(rect.width() / 3.0);
        let left = Rect::from_min_size(rect.min, egui::vec2(handle_width, rect.height()));
        let right = Rect::from_min_size(
            Pos2::new(rect.max.x - handle_width, rect.min.y),
            egui::vec2(handle_width, rect.height()),
        );
        for (handle, is_in) in [(left, true), (right, false)] {
            painter.rect_filled(handle, 2, Color32::from_gray(230));
            let id = ui.id().with(("trim", index, is_in));
            let drag = ui.interact(handle, id, Sense::drag());
            if drag.dragged() {
                trim = Some((index, is_in, drag.drag_delta().x));
            }
            if drag.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
        }
        if response.clicked()
            && let Some(at) = response.interact_pointer_pos()
            && rect.contains(at)
        {
            clicked_clip = Some(index);
        }
        start = end;
    }

    if let Some((index, is_in, dx)) = trim {
        let seconds = f64::from(dx / pixels_per_second);
        project.trim(index, is_in, seconds);
    } else if (response.dragged() || response.clicked())
        && let Some(at) = response.interact_pointer_pos()
    {
        project.player.pause();
        project.player.seek(time_at(at.x));
    }
    if let Some(index) = clicked_clip {
        project.selected = Some(index);
    } else if response.clicked() {
        project.selected = None;
    }

    // The playhead.
    let x = x_at(project.player.position());
    painter.line_segment(
        [Pos2::new(x, area.min.y), Pos2::new(x, area.max.y)],
        Stroke::new(2.0, Color32::from_rgb(255, 80, 80)),
    );
    painter.add(egui::Shape::convex_polygon(
        vec![
            Pos2::new(x - 6.0, area.min.y),
            Pos2::new(x + 6.0, area.min.y),
            Pos2::new(x, area.min.y + 8.0),
        ],
        Color32::from_rgb(255, 80, 80),
        Stroke::NONE,
    ));
}

/// The clip the playhead is over, for the selection after a split.
pub fn clip_under(track: &Track, seconds: f64) -> Option<usize> {
    track.clip_at(seconds).map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::{tick_seconds, timecode};

    #[test]
    fn timecodes_carry_minutes_seconds_and_frames() {
        assert_eq!(timecode(0.0, 30.0), "00:00.00");
        assert_eq!(timecode(65.5, 30.0), "01:05.15");
        assert_eq!(timecode(-1.0, 30.0), "00:00.00");
    }

    #[test]
    fn ticks_keep_at_least_sixty_points_apart() {
        assert_eq!(tick_seconds(1000.0), 0.1);
        assert_eq!(tick_seconds(100.0), 1.0);
        assert_eq!(tick_seconds(10.0), 10.0);
        assert_eq!(tick_seconds(0.1), 120.0);
    }
}
