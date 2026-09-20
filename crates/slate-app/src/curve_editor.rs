//! The tone curve editor: a square drawn with the egui painter. A click on
//! the line adds a point, a drag moves one (its x held between its
//! neighbours), a right-click or the Delete key removes one. The two end
//! points stay at x 0 and 1 and cannot be removed.

use egui::{Color32, Pos2, Sense, Stroke, StrokeKind, Vec2};
use slate_color::curve::{TABLE_SIZE, bake_channel};
use slate_core::look::{Curve, MAX_POINTS};

/// The side of the square, when the panel is wide enough.
const SIDE: f32 = 240.0;

/// How near the pointer must be to grab a point or the line, in points.
const GRAB: f32 = 10.0;

/// The margin between the square and the plot, so the end points are drawn
/// whole.
const MARGIN: f32 = 7.0;

/// The smallest gap in x between two neighbours.
const GAP: f32 = 0.01;

/// What the editor remembers between frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CurveEditorState {
    /// The point being dragged.
    dragged: Option<usize>,
    /// The point the Delete key removes.
    selected: Option<usize>,
}

/// Draws `curve` and lets the pointer edit it. Returns true when it changed.
pub fn show(
    ui: &mut egui::Ui,
    curve: &mut Curve,
    state: &mut CurveEditorState,
    colour: Color32,
) -> bool {
    *curve = curve.sanitised();
    let side = SIDE.min(ui.available_width());
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(side), Sense::click_and_drag());
    let plot = rect.shrink(MARGIN);
    let to_screen = |p: [f32; 2]| {
        Pos2::new(
            plot.left() + p[0] * plot.width(),
            plot.bottom() - p[1] * plot.height(),
        )
    };
    let from_screen = |pos: Pos2| {
        [
            ((pos.x - plot.left()) / plot.width()).clamp(0.0, 1.0),
            ((plot.bottom() - pos.y) / plot.height()).clamp(0.0, 1.0),
        ]
    };
    let nearest_point = |curve: &Curve, pos: Pos2| {
        curve
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, to_screen(*p).distance(pos)))
            .filter(|(_, d)| *d <= GRAB)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    };

    let mut changed = false;
    let last = curve.points.len() - 1;
    state.selected = state.selected.filter(|i| *i <= last);

    if let Some(pos) = response.interact_pointer_pos() {
        if response.drag_started() || response.clicked() {
            state.dragged = nearest_point(curve, pos);
            if state.dragged.is_none() && curve.points.len() < MAX_POINTS {
                let [x, y] = from_screen(pos);
                let on_line = slate_color::curve::evaluate(curve, x);
                let near_line = (to_screen([x, on_line]).y - pos.y).abs() <= GRAB;
                let free = curve.points.iter().all(|p| (p[0] - x).abs() >= GAP);
                if near_line && free {
                    let at = curve.points.partition_point(|p| p[0] < x);
                    curve.points.insert(at, [x, y]);
                    state.dragged = Some(at);
                    changed = true;
                }
            }
            state.selected = state.dragged.or(state.selected);
        }
        if response.dragged()
            && let Some(i) = state.dragged
        {
            let [x, y] = from_screen(pos);
            let last = curve.points.len() - 1;
            let x = if i == 0 {
                0.0
            } else if i == last {
                1.0
            } else {
                x.clamp(curve.points[i - 1][0] + GAP, curve.points[i + 1][0] - GAP)
            };
            if curve.points[i] != [x, y] {
                curve.points[i] = [x, y];
                changed = true;
            }
        }
    }
    if response.drag_stopped() {
        state.dragged = None;
    }

    let last = curve.points.len() - 1;
    let removable = |i: usize| i != 0 && i != last;
    let mut remove = None;
    if response.secondary_clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        remove = nearest_point(curve, pos).filter(|i| removable(*i));
    }
    if response.hovered() && ui.input(|i| i.key_pressed(egui::Key::Delete)) {
        remove = remove.or(state.selected.filter(|i| removable(*i)));
    }
    if let Some(i) = remove {
        curve.points.remove(i);
        state.selected = None;
        state.dragged = None;
        changed = true;
    }

    let painter = ui.painter_at(rect);
    let visuals = ui.visuals();
    painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
    let grid = Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color);
    for i in 1..4 {
        let t = i as f32 / 4.0;
        painter.line_segment([to_screen([t, 0.0]), to_screen([t, 1.0])], grid);
        painter.line_segment([to_screen([0.0, t]), to_screen([1.0, t])], grid);
    }
    painter.line_segment([to_screen([0.0, 0.0]), to_screen([1.0, 1.0])], grid);
    painter.rect_stroke(plot, 0.0, grid, StrokeKind::Inside);

    let table = bake_channel(curve, &Curve::default());
    let line: Vec<Pos2> = (0..TABLE_SIZE)
        .step_by(8)
        .chain([TABLE_SIZE - 1])
        .map(|i| to_screen([i as f32 / (TABLE_SIZE - 1) as f32, table[i]]))
        .collect();
    painter.line(line, Stroke::new(2.0, colour));
    for (i, p) in curve.points.iter().enumerate() {
        let centre = to_screen(*p);
        if state.selected == Some(i) {
            painter.circle_filled(centre, 5.0, colour);
        } else {
            painter.circle_filled(centre, 4.0, visuals.extreme_bg_color);
        }
        painter.circle_stroke(centre, 4.5, Stroke::new(1.5, colour));
    }
    changed
}
