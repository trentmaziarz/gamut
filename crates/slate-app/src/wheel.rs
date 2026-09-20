//! One colour wheel: a disc with a draggable dot, a luminance slider under
//! it and a reset. The rim is coloured with the tint each direction gives,
//! through the same hue basis slate-color grades with.

use std::f32::consts::TAU;

use egui::{Color32, Pos2, Sense, Stroke, Vec2};
use slate_color::wheels::tint;
use slate_core::look::Wheel;

/// The smallest and the largest diameter of the disc.
pub const DIAMETERS: std::ops::RangeInclusive<f32> = 56.0..=96.0;

/// What one wheel takes in the panel beside its diameter.
pub const PADDING: f32 = 8.0;

/// The number of coloured segments on the rim.
const SEGMENTS: usize = 48;

/// The colour of the tint toward `(x, y)` on the disc, for the rim.
fn rim_colour(x: f32, y: f32) -> Color32 {
    let t = tint(x, y);
    let channel = |v: f32| ((0.5 + 0.45 * v).clamp(0.0, 1.0) * 255.0) as u8;
    Color32::from_rgb(channel(t[0]), channel(t[1]), channel(t[2]))
}

/// Draws the wheel and lets the pointer move it. Returns true when it
/// changed.
pub fn show(ui: &mut egui::Ui, label: &str, wheel: &mut Wheel, diameter: f32) -> bool {
    let mut changed = false;
    ui.vertical(|ui| {
        ui.set_width(diameter + PADDING);
        ui.spacing_mut().slider_width = diameter - 24.0;
        ui.label(label);
        let (rect, response) =
            ui.allocate_exact_size(Vec2::splat(diameter), Sense::click_and_drag());
        let centre = rect.center();
        let radius = diameter / 2.0 - 4.0;
        if (response.dragged() || response.clicked())
            && let Some(pos) = response.interact_pointer_pos()
        {
            let mut x = (pos.x - centre.x) / radius;
            let mut y = (centre.y - pos.y) / radius;
            let length = (x * x + y * y).sqrt();
            if length > 1.0 {
                x /= length;
                y /= length;
            }
            if (wheel.x, wheel.y) != (x, y) {
                (wheel.x, wheel.y) = (x, y);
                changed = true;
            }
        }
        if response.double_clicked() && (wheel.x, wheel.y) != (0.0, 0.0) {
            (wheel.x, wheel.y) = (0.0, 0.0);
            changed = true;
        }

        let painter = ui.painter_at(rect);
        let visuals = ui.visuals();
        painter.circle_filled(centre, radius, visuals.extreme_bg_color);
        let on_rim = |k: usize| {
            let angle = k as f32 / SEGMENTS as f32 * TAU;
            (angle.cos(), angle.sin())
        };
        for k in 0..SEGMENTS {
            let (x0, y0) = on_rim(k);
            let (x1, y1) = on_rim(k + 1);
            painter.line_segment(
                [
                    Pos2::new(centre.x + x0 * radius, centre.y - y0 * radius),
                    Pos2::new(centre.x + x1 * radius, centre.y - y1 * radius),
                ],
                Stroke::new(3.0, rim_colour(x0 + x1, y0 + y1)),
            );
        }
        let faint = Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color);
        painter.line_segment(
            [
                Pos2::new(centre.x - radius, centre.y),
                Pos2::new(centre.x + radius, centre.y),
            ],
            faint,
        );
        painter.line_segment(
            [
                Pos2::new(centre.x, centre.y - radius),
                Pos2::new(centre.x, centre.y + radius),
            ],
            faint,
        );
        let dot = Pos2::new(centre.x + wheel.x * radius, centre.y - wheel.y * radius);
        painter.circle_filled(dot, 5.0, rim_colour(wheel.x, wheel.y));
        painter.circle_stroke(dot, 5.0, Stroke::new(1.5, visuals.strong_text_color()));

        changed |= ui
            .add(
                egui::Slider::new(&mut wheel.luminance, -100.0..=100.0)
                    .show_value(false)
                    .text("L"),
            )
            .changed();
        if ui.small_button("Reset").clicked() && !wheel.is_identity() {
            *wheel = Wheel::default();
            changed = true;
        }
    });
    changed
}
