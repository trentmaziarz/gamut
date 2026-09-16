//! The Adjust tab: the nine Basic sliders, the reset button and the crop
//! controls, all bound to the session.

use slate_core::{Crop, CropAspect, PhotoEdit};

use crate::app::Session;

/// The range of the exposure slider in stops.
pub const EXPOSURE_RANGE: std::ops::RangeInclusive<f32> = -5.0..=5.0;

/// The range of every other slider.
pub const SLIDER_RANGE: std::ops::RangeInclusive<f32> = -100.0..=100.0;

pub fn ui(ui: &mut egui::Ui, session: &mut Session) {
    ui.heading("Basic");
    let edit = &mut session.edit;
    let mut changed = false;
    changed |= slider(
        ui,
        &mut edit.white_balance_temperature,
        SLIDER_RANGE,
        "Temperature",
    );
    changed |= slider(ui, &mut edit.white_balance_tint, SLIDER_RANGE, "Tint");
    changed |= slider(ui, &mut edit.exposure, EXPOSURE_RANGE, "Exposure");
    changed |= slider(ui, &mut edit.contrast, SLIDER_RANGE, "Contrast");
    changed |= slider(ui, &mut edit.highlights, SLIDER_RANGE, "Highlights");
    changed |= slider(ui, &mut edit.shadows, SLIDER_RANGE, "Shadows");
    changed |= slider(ui, &mut edit.whites, SLIDER_RANGE, "Whites");
    changed |= slider(ui, &mut edit.blacks, SLIDER_RANGE, "Blacks");
    changed |= slider(ui, &mut edit.vibrance, SLIDER_RANGE, "Vibrance");
    changed |= slider(ui, &mut edit.saturation, SLIDER_RANGE, "Saturation");
    if ui.button("Reset").clicked() && *edit != PhotoEdit::default() {
        *edit = PhotoEdit::default();
        changed = true;
    }
    if changed {
        session.mark_edited();
    }

    ui.separator();
    ui.heading("Crop");
    ui.horizontal(|ui| {
        for aspect in CropAspect::ALL {
            let selected = session.crop.aspect == aspect;
            if ui.selectable_label(selected, aspect.label()).clicked() && !selected {
                session.crop = match session.source_size() {
                    Some((width, height)) => Crop::fitted(aspect, width, height),
                    None => Crop {
                        aspect,
                        ..session.crop
                    },
                };
                session.mark_edited();
            }
        }
    });
    if session.crop.aspect == CropAspect::Feed4x5 {
        ui.checkbox(&mut session.grid_guide, "3:4 grid guide");
    }

    ui.separator();
    match (session.open_name(), session.source_size()) {
        (Some(name), Some((width, height))) => {
            ui.label(format!("{name}, {width}x{height}"));
        }
        _ => {
            ui.weak(
                "Open a photo or a video: File > Open, drop a file on the window, or slate-app <path>.",
            );
        }
    }
    if let Some(status) = &session.status {
        ui.colored_label(ui.visuals().warn_fg_color, status);
    }
}

fn slider(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    label: &str,
) -> bool {
    ui.add(egui::Slider::new(value, range).text(label))
        .changed()
}
