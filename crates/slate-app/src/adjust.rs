//! The Adjust tab: collapsing sections for the Basic sliders, Presence, the
//! tone curve, the HSL mixer, the grading wheels, look presets and named
//! versions, then the crop controls. Every control is bound to the session
//! and every change goes through `Session::mark_edited`.

use egui::{CollapsingHeader, Color32};
use slate_core::look::HSL_NAMES;
use slate_core::{Crop, CropAspect, PhotoEdit};

use crate::app::Session;
use crate::curve_editor::{self, CurveEditorState};
use crate::wheel;

/// The range of the exposure slider in stops.
pub const EXPOSURE_RANGE: std::ops::RangeInclusive<f32> = -5.0..=5.0;

/// The range of every other slider.
pub const SLIDER_RANGE: std::ops::RangeInclusive<f32> = -100.0..=100.0;

/// Which tone curve the editor shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CurveChannel {
    #[default]
    Master,
    Red,
    Green,
    Blue,
}

impl CurveChannel {
    const ALL: [(CurveChannel, &str); 4] = [
        (CurveChannel::Master, "Master"),
        (CurveChannel::Red, "Red"),
        (CurveChannel::Green, "Green"),
        (CurveChannel::Blue, "Blue"),
    ];
}

/// Which row of the mixer the eight sliders show.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MixerRow {
    Hue,
    #[default]
    Saturation,
    Luminance,
}

/// What the Adjust tab remembers between frames that is not part of the
/// edit.
#[derive(Clone, Debug, Default)]
pub struct AdjustState {
    pub curve_channel: CurveChannel,
    pub curve_editor: CurveEditorState,
    pub mixer_row: MixerRow,
}

pub fn ui(ui: &mut egui::Ui, session: &mut Session) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| sections(ui, session));
}

fn sections(ui: &mut egui::Ui, session: &mut Session) {
    let mut changed = false;
    if ui.button("Reset all").clicked() && session.edit != PhotoEdit::default() {
        session.edit = PhotoEdit::default();
        changed = true;
    }
    let Session { edit, adjust, .. } = session;

    CollapsingHeader::new("Basic")
        .default_open(true)
        .show(ui, |ui| {
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
        });

    CollapsingHeader::new("Presence").show(ui, |ui| {
        changed |= slider(ui, &mut edit.texture, SLIDER_RANGE, "Texture");
        changed |= slider(ui, &mut edit.clarity, SLIDER_RANGE, "Clarity");
        changed |= slider(ui, &mut edit.dehaze, SLIDER_RANGE, "Dehaze");
    });

    CollapsingHeader::new("Curve").show(ui, |ui| {
        ui.horizontal(|ui| {
            for (channel, label) in CurveChannel::ALL {
                if ui
                    .selectable_value(&mut adjust.curve_channel, channel, label)
                    .clicked()
                {
                    adjust.curve_editor = CurveEditorState::default();
                }
            }
        });
        let curves = &mut edit.look.curves;
        let (curve, colour) = match adjust.curve_channel {
            CurveChannel::Master => (&mut curves.master, ui.visuals().strong_text_color()),
            CurveChannel::Red => (&mut curves.red, Color32::from_rgb(230, 70, 70)),
            CurveChannel::Green => (&mut curves.green, Color32::from_rgb(70, 190, 90)),
            CurveChannel::Blue => (&mut curves.blue, Color32::from_rgb(80, 130, 240)),
        };
        changed |= curve_editor::show(ui, curve, &mut adjust.curve_editor, colour);
        if ui.small_button("Reset curve").clicked() && !curve.is_identity() {
            *curve = Default::default();
            changed = true;
        }
    });

    CollapsingHeader::new("Mixer").show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut adjust.mixer_row, MixerRow::Hue, "Hue");
            ui.selectable_value(&mut adjust.mixer_row, MixerRow::Saturation, "Saturation");
            ui.selectable_value(&mut adjust.mixer_row, MixerRow::Luminance, "Luminance");
        });
        for (range, name) in edit.look.hsl.iter_mut().zip(HSL_NAMES) {
            let value = match adjust.mixer_row {
                MixerRow::Hue => &mut range.hue,
                MixerRow::Saturation => &mut range.saturation,
                MixerRow::Luminance => &mut range.luminance,
            };
            changed |= slider(ui, value, SLIDER_RANGE, name);
        }
    });

    CollapsingHeader::new("Grading").show(ui, |ui| {
        let wheels = &mut edit.look.wheels;
        // Three across, at the diameter the panel leaves room for.
        let gaps = 2.0 * ui.spacing().item_spacing.x + 3.0 * wheel::PADDING;
        let diameter = ((ui.available_width() - gaps) / 3.0)
            .clamp(*wheel::DIAMETERS.start(), *wheel::DIAMETERS.end());
        ui.horizontal(|ui| {
            changed |= wheel::show(ui, "Shadows", &mut wheels.shadows, diameter);
            changed |= wheel::show(ui, "Midtones", &mut wheels.midtones, diameter);
            changed |= wheel::show(ui, "Highlights", &mut wheels.highlights, diameter);
        });
    });

    CollapsingHeader::new("Presets").show(ui, |ui| {
        ui.weak("No presets yet.");
    });

    CollapsingHeader::new("Versions").show(ui, |ui| {
        ui.weak("No versions yet.");
    });

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
