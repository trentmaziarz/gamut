//! The Adjust tab: collapsing sections for the Basic sliders, Presence, the
//! tone curve, the HSL mixer, the grading wheels, look presets and named
//! versions, then the crop controls. Every control is bound to the session
//! and every change goes through `Session::mark_edited`.

use std::path::PathBuf;

use egui::{CollapsingHeader, Color32};
use slate_core::look::HSL_NAMES;
use slate_core::preset::Groups;
use slate_core::{Crop, CropAspect, LookPreset, NamedVersion, PhotoEdit};

use crate::app::{Session, SwitchAnswer};
use crate::curve_editor::{self, CurveEditorState};
use crate::presets;
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
    /// The name and the groups of the preset about to be saved.
    pub preset_name: String,
    pub preset_groups: Groups,
    /// The presets folder as last read; `None` until the section first
    /// shows and after a save or a delete.
    pub presets: Option<Vec<(PathBuf, LookPreset)>>,
    /// The name of the version about to be saved.
    pub version_name: String,
    /// The version being renamed and its new name so far.
    pub renaming: Option<(String, String)>,
    /// The version a Switch to asked for while the working state held work
    /// its active version does not; the Save, Discard, Cancel prompt is up
    /// while this is set.
    pub pending_switch: Option<String>,
}

/// What a button of the Versions section asked for. It runs after the
/// sections are drawn, when the session is free to change.
enum VersionAction {
    Save(String),
    SwitchTo(String),
    Update(String),
    Rename(String, String),
    Delete(String),
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
    let mut message = None;
    let mut version_action = None;
    let version_dirty = session.version_is_dirty();
    let Session {
        edit,
        adjust,
        versions,
        active_version,
        photo,
        ..
    } = session;

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

    CollapsingHeader::new("Presets").show(ui, |ui| match presets_section(ui, edit, adjust) {
        Ok(applied) => changed |= applied,
        Err(error) => message = Some(error),
    });

    CollapsingHeader::new("Versions").show(ui, |ui| {
        if photo.is_none() {
            ui.weak("Versions belong to a photo.");
            return;
        }
        version_action = versions_section(
            ui,
            versions,
            active_version.as_deref(),
            version_dirty,
            adjust,
        );
    });

    if changed {
        session.mark_edited();
    }
    if let Some(action) = version_action {
        let result = match &action {
            VersionAction::Save(name) => session.with_versions(|s| s.save_version(name)),
            VersionAction::SwitchTo(name) => session.request_switch(name),
            VersionAction::Update(name) => session.update_version(name),
            VersionAction::Rename(name, new_name) => {
                session.with_versions(|s| s.rename_version(name, new_name))
            }
            VersionAction::Delete(name) => session.with_versions(|s| s.delete_version(name)),
        };
        match result {
            Ok(()) => {
                if matches!(action, VersionAction::Save(_)) {
                    session.adjust.version_name.clear();
                }
                session.adjust.renaming = None;
            }
            Err(error) => message = Some(error.to_string()),
        }
    }
    if let Some(answer) = switch_prompt(ui.ctx(), session)
        && let Err(error) = session.answer_switch(answer)
    {
        message = Some(error.to_string());
    }
    if message.is_some() {
        session.status = message;
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

/// The Presets section: the save row with one checkbox per group, then the
/// presets of the folder. Returns whether a preset was applied to `edit`.
fn presets_section(
    ui: &mut egui::Ui,
    edit: &mut PhotoEdit,
    adjust: &mut AdjustState,
) -> Result<bool, String> {
    let Some(folder) = presets::folder() else {
        ui.weak("The environment names no config folder to keep presets in.");
        return Ok(false);
    };
    let mut applied = false;
    let mut result = Ok(());

    ui.add(egui::TextEdit::singleline(&mut adjust.preset_name).hint_text("Preset name"));
    ui.horizontal_wrapped(|ui| {
        let groups = &mut adjust.preset_groups;
        ui.checkbox(&mut groups.basic, "Basic");
        ui.checkbox(&mut groups.presence, "Presence");
        ui.checkbox(&mut groups.curve, "Curve");
        ui.checkbox(&mut groups.mixer, "Mixer");
        ui.checkbox(&mut groups.grading, "Grading");
    });
    let ready = !adjust.preset_name.trim().is_empty() && adjust.preset_groups.any();
    if ui
        .add_enabled(ready, egui::Button::new("Save preset"))
        .clicked()
    {
        let preset = LookPreset::from_edit(&adjust.preset_name, edit, adjust.preset_groups);
        match presets::save(&folder, &preset) {
            Ok(_) => {
                adjust.preset_name.clear();
                adjust.presets = None;
            }
            Err(error) => result = Err(format!("Could not save the preset: {error}")),
        }
    }

    ui.separator();
    let list = adjust.presets.get_or_insert_with(|| presets::list(&folder));
    if list.is_empty() {
        ui.weak("No presets yet.");
    }
    let mut deleted = false;
    for (path, preset) in list.iter() {
        ui.horizontal(|ui| {
            if ui.small_button("Apply").clicked() {
                preset.apply(edit);
                applied = true;
            }
            if ui.small_button("Delete").clicked() {
                match std::fs::remove_file(path) {
                    Ok(()) => deleted = true,
                    Err(error) => result = Err(format!("Could not delete the preset: {error}")),
                }
            }
            ui.label(&preset.name);
        });
    }
    if deleted {
        adjust.presets = None;
    }
    result.map(|()| applied)
}

/// The prompt a Switch to raises over unsaved work: Save into the active
/// version, Discard, or Cancel. Closing it any other way cancels.
fn switch_prompt(ctx: &egui::Context, session: &Session) -> Option<SwitchAnswer> {
    let target = session.adjust.pending_switch.as_deref()?;
    let active = session.active_version.as_deref().unwrap_or_default();
    let mut answer = None;
    let modal = egui::Modal::new(egui::Id::new("version switch prompt")).show(ctx, |ui| {
        ui.set_max_width(360.0);
        ui.heading(format!("Switch to {target}"));
        ui.label(format!(
            "The working state has changes that are not saved into {active}."
        ));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button(format!("Save into {active}")).clicked() {
                answer = Some(SwitchAnswer::Save);
            }
            if ui.button("Discard").clicked() {
                answer = Some(SwitchAnswer::Discard);
            }
            if ui.button("Cancel").clicked() {
                answer = Some(SwitchAnswer::Cancel);
            }
        });
    });
    if answer.is_none() && modal.should_close() {
        answer = Some(SwitchAnswer::Cancel);
    }
    answer
}

/// The Versions section: the save row, then one row per version. Returns
/// what a button asked for.
fn versions_section(
    ui: &mut egui::Ui,
    versions: &[NamedVersion],
    active: Option<&str>,
    dirty: bool,
    adjust: &mut AdjustState,
) -> Option<VersionAction> {
    let mut action = None;
    match (active, dirty) {
        (Some(name), true) => ui.label(format!("Working from {name}, with changes not in it")),
        (Some(name), false) => ui.label(format!("Working from {name}")),
        (None, _) => ui.weak("The working state is not a version."),
    };
    ui.add(egui::TextEdit::singleline(&mut adjust.version_name).hint_text("Version name"));
    let ready = !adjust.version_name.trim().is_empty();
    if ui
        .add_enabled(ready, egui::Button::new("Save as version"))
        .clicked()
    {
        action = Some(VersionAction::Save(adjust.version_name.clone()));
    }

    ui.separator();
    if versions.is_empty() {
        ui.weak("No versions yet.");
    }
    for version in versions {
        let name = &version.name;
        ui.horizontal(|ui| {
            if ui.small_button("Switch to").clicked() {
                action = Some(VersionAction::SwitchTo(name.clone()));
            }
            if ui
                .small_button("Update")
                .on_hover_text("Write the working state into this version")
                .clicked()
            {
                action = Some(VersionAction::Update(name.clone()));
            }
            if ui.small_button("Rename").clicked() {
                adjust.renaming = Some((name.clone(), name.clone()));
            }
            if ui.small_button("Delete").clicked() {
                action = Some(VersionAction::Delete(name.clone()));
            }
            if active == Some(name.as_str()) {
                ui.strong(name);
            } else {
                ui.label(name);
            }
        });
        if let Some((renamed, new_name)) = &mut adjust.renaming
            && renamed == name
        {
            let mut done = false;
            let mut cancelled = false;
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(new_name).desired_width(140.0));
                done = ui.small_button("OK").clicked();
                cancelled = ui.small_button("Cancel").clicked();
            });
            if done {
                action = Some(VersionAction::Rename(renamed.clone(), new_name.clone()));
            } else if cancelled {
                adjust.renaming = None;
            }
        }
    }
    action
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
