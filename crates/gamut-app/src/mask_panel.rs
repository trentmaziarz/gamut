//! The Masks section of the Adjust tab: the list of masks, the buttons that
//! make one of each source, and under the selected mask its opacity, invert
//! and overlay, its components with their operators, and the numbers of each
//! source. A brush component has a Paint button that arms it, the settings
//! of the brush, how many strokes it holds and Clear strokes.
//!
//! Selecting a mask is what points the Basic, Presence, Curve, Mixer and
//! Grading sections at it (see `adjust.rs`), and what shows its handles on
//! the picture (see `mask_handles.rs`). The list operations keep the
//! selection on the same mask while the list changes around it.

use gamut_core::brush::{Brush, MAX_BRUSH_SIZE, MIN_BRUSH_SIZE, MIN_FLOW};
use gamut_core::mask::{
    ColourRange, Component, LinearGradient, LuminanceRange, MAX_COMPONENTS, MAX_MASKS, MAX_RADIUS,
    MIN_RADIUS, Mask, MaskOp, MaskSource, RadialGradient, free_name,
};

use crate::adjust::AdjustState;
use crate::brush_tool::BrushTool;

/// What a button of a brush component asked for. The session carries it out
/// once the section is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushRequest {
    /// The Paint button of this component: arm it, or put it down.
    Paint(usize),
    /// Clear strokes.
    Clear(usize),
}

/// What the section did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// The edit changed: the picture and the file are stale.
    pub edited: bool,
    /// Only what is shown changed (the selection, the overlay): the picture
    /// is stale and the file is not.
    pub view_changed: bool,
    pub brush: Option<BrushRequest>,
}

impl Outcome {
    /// Takes in what a part of the section did, so nothing a part asks for
    /// is lost on the way to the session.
    fn take(&mut self, part: Outcome) {
        self.edited |= part.edited;
        self.view_changed |= part.view_changed;
        self.brush = part.brush.or(self.brush);
    }
}

/// The five sources a new mask or a new component starts from, with the
/// word a button and a mask name use for each.
pub fn new_sources() -> [(&'static str, MaskSource); 5] {
    [
        ("Linear", MaskSource::Linear(LinearGradient::default())),
        ("Radial", MaskSource::Radial(RadialGradient::default())),
        (
            "Luminance",
            MaskSource::Luminance(LuminanceRange::default()),
        ),
        ("Colour", MaskSource::Colour(ColourRange::default())),
        ("Brush", MaskSource::Brush(Brush::default())),
    ]
}

/// Adds a mask of one component at the end of the list under a free name and
/// returns its index, or `None` when the list is full.
pub fn add_mask(masks: &mut Vec<Mask>, base: &str, source: MaskSource) -> Option<usize> {
    if masks.len() >= MAX_MASKS {
        return None;
    }
    let name = free_name(masks, base);
    masks.push(Mask::new(&name, source));
    Some(masks.len() - 1)
}

/// Removes a mask and keeps the selection on the mask it was on; removing
/// the selected mask selects nothing.
pub fn remove_mask(masks: &mut Vec<Mask>, selected: &mut Option<usize>, index: usize) {
    if index >= masks.len() {
        return;
    }
    masks.remove(index);
    *selected = match *selected {
        Some(s) if s == index => None,
        Some(s) if s > index => Some(s - 1),
        other => other,
    };
}

/// Moves a mask one place up (`-1`) or down (`1`) the list, which is the
/// order the masks blend in, and keeps the selection on the same mask.
/// Returns whether it moved.
pub fn move_mask(
    masks: &mut [Mask],
    selected: &mut Option<usize>,
    index: usize,
    by: isize,
) -> bool {
    let Some(to) = index.checked_add_signed(by).filter(|to| *to < masks.len()) else {
        return false;
    };
    if index >= masks.len() || to == index {
        return false;
    }
    masks.swap(index, to);
    *selected = match *selected {
        Some(s) if s == index => Some(to),
        Some(s) if s == to => Some(index),
        other => other,
    };
    true
}

/// The name a rename leaves: trimmed, and the old one when the new one is
/// empty or another mask has it.
pub fn renamed(masks: &[Mask], index: usize, wanted: &str) -> String {
    let wanted = wanted.trim();
    let taken = masks
        .iter()
        .enumerate()
        .any(|(i, m)| i != index && m.name.eq_ignore_ascii_case(wanted));
    if wanted.is_empty() || taken {
        masks[index].name.clone()
    } else {
        wanted.to_string()
    }
}

fn number(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    speed: f32,
) -> bool {
    ui.horizontal(|ui| {
        let changed = ui
            .add(
                egui::DragValue::new(value)
                    .range(range)
                    .speed(speed)
                    .fixed_decimals(1),
            )
            .changed();
        ui.label(label);
        changed
    })
    .inner
}

fn point(ui: &mut egui::Ui, label: &str, value: &mut [f32; 2]) -> bool {
    ui.horizontal(|ui| {
        let mut changed = false;
        for axis in value.iter_mut() {
            changed |= ui
                .add(
                    egui::DragValue::new(axis)
                        .range(-1.0..=2.0)
                        .speed(0.002)
                        .fixed_decimals(3),
                )
                .changed();
        }
        ui.label(label);
        changed
    })
    .inner
}

/// The numeric fields of one source. Returns whether one changed.
fn source_fields(ui: &mut egui::Ui, source: &mut MaskSource) -> bool {
    let mut changed = false;
    match source {
        MaskSource::Linear(gradient) => {
            changed |= point(ui, "Start (no effect)", &mut gradient.start);
            changed |= point(ui, "End (full effect)", &mut gradient.end);
        }
        MaskSource::Radial(gradient) => {
            changed |= point(ui, "Centre", &mut gradient.centre);
            ui.horizontal(|ui| {
                for radius in gradient.radius.iter_mut() {
                    changed |= ui
                        .add(
                            egui::DragValue::new(radius)
                                .range(MIN_RADIUS..=MAX_RADIUS)
                                .speed(0.002)
                                .fixed_decimals(3),
                        )
                        .changed();
                }
                ui.label("Radii");
            });
            changed |= number(ui, "Rotation", &mut gradient.rotation, -180.0..=180.0, 0.5);
            changed |= ui
                .add(egui::Slider::new(&mut gradient.feather, 0.0..=100.0).text("Feather"))
                .changed();
        }
        MaskSource::Luminance(range) => {
            changed |= ui
                .add(
                    egui::Slider::new(&mut range.low, 0.0..=1.0)
                        .text("Low")
                        .fixed_decimals(2),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut range.high, 0.0..=1.0)
                        .text("High")
                        .fixed_decimals(2),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut range.falloff, 0.0..=0.5)
                        .text("Falloff")
                        .fixed_decimals(2),
                )
                .changed();
            if range.low > range.high {
                (range.low, range.high) = (range.high, range.low);
            }
        }
        MaskSource::Colour(range) => {
            changed |= ui
                .add(
                    egui::Slider::new(&mut range.hue, 0.0..=360.0)
                        .text("Hue")
                        .fixed_decimals(0),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut range.hue_width, 0.0..=360.0).text("Width"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut range.falloff, 0.0..=90.0).text("Falloff"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut range.chroma_low, 0.0..=0.3)
                        .text("Least chroma")
                        .fixed_decimals(3),
                )
                .changed();
        }
        // A brush has no numbers of its own: see `brush_fields`.
        MaskSource::Brush(_) => {}
    }
    changed
}

/// The size of the brush as the slider shows it: percent of the longer side.
const SIZE_PERCENT: std::ops::RangeInclusive<f32> = MIN_BRUSH_SIZE * 100.0..=MAX_BRUSH_SIZE * 100.0;

/// What a brush component shows: the Paint button, the settings of the
/// brush, how many strokes it holds and Clear strokes.
fn brush_fields(
    ui: &mut egui::Ui,
    brush: &Brush,
    index: usize,
    tool: &mut BrushTool,
) -> Option<BrushRequest> {
    let mut request = None;
    let armed = tool.armed == Some(index);
    ui.horizontal(|ui| {
        let label = if armed { "Painting..." } else { "Paint" };
        let button = ui
            .add(egui::Button::new(label).selected(armed))
            .on_hover_text("Paint this mask on the picture. Esc or Done puts the brush down.");
        if button.clicked() {
            request = Some(BrushRequest::Paint(index));
        }
        ui.checkbox(&mut tool.erase, "Erase")
            .on_hover_text("Strokes take away what was painted. Alt held does the same.");
    });
    let mut percent = tool.size * 100.0;
    let size = egui::Slider::new(&mut percent, SIZE_PERCENT)
        .logarithmic(true)
        .fixed_decimals(2)
        .suffix(" %")
        .text("Size");
    if ui
        .add(size)
        .on_hover_text("The radius, as a share of the longer side. [ and ]")
        .changed()
    {
        tool.size = (percent / 100.0).clamp(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE);
    }
    ui.add(egui::Slider::new(&mut tool.feather, 0.0..=100.0).text("Feather"))
        .on_hover_text("How much of the radius the edge fades over. Shift+[ and Shift+]");
    ui.add(egui::Slider::new(&mut tool.flow, MIN_FLOW..=100.0).text("Flow"))
        .on_hover_text("How much one dab paints; passes over the same place build up");
    ui.horizontal(|ui| {
        let count = brush.strokes.len();
        ui.label(match count {
            1 => "1 stroke".to_string(),
            count => format!("{count} strokes"),
        });
        if ui
            .add_enabled(count > 0, egui::Button::new("Clear strokes").small())
            .clicked()
        {
            request = Some(BrushRequest::Clear(index));
        }
    });
    request
}

/// The list of masks. Returns whether the edit changed.
fn mask_list(ui: &mut egui::Ui, masks: &mut Vec<Mask>, adjust: &mut AdjustState) -> Outcome {
    let mut outcome = Outcome::default();
    let mut removed = None;
    let mut moved = None;
    let count = masks.len();
    for index in 0..count {
        ui.horizontal_wrapped(|ui| {
            outcome.edited |= ui
                .checkbox(&mut masks[index].enabled, "")
                .on_hover_text("Apply this mask")
                .changed();
            let selected = adjust.selected_mask == Some(index);
            if ui.selectable_label(selected, &masks[index].name).clicked() {
                adjust.select_mask((!selected).then_some(index));
                outcome.view_changed = true;
            }
            if ui
                .add_enabled(index > 0, egui::Button::new("Up").small())
                .on_hover_text("Blend this mask earlier")
                .clicked()
            {
                moved = Some((index, -1));
            }
            if ui
                .add_enabled(index + 1 < count, egui::Button::new("Down").small())
                .on_hover_text("Blend this mask later")
                .clicked()
            {
                moved = Some((index, 1));
            }
            if ui.small_button("Rename").clicked() {
                adjust.mask_renaming = Some((index, masks[index].name.clone()));
            }
            if ui.small_button("Delete").clicked() {
                removed = Some(index);
            }
        });
        if let Some((renaming, name)) = &mut adjust.mask_renaming
            && *renaming == index
        {
            let mut done = false;
            let mut cancelled = false;
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(name).desired_width(140.0));
                done = ui.small_button("OK").clicked();
                cancelled = ui.small_button("Cancel").clicked();
            });
            if done {
                let name = renamed(masks, index, name);
                if masks[index].name != name {
                    masks[index].name = name;
                    outcome.edited = true;
                }
            }
            if done || cancelled {
                adjust.mask_renaming = None;
            }
        }
    }
    if let Some((index, by)) = moved
        && move_mask(masks, &mut adjust.selected_mask, index, by)
    {
        adjust.mask_renaming = None;
        outcome.edited = true;
    }
    if let Some(index) = removed {
        let was_selected = adjust.selected_mask == Some(index);
        remove_mask(masks, &mut adjust.selected_mask, index);
        if was_selected {
            adjust.select_mask(None);
        }
        adjust.mask_renaming = None;
        outcome.edited = true;
    }
    outcome
}

/// What the selected mask holds apart from its adjustments. `can_pick` is
/// false when no photo is open to pick a range from.
fn selected_mask(
    ui: &mut egui::Ui,
    mask: &mut Mask,
    adjust: &mut AdjustState,
    can_pick: bool,
) -> Outcome {
    let mut outcome = Outcome::default();
    outcome.edited |= ui
        .add(egui::Slider::new(&mut mask.opacity, 0.0..=100.0).text("Opacity"))
        .changed();
    ui.horizontal(|ui| {
        outcome.edited |= ui.checkbox(&mut mask.invert, "Invert").changed();
        outcome.view_changed |= ui
            .checkbox(&mut adjust.mask_overlay, "Show overlay")
            .on_hover_text("Show where this mask applies in red")
            .changed();
    });

    let mut removed = None;
    let count = mask.components.len();
    for (index, component) in mask.components.iter_mut().enumerate() {
        ui.separator();
        ui.horizontal(|ui| {
            ui.strong(component.source.label());
            if ui
                .add_enabled(count > 1, egui::Button::new("Delete").small())
                .clicked()
            {
                removed = Some(index);
            }
        });
        ui.horizontal(|ui| {
            for (op, label) in MaskOp::ALL {
                // The first component joins an empty alpha: adding is the
                // only operator that gives it anything.
                let usable = index > 0 || op == MaskOp::Add;
                let chosen = component.op == op;
                if ui
                    .add_enabled(usable, egui::Button::selectable(chosen, label))
                    .clicked()
                    && !chosen
                {
                    component.op = op;
                    outcome.edited = true;
                }
            }
            outcome.edited |= ui.checkbox(&mut component.invert, "Invert").changed();
        });
        outcome.edited |= source_fields(ui, &mut component.source);
        if let MaskSource::Brush(brush) = &component.source
            && let Some(request) = brush_fields(ui, brush, index, &mut adjust.brush)
        {
            outcome.brush = Some(request);
        }
        if component.source.reads_the_pixel() {
            let armed = adjust.picking == Some(index);
            let label = if armed {
                "Click the picture..."
            } else {
                "Pick from the picture"
            };
            let button = ui
                .add_enabled(can_pick, egui::Button::new(label).selected(armed))
                .on_disabled_hover_text("A range is picked from an open photo");
            if button.clicked() {
                adjust.picking = (!armed).then_some(index);
            }
        }
    }
    if let Some(index) = removed {
        mask.components.remove(index);
        if let Some(first) = mask.components.first_mut() {
            first.op = MaskOp::Add;
        }
        adjust.picking = None;
        // The armed component may be the one that went, or have moved up.
        outcome.view_changed |= adjust.put_brush_down();
        outcome.edited = true;
    }

    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label("Add component:");
        let room = mask.components.len() < MAX_COMPONENTS;
        for (word, source) in new_sources() {
            if ui
                .add_enabled(room, egui::Button::new(word).small())
                .clicked()
            {
                mask.components.push(Component::new(source));
                outcome.edited = true;
            }
        }
    });
    outcome
}

/// The whole section.
pub fn show(
    ui: &mut egui::Ui,
    masks: &mut Vec<Mask>,
    adjust: &mut AdjustState,
    can_pick: bool,
) -> Outcome {
    let mut outcome = Outcome::default();
    ui.horizontal_wrapped(|ui| {
        let room = masks.len() < MAX_MASKS;
        for (word, source) in new_sources() {
            let label = match word {
                "Brush" => "New brush".to_string(),
                "Luminance" => "New luminance range".to_string(),
                "Colour" => "New colour range".to_string(),
                word => format!("New {}", word.to_lowercase()),
            };
            if ui.add_enabled(room, egui::Button::new(label)).clicked()
                && let Some(index) = add_mask(masks, word, source)
            {
                adjust.select_mask(Some(index));
                outcome.edited = true;
            }
        }
    });
    if masks.is_empty() {
        ui.weak("No masks yet. A mask adjusts part of the picture on top of the global edit.");
        return outcome;
    }
    if masks.len() >= MAX_MASKS {
        ui.weak(format!("An edit holds {MAX_MASKS} masks."));
    }
    ui.weak("Masks blend in list order, each over the ones above it.");
    let listed = mask_list(ui, masks, adjust);
    outcome.take(listed);

    if let Some(index) = adjust.selected_mask.filter(|index| *index < masks.len()) {
        ui.separator();
        let chosen = selected_mask(ui, &mut masks[index], adjust, can_pick);
        outcome.take(chosen);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(names: &[&str]) -> Vec<Mask> {
        names
            .iter()
            .map(|name| Mask::new(name, MaskSource::default()))
            .collect()
    }

    fn names(masks: &[Mask]) -> Vec<&str> {
        masks.iter().map(|m| m.name.as_str()).collect()
    }

    #[test]
    fn a_new_mask_takes_a_free_name_and_the_list_holds_eight() {
        let mut masks = Vec::new();
        let [(word, source), ..] = new_sources();
        assert_eq!(add_mask(&mut masks, word, source.clone()), Some(0));
        assert_eq!(add_mask(&mut masks, word, source.clone()), Some(1));
        assert_eq!(names(&masks), ["Linear", "Linear 2"]);
        assert!(masks[0].enabled && masks[0].components.len() == 1);
        while masks.len() < MAX_MASKS {
            add_mask(&mut masks, "Radial", source.clone()).expect("room");
        }
        assert_eq!(add_mask(&mut masks, "Radial", source), None);
        assert_eq!(masks.len(), MAX_MASKS);
    }

    #[test]
    fn what_a_part_of_the_section_asks_for_reaches_the_whole() {
        let mut whole = Outcome {
            edited: true,
            ..Outcome::default()
        };
        whole.take(Outcome {
            view_changed: true,
            brush: Some(BrushRequest::Paint(2)),
            ..Outcome::default()
        });
        assert_eq!(
            whole,
            Outcome {
                edited: true,
                view_changed: true,
                brush: Some(BrushRequest::Paint(2)),
            }
        );
        // A part that asks for nothing takes nothing away.
        whole.take(Outcome::default());
        assert_eq!(whole.brush, Some(BrushRequest::Paint(2)));
        whole.take(Outcome {
            brush: Some(BrushRequest::Clear(0)),
            ..Outcome::default()
        });
        assert_eq!(whole.brush, Some(BrushRequest::Clear(0)));
    }

    #[test]
    fn there_is_a_new_button_for_each_of_the_five_sources() {
        let labels: Vec<&str> = new_sources()
            .iter()
            .map(|(_, source)| source.label())
            .collect();
        assert_eq!(
            labels,
            [
                "Linear gradient",
                "Radial gradient",
                "Luminance range",
                "Colour range",
                "Brush"
            ]
        );
    }

    #[test]
    fn removing_a_mask_keeps_the_selection_on_the_same_mask() {
        let mut masks = named(&["A", "B", "C"]);
        let mut selected = Some(2);
        remove_mask(&mut masks, &mut selected, 0);
        assert_eq!((names(&masks), selected), (vec!["B", "C"], Some(1)));
        remove_mask(&mut masks, &mut selected, 1);
        assert_eq!((names(&masks), selected), (vec!["B"], None));
        let mut selected = Some(0);
        remove_mask(&mut masks, &mut selected, 5);
        assert_eq!((masks.len(), selected), (1, Some(0)));
    }

    #[test]
    fn moving_a_mask_reorders_the_blend_and_follows_the_selection() {
        let mut masks = named(&["A", "B", "C"]);
        let mut selected = Some(0);
        assert!(move_mask(&mut masks, &mut selected, 0, 1));
        assert_eq!((names(&masks), selected), (vec!["B", "A", "C"], Some(1)));
        // The mask that was swapped past keeps the selection when it has it.
        let mut selected = Some(2);
        assert!(move_mask(&mut masks, &mut selected, 1, 1));
        assert_eq!((names(&masks), selected), (vec!["B", "C", "A"], Some(1)));
        assert!(!move_mask(&mut masks, &mut selected, 0, -1), "the top");
        assert!(!move_mask(&mut masks, &mut selected, 2, 1), "the bottom");
        assert_eq!(names(&masks), ["B", "C", "A"]);
    }

    #[test]
    fn a_rename_refuses_an_empty_or_a_taken_name() {
        let masks = named(&["Sky", "Face"]);
        assert_eq!(renamed(&masks, 0, "  Clouds "), "Clouds");
        assert_eq!(renamed(&masks, 0, "face"), "Sky");
        assert_eq!(renamed(&masks, 0, "  "), "Sky");
        assert_eq!(
            renamed(&masks, 0, "SKY"),
            "SKY",
            "its own name in another case"
        );
    }
}
