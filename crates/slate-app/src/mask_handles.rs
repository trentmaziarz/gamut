//! The selected mask on the picture: the handles of its gradients, and the
//! click that picks a luminance or a colour range from the photo.
//!
//! Everything here goes between the picture and the screen through
//! [`PictureMap`], the one place that knows where the photo is drawn. The
//! viewer fits the photo to the tab today; a viewer that zooms and pans
//! builds a different map and the handles follow it.
//!
//! A linear gradient shows its start and its end with the line between
//! them. A radial gradient shows its centre, a handle on each radius and a
//! rotation handle past the first radius, over the outline of its ellipse.
//! A range is picked with a click while its Pick button is armed; the click
//! reads the source pixel on the CPU from a small copy of the decoded photo,
//! through slate-color, never from the GPU.

use egui::{Color32, CursorIcon, Pos2, Rect, Sense, Stroke, Vec2};
use slate_color::mask::CHROMA_RAMP;
use slate_color::{SourceSpace, acescct, basic, hue, wheels};
use slate_core::mask::{
    ColourRange, LinearGradient, LuminanceRange, MAX_RADIUS, MIN_RADIUS, MaskSource, RadialGradient,
};
use slate_media::Photo;

use crate::app::Session;

/// The radius of a handle in points, and of the area that grabs it.
const HANDLE_RADIUS: f32 = 6.0;
const GRAB_RADIUS: f32 = 11.0;

/// How far past the first radius the rotation handle sits, in points.
const ROTATION_REACH: f32 = 28.0;

/// Half the length of the bars across the ends of a linear gradient.
const BAR_HALF_LENGTH: f32 = 70.0;

/// The segments of the outline of a radial gradient.
const OUTLINE_SEGMENTS: usize = 72;

/// The longer side of the copy of the photo a pick reads.
const PICK_SIDE: u32 = 512;

/// The narrowest luminance range a pick leaves.
const PICK_LUMINANCE_WIDTH: f32 = 0.1;

/// Where the photo is on the screen. Positions on the picture are
/// normalised to the uncropped photo, as a mask stores them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PictureMap {
    /// The rectangle the whole photo is drawn in, in points.
    pub image: Rect,
}

impl PictureMap {
    /// The picture-to-screen mapping: a normalised position in points.
    pub fn to_screen(&self, at: [f32; 2]) -> Pos2 {
        Pos2::new(
            self.image.min.x + at[0] * self.image.width(),
            self.image.min.y + at[1] * self.image.height(),
        )
    }

    /// The inverse: a point on the screen as a normalised position.
    pub fn to_picture(&self, pos: Pos2) -> [f32; 2] {
        [
            (pos.x - self.image.min.x) / self.image.width().max(1e-6),
            (pos.y - self.image.min.y) / self.image.height().max(1e-6),
        ]
    }

    /// The longer side of the photo in points: what a length stored as a
    /// fraction of the longer side is multiplied by.
    pub fn long_side(&self) -> f32 {
        self.image.width().max(self.image.height())
    }
}

/// Where the four handles of a radial gradient are on the screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialHandles {
    pub centre: Pos2,
    /// On the first radius, along the turned x axis.
    pub radius_x: Pos2,
    /// On the second radius, along the turned y axis.
    pub radius_y: Pos2,
    /// Past the first radius: dragging it around the centre turns the mask.
    pub rotation: Pos2,
}

/// The two axes of a radial gradient on the screen, as unit vectors. The y
/// of the screen runs down, as the y of the picture does, so a positive
/// rotation turns clockwise on both.
fn radial_axes(gradient: &RadialGradient) -> (Vec2, Vec2) {
    let (sin, cos) = gradient.rotation.to_radians().sin_cos();
    (Vec2::new(cos, sin), Vec2::new(-sin, cos))
}

pub fn radial_handles(map: &PictureMap, gradient: &RadialGradient) -> RadialHandles {
    let centre = map.to_screen(gradient.centre);
    let (x_axis, y_axis) = radial_axes(gradient);
    let long = map.long_side();
    RadialHandles {
        centre,
        radius_x: centre + x_axis * (gradient.radius[0] * long),
        radius_y: centre + y_axis * (gradient.radius[1] * long),
        rotation: centre + x_axis * (gradient.radius[0] * long + ROTATION_REACH),
    }
}

/// The radius the pointer asks for on one axis (0 or 1): how far along the
/// axis it is from the centre, either side.
pub fn dragged_radius(
    map: &PictureMap,
    gradient: &RadialGradient,
    axis: usize,
    pointer: Pos2,
) -> f32 {
    let (x_axis, y_axis) = radial_axes(gradient);
    let along = if axis == 0 { x_axis } else { y_axis };
    let reach = (pointer - map.to_screen(gradient.centre)).dot(along).abs();
    (reach / map.long_side().max(1e-6)).clamp(MIN_RADIUS, MAX_RADIUS)
}

/// The rotation in degrees the pointer asks for: the direction from the
/// centre to it.
pub fn dragged_rotation(map: &PictureMap, gradient: &RadialGradient, pointer: Pos2) -> f32 {
    let away = pointer - map.to_screen(gradient.centre);
    if away.length() < 1.0 {
        return gradient.rotation;
    }
    away.y.atan2(away.x).to_degrees()
}

/// The outline of a radial gradient on the screen, closed.
pub fn radial_outline(map: &PictureMap, gradient: &RadialGradient) -> Vec<Pos2> {
    let centre = map.to_screen(gradient.centre);
    let (x_axis, y_axis) = radial_axes(gradient);
    let long = map.long_side();
    (0..=OUTLINE_SEGMENTS)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / OUTLINE_SEGMENTS as f32;
            centre
                + x_axis * (angle.cos() * gradient.radius[0] * long)
                + y_axis * (angle.sin() * gradient.radius[1] * long)
        })
        .collect()
}

/// A small copy of the decoded photo for the pick to read.
#[derive(Clone, Debug)]
pub struct PickImage {
    width: u32,
    height: u32,
    rgb: Vec<[u8; 3]>,
    space: SourceSpace,
}

impl PickImage {
    /// The photo averaged down until its longer side is [`PICK_SIDE`] at
    /// most: each pixel of the copy is the mean of the block it covers.
    pub fn from_photo(photo: &Photo) -> Self {
        let step = photo.width.max(photo.height).div_ceil(PICK_SIDE).max(1);
        let (width, height) = (
            photo.width.div_ceil(step).max(1),
            photo.height.div_ceil(step).max(1),
        );
        let mut rgb = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0u32; 3];
                let mut count = 0u32;
                for sy in (y * step)..((y + 1) * step).min(photo.height) {
                    for sx in (x * step)..((x + 1) * step).min(photo.width) {
                        let at = ((sy * photo.width + sx) * 4) as usize;
                        for (c, total) in sum.iter_mut().enumerate() {
                            *total += u32::from(photo.rgba8[at + c]);
                        }
                        count += 1;
                    }
                }
                rgb.push(sum.map(|total| (total / count.max(1)) as u8));
            }
        }
        PickImage {
            width,
            height,
            rgb,
            space: photo.source,
        }
    }

    /// The source pixel under a normalised position, in linear Rec.2020:
    /// what a range mask measures. `None` outside the photo.
    pub fn source_pixel(&self, at: [f32; 2]) -> Option<[f32; 3]> {
        if !(0.0..1.0).contains(&at[0]) || !(0.0..1.0).contains(&at[1]) {
            return None;
        }
        let x = ((at[0] * self.width as f32) as u32).min(self.width - 1);
        let y = ((at[1] * self.height as f32) as u32).min(self.height - 1);
        let rgb = self.rgb[(y * self.width + x) as usize];
        Some(basic::decode_rgb8(rgb, self.space))
    }
}

/// A range moved onto a picked source pixel. A luminance range keeps its
/// width (no narrower than [`PICK_LUMINANCE_WIDTH`]) and centres on the tone
/// of the pixel. A colour range takes the hue of the pixel and lowers its
/// chroma floor until the pixel is fully inside. A gradient is left as it is.
pub fn picked(source: &MaskSource, px: [f32; 3]) -> MaskSource {
    let v = acescct::encode_pixel(px);
    match *source {
        MaskSource::Luminance(range) => {
            let n = wheels::tone(v);
            let half = (range.high - range.low).max(PICK_LUMINANCE_WIDTH) / 2.0;
            MaskSource::Luminance(LuminanceRange {
                low: (n - half).clamp(0.0, 1.0),
                high: (n + half).clamp(0.0, 1.0),
                ..range
            })
        }
        MaskSource::Colour(range) => {
            let plane = hue::chroma_plane(v);
            MaskSource::Colour(ColourRange {
                hue: hue::hue(plane).to_degrees().rem_euclid(360.0),
                chroma_low: range
                    .chroma_low
                    .min((hue::chroma(plane) - CHROMA_RAMP).max(0.0)),
                ..range
            })
        }
        other => other,
    }
}

fn handle(ui: &egui::Ui, at: Pos2, id: egui::Id) -> egui::Response {
    let area = Rect::from_center_size(at, Vec2::splat(GRAB_RADIUS * 2.0));
    ui.interact(area, id, Sense::drag())
        .on_hover_cursor(CursorIcon::Grab)
}

fn paint_handle(painter: &egui::Painter, at: Pos2, filled: bool, hot: bool) {
    let radius = if hot {
        HANDLE_RADIUS + 1.5
    } else {
        HANDLE_RADIUS
    };
    let fill = if filled {
        Color32::WHITE
    } else {
        Color32::from_black_alpha(140)
    };
    painter.circle(at, radius, fill, Stroke::new(2.0, Color32::from_gray(20)));
    painter.circle_stroke(at, radius - 1.5, Stroke::new(1.5, Color32::WHITE));
}

/// A line drawn twice, dark under light, so it reads on any picture.
fn paint_line(painter: &egui::Painter, points: [Pos2; 2]) {
    painter.line_segment(points, Stroke::new(3.0, Color32::from_black_alpha(150)));
    painter.line_segment(points, Stroke::new(1.0, Color32::WHITE));
}

/// The handles of one linear gradient. Returns the gradient when a drag
/// moved it.
fn linear_handles(
    ui: &egui::Ui,
    map: &PictureMap,
    id: egui::Id,
    gradient: &LinearGradient,
) -> Option<LinearGradient> {
    let mut moved = *gradient;
    let mut changed = false;
    let mut hot = [false; 2];
    for (k, point) in [&mut moved.start, &mut moved.end].into_iter().enumerate() {
        let response = handle(ui, map.to_screen(*point), id.with(k));
        hot[k] = response.hovered() || response.dragged();
        if response.dragged()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            *point = map.to_picture(pointer);
            changed = true;
        }
    }
    let painter = ui.painter().with_clip_rect(map.image.expand(40.0));
    let (start, end) = (map.to_screen(moved.start), map.to_screen(moved.end));
    paint_line(&painter, [start, end]);
    let along = end - start;
    if along.length() > 1.0 {
        let across = Vec2::new(-along.y, along.x).normalized() * BAR_HALF_LENGTH;
        paint_line(&painter, [start - across, start + across]);
        paint_line(&painter, [end - across, end + across]);
    }
    paint_handle(&painter, start, false, hot[0]);
    paint_handle(&painter, end, true, hot[1]);
    changed.then_some(moved)
}

/// The handles of one radial gradient. Returns the gradient when a drag
/// changed it.
fn radial_gradient_handles(
    ui: &egui::Ui,
    map: &PictureMap,
    id: egui::Id,
    gradient: &RadialGradient,
) -> Option<RadialGradient> {
    let mut moved = *gradient;
    let mut changed = false;
    let at = radial_handles(map, gradient);
    let places = [at.centre, at.radius_x, at.radius_y, at.rotation];
    let mut hot = [false; 4];
    for (k, place) in places.into_iter().enumerate() {
        let response = handle(ui, place, id.with(k));
        hot[k] = response.hovered() || response.dragged();
        let Some(pointer) = response
            .interact_pointer_pos()
            .filter(|_| response.dragged())
        else {
            continue;
        };
        match k {
            0 => moved.centre = map.to_picture(pointer),
            1 => moved.radius[0] = dragged_radius(map, gradient, 0, pointer),
            2 => moved.radius[1] = dragged_radius(map, gradient, 1, pointer),
            _ => moved.rotation = dragged_rotation(map, gradient, pointer),
        }
        changed = true;
    }
    let painter = ui.painter().with_clip_rect(map.image.expand(40.0));
    let outline = radial_outline(map, &moved);
    painter.add(egui::Shape::line(
        outline.clone(),
        Stroke::new(3.0, Color32::from_black_alpha(150)),
    ));
    painter.add(egui::Shape::line(outline, Stroke::new(1.0, Color32::WHITE)));
    let at = radial_handles(map, &moved);
    paint_line(&painter, [at.radius_x, at.rotation]);
    paint_handle(&painter, at.centre, true, hot[0]);
    paint_handle(&painter, at.radius_x, false, hot[1]);
    paint_handle(&painter, at.radius_y, false, hot[2]);
    painter.circle(
        at.rotation,
        if hot[3] { 5.5 } else { 4.0 },
        Color32::from_rgb(255, 196, 60),
        Stroke::new(1.5, Color32::from_gray(20)),
    );
    changed.then_some(moved)
}

/// Draws the handles of the selected mask over the picture and takes the
/// pick click while a range is armed. Every change goes through
/// `Session::mark_edited`.
pub fn show(ui: &mut egui::Ui, map: &PictureMap, session: &mut Session) {
    let Some(selected) = session.adjust.selected_mask else {
        return;
    };
    if selected >= session.edit.masks.len() {
        return;
    }
    let base = ui.id().with(("mask handles", selected));
    let mut changed = false;

    // A pick takes the next click on the picture, ahead of any handle.
    let armed = session.adjust.picking.filter(|component| {
        session.edit.masks[selected]
            .components
            .get(*component)
            .is_some_and(|c| c.source.reads_the_pixel())
    });
    if session.adjust.picking.is_some() && armed.is_none() {
        session.adjust.picking = None;
    }
    if let Some(component) = armed {
        let response = ui
            .interact(map.image, base.with("pick"), Sense::click())
            .on_hover_cursor(CursorIcon::Crosshair);
        if response.clicked()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let px = session
                .pick
                .as_ref()
                .and_then(|image| image.source_pixel(map.to_picture(pointer)));
            if let Some(px) = px {
                let slot = &mut session.edit.masks[selected].components[component].source;
                let now = picked(slot, px);
                if *slot != now {
                    *slot = now;
                    changed = true;
                }
            }
            session.adjust.picking = None;
        }
    } else {
        let components = &mut session.edit.masks[selected].components;
        for (index, component) in components.iter_mut().enumerate() {
            let id = base.with(index);
            match &mut component.source {
                MaskSource::Linear(gradient) => {
                    if let Some(moved) = linear_handles(ui, map, id, gradient) {
                        *gradient = moved;
                        changed = true;
                    }
                }
                MaskSource::Radial(gradient) => {
                    if let Some(moved) = radial_gradient_handles(ui, map, id, gradient) {
                        *gradient = moved;
                        changed = true;
                    }
                }
                MaskSource::Luminance(_) | MaskSource::Colour(_) => {}
            }
        }
    }
    if changed {
        session.mark_edited();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> PictureMap {
        PictureMap {
            image: Rect::from_min_size(Pos2::new(100.0, 50.0), Vec2::new(400.0, 200.0)),
        }
    }

    fn near(a: Pos2, b: Pos2) -> bool {
        (a - b).length() < 1e-3
    }

    #[test]
    fn the_picture_maps_onto_the_image_rectangle_and_back() {
        let map = map();
        assert_eq!(map.to_screen([0.0, 0.0]), Pos2::new(100.0, 50.0));
        assert_eq!(map.to_screen([1.0, 1.0]), Pos2::new(500.0, 250.0));
        assert_eq!(map.to_screen([0.25, 0.5]), Pos2::new(200.0, 150.0));
        assert_eq!(map.long_side(), 400.0);
        for at in [[0.0, 0.0], [0.3, 0.8], [1.2, -0.1]] {
            let back = map.to_picture(map.to_screen(at));
            assert!((back[0] - at[0]).abs() < 1e-6 && (back[1] - at[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn the_handles_of_a_radial_gradient_follow_its_radii_and_its_rotation() {
        let map = map();
        let mut gradient = RadialGradient {
            centre: [0.5, 0.5],
            radius: [0.25, 0.1],
            rotation: 0.0,
            feather: 50.0,
        };
        let at = radial_handles(&map, &gradient);
        assert!(near(at.centre, Pos2::new(300.0, 150.0)));
        // A quarter of the longer side is 100 points, on either axis.
        assert!(near(at.radius_x, Pos2::new(400.0, 150.0)));
        assert!(near(at.radius_y, Pos2::new(300.0, 190.0)));
        assert!(near(at.rotation, Pos2::new(400.0 + ROTATION_REACH, 150.0)));
        // Turned a quarter clockwise the first radius points down the screen.
        gradient.rotation = 90.0;
        let at = radial_handles(&map, &gradient);
        assert!(near(at.radius_x, Pos2::new(300.0, 250.0)));
        assert!(near(at.radius_y, Pos2::new(260.0, 150.0)));
    }

    #[test]
    fn dragging_a_radius_handle_gives_the_radius_back() {
        let map = map();
        let gradient = RadialGradient {
            centre: [0.4, 0.6],
            radius: [0.3, 0.12],
            rotation: 35.0,
            feather: 20.0,
        };
        let at = radial_handles(&map, &gradient);
        assert!((dragged_radius(&map, &gradient, 0, at.radius_x) - 0.3).abs() < 1e-5);
        assert!((dragged_radius(&map, &gradient, 1, at.radius_y) - 0.12).abs() < 1e-5);
        // Only the part of the drag along the axis counts.
        let (_, y_axis) = radial_axes(&gradient);
        let aside = at.radius_x + y_axis * 40.0;
        assert!((dragged_radius(&map, &gradient, 0, aside) - 0.3).abs() < 1e-5);
        // A radius never collapses.
        assert_eq!(dragged_radius(&map, &gradient, 0, at.centre), MIN_RADIUS);
    }

    #[test]
    fn dragging_the_rotation_handle_gives_the_rotation_back() {
        let map = map();
        let mut gradient = RadialGradient::default();
        for degrees in [-135.0, -20.0, 0.0, 35.0, 90.0, 170.0] {
            gradient.rotation = degrees;
            let at = radial_handles(&map, &gradient);
            let back = dragged_rotation(&map, &gradient, at.rotation);
            assert!((back - degrees).abs() < 1e-3, "{degrees} gave {back}");
        }
        let centre = radial_handles(&map, &gradient).centre;
        assert_eq!(dragged_rotation(&map, &gradient, centre), gradient.rotation);
    }

    #[test]
    fn the_outline_passes_through_the_radius_handles() {
        let map = map();
        let gradient = RadialGradient {
            centre: [0.5, 0.5],
            radius: [0.2, 0.1],
            rotation: 30.0,
            feather: 0.0,
        };
        let outline = radial_outline(&map, &gradient);
        let at = radial_handles(&map, &gradient);
        assert_eq!(outline.len(), OUTLINE_SEGMENTS + 1);
        assert!(near(outline[0], at.radius_x));
        assert!(near(outline[OUTLINE_SEGMENTS / 4], at.radius_y));
        assert!(near(outline[0], outline[OUTLINE_SEGMENTS]));
    }

    fn photo(width: u32, height: u32, colour: impl Fn(u32, u32) -> [u8; 3]) -> Photo {
        let mut rgba8 = Vec::new();
        for y in 0..height {
            for x in 0..width {
                rgba8.extend_from_slice(&colour(x, y));
                rgba8.push(255);
            }
        }
        Photo {
            width,
            height,
            rgba8,
            source: SourceSpace::Srgb,
            bit_depth: 8,
            has_alpha: false,
        }
    }

    #[test]
    fn the_pick_image_averages_blocks_and_reads_by_position() {
        // Left half dark, right half bright, 2048 wide: blocks of four.
        let photo = photo(2048, 8, |x, _| if x < 1024 { [20; 3] } else { [220; 3] });
        let image = PickImage::from_photo(&photo);
        assert_eq!((image.width, image.height), (512, 2));
        let dark = image.source_pixel([0.25, 0.5]).expect("inside");
        let bright = image.source_pixel([0.75, 0.5]).expect("inside");
        assert_eq!(dark, basic::decode_rgb8([20; 3], SourceSpace::Srgb));
        assert_eq!(bright, basic::decode_rgb8([220; 3], SourceSpace::Srgb));
        assert_eq!(image.source_pixel([1.0, 0.5]), None);
        assert_eq!(image.source_pixel([0.5, -0.01]), None);
        // A photo smaller than the copy is kept as it is.
        let small = PickImage::from_photo(&self::photo(3, 2, |x, y| [x as u8, y as u8, 7]));
        assert_eq!((small.width, small.height), (3, 2));
        assert_eq!(small.rgb[5], [2, 1, 7]);
    }

    #[test]
    fn a_picked_luminance_range_centres_on_the_pixel_and_keeps_its_width() {
        let range = MaskSource::Luminance(LuminanceRange {
            low: 0.1,
            high: 0.4,
            falloff: 0.07,
        });
        let px = [acescct::decode(acescct::denormalise(0.7)); 3];
        let MaskSource::Luminance(now) = picked(&range, px) else {
            panic!("still a luminance range");
        };
        assert!((now.low - 0.55).abs() < 1e-4 && (now.high - 0.85).abs() < 1e-4);
        assert_eq!(now.falloff, 0.07);
        // Near white the range is held at the end of the axis.
        let white = picked(&range, [1.0; 3]);
        let MaskSource::Luminance(now) = white else {
            panic!("still a luminance range");
        };
        assert_eq!(now.high, 1.0);
        assert!((now.low - 0.85).abs() < 1e-4);
    }

    #[test]
    fn a_picked_colour_range_takes_the_hue_and_lets_the_pixel_in() {
        let range = ColourRange {
            hue: 10.0,
            hue_width: 50.0,
            chroma_low: 0.08,
            falloff: 12.0,
        };
        let px = basic::decode_rgb8([40, 90, 200], SourceSpace::Srgb);
        let MaskSource::Colour(now) = picked(&MaskSource::Colour(range), px) else {
            panic!("still a colour range");
        };
        let plane = hue::chroma_plane(acescct::encode_pixel(px));
        assert!((now.hue - hue::hue(plane).to_degrees()).abs() < 1e-3);
        assert!(now.chroma_low <= hue::chroma(plane) - CHROMA_RAMP + 1e-6);
        assert_eq!((now.hue_width, now.falloff), (50.0, 12.0));
        assert!(
            (slate_color::mask::colour(&now, px) - 1.0).abs() < 1e-3,
            "the picked pixel is inside the range"
        );
        // A gradient is not something a pick changes.
        let linear = MaskSource::default();
        assert_eq!(picked(&linear, px), linear);
    }
}
