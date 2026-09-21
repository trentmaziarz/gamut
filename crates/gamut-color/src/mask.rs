//! Masks on the CPU: the alpha of each source, the combiner, what a mask
//! adjusts on top of the global edit, and the ordered blend of a whole
//! image. mask.wgsl and the masked develop pass in gamut-gpu mirror this
//! module; the golden tests hold the two together.
//!
//! A source is measured on what the GPU has at the same point: where the
//! pixel is in the window of the render, and the source pixel in linear
//! Rec.2020 before any operator, so a mask never moves when its own sliders
//! move.
//!
//! Positions are normalised to the uncropped photo. Distances are measured
//! after scaling by [`Geometry::aspect`], in fractions of the longer side, so
//! a gradient is perpendicular to its line and a radial gradient with equal
//! radii is a circle on a photo of any shape.

use gamut_core::mask::{
    ColourRange, LinearGradient, LuminanceRange, Mask, MaskOp, MaskSource, RadialGradient,
};
use gamut_core::{Adjustments, CropRect, EXPOSURE_LIMIT, PhotoEdit, SLIDER_LIMIT, Wheel};

use crate::basic::{self, Neighbourhood, Prepared};
use crate::brush::{self, Dab};
use crate::{acescct, hue, wheels};

/// The chroma above `chroma_low` at which a colour range is fully on.
pub const CHROMA_RAMP: f32 = 0.01;

/// The narrowest fall of a luminance range, so its edges stay a ramp.
pub const LUMINANCE_FALLOFF_FLOOR: f32 = 0.0001;

/// The narrowest fall of a colour range in degrees.
pub const HUE_FALLOFF_FLOOR: f32 = 0.01;

/// The widest the inside of a radial gradient gets, as a share of its
/// radius, so a feather of 0 is still a ramp and not a division by zero.
pub const RADIAL_INNER_CEILING: f32 = 0.999;

/// How much red the overlay of the selected mask mixes in at a full alpha.
pub const OVERLAY_STRENGTH: f32 = 0.5;

/// The shortest squared length a linear gradient is measured along.
pub const LINEAR_LENGTH_FLOOR: f32 = 1e-12;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Where the pixels of a render are on the photo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// The part of the photo the render shows.
    pub window: CropRect,
    /// The size of the render in pixels.
    pub size: (u32, u32),
    /// The size of the uncropped photo in pixels.
    pub photo: (u32, u32),
}

impl Geometry {
    /// A render of the whole photo.
    pub fn full(size: (u32, u32), photo: (u32, u32)) -> Self {
        Geometry {
            window: CropRect::FULL,
            size,
            photo,
        }
    }

    /// The centre of pixel (x, y) of the render, normalised to the photo.
    pub fn position(&self, x: u32, y: u32) -> [f32; 2] {
        [
            self.window.x + (x as f32 + 0.5) / self.size.0 as f32 * self.window.width,
            self.window.y + (y as f32 + 0.5) / self.size.1 as f32 * self.window.height,
        ]
    }

    /// What a normalised position is scaled by to measure distances: each
    /// side of the photo over its longer side.
    pub fn aspect(&self) -> [f32; 2] {
        let longer = self.photo.0.max(self.photo.1).max(1) as f32;
        [self.photo.0 as f32 / longer, self.photo.1 as f32 / longer]
    }
}

/// A linear gradient at a normalised position: 0 at the start, 1 at the end,
/// a smoothstep between, constant along the perpendicular.
pub fn linear(gradient: &LinearGradient, at: [f32; 2], aspect: [f32; 2]) -> f32 {
    let start = [gradient.start[0] * aspect[0], gradient.start[1] * aspect[1]];
    let end = [gradient.end[0] * aspect[0], gradient.end[1] * aspect[1]];
    let p = [at[0] * aspect[0] - start[0], at[1] * aspect[1] - start[1]];
    let along = [end[0] - start[0], end[1] - start[1]];
    let length = (along[0] * along[0] + along[1] * along[1]).max(LINEAR_LENGTH_FLOOR);
    smoothstep(0.0, 1.0, (p[0] * along[0] + p[1] * along[1]) / length)
}

/// A radial gradient at a normalised position: 1 inside, falling to 0 at the
/// rim of the turned ellipse across the feather band.
pub fn radial(gradient: &RadialGradient, at: [f32; 2], aspect: [f32; 2]) -> f32 {
    let p = [
        (at[0] - gradient.centre[0]) * aspect[0],
        (at[1] - gradient.centre[1]) * aspect[1],
    ];
    let (sin, cos) = gradient.rotation.to_radians().sin_cos();
    let q = [
        (p[0] * cos + p[1] * sin) / gradient.radius[0],
        (p[1] * cos - p[0] * sin) / gradient.radius[1],
    ];
    let distance = (q[0] * q[0] + q[1] * q[1]).sqrt();
    let inner = (1.0 - gradient.feather / 100.0).min(RADIAL_INNER_CEILING);
    1.0 - smoothstep(inner, 1.0, distance)
}

/// A luminance range on a source pixel in linear Rec.2020, by the tone of
/// its ACEScct encoding.
pub fn luminance(range: &LuminanceRange, px: [f32; 3]) -> f32 {
    let n = wheels::tone(acescct::encode_pixel(px));
    let falloff = range.falloff.max(LUMINANCE_FALLOFF_FLOOR);
    smoothstep(range.low - falloff, range.low, n)
        * (1.0 - smoothstep(range.high, range.high + falloff, n))
}

/// A colour range on a source pixel in linear Rec.2020, by the hue and the
/// chroma of its ACEScct encoding.
pub fn colour(range: &ColourRange, px: [f32; 3]) -> f32 {
    let plane = hue::chroma_plane(acescct::encode_pixel(px));
    let degrees = hue::hue(plane).to_degrees();
    let away = ((degrees - range.hue + 180.0).rem_euclid(360.0) - 180.0).abs();
    let half = range.hue_width / 2.0;
    let falloff = range.falloff.max(HUE_FALLOFF_FLOOR);
    let by_hue = 1.0 - smoothstep(half, half + falloff, away);
    let by_chroma = smoothstep(
        range.chroma_low,
        range.chroma_low + CHROMA_RAMP,
        hue::chroma(plane),
    );
    by_hue * by_chroma
}

/// The alpha of one source at a normalised position over a source pixel.
pub fn source_alpha(source: &MaskSource, at: [f32; 2], aspect: [f32; 2], px: [f32; 3]) -> f32 {
    match source {
        MaskSource::Linear(gradient) => linear(gradient, at, aspect),
        MaskSource::Radial(gradient) => radial(gradient, at, aspect),
        MaskSource::Luminance(range) => luminance(range, px),
        MaskSource::Colour(range) => colour(range, px),
        MaskSource::Brush(painted) => brush::brush_alpha(painted, at, aspect),
    }
}

/// Joins the alpha built so far, `a`, with a component's, `b`.
pub fn combine(a: f32, b: f32, op: MaskOp) -> f32 {
    match op {
        MaskOp::Add => a + b - a * b,
        MaskOp::Subtract => a * (1.0 - b),
        MaskOp::Intersect => a * b,
    }
}

/// The alpha of a sanitised mask at a normalised position over a source
/// pixel: its components combined in order from 0, then the mask's invert.
/// The opacity is not part of it.
pub fn alpha(mask: &Mask, at: [f32; 2], aspect: [f32; 2], px: [f32; 3]) -> f32 {
    StampedMask::new(mask, aspect).alpha(at, px, &|layer| layer)
}

/// A sanitised mask with the dabs of its brushes stamped once, for the alpha
/// of many positions.
pub struct StampedMask<'a> {
    mask: &'a Mask,
    aspect: [f32; 2],
    /// The dabs of each component that is a brush.
    dabs: Vec<Option<Vec<Dab>>>,
}

impl<'a> StampedMask<'a> {
    pub fn new(mask: &'a Mask, aspect: [f32; 2]) -> Self {
        let dabs = mask
            .components
            .iter()
            .map(|component| match &component.source {
                MaskSource::Brush(painted) => Some(brush::dabs(painted, aspect)),
                _ => None,
            })
            .collect();
        StampedMask { mask, aspect, dabs }
    }

    /// The alpha at a normalised position over a source pixel. `layer_store`
    /// is the rounding of the layer a brush is stamped into, applied after
    /// every dab.
    pub fn alpha(&self, at: [f32; 2], px: [f32; 3], layer_store: &dyn Fn(f32) -> f32) -> f32 {
        let components = self.mask.components.iter().zip(&self.dabs);
        let combined = components.fold(0.0, |a, (component, dabs)| {
            let b = match dabs {
                Some(dabs) => brush::dabs_alpha(dabs, at, self.aspect, layer_store),
                None => source_alpha(&component.source, at, self.aspect, px),
            };
            combine(a, if component.invert { 1.0 - b } else { b }, component.op)
        });
        if self.mask.invert {
            1.0 - combined
        } else {
            combined
        }
    }
}

/// An alpha as the r8unorm texture of the GPU holds it.
pub fn stored_alpha(alpha: f32) -> f32 {
    stored_alpha_stepping(alpha, 0.0)
}

/// How far from the half code a GPU may step from one unorm code to the
/// next: Direct3D and Vulkan give the conversion from a float 0.6 of a code,
/// so an alpha of 4.45 codes may be stored as 5 and one of 4.55 as 4.
pub const UNORM_STEP_TOLERANCE: f32 = 0.1;

/// [`stored_alpha`] on a GPU whose step between two codes lies `bias` of a
/// code under the half, inside [`UNORM_STEP_TOLERANCE`]. A reference that
/// shows the alpha itself, as the overlay does, accepts the whole tolerance.
pub fn stored_alpha_stepping(alpha: f32, bias: f32) -> f32 {
    ((alpha.clamp(0.0, 1.0) * 255.0 + bias).round() / 255.0).clamp(0.0, 1.0)
}

/// The alpha of a mask over every pixel of a render, as the GPU stores it.
pub fn alpha_image(mask: &Mask, pixels: &[[f32; 3]], geometry: &Geometry) -> Vec<f32> {
    alpha_image_with(mask, pixels, geometry, &|layer| layer)
}

/// [`alpha_image`] with the rounding of the layer a brush is stamped into.
pub fn alpha_image_with(
    mask: &Mask,
    pixels: &[[f32; 3]],
    geometry: &Geometry,
    layer_store: &dyn Fn(f32) -> f32,
) -> Vec<f32> {
    alpha_image_before_the_store(mask, pixels, geometry, layer_store)
        .into_iter()
        .map(stored_alpha)
        .collect()
}

/// The alpha of a mask over every pixel of a render before the r8unorm
/// store rounds it.
pub fn alpha_image_before_the_store(
    mask: &Mask,
    pixels: &[[f32; 3]],
    geometry: &Geometry,
    layer_store: &dyn Fn(f32) -> f32,
) -> Vec<f32> {
    let mask = mask.sanitised();
    let stamped = StampedMask::new(&mask, geometry.aspect());
    let width = geometry.size.0;
    pixels
        .iter()
        .enumerate()
        .map(|(i, px)| {
            let at = geometry.position(i as u32 % width, i as u32 / width);
            stamped.alpha(at, *px, layer_store)
        })
        .collect()
}

fn add_wheel(global: Wheel, mask: Wheel) -> Wheel {
    let (x, y) = (global.x + mask.x, global.y + mask.y);
    let length = (x * x + y * y).sqrt();
    let pull = if length > 1.0 { 1.0 / length } else { 1.0 };
    Wheel {
        x: x * pull,
        y: y * pull,
        luminance: (global.luminance + mask.luminance).clamp(-SLIDER_LIMIT, SLIDER_LIMIT),
    }
}

/// What a mask develops with: the global adjustments with the mask's added.
/// Every scalar is added and held inside its slider's range, the mixer is
/// added field by field, and a wheel adds its disc point (pulled back to the
/// rim) and its luminance. The tone curves do not add; they compose, which
/// [`Prepared::composed`] does, so the curves of the result are the global
/// ones. A mask whose adjustments are the default gives the global
/// adjustments back.
pub fn effective_adjustments(global: &Adjustments, mask: &Adjustments) -> Adjustments {
    let slider = |a: f32, b: f32| (a + b).clamp(-SLIDER_LIMIT, SLIDER_LIMIT);
    let mut out = global.clone();
    out.white_balance_temperature = slider(
        global.white_balance_temperature,
        mask.white_balance_temperature,
    );
    out.white_balance_tint = slider(global.white_balance_tint, mask.white_balance_tint);
    out.exposure = (global.exposure + mask.exposure).clamp(-EXPOSURE_LIMIT, EXPOSURE_LIMIT);
    out.contrast = slider(global.contrast, mask.contrast);
    out.highlights = slider(global.highlights, mask.highlights);
    out.shadows = slider(global.shadows, mask.shadows);
    out.whites = slider(global.whites, mask.whites);
    out.blacks = slider(global.blacks, mask.blacks);
    out.vibrance = slider(global.vibrance, mask.vibrance);
    out.saturation = slider(global.saturation, mask.saturation);
    out.texture = slider(global.texture, mask.texture);
    out.clarity = slider(global.clarity, mask.clarity);
    out.dehaze = slider(global.dehaze, mask.dehaze);
    for (range, added) in out.look.hsl.iter_mut().zip(&mask.look.hsl) {
        range.hue = slider(range.hue, added.hue);
        range.saturation = slider(range.saturation, added.saturation);
        range.luminance = slider(range.luminance, added.luminance);
    }
    let (wheels, added) = (&mut out.look.wheels, &mask.look.wheels);
    wheels.shadows = add_wheel(wheels.shadows, added.shadows);
    wheels.midtones = add_wheel(wheels.midtones, added.midtones);
    wheels.highlights = add_wheel(wheels.highlights, added.highlights);
    out
}

/// The masks of an edit that change the picture, sanitised, in list order,
/// each with the index it has in the edit.
pub fn active_masks(edit: &PhotoEdit) -> Vec<(usize, Mask)> {
    gamut_core::mask::sanitised(&edit.masks)
        .into_iter()
        .enumerate()
        .filter(|(_, mask)| mask.is_active())
        .collect()
}

/// The overlay of the selected mask on an output pixel in linear sRGB, after
/// the clip: red mixed in by the stored alpha at [`OVERLAY_STRENGTH`].
pub fn overlay(srgb: [f32; 3], alpha: f32) -> [f32; 3] {
    let a = alpha * OVERLAY_STRENGTH;
    [
        srgb[0] * (1.0 - a) + a,
        srgb[1] * (1.0 - a),
        srgb[2] * (1.0 - a),
    ]
}

/// One step of the ordered blend: `developed` over `result` by the stored
/// alpha of the mask times its opacity.
pub fn blend(result: [f32; 3], developed: [f32; 3], alpha: f32, opacity: f32) -> [f32; 3] {
    let a = alpha * (opacity / 100.0);
    [0, 1, 2].map(|c| developed[c] * a + result[c] * (1.0 - a))
}

/// What the develop chain of a whole render reads: the source pixels in
/// linear Rec.2020 and the three blurred products under them, all at the
/// size of `geometry`.
pub struct Image<'a> {
    pub pixels: &'a [[f32; 3]],
    pub base: &'a [f32],
    pub texture: &'a [f32],
    pub transmission: &'a [f32],
    pub geometry: Geometry,
}

/// The develop of a whole render with its masks: the global develop of every
/// pixel, then for each active mask in list order the develop of the source
/// pixel with the mask's effective adjustments, blended over the result by
/// the mask's alpha times its opacity. Each mask starts from the source
/// pixel, not from the result so far.
///
/// `store` is applied where the GPU writes the developed texture: once after
/// the global develop and once after every blend. A golden test passes the
/// rounding of a half float there.
pub fn develop_image(
    image: &Image,
    edit: &PhotoEdit,
    atmosphere: [f32; 3],
    store: &dyn Fn(f32) -> f32,
) -> Vec<[f32; 3]> {
    develop_image_with(image, edit, atmosphere, store, &|layer| layer)
}

/// [`develop_image`] with the rounding of the layer a brush is stamped into,
/// applied after every dab.
pub fn develop_image_with(
    image: &Image,
    edit: &PhotoEdit,
    atmosphere: [f32; 3],
    store: &dyn Fn(f32) -> f32,
    layer_store: &dyn Fn(f32) -> f32,
) -> Vec<[f32; 3]> {
    let around = |i: usize| Neighbourhood {
        base_luma: image.base[i],
        texture_luma: image.texture[i],
        transmission: image.transmission[i],
    };
    let prepared = Prepared::new(edit, atmosphere);
    let mut result: Vec<[f32; 3]> = image
        .pixels
        .iter()
        .enumerate()
        .map(|(i, px)| basic::develop_pixel_with(*px, &around(i), edit, &prepared).map(store))
        .collect();
    for (_, mask) in active_masks(edit) {
        let alphas = alpha_image_with(&mask, image.pixels, &image.geometry, layer_store);
        let effective = PhotoEdit::from(effective_adjustments(&edit.adjust, &mask.adjust));
        let prepared = Prepared::composed(&edit.adjust, &mask.adjust, atmosphere);
        for (i, px) in image.pixels.iter().enumerate() {
            let developed = basic::develop_pixel_with(*px, &around(i), &effective, &prepared);
            result[i] = blend(result[i], developed, alphas[i], mask.opacity).map(store);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve;
    use gamut_core::look::Curve;
    use gamut_core::mask::Component;

    const SQUARE: [f32; 2] = [1.0, 1.0];
    const GREY: [f32; 3] = [0.18; 3];

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn a_linear_gradient_runs_from_0_at_its_start_to_1_at_its_end() {
        let gradient = LinearGradient {
            start: [0.2, 0.3],
            end: [0.8, 0.7],
        };
        assert_eq!(linear(&gradient, gradient.start, SQUARE), 0.0);
        assert!(close(linear(&gradient, [0.5, 0.5], SQUARE), 0.5));
        assert!(close(linear(&gradient, gradient.end, SQUARE), 1.0));
        assert_eq!(
            linear(&gradient, [0.0, 0.0], SQUARE),
            0.0,
            "before the start"
        );
        assert_eq!(linear(&gradient, [1.0, 1.0], SQUARE), 1.0, "past the end");
        // A quarter of the way along, a smoothstep: 3 t^2 - 2 t^3.
        assert!(close(linear(&gradient, [0.35, 0.4], SQUARE), 0.15625));
    }

    #[test]
    fn a_linear_gradient_is_constant_along_its_perpendicular() {
        let gradient = LinearGradient {
            start: [0.2, 0.3],
            end: [0.8, 0.7],
        };
        // On a photo twice as wide as it is high the perpendicular of the
        // line is measured in pixels, not in normalised units.
        let aspect = [1.0, 0.5];
        let along = [0.6 * aspect[0], 0.4 * aspect[1]];
        let across = [-along[1], along[0]];
        let middle = [0.5, 0.5];
        let expected = linear(&gradient, middle, aspect);
        for step in [-0.3, -0.1, 0.2, 0.4] {
            let at = [
                middle[0] + step * across[0] / aspect[0],
                middle[1] + step * across[1] / aspect[1],
            ];
            assert!(close(linear(&gradient, at, aspect), expected), "at {at:?}");
        }
        assert!(close(expected, 0.5));
    }

    #[test]
    fn a_linear_gradient_of_no_length_is_not_a_division_by_zero() {
        let gradient = LinearGradient {
            start: [0.5, 0.5],
            end: [0.5, 0.5],
        };
        for at in [[0.5, 0.5], [0.1, 0.9]] {
            assert!(linear(&gradient, at, SQUARE).is_finite());
        }
    }

    #[test]
    fn a_radial_gradient_honours_both_radii_and_the_feather() {
        let gradient = RadialGradient {
            centre: [0.5, 0.5],
            radius: [0.4, 0.2],
            rotation: 0.0,
            feather: 50.0,
        };
        assert_eq!(radial(&gradient, [0.5, 0.5], SQUARE), 1.0);
        // Inside the inner half of each radius the alpha is 1.
        assert_eq!(radial(&gradient, [0.69, 0.5], SQUARE), 1.0);
        assert_eq!(radial(&gradient, [0.5, 0.59], SQUARE), 1.0);
        // Three quarters of the way out, the middle of the feather band.
        assert!(close(radial(&gradient, [0.8, 0.5], SQUARE), 0.5));
        assert!(close(radial(&gradient, [0.5, 0.65], SQUARE), 0.5));
        // At the rim and past it the alpha is 0.
        assert!(close(radial(&gradient, [0.9, 0.5], SQUARE), 0.0));
        assert_eq!(radial(&gradient, [0.5, 0.75], SQUARE), 0.0);
        assert_eq!(radial(&gradient, [0.95, 0.5], SQUARE), 0.0);
    }

    #[test]
    fn a_radial_gradient_honours_its_rotation() {
        let flat = RadialGradient {
            centre: [0.5, 0.5],
            radius: [0.4, 0.1],
            rotation: 0.0,
            feather: 0.0,
        };
        let turned = RadialGradient {
            rotation: 90.0,
            ..flat
        };
        let right = [0.8, 0.5];
        let below = [0.5, 0.8];
        assert_eq!(radial(&flat, right, SQUARE), 1.0);
        assert_eq!(radial(&flat, below, SQUARE), 0.0);
        assert_eq!(radial(&turned, right, SQUARE), 0.0);
        assert_eq!(radial(&turned, below, SQUARE), 1.0);
        // A positive rotation turns the long axis clockwise on the screen,
        // where y runs down: at 45 degrees it points down and to the right.
        let diagonal = RadialGradient {
            rotation: 45.0,
            ..flat
        };
        assert_eq!(radial(&diagonal, [0.7, 0.7], SQUARE), 1.0);
        assert_eq!(radial(&diagonal, [0.7, 0.3], SQUARE), 0.0);
    }

    #[test]
    fn equal_radii_are_a_circle_on_a_photo_of_any_shape() {
        let gradient = RadialGradient {
            centre: [0.5, 0.5],
            radius: [0.25, 0.25],
            rotation: 0.0,
            feather: 0.0,
        };
        let geometry = Geometry::full((200, 100), (4000, 2000));
        let aspect = geometry.aspect();
        assert_eq!(aspect, [1.0, 0.5]);
        // 0.2 of the longer side to the right, and the same distance down,
        // which is 0.4 of the height.
        assert_eq!(radial(&gradient, [0.7, 0.5], aspect), 1.0);
        assert_eq!(radial(&gradient, [0.5, 0.9], aspect), 1.0);
        assert_eq!(radial(&gradient, [0.8, 0.5], aspect), 0.0);
    }

    #[test]
    fn the_geometry_places_pixel_centres_inside_the_window() {
        let geometry = Geometry {
            window: CropRect {
                x: 0.25,
                y: 0.5,
                width: 0.5,
                height: 0.25,
            },
            size: (100, 50),
            photo: (3000, 4000),
        };
        assert_eq!(geometry.position(0, 0), [0.2525, 0.5025]);
        let last = geometry.position(99, 49);
        assert!(close(last[0], 0.7475) && close(last[1], 0.7475));
        assert_eq!(geometry.aspect(), [0.75, 1.0]);
    }

    /// A grey whose tone on the normalised axis is `n`.
    fn grey_at(n: f32) -> [f32; 3] {
        [acescct::decode(acescct::denormalise(n)); 3]
    }

    #[test]
    fn a_luminance_range_is_1_inside_and_0_past_the_falloff() {
        let range = LuminanceRange {
            low: 0.4,
            high: 0.6,
            falloff: 0.1,
        };
        for n in [0.4, 0.5, 0.6] {
            assert!(close(luminance(&range, grey_at(n)), 1.0), "at {n}");
        }
        for n in [0.0, 0.29, 0.71, 1.0] {
            assert_eq!(luminance(&range, grey_at(n)), 0.0, "at {n}");
        }
        assert!((luminance(&range, grey_at(0.35)) - 0.5).abs() < 1e-3);
        assert!((luminance(&range, grey_at(0.65)) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn a_luminance_range_reaches_black_and_white_and_survives_no_falloff() {
        let all = LuminanceRange {
            low: 0.0,
            high: 1.0,
            falloff: 0.0,
        };
        for px in [[0.0; 3], GREY, [1.0; 3], [4.0; 3]] {
            assert_eq!(luminance(&all, px), 1.0, "{px:?}");
        }
        let hard = LuminanceRange {
            low: 0.5,
            high: 1.0,
            falloff: 0.0,
        };
        assert_eq!(luminance(&hard, grey_at(0.4)), 0.0);
        assert_eq!(luminance(&hard, grey_at(0.6)), 1.0);
    }

    /// A pixel of the given hue in degrees and chroma around a middle grey,
    /// in linear Rec.2020.
    fn coloured(degrees: f32, chroma: f32) -> [f32; 3] {
        let (sin, cos) = degrees.to_radians().sin_cos();
        let tint = hue::from_plane([chroma * cos, chroma * sin]);
        acescct::decode_pixel([0.4 + tint[0], 0.4 + tint[1], 0.4 + tint[2]])
    }

    #[test]
    fn a_colour_range_follows_the_hue_across_the_wrap_at_0() {
        let range = ColourRange {
            hue: 5.0,
            hue_width: 40.0,
            chroma_low: 0.02,
            falloff: 10.0,
        };
        // 20 degrees either side of 5 is inside: 345 to 25, across 0.
        for degrees in [350.0, 0.0, 5.0, 20.0] {
            let a = colour(&range, coloured(degrees, 0.1));
            assert!((a - 1.0).abs() < 1e-3, "{degrees}: {a}");
        }
        for degrees in [335.0, 320.0, 180.0, 40.0] {
            let a = colour(&range, coloured(degrees, 0.1));
            assert!(a < 1e-3, "{degrees}: {a}");
        }
        let edge = colour(&range, coloured(340.0, 0.1));
        assert!((edge - 0.5).abs() < 0.01, "the middle of the fall: {edge}");
    }

    #[test]
    fn a_colour_range_ignores_pixels_under_chroma_low() {
        let range = ColourRange {
            hue: 120.0,
            hue_width: 60.0,
            chroma_low: 0.05,
            falloff: 10.0,
        };
        assert_eq!(colour(&range, coloured(120.0, 0.04)), 0.0);
        assert_eq!(colour(&range, GREY), 0.0);
        assert!((colour(&range, coloured(120.0, 0.055)) - 0.5).abs() < 0.01);
        assert!((colour(&range, coloured(120.0, 0.07)) - 1.0).abs() < 1e-3);
        let all_hues = ColourRange {
            hue_width: 360.0,
            ..range
        };
        assert!((colour(&all_hues, coloured(300.0, 0.2)) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn add_subtract_and_intersect_match_hand_values() {
        assert!(close(combine(0.5, 0.4, MaskOp::Add), 0.7));
        assert!(close(combine(0.5, 0.4, MaskOp::Subtract), 0.3));
        assert!(close(combine(0.5, 0.4, MaskOp::Intersect), 0.2));
        assert_eq!(combine(0.0, 0.6, MaskOp::Add), 0.6, "the first component");
        assert_eq!(combine(1.0, 1.0, MaskOp::Add), 1.0);
        assert_eq!(combine(0.8, 1.0, MaskOp::Subtract), 0.0);
        assert_eq!(combine(0.0, 0.9, MaskOp::Intersect), 0.0);
    }

    fn radial_at(centre: [f32; 2]) -> MaskSource {
        MaskSource::Radial(RadialGradient {
            centre,
            radius: [0.2, 0.2],
            rotation: 0.0,
            feather: 0.0,
        })
    }

    #[test]
    fn components_combine_in_order_with_their_inverts() {
        let mut mask = Mask::new("Two discs", radial_at([0.4, 0.5]));
        mask.components.push(Component {
            op: MaskOp::Subtract,
            source: radial_at([0.6, 0.5]),
            invert: false,
        });
        let left = [0.3, 0.5];
        let both = [0.5, 0.5];
        let right = [0.7, 0.5];
        let nowhere = [0.5, 0.05];
        let at = |mask: &Mask, p| alpha(mask, p, SQUARE, GREY);
        assert_eq!(
            [left, both, right, nowhere].map(|p| at(&mask, p)),
            [1.0, 0.0, 0.0, 0.0]
        );
        mask.components[1].op = MaskOp::Intersect;
        assert_eq!(
            [left, both, right, nowhere].map(|p| at(&mask, p)),
            [0.0, 1.0, 0.0, 0.0]
        );
        mask.components[1].op = MaskOp::Add;
        assert_eq!(
            [left, both, right, nowhere].map(|p| at(&mask, p)),
            [1.0, 1.0, 1.0, 0.0]
        );
        // A component's invert flips its own alpha before it joins.
        mask.components[1].invert = true;
        mask.components[1].op = MaskOp::Intersect;
        assert_eq!(
            [left, both, right, nowhere].map(|p| at(&mask, p)),
            [1.0, 0.0, 0.0, 0.0]
        );
        // The mask's invert flips the whole.
        mask.invert = true;
        assert_eq!(
            [left, both, right, nowhere].map(|p| at(&mask, p)),
            [0.0, 1.0, 1.0, 1.0]
        );
        // No components: nothing, or with the invert everything.
        let mut empty = Mask::default();
        assert_eq!(at(&empty, both), 0.0);
        empty.invert = true;
        assert_eq!(at(&empty, both), 1.0);
    }

    #[test]
    fn the_overlay_mixes_red_in_by_half_the_alpha() {
        assert_eq!(overlay([0.2, 0.4, 0.6], 0.0), [0.2, 0.4, 0.6]);
        assert_eq!(overlay([0.2, 0.4, 0.6], 1.0), [0.6, 0.2, 0.3]);
        assert_eq!(overlay([0.0, 0.0, 0.0], 0.5), [0.25, 0.0, 0.0]);
        assert_eq!(overlay([1.0, 1.0, 1.0], 1.0), [1.0, 0.5, 0.5]);
    }

    #[test]
    fn a_stored_alpha_is_one_of_256_levels() {
        assert_eq!(stored_alpha(0.0), 0.0);
        assert_eq!(stored_alpha(1.0), 1.0);
        assert_eq!(stored_alpha(0.5), 128.0 / 255.0);
        assert_eq!(stored_alpha(0.001), 0.0);
        assert_eq!(stored_alpha(1.7), 1.0);
    }

    fn busy_global() -> Adjustments {
        let mut global = Adjustments {
            exposure: 0.5,
            contrast: 20.0,
            texture: 30.0,
            dehaze: 10.0,
            saturation: -10.0,
            ..Adjustments::default()
        };
        global.look.curves.master = Curve {
            points: vec![[0.0, 0.0], [0.25, 0.2], [0.75, 0.8], [1.0, 1.0]],
        };
        global.look.hsl[4].saturation = -40.0;
        global.look.wheels.shadows.x = -0.4;
        global
    }

    #[test]
    fn default_mask_adjustments_give_the_global_ones_back() {
        let global = busy_global();
        assert_eq!(
            effective_adjustments(&global, &Adjustments::default()),
            global
        );
        let atmosphere = [0.9, 0.95, 1.0];
        assert_eq!(
            Prepared::composed(&global, &Adjustments::default(), atmosphere),
            Prepared::new(&PhotoEdit::from(global), atmosphere)
        );
    }

    #[test]
    fn adding_scalars_clamps_at_the_slider_range() {
        let global = Adjustments {
            exposure: 4.0,
            contrast: 80.0,
            shadows: -70.0,
            ..Adjustments::default()
        };
        let mask = Adjustments {
            exposure: 3.0,
            contrast: 50.0,
            shadows: -60.0,
            clarity: 25.0,
            ..Adjustments::default()
        };
        let out = effective_adjustments(&global, &mask);
        assert_eq!(out.exposure, EXPOSURE_LIMIT);
        assert_eq!(out.contrast, SLIDER_LIMIT);
        assert_eq!(out.shadows, -SLIDER_LIMIT);
        assert_eq!(out.clarity, 25.0);
        assert_eq!(out.vibrance, 0.0);
    }

    #[test]
    fn the_mixer_and_the_wheels_add_field_by_field() {
        let mut global = Adjustments::default();
        global.look.hsl[2].hue = 30.0;
        global.look.hsl[2].saturation = 90.0;
        global.look.wheels.shadows = Wheel {
            x: 0.8,
            y: 0.0,
            luminance: 60.0,
        };
        let mut mask = Adjustments::default();
        mask.look.hsl[2].hue = -10.0;
        mask.look.hsl[2].saturation = 40.0;
        mask.look.hsl[5].luminance = -20.0;
        mask.look.wheels.shadows = Wheel {
            x: 0.8,
            y: 0.0,
            luminance: 70.0,
        };
        mask.look.wheels.highlights.y = 0.3;
        let out = effective_adjustments(&global, &mask);
        assert_eq!(out.look.hsl[2].hue, 20.0);
        assert_eq!(out.look.hsl[2].saturation, 100.0);
        assert_eq!(out.look.hsl[5].luminance, -20.0);
        let shadows = out.look.wheels.shadows;
        assert!(
            close(shadows.x, 1.0) && shadows.y == 0.0,
            "pulled to the rim"
        );
        assert_eq!(shadows.luminance, 100.0);
        assert_eq!(out.look.wheels.highlights.y, 0.3);
    }

    #[test]
    fn the_composed_table_of_an_identity_mask_curve_is_the_global_table() {
        let global = busy_global().look.curves;
        let composed = curve::bake_composed(&global, &Default::default());
        assert_eq!(composed, curve::bake(&global));
    }

    #[test]
    fn a_mask_curve_runs_after_the_global_curves() {
        let global = busy_global().look.curves;
        let mask = gamut_core::ToneCurves {
            red: Curve {
                points: vec![[0.0, 0.1], [1.0, 0.9]],
            },
            master: Curve {
                points: vec![[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]],
            },
            ..Default::default()
        };
        let composed = curve::bake_composed(&global, &mask);
        let plain = curve::bake(&global);
        for i in [0, 100, 511, 900, 1023] {
            let after_red = curve::evaluate(&mask.red, plain.red[i]);
            let expected = curve::evaluate(&mask.master, after_red);
            assert!(close(composed.red[i], expected), "red entry {i}");
            let expected = curve::evaluate(&mask.master, plain.green[i]);
            assert!(close(composed.green[i], expected), "green entry {i}");
        }
        // With no global curve the mask's curves are the whole table.
        let alone = curve::bake_composed(&Default::default(), &mask);
        let whole = curve::bake(&mask);
        for i in 0..curve::TABLE_SIZE {
            assert!(close(alone.red[i], whole.red[i]), "red entry {i}");
            assert!(close(alone.blue[i], whole.blue[i]), "blue entry {i}");
        }
    }

    fn flat_image(pixels: &[[f32; 3]], size: (u32, u32)) -> (Vec<f32>, Vec<f32>, Geometry) {
        let luma: Vec<f32> = pixels.iter().map(|px| basic::luma(*px)).collect();
        let clear = vec![1.0; pixels.len()];
        (luma, clear, Geometry::full(size, size))
    }

    #[test]
    fn a_mask_with_default_adjustments_is_the_identity_of_the_blend() {
        let pixels: Vec<[f32; 3]> = (0..16)
            .map(|i| [0.05 * i as f32, 0.3, 0.9 - 0.05 * i as f32])
            .collect();
        let (luma, clear, geometry) = flat_image(&pixels, (4, 4));
        let image = Image {
            pixels: &pixels,
            base: &luma,
            texture: &luma,
            transmission: &clear,
            geometry,
        };
        let plain = PhotoEdit::from(busy_global());
        let mut masked = plain.clone();
        masked.masks.push(Mask::new("Idle", radial_at([0.5, 0.5])));
        let atmosphere = [1.0; 3];
        let identity = |v: f32| v;
        assert!(active_masks(&masked).is_empty());
        assert_eq!(
            develop_image(&image, &masked, atmosphere, &identity),
            develop_image(&image, &plain, atmosphere, &identity)
        );
    }

    #[test]
    fn masks_blend_in_list_order_each_from_the_source_pixel() {
        let pixels = vec![[0.2, 0.2, 0.2]; 4];
        let (luma, clear, geometry) = flat_image(&pixels, (2, 2));
        let image = Image {
            pixels: &pixels,
            base: &luma,
            texture: &luma,
            transmission: &clear,
            geometry,
        };
        let everywhere = || Mask {
            invert: true,
            ..Mask::default()
        };
        let mut brighter = everywhere();
        brighter.name = "Brighter".to_string();
        brighter.adjust.exposure = 1.0;
        brighter.opacity = 50.0;
        let mut darker = everywhere();
        darker.name = "Darker".to_string();
        darker.adjust.exposure = -1.0;
        darker.opacity = 50.0;

        let mut edit = PhotoEdit {
            masks: vec![brighter.clone(), darker.clone()],
            ..PhotoEdit::default()
        };
        let identity = |v: f32| v;
        let one_way = develop_image(&image, &edit, [1.0; 3], &identity);
        // Global 0.2; brighter at 50 percent: 0.3; darker at 50 percent over
        // that, from the source pixel: 0.5 * 0.1 + 0.5 * 0.3 = 0.2.
        assert!(close(one_way[0][0], 0.2), "{:?}", one_way[0]);
        edit.masks = vec![darker.clone(), brighter.clone()];
        let other_way = develop_image(&image, &edit, [1.0; 3], &identity);
        // Darker first: 0.15; brighter over it: 0.5 * 0.4 + 0.5 * 0.15.
        assert!(close(other_way[0][0], 0.275), "{:?}", other_way[0]);

        // A disabled mask and a mask of no opacity cost nothing.
        edit.masks[0].enabled = false;
        edit.masks[1].opacity = 0.0;
        assert!(active_masks(&edit).is_empty());
        let plain = develop_image(&image, &edit, [1.0; 3], &identity);
        assert_eq!(plain[0], [0.2, 0.2, 0.2]);
    }

    #[test]
    fn the_store_runs_after_the_global_develop_and_after_every_blend() {
        let pixels = vec![[0.2, 0.2, 0.2]; 1];
        let (luma, clear, geometry) = flat_image(&pixels, (1, 1));
        let image = Image {
            pixels: &pixels,
            base: &luma,
            texture: &luma,
            transmission: &clear,
            geometry,
        };
        let mut mask = Mask {
            invert: true,
            ..Mask::default()
        };
        mask.adjust.exposure = 1.0;
        let edit = PhotoEdit {
            masks: vec![mask.clone(), mask],
            ..PhotoEdit::default()
        };
        let count = std::cell::Cell::new(0);
        let counting = |v: f32| {
            count.set(count.get() + 1);
            v
        };
        develop_image(&image, &edit, [1.0; 3], &counting);
        assert_eq!(count.get(), 3 * 3, "three channels, three stores");
    }

    #[test]
    fn a_brush_combines_with_a_gradient_by_each_op() {
        use gamut_core::brush::{Brush, SharedStroke, Stroke};
        let painted = MaskSource::Brush(Brush {
            strokes: vec![SharedStroke::new(&Stroke {
                points: vec![[0.3, 0.5], [0.7, 0.5]],
                size: 0.1,
                feather: 100.0,
                flow: 30.0,
                ..Stroke::default()
            })],
        });
        let gradient = MaskSource::Linear(LinearGradient {
            start: [0.2, 0.5],
            end: [0.8, 0.5],
        });
        let at = [0.6, 0.53];
        let b = source_alpha(&painted, at, SQUARE, GREY);
        let g = source_alpha(&gradient, at, SQUARE, GREY);
        assert!(b > 0.1 && b < 0.95 && g > 0.1 && g < 0.95, "{b} {g}");
        for (op, expected) in [
            (MaskOp::Add, g + b - g * b),
            (MaskOp::Subtract, g * (1.0 - b)),
            (MaskOp::Intersect, g * b),
        ] {
            let mut mask = Mask::new("Both", gradient.clone());
            mask.components.push(Component {
                op,
                source: painted.clone(),
                invert: false,
            });
            assert!(close(alpha(&mask, at, SQUARE, GREY), expected), "{op:?}");
            // The other way round the brush is what the gradient joins.
            let mut mask = Mask::new("Both", painted.clone());
            mask.components.push(Component {
                op,
                source: gradient.clone(),
                invert: false,
            });
            let expected = combine(b, g, op);
            assert!(close(alpha(&mask, at, SQUARE, GREY), expected), "{op:?}");
        }
        // A whole image stamps the brush once and agrees with the one pixel.
        let geometry = Geometry::full((8, 8), (800, 800));
        let mask = Mask::new("Painted", painted);
        let image = alpha_image(&mask, &[GREY; 64], &geometry);
        let pixel = geometry.position(5, 4);
        assert_eq!(
            image[4 * 8 + 5],
            stored_alpha(alpha(&mask, pixel, SQUARE, GREY))
        );
        assert!(image[4 * 8 + 5] > 0.0);
    }
}
