//! Hue and chroma of an ACEScct pixel, shared by the HSL mixer and the colour
//! wheels. The model is the one vibrance uses, luminance plus a chroma vector;
//! hue is the angle of that vector on a fixed orthonormal basis of the plane
//! normal to grey.

use std::f32::consts::{PI, TAU};

use slate_core::look::HSL_RANGES;

use crate::SourceSpace;
use crate::acescct;
use crate::basic::decode_rgb8;

/// The first basis vector, (2, -1, -1) over the square root of 6. Red sits at
/// angle 0.
pub const E1: [f32; 3] = [0.816_496_6, -0.408_248_3, -0.408_248_3];

/// The second basis vector, (0, 1, -1) over the square root of 2. Green sits
/// near a third of a turn, blue near two thirds.
pub const E2: [f32; 3] = [0.0, 0.707_106_77, -0.707_106_77];

/// The sRGB colours whose hue angles are the centres of the eight ranges:
/// red, orange, yellow, green, aqua, blue, purple, magenta.
pub const CENTRE_COLOURS: [[u8; 3]; HSL_RANGES] = [
    [255, 0, 0],
    [255, 128, 0],
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
    [0, 0, 255],
    [128, 0, 255],
    [255, 0, 255],
];

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The chroma vector of an ACEScct pixel on the basis. Both basis vectors
/// are normal to grey, so the luminance drops out of the projection.
pub fn chroma_plane(v: [f32; 3]) -> [f32; 2] {
    [dot(v, E1), dot(v, E2)]
}

/// The pixel made of a point on the basis, with zero mean across channels.
pub fn from_plane(p: [f32; 2]) -> [f32; 3] {
    [
        p[0] * E1[0] + p[1] * E2[0],
        p[0] * E1[1] + p[1] * E2[1],
        p[0] * E1[2] + p[1] * E2[2],
    ]
}

/// The length of the chroma vector.
pub fn chroma(p: [f32; 2]) -> f32 {
    (p[0] * p[0] + p[1] * p[1]).sqrt()
}

/// The hue angle in 0 to a full turn, in radians.
pub fn hue(p: [f32; 2]) -> f32 {
    wrap(p[1].atan2(p[0]))
}

/// An angle brought into 0 to a full turn.
pub fn wrap(angle: f32) -> f32 {
    let wrapped = angle - TAU * (angle / TAU).floor();
    if wrapped >= TAU { 0.0 } else { wrapped }
}

/// The hue angle of an 8-bit sRGB colour carried through the working
/// transform: linear Rec.2020, then ACEScct.
pub fn hue_of_srgb8(rgb: [u8; 3]) -> f32 {
    let linear = decode_rgb8(rgb, SourceSpace::Srgb);
    hue(chroma_plane(acescct::encode_pixel(linear)))
}

/// The centres of the eight ranges, in radians, rising from red. The shader
/// holds the same list as constants and a test in slate-gpu keeps the two
/// equal.
pub fn range_centres() -> [f32; HSL_RANGES] {
    CENTRE_COLOURS.map(hue_of_srgb8)
}

/// The two ranges a hue falls between and the weight of the second: the
/// range whose centre is the nearest one at or below the hue, its upper
/// neighbour, and a raised cosine that is 0 at the first centre and 1 at the
/// second. The first range's weight is one minus that, so the eight weights
/// sum to 1 at every hue.
pub fn range_pair(centres: &[f32; HSL_RANGES], hue: f32) -> (usize, usize, f32) {
    let mut lower = 0;
    let mut distance = TAU;
    for (k, centre) in centres.iter().enumerate() {
        let d = wrap(hue - centre);
        if d < distance {
            distance = d;
            lower = k;
        }
    }
    let upper = (lower + 1) % HSL_RANGES;
    let span = wrap(centres[upper] - centres[lower]);
    let t = (distance / span).min(1.0);
    (lower, upper, 0.5 - 0.5 * (PI * t).cos())
}

/// The weight of every range at a hue.
pub fn range_weights(centres: &[f32; HSL_RANGES], hue: f32) -> [f32; HSL_RANGES] {
    let (lower, upper, weight) = range_pair(centres, hue);
    let mut weights = [0.0; HSL_RANGES];
    weights[lower] = 1.0 - weight;
    weights[upper] += weight;
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_basis_is_orthonormal_and_normal_to_grey() {
        assert!((dot(E1, E1) - 1.0).abs() < 1e-6);
        assert!((dot(E2, E2) - 1.0).abs() < 1e-6);
        assert!(dot(E1, E2).abs() < 1e-6);
        assert!(dot(E1, [1.0; 3]).abs() < 1e-6);
        assert!(dot(E2, [1.0; 3]).abs() < 1e-6);
    }

    #[test]
    fn the_centres_rise_from_red_around_the_circle() {
        let centres = range_centres();
        println!("centres: {centres:?}");
        let spans: f32 = (0..HSL_RANGES)
            .map(|k| wrap(centres[(k + 1) % HSL_RANGES] - centres[k]))
            .sum();
        assert!((spans - TAU).abs() < 1e-4, "the spans add up to {spans}");
        for k in 0..HSL_RANGES {
            let span = wrap(centres[(k + 1) % HSL_RANGES] - centres[k]);
            assert!(span > 0.2 && span < 1.8, "range {k} spans {span}");
        }
    }

    #[test]
    fn the_weights_sum_to_one_and_each_centre_owns_its_range() {
        let centres = range_centres();
        for degree in 0..360 {
            let weights = range_weights(&centres, (degree as f32).to_radians());
            let sum: f32 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "{degree} degrees sums to {sum}");
            assert!(weights.iter().all(|w| (0.0..=1.0).contains(w)));
        }
        for (k, centre) in centres.iter().enumerate() {
            let weights = range_weights(&centres, *centre);
            assert!((weights[k] - 1.0).abs() < 1e-6, "range {k}: {weights:?}");
        }
    }

    #[test]
    fn a_pixel_is_its_mean_plus_its_point_on_the_plane() {
        let v = [0.52, 0.31, 0.44];
        let mean = (v[0] + v[1] + v[2]) / 3.0;
        let back = from_plane(chroma_plane(v)).map(|c| c + mean);
        for (a, b) in v.iter().zip(back) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
