//! Golden tests: every GPU operator against its CPU twin in gamut-color,
//! through the same input and output transforms, on a 64 by 64 synthetic
//! photo that covers hue, brightness and neutral ramps. The tests skip,
//! and say so, when the machine has no adapter.

use std::sync::Mutex;

use gamut_color::SourceSpace;
use gamut_color::basic;
use gamut_color::brush::Proxy;
use gamut_color::mask::{self as mask_twin, Geometry, Image};
use gamut_color::{dehaze, local, matrices, transfer};
use gamut_core::brush::{Brush, SharedStroke, Stroke};
use gamut_core::look::{Curve, HslRange, Wheel};
use gamut_core::mask::{
    ColourRange, Component, LinearGradient, LuminanceRange, MaskOp, MaskSource, RadialGradient,
};
use gamut_core::{Adjustments, CropRect, ExportPreset, Mask, PhotoEdit};
use gamut_gpu::{Develop, Headless, Readback, ViewWindow};
use gamut_media::Photo;
use half::f16;

const SIZE: u32 = 64;

/// The largest difference allowed in any channel of any pixel, in 8-bit codes.
const MAX_DIFFERENCE: i32 = 2;

/// The mean absolute difference allowed over all channels, in 8-bit codes.
const MEAN_DIFFERENCE: f64 = 0.5;

/// Hue across, brightness down, with a grey ramp in the first columns and
/// a near-black band at the bottom.
fn synthetic_photo() -> Photo {
    let mut rgba8 = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let fx = x as f32 / (SIZE - 1) as f32;
            let fy = y as f32 / (SIZE - 1) as f32;
            let rgb = if x < 6 {
                [fy, fy, fy]
            } else if y >= SIZE - 6 {
                [0.02 + 0.03 * fx, 0.02, 0.04]
            } else {
                let hue = fx * 6.0;
                let c = 1.0 - (hue % 2.0 - 1.0).abs();
                let (r, g, b) = match hue as u32 {
                    0 => (1.0, c, 0.0),
                    1 => (c, 1.0, 0.0),
                    2 => (0.0, 1.0, c),
                    3 => (0.0, c, 1.0),
                    4 => (c, 0.0, 1.0),
                    _ => (1.0, 0.0, c),
                };
                let mix = 0.3 + 0.7 * fy;
                let wash = 0.5 * (1.0 - fy);
                [
                    (r * mix + wash).min(1.0),
                    (g * mix + wash).min(1.0),
                    (b * mix + wash).min(1.0),
                ]
            };
            for c in rgb {
                rgba8.push((c * 255.0).round() as u8);
            }
            rgba8.push(255);
        }
    }
    Photo {
        width: SIZE,
        height: SIZE,
        rgba8,
        source: SourceSpace::Srgb,
        bit_depth: 8,
        has_alpha: false,
    }
}

/// A second photo for the look: the eight centre colours of the HSL mixer in
/// columns with fine detail over them, under a haze that thickens toward the
/// top, below a band of bright sky.
fn hazy_photo() -> Photo {
    const CENTRES: [[f32; 3]; 8] = [
        [1.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];
    const HAZE: [f32; 3] = [0.82, 0.86, 0.9];
    let mut rgba8 = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let fy = y as f32 / (SIZE - 1) as f32;
            let rgb = if y < 8 {
                HAZE
            } else {
                let colour = CENTRES[(x / 8) as usize];
                let detail = if (x / 2 + y / 2) % 2 == 0 { 1.0 } else { 0.6 };
                let level = (0.25 + 0.6 * fy) * detail;
                let clear = 0.35 + 0.65 * fy;
                [0, 1, 2].map(|c| colour[c] * level * clear + HAZE[c] * (1.0 - clear))
            };
            for c in rgb {
                rgba8.push((c * 255.0).round() as u8);
            }
            rgba8.push(255);
        }
    }
    Photo {
        width: SIZE,
        height: SIZE,
        rgba8,
        source: SourceSpace::Srgb,
        bit_depth: 8,
        has_alpha: false,
    }
}

/// How a GPU rounds when it stores a half float. Vulkan and D3D allow
/// either mode for render target writes; this machine's driver truncates,
/// WARP may not, so the reference is built both ways and the GPU has to
/// match one of them.
#[derive(Clone, Copy)]
enum Rounding {
    Nearest,
    TowardZero,
}

/// Rounds a value the way the rgba16float and r16float textures store it.
/// The GPU pipeline keeps its working values in half floats, so the CPU
/// reference rounds at the same texture boundaries; otherwise the rounding
/// of a bright channel leaks through the output matrix into a near-black
/// channel, where the sRGB curve turns it into several codes.
fn half(value: f32, rounding: Rounding) -> f32 {
    let nearest = f16::from_f32(value);
    match rounding {
        Rounding::Nearest => nearest.to_f32(),
        Rounding::TowardZero
            if nearest.to_f32().abs() > value.abs() && nearest.to_bits() & 0x7fff != 0 =>
        {
            f16::from_bits(nearest.to_bits() - 1).to_f32()
        }
        Rounding::TowardZero => nearest.to_f32(),
    }
}

/// A half float moved by whole steps of its own grid, held at 0.
fn half_steps(value: f32, steps: i32) -> f32 {
    let bits = i32::from(f16::from_f32(value).to_bits()) + steps;
    f16::from_bits(bits.clamp(0, 0x7bff) as u16).to_f32()
}

/// The steps of the half float grid the stored transmission map may sit
/// from the modelled one. The map comes out of a minimum filter and two
/// blur passes, where the order of the sums decides which side of a
/// rounding boundary a value lands on, and dehaze divides by it next to a
/// subtraction of the atmosphere: on a saturated pixel whose red is near
/// black after the output matrix, one step of the map is 2 to 3 codes. So
/// with dehaze on the GPU has to match the reference at the modelled map or
/// at the map one step either way.
fn transmission_steps(edit: &PhotoEdit) -> &'static [i32] {
    let dehazes = edit.dehaze != 0.0
        || mask_twin::active_masks(edit).iter().any(|(_, mask)| {
            mask_twin::effective_adjustments(&edit.adjust, &mask.adjust).dehaze != 0.0
        });
    if dehazes { &[0, -1, 1] } else { &[0] }
}

fn cpu_reference(
    photo: &Photo,
    edit: &PhotoEdit,
    rounding: Rounding,
    transmission_step: i32,
) -> Vec<[u8; 3]> {
    let linear: Vec<[f32; 3]> = photo
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| {
            basic::decode_rgb8([px[0], px[1], px[2]], photo.source).map(|c| half(c, rounding))
        })
        .collect();
    let (width, height) = (photo.width, photo.height);
    // Every blurred layer passes through two half float textures.
    let store = |v: f32| half(v, rounding);
    let luma: Vec<f32> = linear.iter().map(|px| basic::luma(*px)).collect();
    let base = basic::gaussian_stored(
        &luma,
        width,
        height,
        basic::base_sigma(width, height),
        &store,
    );
    let texture = basic::gaussian_stored(
        &luma,
        width,
        height,
        local::texture_sigma(width, height),
        &store,
    );
    // The atmospheric light is read from the photo's bytes on the CPU, before
    // any half float store, exactly as Develop::set_source reads it.
    let atmosphere = dehaze::atmosphere_rgba8(&photo.rgba8, width, height, photo.source);
    let transmission: Vec<f32> =
        dehaze::transmission_stored(&linear, width, height, atmosphere, &store)
            .into_iter()
            .map(|t| half_steps(t, transmission_step))
            .collect();
    // What an auto stroke reads its references from: the working pixels of
    // the whole source under a box filter, kept in a half float texture.
    let proxy = Proxy::from_source(&linear, (width, height), &store);
    // The global develop and the ordered blend of the masks live in
    // gamut-color; the developed texture is a half float store after the
    // global pass and after every blend.
    let image = Image {
        pixels: &linear,
        base: &base,
        texture: &texture,
        transmission: &transmission,
        geometry: Geometry::full((width, height), (width, height)),
        proxy: Some(&proxy),
    };
    // A brush layer is a half float the dabs blend into: a store a dab.
    mask_twin::develop_image_with(&image, edit, atmosphere, &store, &store)
        .into_iter()
        .map(basic::output_srgb8)
        .collect()
}

fn gpu_render(gpu: &Headless, photo: &Photo, edit: &PhotoEdit) -> Vec<[u8; 3]> {
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(photo);
    let size = (photo.width, photo.height);
    let view = develop
        .render(edit, CropRect::FULL, size, size)
        .expect("a source is set");
    Readback::new(&gpu.device)
        .read(&gpu.device, &gpu.queue, view, size.0, size.1)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| [px[0], px[1], px[2]])
        .collect()
}

/// The tests run one at a time: eleven threads each opening a device of
/// its own at once hung a driver on this machine.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn check(name: &str, edit: &PhotoEdit) {
    check_on(name, &synthetic_photo(), edit);
    check_on(&format!("{name}, hazy photo"), &hazy_photo(), edit);
}

fn check_on(name: &str, photo: &Photo, edit: &PhotoEdit) {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let references: Vec<Vec<[u8; 3]>> = transmission_steps(edit)
        .iter()
        .flat_map(|step| {
            [Rounding::Nearest, Rounding::TowardZero]
                .map(|rounding| cpu_reference(photo, edit, rounding, *step))
        })
        .collect();
    let gpu_pixels = gpu_render(&gpu, photo, edit);
    assert_eq!(references[0].len(), gpu_pixels.len());
    let mut max = 0;
    let mut sum = 0u64;
    let mut worst = (0usize, [0u8; 3], [0u8; 3]);
    for (i, g) in gpu_pixels.iter().enumerate() {
        for k in 0..3 {
            let d = references
                .iter()
                .map(|reference| (i32::from(reference[i][k]) - i32::from(g[k])).abs())
                .min()
                .expect("at least one reference");
            sum += d as u64;
            if d > max {
                max = d;
                worst = (i, references[0][i], *g);
            }
        }
    }
    let mean = sum as f64 / (gpu_pixels.len() * 3) as f64;
    println!(
        "{name}: max {max} at pixel {} (cpu {:?}, gpu {:?}), mean {mean:.3}",
        worst.0, worst.1, worst.2
    );
    assert!(
        max <= MAX_DIFFERENCE,
        "{name}: max difference {max} exceeds {MAX_DIFFERENCE}"
    );
    assert!(
        mean <= MEAN_DIFFERENCE,
        "{name}: mean difference {mean} exceeds {MEAN_DIFFERENCE}"
    );
}

#[test]
fn identity_matches() {
    check("identity", &PhotoEdit::default());
}

#[test]
fn white_balance_matches() {
    check(
        "white balance",
        &PhotoEdit::from(Adjustments {
            white_balance_temperature: 40.0,
            white_balance_tint: -20.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn exposure_matches() {
    check(
        "exposure",
        &PhotoEdit::from(Adjustments {
            exposure: 1.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn contrast_matches() {
    check(
        "contrast",
        &PhotoEdit::from(Adjustments {
            contrast: 60.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn highlights_match() {
    check(
        "highlights",
        &PhotoEdit::from(Adjustments {
            highlights: -70.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn shadows_match() {
    check(
        "shadows",
        &PhotoEdit::from(Adjustments {
            shadows: 60.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn whites_match() {
    check(
        "whites",
        &PhotoEdit::from(Adjustments {
            whites: 40.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn blacks_match() {
    check(
        "blacks",
        &PhotoEdit::from(Adjustments {
            blacks: -40.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn vibrance_matches() {
    check(
        "vibrance",
        &PhotoEdit::from(Adjustments {
            vibrance: 60.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn saturation_matches() {
    check(
        "saturation",
        &PhotoEdit::from(Adjustments {
            saturation: -50.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn every_slider_together_matches_develop_pixel() {
    check(
        "all sliders",
        &PhotoEdit::from(Adjustments {
            white_balance_temperature: -30.0,
            white_balance_tint: 15.0,
            exposure: 0.6,
            contrast: 35.0,
            highlights: -50.0,
            shadows: 40.0,
            whites: 20.0,
            blacks: -15.0,
            vibrance: 30.0,
            saturation: 10.0,
            ..Adjustments::default()
        }),
    );
}

fn curve(points: &[[f32; 2]]) -> Curve {
    Curve {
        points: points.to_vec(),
    }
}

fn s_curve() -> Curve {
    curve(&[[0.0, 0.0], [0.25, 0.17], [0.75, 0.85], [1.0, 1.0]])
}

fn with_hsl(index: usize, range: HslRange) -> PhotoEdit {
    let mut edit = PhotoEdit::default();
    edit.look.hsl[index] = range;
    edit
}

/// A diagonal curve with a third point is not the default, so the pixel
/// goes through ACEScct, the table and back, and must land where it began.
#[test]
fn the_acescct_round_trip_matches_and_changes_nothing() {
    let mut edit = PhotoEdit::default();
    edit.look.curves.master = curve(&[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]);
    assert!(!edit.look.curves.is_identity());
    check("acescct round trip", &edit);
    for photo in [synthetic_photo(), hazy_photo()] {
        let through = cpu_reference(&photo, &edit, Rounding::Nearest, 0);
        let plain = cpu_reference(&photo, &PhotoEdit::default(), Rounding::Nearest, 0);
        for (a, b) in through.iter().zip(&plain) {
            for k in 0..3 {
                assert!(
                    (i32::from(a[k]) - i32::from(b[k])).abs() <= 1,
                    "{a:?} and {b:?}"
                );
            }
        }
    }
}

#[test]
fn a_master_s_curve_matches() {
    let mut edit = PhotoEdit::default();
    edit.look.curves.master = s_curve();
    check("master s-curve", &edit);
}

#[test]
fn a_red_channel_curve_matches() {
    let mut edit = PhotoEdit::default();
    edit.look.curves.red = curve(&[[0.0, 0.05], [0.4, 0.55], [1.0, 0.95]]);
    check("red curve", &edit);
}

#[test]
fn an_hsl_hue_shift_matches() {
    check(
        "hsl hue",
        &with_hsl(
            3,
            HslRange {
                hue: 80.0,
                ..HslRange::default()
            },
        ),
    );
}

#[test]
fn an_hsl_saturation_change_matches() {
    check(
        "hsl saturation",
        &with_hsl(
            1,
            HslRange {
                saturation: -60.0,
                ..HslRange::default()
            },
        ),
    );
}

#[test]
fn an_hsl_luminance_change_matches() {
    check(
        "hsl luminance",
        &with_hsl(
            5,
            HslRange {
                luminance: 70.0,
                ..HslRange::default()
            },
        ),
    );
}

#[test]
fn the_shadows_wheel_matches() {
    let mut edit = PhotoEdit::default();
    edit.look.wheels.shadows = Wheel {
        x: -0.5,
        y: -0.3,
        luminance: -20.0,
    };
    check("shadows wheel", &edit);
}

#[test]
fn the_midtones_wheel_matches() {
    let mut edit = PhotoEdit::default();
    edit.look.wheels.midtones = Wheel {
        x: 0.3,
        y: 0.5,
        luminance: 25.0,
    };
    check("midtones wheel", &edit);
}

#[test]
fn the_highlights_wheel_matches() {
    let mut edit = PhotoEdit::default();
    edit.look.wheels.highlights = Wheel {
        x: 0.6,
        y: 0.2,
        luminance: -15.0,
    };
    check("highlights wheel", &edit);
}

#[test]
fn texture_matches() {
    for texture in [70.0, -70.0] {
        check(
            "texture",
            &PhotoEdit::from(Adjustments {
                texture,
                ..Adjustments::default()
            }),
        );
    }
}

#[test]
fn clarity_matches() {
    for clarity in [60.0, -60.0] {
        check(
            "clarity",
            &PhotoEdit::from(Adjustments {
                clarity,
                ..Adjustments::default()
            }),
        );
    }
}

#[test]
fn positive_dehaze_matches() {
    check(
        "dehaze +",
        &PhotoEdit::from(Adjustments {
            dehaze: 70.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn negative_dehaze_matches() {
    check(
        "dehaze -",
        &PhotoEdit::from(Adjustments {
            dehaze: -50.0,
            ..Adjustments::default()
        }),
    );
}

#[test]
fn every_operator_together_matches_develop_pixel() {
    let mut edit = PhotoEdit::from(Adjustments {
        white_balance_temperature: -20.0,
        white_balance_tint: 10.0,
        exposure: 0.4,
        contrast: 25.0,
        highlights: -40.0,
        shadows: 30.0,
        whites: 15.0,
        blacks: -10.0,
        vibrance: 20.0,
        saturation: 8.0,
        texture: 40.0,
        clarity: 35.0,
        dehaze: 30.0,
        ..Adjustments::default()
    });
    edit.look.curves.master = s_curve();
    edit.look.curves.blue = curve(&[[0.0, 0.03], [0.5, 0.46], [1.0, 1.0]]);
    edit.look.hsl[1].saturation = -40.0;
    edit.look.hsl[4].hue = 50.0;
    edit.look.hsl[5].luminance = -30.0;
    edit.look.wheels.shadows = Wheel {
        x: -0.4,
        y: -0.3,
        luminance: 0.0,
    };
    edit.look.wheels.midtones = Wheel {
        x: 0.1,
        y: 0.2,
        luminance: 10.0,
    };
    edit.look.wheels.highlights = Wheel {
        x: 0.4,
        y: 0.15,
        luminance: 0.0,
    };
    check("every operator", &edit);
}

/// Texture and dehaze switched on after a first render draw their head-pass
/// products then, and switching them off again returns the first picture.
#[test]
fn a_product_switched_on_later_matches_a_fresh_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = hazy_photo();
    let on = PhotoEdit::from(Adjustments {
        texture: 50.0,
        dehaze: 60.0,
        ..Adjustments::default()
    });
    let fresh = gpu_render(&gpu, &photo, &on);

    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let mut render = |edit: &PhotoEdit| -> Vec<u8> {
        let view = develop
            .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
            .expect("a source is set");
        Readback::new(&gpu.device).read(&gpu.device, &gpu.queue, view, SIZE, SIZE)
    };
    let first = render(&PhotoEdit::default());
    let later: Vec<[u8; 3]> = render(&on)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| [px[0], px[1], px[2]])
        .collect();
    assert_eq!(later, fresh);
    assert_eq!(render(&PhotoEdit::default()), first);
}

fn linear_source() -> MaskSource {
    MaskSource::Linear(LinearGradient {
        start: [0.2, 0.8],
        end: [0.7, 0.3],
    })
}

fn radial_source() -> MaskSource {
    MaskSource::Radial(RadialGradient {
        centre: [0.55, 0.45],
        radius: [0.35, 0.2],
        rotation: 30.0,
        feather: 40.0,
    })
}

fn luminance_source() -> MaskSource {
    MaskSource::Luminance(LuminanceRange {
        low: 0.45,
        high: 0.8,
        falloff: 0.1,
    })
}

fn colour_source() -> MaskSource {
    MaskSource::Colour(ColourRange {
        hue: 250.0,
        hue_width: 80.0,
        chroma_low: 0.03,
        falloff: 20.0,
    })
}

/// A mask of one source that lifts the exposure.
fn exposure_mask(name: &str, source: MaskSource) -> Mask {
    let mut mask = Mask::new(name, source);
    mask.adjust.exposure = 1.2;
    mask
}

fn masked(masks: Vec<Mask>) -> PhotoEdit {
    PhotoEdit {
        masks,
        ..PhotoEdit::default()
    }
}

/// A mask has to move the picture for its golden test to mean anything.
fn assert_the_masks_show(photo: &Photo, edit: &PhotoEdit) {
    let with = cpu_reference(photo, edit, Rounding::Nearest, 0);
    let without = cpu_reference(
        photo,
        &PhotoEdit::from(edit.adjust.clone()),
        Rounding::Nearest,
        0,
    );
    let moved = with.iter().zip(&without).filter(|(a, b)| a != b).count();
    assert!(
        moved * 20 > with.len(),
        "the masks move only {moved} of {} pixels",
        with.len()
    );
    assert!(moved < with.len(), "the masks move every pixel");
}

fn check_masks(name: &str, edit: &PhotoEdit) {
    assert_the_masks_show(&synthetic_photo(), edit);
    check(name, edit);
}

#[test]
fn a_linear_gradient_mask_matches() {
    check_masks(
        "linear gradient mask",
        &masked(vec![exposure_mask("Linear", linear_source())]),
    );
}

#[test]
fn a_radial_gradient_mask_matches() {
    check_masks(
        "radial gradient mask",
        &masked(vec![exposure_mask("Radial", radial_source())]),
    );
}

#[test]
fn a_luminance_range_mask_matches() {
    check_masks(
        "luminance range mask",
        &masked(vec![exposure_mask("Luminance", luminance_source())]),
    );
}

#[test]
fn a_colour_range_mask_matches() {
    check_masks(
        "colour range mask",
        &masked(vec![exposure_mask("Colour", colour_source())]),
    );
}

#[test]
fn an_inverted_mask_matches() {
    let mut mask = exposure_mask("Outside", radial_source());
    mask.invert = true;
    let inverted = masked(vec![mask]);
    check_masks("inverted mask", &inverted);
    let photo = synthetic_photo();
    let plain = masked(vec![exposure_mask("Inside", radial_source())]);
    assert_ne!(
        cpu_reference(&photo, &inverted, Rounding::Nearest, 0),
        cpu_reference(&photo, &plain, Rounding::Nearest, 0)
    );
}

#[test]
fn a_mask_at_half_opacity_matches() {
    let mut mask = exposure_mask("Half", radial_source());
    mask.opacity = 50.0;
    check_masks("mask at opacity 50", &masked(vec![mask]));
}

fn linear_and_radial(op: MaskOp) -> PhotoEdit {
    let mut mask = exposure_mask("Both", linear_source());
    mask.components.push(Component {
        op,
        source: radial_source(),
        invert: false,
    });
    masked(vec![mask])
}

#[test]
fn a_linear_gradient_with_a_radial_added_matches() {
    check_masks("linear add radial", &linear_and_radial(MaskOp::Add));
}

#[test]
fn a_linear_gradient_with_a_radial_subtracted_matches() {
    check_masks(
        "linear subtract radial",
        &linear_and_radial(MaskOp::Subtract),
    );
}

#[test]
fn a_linear_gradient_intersected_with_a_radial_matches() {
    check_masks(
        "linear intersect radial",
        &linear_and_radial(MaskOp::Intersect),
    );
}

#[test]
fn two_overlapping_masks_blend_in_list_order() {
    let mut warm = Mask::new("Warm", radial_source());
    warm.adjust.white_balance_temperature = 60.0;
    warm.adjust.exposure = 0.8;
    let mut dark = Mask::new("Dark", linear_source());
    dark.adjust.exposure = -1.0;
    dark.adjust.saturation = -50.0;
    dark.opacity = 70.0;
    let one_way = masked(vec![warm.clone(), dark.clone()]);
    let other_way = masked(vec![dark, warm]);
    check_masks("two masks in order", &one_way);
    check_masks("the same two masks swapped", &other_way);

    // The order is part of the picture, on the CPU and on the GPU.
    let photo = synthetic_photo();
    assert_ne!(
        cpu_reference(&photo, &one_way, Rounding::Nearest, 0),
        cpu_reference(&photo, &other_way, Rounding::Nearest, 0)
    );
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    assert_ne!(
        gpu_render(&gpu, &photo, &one_way),
        gpu_render(&gpu, &photo, &other_way)
    );
}

/// The edit of `every_operator_together_matches_develop_pixel`.
fn everything_global() -> PhotoEdit {
    let mut edit = PhotoEdit::from(Adjustments {
        white_balance_temperature: -20.0,
        white_balance_tint: 10.0,
        exposure: 0.4,
        contrast: 25.0,
        highlights: -40.0,
        shadows: 30.0,
        whites: 15.0,
        blacks: -10.0,
        vibrance: 20.0,
        saturation: 8.0,
        texture: 40.0,
        clarity: 35.0,
        dehaze: 30.0,
        ..Adjustments::default()
    });
    edit.look.curves.master = s_curve();
    edit.look.curves.blue = curve(&[[0.0, 0.03], [0.5, 0.46], [1.0, 1.0]]);
    edit.look.hsl[1].saturation = -40.0;
    edit.look.hsl[4].hue = 50.0;
    edit.look.hsl[5].luminance = -30.0;
    edit.look.wheels.shadows = Wheel {
        x: -0.4,
        y: -0.3,
        luminance: 0.0,
    };
    edit.look.wheels.midtones = Wheel {
        x: 0.1,
        y: 0.2,
        luminance: 10.0,
    };
    edit.look.wheels.highlights = Wheel {
        x: 0.4,
        y: 0.15,
        luminance: 0.0,
    };
    edit
}

/// Adjustments that set every group: Basic, Presence, curves, mixer, wheels.
fn everything_in_a_mask() -> Adjustments {
    let mut adjust = Adjustments {
        white_balance_temperature: 35.0,
        white_balance_tint: -15.0,
        exposure: -0.6,
        contrast: -20.0,
        highlights: 30.0,
        shadows: -25.0,
        whites: -10.0,
        blacks: 12.0,
        vibrance: -30.0,
        saturation: 25.0,
        texture: -30.0,
        clarity: 20.0,
        dehaze: 25.0,
        ..Adjustments::default()
    };
    adjust.look.curves.master = curve(&[[0.0, 0.05], [0.4, 0.5], [1.0, 0.95]]);
    adjust.look.curves.red = curve(&[[0.0, 0.0], [0.5, 0.58], [1.0, 1.0]]);
    adjust.look.hsl[1].saturation = 60.0;
    adjust.look.hsl[5].hue = -40.0;
    adjust.look.hsl[7].luminance = 35.0;
    adjust.look.wheels.shadows = Wheel {
        x: 0.5,
        y: 0.2,
        luminance: 15.0,
    };
    adjust.look.wheels.highlights = Wheel {
        x: -0.3,
        y: -0.4,
        luminance: -20.0,
    };
    adjust
}

#[test]
fn a_mask_carrying_everything_over_a_global_edit_with_everything_matches() {
    let mut edit = everything_global();
    let mut mask = Mask::new("Everything", radial_source());
    mask.adjust = everything_in_a_mask();
    mask.opacity = 85.0;
    edit.masks.push(mask);
    check_masks("a mask carrying everything", &edit);
}

#[test]
fn a_mask_that_alone_turns_on_dehaze_and_texture_matches() {
    let mut mask = Mask::new("Haze", linear_source());
    mask.adjust.dehaze = 60.0;
    mask.adjust.texture = 50.0;
    check_masks("a mask alone with dehaze and texture", &masked(vec![mask]));
}

/// A render of only the crop window has to show the masks where the full
/// render shows them: the alpha is laid out on the photo, not on the render.
#[test]
fn masks_under_a_crop_window_match_the_full_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let crop = CropRect {
        x: 0.25,
        y: 0.25,
        width: 0.5,
        height: 0.5,
    };
    let output = (32, 32);
    let mut colour = exposure_mask("Colour", colour_source());
    colour.adjust.saturation = -60.0;
    let edit = masked(vec![
        exposure_mask("Linear", linear_source()),
        exposure_mask("Radial", radial_source()),
        colour,
    ]);
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let read =
        |view: &wgpu::TextureView| readback.read(&gpu.device, &gpu.queue, view, output.0, output.1);
    let full = develop
        .render(&edit, crop, (SIZE, SIZE), output)
        .expect("a source is set");
    let full = read(full);
    let windowed = develop
        .render_crop(&edit, crop, output)
        .expect("a source is set");
    let windowed = read(windowed);
    let plain = develop
        .render_crop(&PhotoEdit::default(), crop, output)
        .expect("a source is set");
    let plain = read(plain);
    let max = full
        .iter()
        .zip(&windowed)
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
        .max()
        .unwrap_or(0);
    println!("masked windowed render against the full render: max difference {max}");
    assert!(max <= 1, "max difference {max}");
    assert_ne!(windowed, plain, "the masks show inside the window");
}

/// The edit the zoomed window tests develop: everything on in the global
/// edit, a positional mask and a range mask over it.
fn zoomed_edit() -> PhotoEdit {
    let mut edit = everything_global();
    let mut radial = exposure_mask("Radial", radial_source());
    radial.adjust.clarity = 30.0;
    let mut range = exposure_mask("Luminance", luminance_source());
    range.adjust.saturation = -50.0;
    edit.masks = vec![radial, range];
    edit
}

/// The whole picture at 100 and at 200 percent of a fitted size of 32, with
/// the padded window and the visible part a zoomed viewer would ask for.
fn zoomed_views() -> [ViewWindow; 2] {
    [
        ViewWindow {
            full: (32, 32),
            window: (8, 6, 18, 20),
            visible: (11, 9, 10, 12),
        },
        ViewWindow {
            full: (SIZE, SIZE),
            window: (12, 10, 40, 40),
            visible: (20, 18, 24, 20),
        },
    ]
}

/// The part of the full render at `view.full` that `view.visible` names.
fn full_render_of(
    develop: &mut Develop,
    gpu: &Headless,
    readback: &Readback,
    edit: &PhotoEdit,
    view: &ViewWindow,
) -> Vec<u8> {
    let (x, y, w, h) = view.visible;
    let (fw, fh) = (view.full.0 as f32, view.full.1 as f32);
    let crop = CropRect {
        x: x as f32 / fw,
        y: y as f32 / fh,
        width: w as f32 / fw,
        height: h as f32 / fh,
    };
    let full = develop
        .render(edit, crop, view.full, (w, h))
        .expect("a source is set");
    readback.read(&gpu.device, &gpu.queue, full, w, h)
}

fn view_render_of(
    develop: &mut Develop,
    gpu: &Headless,
    readback: &Readback,
    edit: &PhotoEdit,
    view: &ViewWindow,
) -> Vec<u8> {
    let zoomed = develop.render_view(edit, view).expect("a source is set");
    readback.read(
        &gpu.device,
        &gpu.queue,
        zoomed,
        view.visible.2,
        view.visible.3,
    )
}

fn max_difference(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
        .max()
        .unwrap_or(0)
}

/// What a zoomed viewer renders, a padded window of the source with the
/// visible part taken out of it, is the same picture the full render shows
/// there: the sigmas come from the full size and the masks lie on the photo.
#[test]
fn a_zoomed_window_matches_the_full_render_at_100_and_200_percent_of_the_fit() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let edit = zoomed_edit();
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    for view in zoomed_views() {
        let full = full_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let zoomed = view_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let plain = view_render_of(&mut develop, &gpu, &readback, &PhotoEdit::default(), &view);
        let max = max_difference(&full, &zoomed);
        println!(
            "zoomed window at {:?} against the full render: max difference {max}",
            view.full
        );
        assert!(max <= 1, "max difference {max} at {:?}", view.full);
        assert_ne!(zoomed, plain, "the edit shows inside the window");
    }
}

/// A pan that stays inside the padded window moves the output crop and
/// nothing else: no mask alpha is drawn again, and the picture is still the
/// full render's. A window somewhere else draws them again.
#[test]
fn a_pan_inside_the_zoomed_window_redraws_no_product_and_matches() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let edit = zoomed_edit();
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let [_, first] = zoomed_views();
    view_render_of(&mut develop, &gpu, &readback, &edit, &first);
    let builds = develop.mask_alpha_builds();
    assert_eq!(builds, 2, "one alpha per mask");
    for visible in [(12, 10, 24, 20), (28, 30, 24, 20), (21, 19, 24, 20)] {
        let panned = ViewWindow { visible, ..first };
        let zoomed = view_render_of(&mut develop, &gpu, &readback, &edit, &panned);
        assert_eq!(
            develop.mask_alpha_builds(),
            builds,
            "a pan to {visible:?} inside the window"
        );
        // The reference replaces the frame, so it is drawn on its own graph.
        let mut other = Develop::new(&gpu.device, &gpu.queue);
        other.set_source(&photo);
        let full = full_render_of(&mut other, &gpu, &readback, &edit, &panned);
        let max = max_difference(&full, &zoomed);
        println!("panned to {visible:?} against the full render: max difference {max}");
        assert!(max <= 1, "max difference {max} at {visible:?}");
    }
    let moved = ViewWindow {
        window: (20, 20, 40, 40),
        visible: (30, 30, 24, 20),
        ..first
    };
    view_render_of(&mut develop, &gpu, &readback, &edit, &moved);
    assert_eq!(
        develop.mask_alpha_builds(),
        builds + 2,
        "a window somewhere else draws both alphas again"
    );
}

/// The red overlay of a mask under a zoomed window is the overlay the full
/// render shows there.
#[test]
fn the_overlay_under_a_zoomed_window_matches_the_full_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let edit = zoomed_edit();
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    for view in zoomed_views() {
        develop.set_overlay(None);
        let bare = view_render_of(&mut develop, &gpu, &readback, &edit, &view);
        develop.set_overlay(Some(0));
        let full = full_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let zoomed = view_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let max = max_difference(&full, &zoomed);
        println!(
            "overlay under a zoomed window at {:?}: max difference {max}",
            view.full
        );
        assert!(max <= 1, "max difference {max} at {:?}", view.full);
        assert_ne!(zoomed, bare, "the overlay shows inside the window");
    }
}

/// The alpha of a mask is a head-pass product: a slider of the mask, its
/// opacity or the global edit never draws it again; its shape does, and so
/// does a new source.
#[test]
fn a_mask_slider_does_not_rebuild_its_alpha_and_a_component_does() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let mut builds_after = |edit: &PhotoEdit| -> u64 {
        develop
            .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
            .expect("a source is set");
        develop.mask_alpha_builds()
    };
    let mut edit = masked(vec![
        exposure_mask("Radial", radial_source()),
        exposure_mask("Luminance", luminance_source()),
    ]);
    assert_eq!(builds_after(&PhotoEdit::default()), 0, "no mask, no alpha");
    assert_eq!(builds_after(&edit), 2, "one alpha per mask");
    assert_eq!(builds_after(&edit), 2, "the same edit again");

    edit.masks[0].adjust.exposure = -0.5;
    edit.masks[0].adjust.look.curves.master = s_curve();
    edit.masks[1].adjust.look.wheels.shadows.x = 0.3;
    assert_eq!(builds_after(&edit), 2, "sliders of the masks");
    edit.masks[0].opacity = 35.0;
    edit.masks[0].name = "Renamed".to_string();
    edit.exposure = 0.7;
    assert_eq!(
        builds_after(&edit),
        2,
        "the opacity, the name, a global slider"
    );

    edit.masks[0].components[0].invert = true;
    assert_eq!(builds_after(&edit), 3, "a component of the first mask");
    edit.masks[1].invert = true;
    assert_eq!(builds_after(&edit), 4, "the invert of the second mask");
    edit.masks[1]
        .components
        .push(Component::new(linear_source()));
    assert_eq!(builds_after(&edit), 5, "a new component");

    edit.masks[1].enabled = false;
    assert_eq!(builds_after(&edit), 5, "a disabled mask costs nothing");
    edit.masks[1].enabled = true;
    assert_eq!(builds_after(&edit), 5, "and comes back with its alpha kept");

    develop.set_source(&hazy_photo());
    develop
        .render(&edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
        .expect("a source is set");
    assert_eq!(
        develop.mask_alpha_builds(),
        7,
        "a new source draws both again"
    );
}

/// The red overlay of the selected mask: the output pass mixes red in by the
/// alpha of the mask, on a mask that adjusts nothing as much as on one that
/// does, and an export never shows it.
#[test]
fn the_overlay_of_a_mask_matches_and_stays_out_of_an_export() {
    check_overlay(Mask::new("Idle", radial_source()));
}

/// The overlay of `idle`, a mask that adjusts nothing and covers the middle
/// of the photo but not its first pixel.
fn check_overlay(idle: Mask) {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let mut edit = PhotoEdit::from(Adjustments {
        exposure: 0.3,
        ..Adjustments::default()
    });
    edit.masks = vec![exposure_mask("Linear", linear_source()), idle.clone()];

    // The reference: the developed picture through the output transform
    // with the overlay of the second mask between the clip and the curve.
    let linear: Vec<[f32; 3]> = photo
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| basic::decode_rgb8([px[0], px[1], px[2]], photo.source))
        .collect();
    let geometry = Geometry::full((SIZE, SIZE), (SIZE, SIZE));
    // The overlay shows the alpha itself, and near black one code of alpha
    // is six of the output, so the reference takes the unorm store at every
    // step the conversion is allowed.
    let reference = |rounding: Rounding, step: f32| -> Vec<[u8; 3]> {
        let stored: Vec<[f32; 3]> = linear
            .iter()
            .map(|px| px.map(|c| half(c, rounding)))
            .collect();
        let store = |v: f32| half(v, rounding);
        let alphas: Vec<f32> =
            mask_twin::alpha_image_before_the_store(&idle, &stored, &geometry, None, &store)
                .into_iter()
                .map(|alpha| mask_twin::stored_alpha_stepping(alpha, step))
                .collect();
        let luma: Vec<f32> = stored.iter().map(|px| basic::luma(*px)).collect();
        let base = basic::gaussian_stored(&luma, SIZE, SIZE, basic::base_sigma(SIZE, SIZE), &store);
        let clear = vec![1.0; stored.len()];
        let image = Image {
            pixels: &stored,
            base: &base,
            texture: &luma,
            transmission: &clear,
            geometry,
            proxy: None,
        };
        mask_twin::develop_image_with(&image, &edit, [1.0; 3], &store, &store)
            .into_iter()
            .zip(alphas)
            .map(|(px, alpha)| {
                let srgb = matrices::rec2020_to_srgb()
                    .apply(px)
                    .map(|c| c.clamp(0.0, 1.0));
                mask_twin::overlay(srgb, alpha).map(transfer::linear_to_srgb8)
            })
            .collect()
    };
    let tolerance = mask_twin::UNORM_STEP_TOLERANCE;
    let references: Vec<Vec<[u8; 3]>> = [Rounding::Nearest, Rounding::TowardZero]
        .into_iter()
        .flat_map(|rounding| [-tolerance, 0.0, tolerance].map(|step| reference(rounding, step)))
        .collect();

    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let render = |develop: &mut Develop| -> Vec<[u8; 3]> {
        let view = develop
            .render(&edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
            .expect("a source is set");
        readback
            .read(&gpu.device, &gpu.queue, view, SIZE, SIZE)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| [px[0], px[1], px[2]])
            .collect()
    };
    let plain = render(&mut develop);
    develop.set_overlay(Some(1));
    let overlaid = render(&mut develop);
    let mut max = 0;
    let mut sum = 0u64;
    for (i, g) in overlaid.iter().enumerate() {
        for k in 0..3 {
            let d = references
                .iter()
                .map(|r| (i32::from(r[i][k]) - i32::from(g[k])).abs())
                .min()
                .expect("six references");
            max = max.max(d);
            sum += d as u64;
        }
    }
    let mean = sum as f64 / (overlaid.len() * 3) as f64;
    println!("overlay: max {max}, mean {mean:.3}");
    assert!(max <= MAX_DIFFERENCE, "overlay: max difference {max}");
    assert!(mean <= MEAN_DIFFERENCE, "overlay: mean difference {mean}");

    // Red where the mask is, the plain picture where it is not.
    let centre = (SIZE * (SIZE * 45 / 100) + SIZE * 55 / 100) as usize;
    assert!(
        overlaid[centre][0] > plain[centre][0],
        "red inside the mask"
    );
    assert!(overlaid[centre][1] < plain[centre][1], "less green inside");
    assert_eq!(overlaid[0], plain[0], "nothing outside the mask");

    // Off again, the first picture comes back; an out of range mask shows
    // nothing; an export ignores the overlay.
    develop.set_overlay(None);
    assert_eq!(render(&mut develop), plain);
    develop.set_overlay(Some(5));
    assert_eq!(render(&mut develop), plain);
    develop.set_overlay(Some(1));
    let shown = develop
        .render_export(&edit, CropRect::FULL, ExportPreset::ALL[0])
        .expect("a source is set");
    assert_eq!(develop.overlay(), Some(1), "the export leaves the choice");
    develop.set_overlay(None);
    let hidden = develop
        .render_export(&edit, CropRect::FULL, ExportPreset::ALL[0])
        .expect("a source is set");
    assert_eq!(shown, hidden);
}

fn stroke(points: &[[f32; 2]], size: f32, feather: f32, flow: f32) -> Stroke {
    Stroke {
        points: points.to_vec(),
        size,
        feather,
        flow,
        ..Stroke::default()
    }
}

fn brush_source(strokes: &[Stroke]) -> MaskSource {
    MaskSource::Brush(Brush {
        strokes: strokes.iter().map(SharedStroke::new).collect(),
    })
}

/// Two crossing strokes of different brushes over the middle of the photo,
/// one of them built up at a part flow.
fn painted_strokes() -> Vec<Stroke> {
    vec![
        stroke(&[[0.2, 0.3], [0.5, 0.45], [0.8, 0.4]], 0.12, 60.0, 100.0),
        stroke(&[[0.6, 0.15], [0.55, 0.5], [0.35, 0.85]], 0.08, 30.0, 45.0),
    ]
}

fn painted_source() -> MaskSource {
    brush_source(&painted_strokes())
}

#[test]
fn a_brush_mask_matches() {
    check_masks(
        "brush mask",
        &masked(vec![exposure_mask("Brush", painted_source())]),
    );
}

#[test]
fn a_brush_with_an_erase_stroke_matches() {
    let mut strokes = painted_strokes();
    strokes.push(Stroke {
        erase: true,
        ..stroke(&[[0.3, 0.2], [0.7, 0.7]], 0.07, 50.0, 80.0)
    });
    // Painted again after the erase: the order is part of the picture.
    strokes.push(stroke(&[[0.5, 0.45]], 0.05, 80.0, 60.0));
    let erased = masked(vec![exposure_mask("Erased", brush_source(&strokes))]);
    let photo = synthetic_photo();
    assert_ne!(
        cpu_reference(&photo, &erased, Rounding::Nearest, 0),
        cpu_reference(
            &photo,
            &masked(vec![exposure_mask("Brush", painted_source())]),
            Rounding::Nearest,
            0
        ),
        "the erase shows"
    );
    check_masks("brush with an erase stroke", &erased);
}

#[test]
fn a_brush_at_feather_0_and_at_feather_100_matches() {
    for feather in [0.0, 100.0] {
        let strokes = [
            stroke(&[[0.25, 0.3], [0.7, 0.6]], 0.13, feather, 100.0),
            stroke(&[[0.3, 0.75]], 0.1, feather, 70.0),
        ];
        check_masks(
            &format!("brush at feather {feather}"),
            &masked(vec![exposure_mask("Brush", brush_source(&strokes))]),
        );
    }
}

/// A low flow built up by many passes has to keep building: the layer is a
/// half float because a dab of 5 percent adds under half of an 8 bit code
/// once the alpha passes 0.96.
#[test]
fn a_low_flow_built_up_by_many_strokes_matches() {
    let pass = stroke(&[[0.3, 0.5], [0.7, 0.5]], 0.15, 40.0, 5.0);
    let strokes = vec![pass; 20];
    let geometry = Geometry::full((SIZE, SIZE), (SIZE, SIZE));
    let mask = exposure_mask("Built up", brush_source(&strokes));
    let middle = mask_twin::alpha(&mask, [0.5, 0.5], geometry.aspect(), [0.18; 3]);
    assert!(middle > 0.99, "twenty passes of 5 percent reach {middle}");
    check_masks("low flow built up", &masked(vec![mask]));
}

fn brush_and(other: MaskSource, op: MaskOp, brush_first: bool) -> PhotoEdit {
    let (first, second) = if brush_first {
        (painted_source(), other)
    } else {
        (other, painted_source())
    };
    let mut mask = exposure_mask("Both", first);
    mask.components.push(Component {
        op,
        source: second,
        invert: false,
    });
    masked(vec![mask])
}

#[test]
fn a_brush_subtracted_from_a_linear_gradient_matches() {
    let edit = brush_and(linear_source(), MaskOp::Subtract, false);
    let photo = synthetic_photo();
    assert_ne!(
        cpu_reference(&photo, &edit, Rounding::Nearest, 0),
        cpu_reference(
            &photo,
            &masked(vec![exposure_mask("Linear", linear_source())]),
            Rounding::Nearest,
            0
        ),
        "the brush takes something away"
    );
    check_masks("brush subtracted from a linear gradient", &edit);
}

#[test]
fn a_brush_intersected_with_a_radial_gradient_matches() {
    check_masks(
        "brush intersected with a radial gradient",
        &brush_and(radial_source(), MaskOp::Intersect, true),
    );
}

#[test]
fn two_brush_components_in_one_mask_match() {
    let mut mask = exposure_mask("Two brushes", painted_source());
    mask.components.push(Component::new(linear_source()));
    mask.components.push(Component {
        op: MaskOp::Subtract,
        source: brush_source(&[stroke(&[[0.2, 0.2], [0.8, 0.8]], 0.1, 70.0, 90.0)]),
        invert: false,
    });
    check_masks("two brush components in one mask", &masked(vec![mask]));
}

#[test]
fn a_brush_mask_carrying_everything_over_a_global_edit_with_everything_matches() {
    let mut edit = everything_global();
    let mut mask = Mask::new("Everything", painted_source());
    mask.adjust = everything_in_a_mask();
    mask.opacity = 85.0;
    edit.masks.push(mask);
    check_masks("a brush mask carrying everything", &edit);
}

#[test]
fn the_overlay_of_a_brush_mask_matches_and_stays_out_of_an_export() {
    check_overlay(Mask::new("Idle", painted_source()));
}

/// A brush is laid out on the photo: under a crop window and under a zoomed
/// window at 100 and at 200 percent of the fit it is where the full render
/// has it.
#[test]
fn a_brush_under_a_crop_window_and_a_zoomed_window_matches_the_full_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let mut brush = exposure_mask("Brush", painted_source());
    brush.adjust.clarity = 30.0;
    let masks = vec![brush, exposure_mask("Radial", radial_source())];
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);

    let crop = CropRect {
        x: 0.25,
        y: 0.25,
        width: 0.5,
        height: 0.5,
    };
    let output = (32, 32);
    // The crop window carries the masks alone, as the crop test of the other
    // sources does: a crop render has no reach for the blurs of a global
    // edit, which is what the zoomed window adds.
    let masks_only = masked(masks.clone());
    let full = develop
        .render(&masks_only, crop, (SIZE, SIZE), output)
        .expect("a source is set");
    let full = readback.read(&gpu.device, &gpu.queue, full, output.0, output.1);
    let windowed = develop
        .render_crop(&masks_only, crop, output)
        .expect("a source is set");
    let windowed = readback.read(&gpu.device, &gpu.queue, windowed, output.0, output.1);
    let max = max_difference(&full, &windowed);
    println!("brush under a crop window against the full render: max difference {max}");
    assert!(max <= 1, "max difference {max}");

    let mut edit = everything_global();
    edit.masks = masks;
    let mut unpainted = edit.clone();
    unpainted.masks.remove(0);
    for view in zoomed_views() {
        let full = full_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let zoomed = view_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let without = view_render_of(&mut develop, &gpu, &readback, &unpainted, &view);
        let max = max_difference(&full, &zoomed);
        println!(
            "brush under a zoomed window at {:?} against the full render: max difference {max}",
            view.full
        );
        assert!(max <= 1, "max difference {max} at {:?}", view.full);
        assert_ne!(zoomed, without, "the brush shows inside the window");
    }
}

/// A brush layer is stamped once and kept: no slider, no opacity and no pan
/// inside the zoomed window stamps it again; its strokes do.
#[test]
fn a_brush_layer_is_kept_until_its_strokes_change() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let mut edit = masked(vec![
        exposure_mask("Brush", painted_source()),
        exposure_mask("Radial", radial_source()),
    ]);
    let [view, _] = zoomed_views();
    let mut counts_after = |edit: &PhotoEdit, view: &ViewWindow| -> (u64, u64, u64) {
        develop.render_view(edit, view).expect("a source is set");
        (
            develop.brush_layer_builds(),
            develop.brush_layer_appends(),
            develop.mask_alpha_builds(),
        )
    };
    assert_eq!(
        counts_after(&edit, &view),
        (1, 0, 2),
        "one layer, two alphas"
    );
    assert_eq!(counts_after(&edit, &view), (1, 0, 2), "the same edit again");

    edit.masks[0].adjust.exposure = -0.4;
    edit.masks[0].adjust.look.curves.master = s_curve();
    assert_eq!(
        counts_after(&edit, &view),
        (1, 0, 2),
        "a slider of the brush mask"
    );
    edit.masks[0].opacity = 40.0;
    edit.exposure = 0.6;
    assert_eq!(
        counts_after(&edit, &view),
        (1, 0, 2),
        "its opacity, a global slider"
    );

    let panned = ViewWindow {
        visible: (9, 7, 10, 12),
        ..view
    };
    assert_eq!(
        counts_after(&edit, &panned),
        (1, 0, 2),
        "a pan inside the window"
    );

    // A new stroke is stamped onto the layer that is there, and the alpha of
    // that mask alone is drawn again.
    let MaskSource::Brush(brush) = &mut edit.masks[0].components[0].source else {
        panic!("a brush");
    };
    brush
        .strokes
        .push(SharedStroke::new(&stroke(&[[0.4, 0.4]], 0.05, 50.0, 100.0)));
    assert_eq!(counts_after(&edit, &panned), (1, 1, 3), "a new stroke");

    // An undo takes a stroke away: the layer starts again.
    let MaskSource::Brush(brush) = &mut edit.masks[0].components[0].source else {
        panic!("a brush");
    };
    brush.strokes.truncate(1);
    assert_eq!(counts_after(&edit, &panned), (2, 1, 4), "a stroke removed");

    // A window somewhere else is a new frame: everything is drawn again.
    let elsewhere = ViewWindow {
        window: (2, 2, 18, 20),
        visible: (4, 4, 10, 12),
        ..view
    };
    assert_eq!(counts_after(&edit, &elsewhere), (3, 1, 6), "another window");
}

/// A stroke painted piece by piece, the way the window paints it, is the
/// stroke drawn whole: new dabs go onto the layer that is there, and the
/// alpha and the develop passes run over what those dabs reach only. Dabs
/// build in order, so nothing is lost by it. A second brush mask that does
/// not grow is never stamped again.
#[test]
fn a_stroke_painted_in_ten_appended_pieces_equals_the_stroke_drawn_whole() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let readback = Readback::new(&gpu.device);
    let path: Vec<[f32; 2]> = (0..=40)
        .map(|i| {
            let t = i as f32 / 40.0;
            [0.15 + t * 0.7, 0.5 + (t * 7.0).sin() * 0.25]
        })
        .collect();
    let eraser: Vec<[f32; 2]> = (0..=20)
        .map(|i| {
            [
                0.5 + (i as f32 / 20.0 - 0.5) * 0.1,
                0.1 + i as f32 / 20.0 * 0.8,
            ]
        })
        .collect();
    // The edit with the first `painted` points of the stroke and the first
    // `erased` points of the erase stroke after it.
    let edit_at = |painted: usize, erased: usize| -> PhotoEdit {
        let mut strokes = painted_strokes();
        strokes.push(stroke(&path[..painted], 0.07, 45.0, 60.0));
        if erased > 0 {
            strokes.push(Stroke {
                erase: true,
                ..stroke(&eraser[..erased], 0.05, 30.0, 70.0)
            });
        }
        let mut edit = everything_global();
        let mut growing = exposure_mask("Growing", brush_source(&strokes));
        growing.adjust.clarity = 25.0;
        let resting = exposure_mask(
            "Resting",
            brush_source(&[stroke(&[[0.2, 0.8], [0.8, 0.85]], 0.06, 50.0, 80.0)]),
        );
        edit.masks = vec![growing, resting, exposure_mask("Radial", radial_source())];
        edit
    };
    let [_, view] = zoomed_views();
    for zoomed in [false, true] {
        let render = |develop: &mut Develop, edit: &PhotoEdit| -> Vec<u8> {
            if zoomed {
                view_render_of(develop, &gpu, &readback, edit, &view)
            } else {
                let drawn = develop
                    .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
                    .expect("a source is set");
                readback.read(&gpu.device, &gpu.queue, drawn, SIZE, SIZE)
            }
        };
        let mut develop = Develop::new(&gpu.device, &gpu.queue);
        develop.set_source(&photo);
        let mut pieces = Vec::new();
        for piece in 1..=10 {
            pieces.push(render(&mut develop, &edit_at(piece * 4 + 1, 0)));
        }
        for piece in 1..=5 {
            pieces.push(render(&mut develop, &edit_at(41, piece * 4 + 1)));
        }
        assert_ne!(pieces[0], pieces[9], "the stroke grew on the picture");
        assert_ne!(pieces[9], pieces[14], "and the eraser took some of it away");
        assert_eq!(
            develop.brush_layer_builds(),
            2,
            "each layer stamped whole once"
        );
        assert_eq!(
            develop.brush_layer_appends(),
            14,
            "and the growing one added to"
        );
        // Fitted, every piece is seen and is drawn over its own reach. Zoomed,
        // a piece outside what is seen develops nothing at all.
        let (alphas, develops) = develop.brush_patches();
        if zoomed {
            assert!(alphas > 0 && alphas <= 14 && develops > 0 && develops <= alphas);
        } else {
            assert_eq!((alphas, develops), (14, 14), "over the new dabs only");
        }
        assert_eq!(develop.mask_alpha_builds(), 3 + 14);

        let mut fresh = Develop::new(&gpu.device, &gpu.queue);
        fresh.set_source(&photo);
        let whole = render(&mut fresh, &edit_at(41, 21));
        assert_eq!(fresh.brush_layer_appends(), 0);
        let max = max_difference(&pieces[14], &whole);
        println!(
            "a stroke in appended pieces against the stroke whole, zoomed {zoomed}: max difference {max}"
        );
        assert!(max <= 1, "max difference {max}, zoomed {zoomed}");

        // An undo takes the erase stroke away: the layer starts again and
        // the picture is the one from before it.
        let undone = render(&mut develop, &edit_at(41, 0));
        assert_eq!(develop.brush_layer_builds(), 3);
        assert!(max_difference(&undone, &pieces[9]) <= 1);
    }
}

fn auto(stroke: Stroke, sensitivity: f32) -> Stroke {
    Stroke {
        auto: true,
        sensitivity,
        ..stroke
    }
}

/// Two auto strokes that run along an edge of the photo, where a gate shows:
/// one down the colours beside the grey ramp of the first columns, one along
/// the colours above the near-black band at the bottom, each wide enough to
/// reach over its edge. A stroke across the smooth hues would show almost
/// nothing: every dab takes the colour under its own centre.
fn auto_strokes(sensitivity: f32) -> Vec<Stroke> {
    vec![
        auto(
            stroke(&[[0.17, 0.1], [0.17, 0.8]], 0.12, 60.0, 100.0),
            sensitivity,
        ),
        auto(
            stroke(&[[0.25, 0.82], [0.9, 0.82]], 0.1, 30.0, 100.0),
            sensitivity,
        ),
    ]
}

fn assert_the_gate_shows(edit: &PhotoEdit) {
    let mut plain = edit.clone();
    for mask in &mut plain.masks {
        for component in &mut mask.components {
            if let MaskSource::Brush(brush) = &mut component.source {
                for stroke in &mut brush.strokes {
                    *stroke = SharedStroke::new(&Stroke {
                        auto: false,
                        ..(**stroke).clone()
                    });
                }
            }
        }
    }
    let photo = synthetic_photo();
    let gated = cpu_reference(&photo, edit, Rounding::Nearest, 0);
    let ungated = cpu_reference(&photo, &plain, Rounding::Nearest, 0);
    let moved = gated.iter().zip(&ungated).filter(|(a, b)| a != b).count();
    assert!(
        moved * 50 > gated.len(),
        "the gate moves only {moved} of {} pixels",
        gated.len()
    );
}

#[test]
fn an_auto_brush_mask_matches() {
    let edit = masked(vec![exposure_mask(
        "Auto",
        brush_source(&auto_strokes(70.0)),
    )]);
    assert_the_gate_shows(&edit);
    check_masks("auto brush mask", &edit);
}

#[test]
fn an_auto_brush_at_a_sensitivity_of_0_and_of_100_matches() {
    let at = |sensitivity: f32| {
        masked(vec![exposure_mask(
            "Auto",
            brush_source(&auto_strokes(sensitivity)),
        )])
    };
    let photo = synthetic_photo();
    assert_ne!(
        cpu_reference(&photo, &at(0.0), Rounding::Nearest, 0),
        cpu_reference(&photo, &at(100.0), Rounding::Nearest, 0),
        "the sensitivity shows"
    );
    check_masks("auto brush at sensitivity 0", &at(0.0));
    assert_the_gate_shows(&at(100.0));
    // At 100 the gate lets so little through that the mask moves under a
    // twentieth of the photo, which check_masks asks for; the twin is held
    // all the same.
    check("auto brush at sensitivity 100", &at(100.0));
}

#[test]
fn an_auto_erase_stroke_matches() {
    // Painted over the grey columns and the colours beside them, then erased
    // down the colours: the grey keeps its paint.
    let strokes = [
        stroke(&[[0.12, 0.1], [0.12, 0.8]], 0.12, 40.0, 100.0),
        Stroke {
            erase: true,
            ..auto(
                stroke(&[[0.17, 0.15], [0.17, 0.75]], 0.12, 50.0, 80.0),
                70.0,
            )
        },
    ];
    let erased = masked(vec![exposure_mask("Erased", brush_source(&strokes))]);
    assert_the_gate_shows(&erased);
    check_masks("auto erase stroke", &erased);
}

#[test]
fn an_auto_brush_subtracted_from_a_radial_gradient_matches() {
    // A radial gradient over the grey columns and the colours beside them;
    // the auto strokes take the colours out of it and leave the grey.
    let over_the_edge = MaskSource::Radial(RadialGradient {
        centre: [0.15, 0.45],
        radius: [0.25, 0.35],
        rotation: 0.0,
        feather: 40.0,
    });
    let mut mask = exposure_mask("Both", over_the_edge);
    mask.components.push(Component {
        op: MaskOp::Subtract,
        source: brush_source(&auto_strokes(60.0)),
        invert: false,
    });
    let edit = masked(vec![mask]);
    assert_the_gate_shows(&edit);
    check_masks("auto brush subtracted from a radial gradient", &edit);
}

/// A stroke of a pen that pressed harder as it went, with the two flags.
fn pen_strokes(size: bool, flow: bool) -> Vec<Stroke> {
    let pen = |stroke: Stroke, pressure: &[f32]| Stroke {
        pressure: pressure.to_vec(),
        pressure_size: size,
        pressure_flow: flow,
        ..stroke
    };
    let [first, second] = painted_strokes().try_into().expect("two strokes");
    vec![
        pen(first, &[0.1, 0.55, 1.0]),
        pen(second, &[1.0, 0.3, 0.05]),
    ]
}

#[test]
fn a_pressure_stroke_matches_with_the_flow_with_the_size_and_with_both() {
    let photo = synthetic_photo();
    let render = |size: bool, flow: bool| {
        let edit = masked(vec![exposure_mask(
            "Pen",
            brush_source(&pen_strokes(size, flow)),
        )]);
        cpu_reference(&photo, &edit, Rounding::Nearest, 0)
    };
    let unpressed = render(false, false);
    assert_eq!(
        unpressed,
        cpu_reference(
            &photo,
            &masked(vec![exposure_mask("Brush", painted_source())]),
            Rounding::Nearest,
            0
        ),
        "with both flags off the pressure changes nothing"
    );
    for (size, flow) in [(false, true), (true, false), (true, true)] {
        assert_ne!(render(size, flow), unpressed, "the pressure shows");
        check_masks(
            &format!("pressure stroke, size {size}, flow {flow}"),
            &masked(vec![exposure_mask(
                "Pen",
                brush_source(&pen_strokes(size, flow)),
            )]),
        );
    }
    let mut both = pen_strokes(true, true);
    both[0] = auto(both[0].clone(), 50.0);
    check_masks(
        "auto pressure stroke",
        &masked(vec![exposure_mask("Pen", brush_source(&both))]),
    );
}

/// A photo wider than the proxy. At 1536 by 48 the proxy is 1024 by 32 and
/// each of its pixels covers one and a half source pixels each way. Blocks
/// of hue across, brighter down, under stripes three pixels wide, so a proxy
/// that sampled a point and one that averaged a box read different colours.
fn wide_photo(width: u32, height: u32) -> Photo {
    let mut rgba8 = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let fy = y as f32 / (height - 1) as f32;
            let (r, g, b) = match (x / 128) % 6 {
                0 => (1.0, 0.25, 0.1),
                1 => (0.9, 0.85, 0.1),
                2 => (0.15, 0.9, 0.2),
                3 => (0.1, 0.8, 0.9),
                4 => (0.2, 0.25, 1.0),
                _ => (0.9, 0.15, 0.85),
            };
            let stripe = if x % 3 == 0 { 0.55 } else { 1.0 };
            let level = (0.35 + 0.6 * fy) * stripe;
            for c in [r, g, b] {
                rgba8.push((c * level * 255.0_f32).round() as u8);
            }
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

/// The proxy of a source larger than it is a real reduction: the tiles, the
/// box filter of `proxy.wgsl` and the bilinear read of `brush.wgsl` against
/// `Proxy::from_source` and `Proxy::sample`.
#[test]
fn an_auto_brush_on_a_photo_wider_than_the_proxy_matches() {
    let photo = wide_photo(1536, 48);
    // The radius is a share of the longer side: 0.012 is 18 pixels.
    let strokes = [
        auto(
            stroke(&[[0.03, 0.5], [0.97, 0.5]], 0.012, 40.0, 100.0),
            60.0,
        ),
        auto(stroke(&[[0.2, 0.2], [0.6, 0.8]], 0.02, 70.0, 60.0), 20.0),
    ];
    let edit = masked(vec![exposure_mask("Auto", brush_source(&strokes))]);
    let with = cpu_reference(&photo, &edit, Rounding::Nearest, 0);
    let without = cpu_reference(&photo, &PhotoEdit::default(), Rounding::Nearest, 0);
    let moved = with.iter().zip(&without).filter(|(a, b)| a != b).count();
    assert!(
        moved * 20 > with.len(),
        "the mask moves only {moved} pixels"
    );
    check_on("auto brush on a photo wider than the proxy", &photo, &edit);
}

/// A photo wider than a tile: at 4608 by 24 the proxy is 1024 by 5, each of
/// its pixels covers four and a half source pixels across, and the source is
/// drawn in three tiles. A reference read across a seam is the one the twin
/// reads, which has no tiles.
#[test]
fn an_auto_brush_on_a_photo_of_several_proxy_tiles_matches() {
    let photo = wide_photo(4608, 24);
    assert_eq!(Proxy::size_for((4608, 24)), (1024, 5));
    // The radius is a share of the longer side: 0.002 is 9 pixels.
    let strokes = [
        auto(
            stroke(&[[0.01, 0.5], [0.99, 0.5]], 0.002, 40.0, 100.0),
            60.0,
        ),
        auto(stroke(&[[0.3, 0.2], [0.7, 0.8]], 0.004, 70.0, 60.0), 20.0),
    ];
    let edit = masked(vec![exposure_mask("Auto", brush_source(&strokes))]);
    let with = cpu_reference(&photo, &edit, Rounding::Nearest, 0);
    let without = cpu_reference(&photo, &PhotoEdit::default(), Rounding::Nearest, 0);
    let moved = with.iter().zip(&without).filter(|(a, b)| a != b).count();
    assert!(
        moved * 20 > with.len(),
        "the mask moves only {moved} pixels"
    );
    check_on(
        "auto brush on a photo of several proxy tiles",
        &photo,
        &edit,
    );
}

/// An auto brush lies on the photo and reads its references from the proxy
/// of the whole source: under a crop window and under a zoomed window at 100
/// and at 200 percent of the fit it is where the full render has it, the dab
/// whose centre the window leaves outside included.
#[test]
fn an_auto_brush_under_a_crop_window_and_a_zoomed_window_matches_the_full_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let mut strokes = auto_strokes(60.0);
    strokes.extend(
        painted_strokes()
            .into_iter()
            .map(|stroke| auto(stroke, 40.0)),
    );
    // Its centre is left of every window below and it reaches into what is
    // seen. At a sensitivity of 0 the colours it reaches lie on the fall of
    // its gate, where a reference read from anywhere else would show.
    strokes.push(auto(stroke(&[[0.12, 0.45]], 0.3, 20.0, 80.0), 0.0));
    let mut brush = exposure_mask("Auto", brush_source(&strokes));
    brush.adjust.clarity = 30.0;
    let masks = vec![brush, exposure_mask("Radial", radial_source())];
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);

    let crop = CropRect {
        x: 0.25,
        y: 0.25,
        width: 0.5,
        height: 0.5,
    };
    let output = (32, 32);
    let masks_only = masked(masks.clone());
    let full = develop
        .render(&masks_only, crop, (SIZE, SIZE), output)
        .expect("a source is set");
    let full = readback.read(&gpu.device, &gpu.queue, full, output.0, output.1);
    let windowed = develop
        .render_crop(&masks_only, crop, output)
        .expect("a source is set");
    let windowed = readback.read(&gpu.device, &gpu.queue, windowed, output.0, output.1);
    let max = max_difference(&full, &windowed);
    println!("auto brush under a crop window against the full render: max difference {max}");
    assert!(max <= 1, "max difference {max}");

    let mut edit = everything_global();
    edit.masks = masks;
    let mut without_the_far_dab = edit.clone();
    let MaskSource::Brush(painted) = &mut without_the_far_dab.masks[0].components[0].source else {
        panic!("a brush");
    };
    painted.strokes.pop();
    for view in zoomed_views() {
        let full = full_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let zoomed = view_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let without = view_render_of(&mut develop, &gpu, &readback, &without_the_far_dab, &view);
        let max = max_difference(&full, &zoomed);
        println!(
            "auto brush under a zoomed window at {:?} against the full render: max difference {max}",
            view.full
        );
        assert!(max <= 1, "max difference {max} at {:?}", view.full);
        assert_ne!(
            zoomed, without,
            "the dab whose centre is outside shows inside the window"
        );
    }
    assert_eq!(develop.proxy_builds(), 1, "one source, one proxy");
}

/// A layer with no auto stroke reads no source pixel and outlives a new
/// content on the same frame; a layer with one does not. The proxy is built
/// once a source content, by no window and by no slider.
#[test]
fn an_auto_layer_is_stamped_again_by_a_new_source_content_and_a_plain_one_is_not() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    use gamut_color::video::{PlaneFormat, Transfer, VideoColour, YuvSpace};
    use gamut_media::VideoFrame;
    // A video is the source whose content changes under a frame that stays:
    // two frames of one size.
    let frame = |luma: u8| VideoFrame {
        pts_seconds: 0.0,
        format: PlaneFormat::Nv12,
        width: SIZE,
        height: SIZE,
        y: (0..SIZE * SIZE)
            .map(|i| luma.wrapping_add((i % SIZE) as u8))
            .collect(),
        uv: vec![128; (SIZE * SIZE / 2) as usize],
        y_stride: SIZE as usize,
        uv_stride: SIZE as usize,
    };
    let colour = VideoColour {
        space: YuvSpace::Bt709,
        transfer: Transfer::Sdr,
        full_range: false,
    };
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_video_frame(&frame(40), colour, 0);
    let plain = masked(vec![exposure_mask("Plain", painted_source())]);
    let gated = masked(vec![
        exposure_mask("Plain", painted_source()),
        exposure_mask("Auto", brush_source(&auto_strokes(50.0))),
    ]);
    let [view, _] = zoomed_views();
    let counts_after = |develop: &mut Develop, edit: &PhotoEdit, view: &ViewWindow| {
        develop.render_view(edit, view).expect("a source is set");
        (develop.brush_layer_builds(), develop.proxy_builds())
    };
    assert_eq!(
        counts_after(&mut develop, &plain, &view),
        (1, 0),
        "no auto stroke, no proxy"
    );
    develop.set_video_frame(&frame(90), colour, 0);
    assert_eq!(
        counts_after(&mut develop, &plain, &view),
        (1, 0),
        "a plain layer outlives a new content"
    );

    assert_eq!(
        counts_after(&mut develop, &gated, &view),
        (2, 1),
        "the auto layer and its proxy"
    );
    assert_eq!(
        counts_after(&mut develop, &gated, &view),
        (2, 1),
        "the same again"
    );
    let mut slid = gated.clone();
    slid.masks[1].adjust.exposure = -0.5;
    slid.exposure = 0.3;
    assert_eq!(
        counts_after(&mut develop, &slid, &view),
        (2, 1),
        "a slider draws neither"
    );

    develop.set_video_frame(&frame(140), colour, 0);
    assert_eq!(
        counts_after(&mut develop, &slid, &view),
        (3, 2),
        "a new content: the auto layer and the proxy, not the plain layer"
    );

    // A window replaced is a new frame: both layers again, the proxy not.
    let elsewhere = ViewWindow {
        window: (2, 2, 18, 20),
        visible: (4, 4, 10, 12),
        ..view
    };
    assert_eq!(
        counts_after(&mut develop, &slid, &elsewhere),
        (5, 2),
        "a window replaced builds no proxy"
    );
}

/// The same for a stroke that reads the source and scales with the pen: an
/// auto stroke with pressure on the size and on the flow, painted in ten
/// appended pieces, and an auto erase stroke in five after it, are the
/// strokes drawn whole. With the pressure on the size the dabs of a path are
/// still the first dabs of the path continued, and a gate is a matter of the
/// dab and the pixel alone, so new dabs go onto the layer that is there and
/// nothing that was stamped is stamped again.
#[test]
fn an_auto_pressure_stroke_painted_in_appended_pieces_equals_the_stroke_drawn_whole() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let readback = Readback::new(&gpu.device);
    // Down beside the grey columns and then across the colours, so the gate
    // closes on part of what the dabs reach.
    let path: Vec<[f32; 2]> = (0..=40)
        .map(|i| {
            let t = i as f32 / 40.0;
            [0.14 + t * 0.7, 0.5 + (t * 7.0).sin() * 0.25]
        })
        .collect();
    let pressure: Vec<f32> = (0..=40)
        .map(|i| 0.15 + 0.85 * (i as f32 / 40.0 * 5.0).sin().abs())
        .collect();
    let eraser: Vec<[f32; 2]> = (0..=20)
        .map(|i| {
            [
                0.5 + (i as f32 / 20.0 - 0.5) * 0.1,
                0.1 + i as f32 / 20.0 * 0.8,
            ]
        })
        .collect();
    let edit_at = |painted: usize, erased: usize| -> PhotoEdit {
        let mut strokes = painted_strokes();
        strokes.push(Stroke {
            pressure: pressure[..painted].to_vec(),
            pressure_size: true,
            pressure_flow: true,
            ..auto(stroke(&path[..painted], 0.09, 45.0, 100.0), 60.0)
        });
        if erased > 0 {
            strokes.push(Stroke {
                erase: true,
                ..auto(stroke(&eraser[..erased], 0.06, 30.0, 100.0), 30.0)
            });
        }
        let mut edit = everything_global();
        let mut growing = exposure_mask("Growing", brush_source(&strokes));
        growing.adjust.clarity = 25.0;
        let resting = exposure_mask(
            "Resting",
            brush_source(&[stroke(&[[0.2, 0.8], [0.8, 0.85]], 0.06, 50.0, 80.0)]),
        );
        edit.masks = vec![growing, resting, exposure_mask("Radial", radial_source())];
        edit
    };
    let [_, view] = zoomed_views();
    for zoomed in [false, true] {
        let render = |develop: &mut Develop, edit: &PhotoEdit| -> Vec<u8> {
            if zoomed {
                view_render_of(develop, &gpu, &readback, edit, &view)
            } else {
                let drawn = develop
                    .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
                    .expect("a source is set");
                readback.read(&gpu.device, &gpu.queue, drawn, SIZE, SIZE)
            }
        };
        let mut develop = Develop::new(&gpu.device, &gpu.queue);
        develop.set_source(&photo);
        let mut pieces = Vec::new();
        for piece in 1..=10 {
            pieces.push(render(&mut develop, &edit_at(piece * 4 + 1, 0)));
        }
        for piece in 1..=5 {
            pieces.push(render(&mut develop, &edit_at(41, piece * 4 + 1)));
        }
        assert_ne!(pieces[0], pieces[9], "the stroke grew on the picture");
        assert_ne!(pieces[9], pieces[14], "and the eraser took some of it away");
        assert_eq!(
            develop.brush_layer_builds(),
            2,
            "each layer stamped whole once"
        );
        assert_eq!(
            develop.brush_layer_appends(),
            14,
            "and the growing one added to: new dabs only"
        );
        assert_eq!(develop.proxy_builds(), 1, "one proxy for all of it");
        let (alphas, develops) = develop.brush_patches();
        if zoomed {
            assert!(alphas > 0 && alphas <= 14 && develops > 0 && develops <= alphas);
        } else {
            assert_eq!((alphas, develops), (14, 14), "over the new dabs only");
        }
        assert_eq!(develop.mask_alpha_builds(), 3 + 14);

        let mut fresh = Develop::new(&gpu.device, &gpu.queue);
        fresh.set_source(&photo);
        let whole = render(&mut fresh, &edit_at(41, 21));
        assert_eq!(fresh.brush_layer_appends(), 0);
        let max = max_difference(&pieces[14], &whole);
        println!(
            "an auto pressure stroke in appended pieces against the stroke whole, zoomed {zoomed}: max difference {max}"
        );
        assert!(max <= 1, "max difference {max}, zoomed {zoomed}");

        // The gate and the pressure are in what was compared: without either
        // the picture is another.
        let mut plain = edit_at(41, 21);
        let MaskSource::Brush(painted) = &mut plain.masks[0].components[0].source else {
            panic!("a brush");
        };
        for held in &mut painted.strokes {
            *held = SharedStroke::new(&Stroke {
                auto: false,
                pressure: Vec::new(),
                ..(**held).clone()
            });
        }
        assert_ne!(render(&mut fresh, &plain), whole);

        // An undo takes the erase stroke away: the layer starts again and
        // the picture is the one from before it.
        let undone = render(&mut develop, &edit_at(41, 0));
        assert_eq!(develop.brush_layer_builds(), 3);
        assert!(max_difference(&undone, &pieces[9]) <= 1);
    }
}
