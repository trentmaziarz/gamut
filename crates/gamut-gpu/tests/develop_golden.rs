//! Golden tests: every GPU operator against its CPU twin in gamut-color,
//! through the same input and output transforms, on a 64 by 64 synthetic
//! photo that covers hue, brightness and neutral ramps. The tests skip,
//! and say so, when the machine has no adapter.

use std::sync::Mutex;

use gamut_color::SourceSpace;
use gamut_color::basic;
use gamut_color::brush::Proxy;
use gamut_color::mask::{self as mask_twin, Geometry, Image};
use gamut_color::refine::Plan;
use gamut_color::{dehaze, local, matrices, transfer};
use gamut_core::brush::{Brush, SharedStroke, Stroke};
use gamut_core::look::{Curve, HslRange, Wheel};
use gamut_core::mask::{
    ColourRange, Component, Edge, LinearGradient, LuminanceRange, MaskOp, MaskSource,
    RadialGradient, Refine,
};
use gamut_core::{Adjustments, CropRect, ExportPreset, Mask, PhotoEdit};
use gamut_gpu::{Develop, EdgePasses, Headless, Readback, ViewWindow};
use gamut_media::{Photo, fixtures, open_photo};
use half::f16;

const SIZE: u32 = 64;

/// The largest difference allowed in any channel of any pixel, in 8-bit codes.
const MAX_DIFFERENCE: i32 = 2;

/// The mean absolute difference allowed over all channels, in 8-bit codes.
const MEAN_DIFFERENCE: f64 = 0.5;

/// The flow that puts the most alphas of the saturated row strokes beside a
/// half code, and how many pixels that is at least.
const HALF_CODE_FLOW: f32 = 54.0;
const HALF_CODE_PIXELS: usize = 400;

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
    let gpu_pixels = gpu_render(&gpu, photo, edit);
    assert_matches_the_twin(name, photo, edit, &gpu_pixels);
}

/// Holds `gpu_pixels`, a render of `edit` on `photo`, to the twin within the
/// golden tolerances.
fn assert_matches_the_twin(name: &str, photo: &Photo, edit: &PhotoEdit, gpu_pixels: &[[u8; 3]]) {
    let references: Vec<Vec<[u8; 3]>> = transmission_steps(edit)
        .iter()
        .flat_map(|step| {
            [Rounding::Nearest, Rounding::TowardZero]
                .map(|rounding| cpu_reference(photo, edit, rounding, *step))
        })
        .collect();
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

/// A pan that leaves its window onto one built ahead: in each of the four
/// directions the window ahead is built in a submit of its own while the
/// current one still renders, the pan steps inside the current window until
/// what is seen leaves it, and the render there swaps the frame built ahead
/// in. It replaces no frame, and its output equals a fresh render of that
/// window on a graph of its own byte for byte, and the full render at the
/// golden tolerances.
#[test]
fn a_window_built_ahead_and_swapped_in_equals_a_fresh_render_of_it() {
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
    const FULL: (u32, u32) = (SIZE, SIZE);
    const PAD: (u32, u32) = (8, 8);
    const GRID: u32 = 4;
    const STEP: i64 = 4;
    let start = (20, 22, 24, 20);
    let window = gamut_gpu::develop::padded_window(FULL, start, PAD, GRID);
    let at = |rect: (u32, u32, u32, u32), (dx, dy): (i64, i64)| {
        (
            (i64::from(rect.0) + dx) as u32,
            (i64::from(rect.1) + dy) as u32,
            rect.2,
            rect.3,
        )
    };
    for step in [(STEP, 0), (0, STEP), (-STEP, 0), (0, -STEP)] {
        let mut develop = Develop::new(&gpu.device, &gpu.queue);
        develop.set_source(&photo);
        let first = ViewWindow {
            full: FULL,
            window,
            visible: start,
        };
        let before = view_render_of(&mut develop, &gpu, &readback, &edit, &first);
        assert_eq!(develop.window_replaces(), 1, "the first render replaces");
        let previous = at(start, (-step.0, -step.1));
        let ahead = gamut_gpu::develop::window_ahead(FULL, window, previous, start, PAD, GRID)
            .expect("a pan that moves leaves the window");
        assert!(develop.build_ahead(&edit, FULL, ahead), "built ahead");
        assert_eq!(develop.window_ahead(), Some((FULL, ahead)));
        assert!(
            !develop.build_ahead(&edit, FULL, ahead),
            "kept, not built again"
        );
        // The window built ahead leaves the current one as it was.
        let still = view_render_of(&mut develop, &gpu, &readback, &edit, &first);
        assert_eq!(still, before, "the current window after a build ahead");
        let mut visible = start;
        while gamut_gpu::develop::holds(window, visible) {
            let panned = ViewWindow { visible, ..first };
            view_render_of(&mut develop, &gpu, &readback, &edit, &panned);
            visible = at(visible, step);
        }
        assert!(
            gamut_gpu::develop::holds(ahead, visible),
            "{ahead:?} holds {visible:?}"
        );
        let (replaces, swaps) = (develop.window_replaces(), develop.window_swaps());
        let reached = ViewWindow {
            full: FULL,
            window: ahead,
            visible,
        };
        let swapped = view_render_of(&mut develop, &gpu, &readback, &edit, &reached);
        assert_eq!(
            develop.window_replaces(),
            replaces,
            "the swap replaces no frame"
        );
        assert_eq!(
            develop.window_swaps(),
            swaps + 1,
            "the frame built ahead is taken"
        );
        assert_eq!(
            develop.window_ahead(),
            None,
            "the frame built ahead is in use"
        );
        let mut fresh = Develop::new(&gpu.device, &gpu.queue);
        fresh.set_source(&photo);
        let alone = view_render_of(&mut fresh, &gpu, &readback, &edit, &reached);
        assert_eq!(fresh.window_swaps(), 0);
        let differing = swapped.iter().zip(&alone).filter(|(a, b)| a != b).count();
        println!("pan {step:?}: window {window:?}, ahead {ahead:?}, swapped in at {visible:?}");
        println!(
            "pan {step:?}: bytes that differ from a fresh render of it: {differing} of {}",
            alone.len()
        );
        assert_eq!(
            swapped, alone,
            "the swapped window at {visible:?} for the pan {step:?}"
        );
        // The full render: a window of the source is not the whole of it,
        // so it is held at the golden tolerances.
        let full = full_render_of(&mut fresh, &gpu, &readback, &edit, &reached);
        let max = max_difference(&full, &swapped);
        let mean = full
            .iter()
            .zip(&swapped)
            .map(|(a, b)| f64::from(a.abs_diff(*b)))
            .sum::<f64>()
            / full.len() as f64;
        println!("pan {step:?}: against the full render, max difference {max}, mean {mean:.4}");
        assert!(
            max <= MAX_DIFFERENCE && mean <= MEAN_DIFFERENCE,
            "max difference {max}, mean {mean:.4} for the pan {step:?}"
        );
    }
}

/// A pan across the edge of three windows in a row, driven as the viewer
/// drives it: after each render the window ahead of the pan is built, in
/// slices, while the pan steps inside the current window, and the render
/// where what is seen leaves it swaps that frame in. The frame a swap takes
/// out of use keeps its textures, and the next window built ahead is drawn
/// into them, so the pan makes two frames of textures in all: the first
/// window's and the first window built ahead. At every crossing the swapped
/// window equals a fresh render of it on a graph of its own, byte for byte,
/// with every mask product drawn again on the reused frame: the four masks
/// and a brush mask, then a refined and an edged mask.
#[test]
fn a_pan_across_three_windows_draws_each_ahead_into_the_frame_a_swap_left() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    use gamut_gpu::develop::{holds, padded_window, pan_exit, window_ahead};
    // With Refine edges on, the first frames of the pan reach the left edge
    // of the picture and every frame reaches its top and bottom edges: a
    // frame there is moved inside the picture at the size of every other
    // frame of the zoom, so it is drawn into as well.
    let (width, height) = (1100, 600);
    let photo = blocky_photo(width, height);
    let readback = Readback::new(&gpu.device);
    const PAD: (u32, u32) = (40, 30);
    const GRID: u32 = 8;
    const STEP: u32 = 8;
    let full = (width, height);
    let mut five = everything_global();
    five.masks = vec![
        exposure_mask("Linear", linear_source()),
        exposure_mask("Radial", radial_source()),
        exposure_mask("Luminance", luminance_source()),
        exposure_mask("Colour", colour_source()),
        exposure_mask("Brush", painted_source()),
    ];
    let radial = MaskSource::Radial(RadialGradient {
        centre: [0.393, 0.25],
        radius: [0.049, 0.0327],
        rotation: 0.0,
        feather: 12.0,
    });
    let brush = brush_source(&[stroke(&[[0.425, 0.05], [0.425, 0.5]], 0.016, 20.0, 100.0)]);
    let mut edged = everything_global();
    edged.masks = vec![
        edged_at(
            refined_at(exposure_mask("Radial", radial), 100.0, 0.03, 50.0),
            0.03,
            0.02,
            40.0,
        ),
        edged_at(exposure_mask("Brush", brush), -0.03, 0.02, 0.0),
    ];
    for (name, edit, texels) in [
        ("four masks and a brush mask", &five, 400_000),
        ("a refined and an edged mask", &edged, 200_000),
    ] {
        let mut develop = Develop::new(&gpu.device, &gpu.queue);
        develop.set_source(&photo);
        develop.set_ahead_slice_texels(texels);
        let start = (232, 213, 180, 140);
        let first = ViewWindow {
            full,
            window: padded_window(full, start, PAD, GRID),
            visible: start,
        };
        view_render_of(&mut develop, &gpu, &readback, edit, &first);
        assert_eq!(develop.frame_makes(), 1, "{name}: the first window");
        println!(
            "{name}: first window {:?}, frame {:?}",
            first.window,
            develop.frame_textures(false).first().map(|t| t.2)
        );
        let (mut window, mut previous) = (first.window, start);
        let mut visible = start;
        let mut crossings = 0;
        let mut steps = 0;
        while crossings < 3 {
            steps += 1;
            assert!(steps < 1000, "{name}: the pan does not cross three times");
            visible.0 += STEP;
            let crossed = !holds(window, visible);
            let asked = if crossed {
                develop
                    .window_ahead()
                    .filter(|&(held, ahead)| held == full && holds(ahead, visible))
                    .map(|(_, ahead)| ahead)
                    .expect("the window built ahead holds what is seen")
            } else {
                window
            };
            let view = ViewWindow {
                full,
                window: asked,
                visible,
            };
            let (replaces, swaps) = (develop.window_replaces(), develop.window_swaps());
            let shown = view_render_of(&mut develop, &gpu, &readback, edit, &view);
            if crossed {
                crossings += 1;
                assert_eq!(
                    develop.window_replaces(),
                    replaces,
                    "{name}: crossing {crossings} replaces no frame"
                );
                assert_eq!(
                    develop.window_swaps(),
                    swaps + 1,
                    "{name}: crossing {crossings} takes the frame built ahead"
                );
                let mut fresh = Develop::new(&gpu.device, &gpu.queue);
                fresh.set_source(&photo);
                let alone = view_render_of(&mut fresh, &gpu, &readback, edit, &view);
                let differing = shown.iter().zip(&alone).filter(|(a, b)| a != b).count();
                let frame = develop.frame_textures(false).first().map(|t| t.2);
                println!(
                    "{name}: crossing {crossings} into {asked:?} at {visible:?}: frame {frame:?}, frame makes {}, bytes that differ from a fresh render of it: {differing} of {}",
                    develop.frame_makes(),
                    alone.len()
                );
                assert_eq!(
                    shown, alone,
                    "{name}: the window swapped in at crossing {crossings}"
                );
            }
            // The build ahead as the viewer drives it: the window held ahead
            // while it holds where the pan leaves this one, else a new one,
            // built to its end before the next step.
            let exit = pan_exit(full, asked, previous, visible);
            let held = develop
                .window_ahead()
                .filter(|&(held, _)| held == full)
                .map(|(_, ahead)| ahead);
            let wanted = exit.and_then(|exit| match held {
                Some(held) if holds(held, exit) => Some(held),
                _ => window_ahead(full, asked, previous, visible, PAD, GRID),
            });
            if let Some(ahead) = wanted {
                let mut calls = 0;
                while !develop.build_ahead(edit, full, ahead) && develop.ahead_building() {
                    calls += 1;
                    assert!(calls < 100_000, "{name}: the build ahead does not end");
                }
            }
            (window, previous) = (asked, visible);
        }
        println!(
            "{name}: {crossings} crossings in {steps} steps, {} swaps, {} replaces, {} frame makes",
            develop.window_swaps(),
            develop.window_replaces(),
            develop.frame_makes()
        );
        assert_eq!(
            develop.window_replaces(),
            1,
            "{name}: only the first window"
        );
        assert_eq!(
            develop.frame_makes(),
            2,
            "{name}: the first window and the first window built ahead make frames, and each later one takes the textures a swap left"
        );
    }
}

/// A window built ahead in slices, each in a submit of its own, holds what
/// the same window built in one submit holds, byte for byte in all seven
/// textures of its frame. The slices are one row of a pass each, then 3001
/// texels each, which cuts passes and runs strips of two passes in one
/// slice. Between two slices the current frame renders another window or
/// its own, which writes the input uniform with another window, so a slice
/// that read a render's uniform would differ here. Swapped in, the frame
/// built in slices renders what a fresh render of its window renders, and
/// its textures are that render's, byte for byte.
#[test]
fn a_window_built_ahead_in_slices_equals_one_built_in_one_submit() {
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
    const FULL: (u32, u32) = (SIZE, SIZE);
    const PAD: (u32, u32) = (8, 8);
    const GRID: u32 = 4;
    const STEP: (i64, i64) = (4, 0);
    let start = (20, 22, 24, 20);
    let window = gamut_gpu::develop::padded_window(FULL, start, PAD, GRID);
    let at = |rect: (u32, u32, u32, u32), (dx, dy): (i64, i64)| {
        (
            (i64::from(rect.0) + dx) as u32,
            (i64::from(rect.1) + dy) as u32,
            rect.2,
            rect.3,
        )
    };
    let first = ViewWindow {
        full: FULL,
        window,
        visible: start,
    };
    let elsewhere = ViewWindow {
        full: FULL,
        window: (0, 0, 32, 32),
        visible: (4, 4, 20, 16),
    };
    let previous = at(start, (-STEP.0, -STEP.1));
    let ahead = gamut_gpu::develop::window_ahead(FULL, window, previous, start, PAD, GRID)
        .expect("a pan that moves leaves the window");
    let read = |develop: &Develop, ahead: bool| -> Vec<(&'static str, Vec<u8>)> {
        develop
            .frame_textures(ahead)
            .into_iter()
            .map(|(name, view, (w, h))| (name, readback.read(&gpu.device, &gpu.queue, view, w, h)))
            .collect()
    };

    // The window built in one submit.
    let mut whole = Develop::new(&gpu.device, &gpu.queue);
    whole.set_source(&photo);
    view_render_of(&mut whole, &gpu, &readback, &edit, &first);
    whole.set_ahead_slice_texels(u64::MAX);
    assert!(whole.build_ahead(&edit, FULL, ahead), "one call builds it");
    assert_eq!(whole.ahead_slices(), 1, "in one submit");
    assert!(!whole.ahead_building());
    let one = read(&whole, true);
    assert_eq!(one.len(), 7, "the seven textures of a frame");

    // The pan's exit from the window, where the swap happens.
    let mut visible = start;
    while gamut_gpu::develop::holds(window, visible) {
        visible = at(visible, STEP);
    }
    let reached = ViewWindow {
        full: FULL,
        window: ahead,
        visible,
    };
    let mut fresh = Develop::new(&gpu.device, &gpu.queue);
    fresh.set_source(&photo);
    let alone = view_render_of(&mut fresh, &gpu, &readback, &edit, &reached);
    assert_eq!(fresh.window_swaps(), 0);
    let rendered = read(&fresh, false);

    for texels in [1, 3001] {
        let mut sliced = Develop::new(&gpu.device, &gpu.queue);
        sliced.set_source(&photo);
        view_render_of(&mut sliced, &gpu, &readback, &edit, &first);
        sliced.set_ahead_slice_texels(texels);
        let mut calls = 0u64;
        loop {
            calls += 1;
            let done = sliced.build_ahead(&edit, FULL, ahead);
            assert_eq!(done, !sliced.ahead_building(), "call {calls}");
            assert_eq!(sliced.window_ahead(), Some((FULL, ahead)));
            if done {
                break;
            }
            let view = if calls % 2 == 1 { &elsewhere } else { &first };
            view_render_of(&mut sliced, &gpu, &readback, &edit, view);
        }
        assert_eq!(sliced.ahead_slices(), calls, "one submit a call");
        assert!(calls > 9, "the nine head passes cut over {calls} submits");
        assert!(
            !sliced.build_ahead(&edit, FULL, ahead),
            "complete, nothing left to build"
        );
        assert_eq!(sliced.ahead_slices(), calls, "no slice once complete");
        let pieces = read(&sliced, true);
        assert_eq!(pieces.len(), one.len());
        for ((name, a), (_, b)) in pieces.iter().zip(&one) {
            let differing = a.iter().zip(b).filter(|(a, b)| a != b).count();
            println!(
                "slices of {texels} texels, {calls} submits: {name}: bytes that differ from one submit: {differing} of {}",
                b.len()
            );
            assert_eq!(a, b, "{name} in slices of {texels} texels");
        }

        // Swapped in where the pan leaves the current window.
        view_render_of(&mut sliced, &gpu, &readback, &edit, &first);
        let (replaces, swaps) = (sliced.window_replaces(), sliced.window_swaps());
        let swapped = view_render_of(&mut sliced, &gpu, &readback, &edit, &reached);
        assert_eq!(sliced.window_replaces(), replaces, "the swap replaces none");
        assert_eq!(sliced.window_swaps(), swaps + 1, "the frame ahead is taken");
        let differing = swapped.iter().zip(&alone).filter(|(a, b)| a != b).count();
        println!(
            "slices of {texels} texels: swapped in at {visible:?}, bytes that differ from a fresh render of it: {differing} of {}",
            alone.len()
        );
        assert_eq!(swapped, alone, "the swapped window, slices of {texels}");
        let now = read(&sliced, false);
        assert_eq!(now.len(), rendered.len());
        for ((name, a), (_, b)) in now.iter().zip(&rendered) {
            let differing = a.iter().zip(b).filter(|(a, b)| a != b).count();
            println!(
                "slices of {texels} texels: swapped {name}: bytes that differ from a fresh render's: {differing} of {}",
                b.len()
            );
            assert_eq!(a, b, "swapped {name} in slices of {texels} texels");
        }
    }
}

/// What the graph has drawn of the mask products, counter by counter: the
/// alphas (whole and patched), the brush layers (stamped whole and appended
/// to), the proxies, the refined alphas (whole and patched), the moments of
/// the source (whole tiles and strips), and the three edge stages (whole and
/// patched).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ProductBuilds {
    mask_alpha_builds: u64,
    alpha_patches: u64,
    brush_layer_builds: u64,
    brush_layer_appends: u64,
    proxy_builds: u64,
    refine_builds: u64,
    refine_patches: u64,
    refine_source_builds: u64,
    refine_source_strips: u64,
    edge_builds: [u64; 3],
    edge_patches: [u64; 3],
}

impl ProductBuilds {
    fn of(develop: &Develop) -> Self {
        let (refine_builds, refine_patches) = develop.refine_builds();
        let (edge_builds, edge_patches) = develop.edge_builds();
        ProductBuilds {
            mask_alpha_builds: develop.mask_alpha_builds(),
            alpha_patches: develop.brush_patches().0,
            brush_layer_builds: develop.brush_layer_builds(),
            brush_layer_appends: develop.brush_layer_appends(),
            proxy_builds: develop.proxy_builds(),
            refine_builds,
            refine_patches,
            refine_source_builds: develop.refine_source_builds(),
            refine_source_strips: develop.refine_source_strips(),
            edge_builds,
            edge_patches,
        }
    }

    /// What was drawn since `before`.
    fn since(self, before: ProductBuilds) -> ProductBuilds {
        let three = |a: [u64; 3], b: [u64; 3]| [0, 1, 2].map(|i| a[i] - b[i]);
        ProductBuilds {
            mask_alpha_builds: self.mask_alpha_builds - before.mask_alpha_builds,
            alpha_patches: self.alpha_patches - before.alpha_patches,
            brush_layer_builds: self.brush_layer_builds - before.brush_layer_builds,
            brush_layer_appends: self.brush_layer_appends - before.brush_layer_appends,
            proxy_builds: self.proxy_builds - before.proxy_builds,
            refine_builds: self.refine_builds - before.refine_builds,
            refine_patches: self.refine_patches - before.refine_patches,
            refine_source_builds: self.refine_source_builds - before.refine_source_builds,
            refine_source_strips: self.refine_source_strips - before.refine_source_strips,
            edge_builds: three(self.edge_builds, before.edge_builds),
            edge_patches: three(self.edge_patches, before.edge_patches),
        }
    }
}

/// A pan onto a window built ahead with the products of its masks: the
/// picture at `full`, the part seen at the start, the pad and the grid of
/// the padded window, the step of the pan, and the texels of a slice.
struct AheadCase {
    name: &'static str,
    full: (u32, u32),
    start: (u32, u32, u32, u32),
    pad: (u32, u32),
    grid: u32,
    step: (i64, i64),
    texels: u64,
}

/// Renders the window of `case.start`, builds the window a pan by
/// `case.step` leaves it for in slices of `case.texels`, rendering the
/// current window between two slices, pans inside the current window until
/// what is seen leaves it, and swaps the frame built ahead in. The slices
/// draw every alpha of `edit` and no proxy; the swap draws no mask product
/// again; its output equals a fresh render of that window byte for byte and
/// the full render at the golden tolerances; and a slider step of the global
/// edit and of the first mask after it draws no mask product again either,
/// and equals a fresh render of the stepped edit byte for byte. Returns
/// what the slices drew.
fn assert_products_built_ahead(
    gpu: &Headless,
    photo: &Photo,
    edit: &PhotoEdit,
    case: &AheadCase,
) -> ProductBuilds {
    let name = case.name;
    let readback = Readback::new(&gpu.device);
    let at = |rect: (u32, u32, u32, u32), (dx, dy): (i64, i64)| {
        (
            (i64::from(rect.0) + dx) as u32,
            (i64::from(rect.1) + dy) as u32,
            rect.2,
            rect.3,
        )
    };
    let window = gamut_gpu::develop::padded_window(case.full, case.start, case.pad, case.grid);
    let first = ViewWindow {
        full: case.full,
        window,
        visible: case.start,
    };
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(photo);
    view_render_of(&mut develop, gpu, &readback, edit, &first);
    let previous = at(case.start, (-case.step.0, -case.step.1));
    let ahead = gamut_gpu::develop::window_ahead(
        case.full, window, previous, case.start, case.pad, case.grid,
    )
    .expect("a pan that moves leaves the window");

    // Built ahead in slices, the current window rendered between them.
    develop.set_ahead_slice_texels(case.texels);
    let before = ProductBuilds::of(&develop);
    let slices = develop.ahead_slices();
    let mut calls = 0u64;
    loop {
        calls += 1;
        let done = develop.build_ahead(edit, case.full, ahead);
        assert_eq!(done, !develop.ahead_building(), "{name}: call {calls}");
        if done {
            break;
        }
        assert!(calls < 100_000, "{name}: the build ahead does not end");
        view_render_of(&mut develop, gpu, &readback, edit, &first);
    }
    assert_eq!(
        develop.ahead_slices() - slices,
        calls,
        "{name}: one submit a call"
    );
    let built = ProductBuilds::of(&develop).since(before);
    println!(
        "{name}: built ahead in {calls} slices of {} texels: {built:?}",
        case.texels
    );
    assert_eq!(
        built.mask_alpha_builds,
        edit.masks.len() as u64,
        "{name}: the slices draw every alpha once"
    );
    assert_eq!(built.proxy_builds, 0, "{name}: no slice builds the proxy");
    assert!(
        !develop.build_ahead(edit, case.full, ahead),
        "{name}: complete, nothing left to build"
    );
    assert_eq!(
        develop.ahead_slices() - slices,
        calls,
        "{name}: no slice once complete"
    );

    // The pan inside the current window, then onto the window built ahead.
    let mut visible = case.start;
    while gamut_gpu::develop::holds(window, visible) {
        let panned = ViewWindow { visible, ..first };
        view_render_of(&mut develop, gpu, &readback, edit, &panned);
        visible = at(visible, case.step);
    }
    assert!(
        gamut_gpu::develop::holds(ahead, visible),
        "{name}: {ahead:?} holds {visible:?}"
    );
    let reached = ViewWindow {
        full: case.full,
        window: ahead,
        visible,
    };
    let (replaces, swaps) = (develop.window_replaces(), develop.window_swaps());
    let before = ProductBuilds::of(&develop);
    let swapped = view_render_of(&mut develop, gpu, &readback, edit, &reached);
    let at_swap = ProductBuilds::of(&develop).since(before);
    println!("{name}: the swap at {visible:?} drew {at_swap:?}");
    assert_eq!(
        develop.window_replaces(),
        replaces,
        "{name}: the swap replaces none"
    );
    assert_eq!(
        develop.window_swaps(),
        swaps + 1,
        "{name}: the frame ahead is taken"
    );
    assert_eq!(
        at_swap,
        ProductBuilds::default(),
        "{name}: the swap draws no mask product again"
    );

    // A fresh render of the window reached, and the full render.
    let mut fresh = Develop::new(&gpu.device, &gpu.queue);
    fresh.set_source(photo);
    let alone = view_render_of(&mut fresh, gpu, &readback, edit, &reached);
    assert_eq!(fresh.window_swaps(), 0);
    let differing = swapped.iter().zip(&alone).filter(|(a, b)| a != b).count();
    println!(
        "{name}: bytes of the swap that differ from a fresh render of the window: {differing} of {}",
        alone.len()
    );
    assert_eq!(swapped, alone, "{name}: the swapped window");
    let full = full_render_of(&mut fresh, gpu, &readback, edit, &reached);
    let max = max_difference(&full, &swapped);
    let mean = full
        .iter()
        .zip(&swapped)
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / full.len() as f64;
    println!("{name}: the swap against the full render: max difference {max}, mean {mean:.4}");
    assert!(
        max <= MAX_DIFFERENCE && mean <= MEAN_DIFFERENCE,
        "{name}: max difference {max}, mean {mean:.4}"
    );

    // A slider step after the swap: the global exposure and the first
    // mask's, neither of them part of a mask's shape.
    let mut stepped = edit.clone();
    stepped.adjust.exposure += 0.1;
    stepped.masks[0].adjust.exposure += 0.1;
    let before = ProductBuilds::of(&develop);
    let slid = view_render_of(&mut develop, gpu, &readback, &stepped, &reached);
    let at_step = ProductBuilds::of(&develop).since(before);
    println!("{name}: a slider step after the swap drew {at_step:?}");
    assert_eq!(
        at_step,
        ProductBuilds::default(),
        "{name}: a slider step after the swap draws no mask product again"
    );
    let mut fresh = Develop::new(&gpu.device, &gpu.queue);
    fresh.set_source(photo);
    let alone = view_render_of(&mut fresh, gpu, &readback, &stepped, &reached);
    let differing = slid.iter().zip(&alone).filter(|(a, b)| a != b).count();
    println!(
        "{name}: bytes of the slider step that differ from a fresh render of it: {differing} of {}",
        alone.len()
    );
    assert_eq!(slid, alone, "{name}: the slider step after the swap");
    assert_ne!(slid, swapped, "{name}: the slider step shows");
    built
}

/// The four full masks and a brush mask, every operator on: their alphas and
/// the brush layer are built ahead with the frame in slices of one row or
/// one product, and the swap and a slider step after it draw none of them
/// again. The brush is painted, then painted with auto strokes, whose layer
/// reads the proxy of the source the current window's render built.
#[test]
fn the_four_masks_and_a_brush_mask_built_ahead_equal_a_fresh_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let case = |name| AheadCase {
        name,
        full: (SIZE, SIZE),
        start: (20, 22, 24, 20),
        pad: (8, 8),
        grid: 4,
        step: (4, 0),
        texels: 1,
    };
    for (name, brush) in [
        ("four masks and a brush mask", painted_source()),
        (
            "four masks and an auto brush mask",
            brush_source(&auto_strokes(60.0)),
        ),
    ] {
        let mut edit = everything_global();
        edit.masks = vec![
            exposure_mask("Linear", linear_source()),
            exposure_mask("Radial", radial_source()),
            exposure_mask("Luminance", luminance_source()),
            exposure_mask("Colour", colour_source()),
            exposure_mask("Brush", brush),
        ];
        let built = assert_products_built_ahead(&gpu, &photo, &edit, &case(name));
        assert_eq!(
            built.brush_layer_builds, 1,
            "{name}: the layer is stamped ahead"
        );
    }
}

/// A refined radial gradient and a refined brush, every operator on: the
/// refine scratch, the moments of the source and each refined alpha are
/// built ahead with the frame, and the swap and a slider step after it draw
/// none of them again. The masks of the refined window test.
#[test]
fn refined_masks_built_ahead_equal_a_fresh_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let (width, height) = (1100, 600);
    let photo = blocky_photo(width, height);
    let radial = MaskSource::Radial(RadialGradient {
        centre: [0.393, 0.25],
        radius: [0.049, 0.0327],
        rotation: 0.0,
        feather: 12.0,
    });
    let brush = brush_source(&[stroke(&[[0.425, 0.05], [0.425, 0.5]], 0.016, 20.0, 100.0)]);
    let mut edit = everything_global();
    edit.masks = vec![
        refined_at(exposure_mask("Radial", radial), 100.0, 0.03, 50.0),
        refined_at(exposure_mask("Brush", brush), 90.0, 0.02, 70.0),
    ];
    // The second budget cuts each pass of Refine edges into strips of rows
    // over several slices.
    for (name, texels) in [
        ("refined masks", 2_000_000),
        ("refined masks in strips", 20_000),
    ] {
        let built = assert_products_built_ahead(
            &gpu,
            &photo,
            &edit,
            &AheadCase {
                name,
                full: (width, height),
                start: (401, 113, 180, 140),
                pad: (40, 30),
                grid: 8,
                step: (8, 0),
                texels,
            },
        );
        assert_eq!(
            built.refine_builds, 2,
            "{name}: each refined alpha is built ahead"
        );
        assert!(
            built.refine_source_builds > 0,
            "{name}: the moments of the source are taken ahead"
        );
    }
}

/// A refined radial gradient under Shift edge, Feather and Contrast and a
/// brush under Shift edge and Feather, every operator on: the shifted
/// alphas, the Feather cells, the finished alphas and the frame's edge
/// scratch are built ahead with the frame, and the swap and a slider step
/// after it draw none of them again. The masks of the edged window test.
#[test]
fn edged_masks_built_ahead_equal_a_fresh_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let (width, height) = (1100, 600);
    let photo = blocky_photo(width, height);
    let radial = MaskSource::Radial(RadialGradient {
        centre: [0.393, 0.25],
        radius: [0.049, 0.0327],
        rotation: 0.0,
        feather: 12.0,
    });
    let brush = brush_source(&[stroke(&[[0.425, 0.05], [0.425, 0.5]], 0.016, 20.0, 100.0)]);
    let mut edit = everything_global();
    edit.masks = vec![
        edged_at(
            refined_at(exposure_mask("Radial", radial), 100.0, 0.03, 50.0),
            0.03,
            0.02,
            40.0,
        ),
        edged_at(exposure_mask("Brush", brush), -0.03, 0.02, 0.0),
    ];
    // The second budget cuts each edge pass and each pass of Refine edges
    // into strips of rows over several slices.
    for (name, texels) in [
        ("edged masks", 2_000_000),
        ("edged masks in strips", 20_000),
    ] {
        let built = assert_products_built_ahead(
            &gpu,
            &photo,
            &edit,
            &AheadCase {
                name,
                full: (width, height),
                start: (401, 113, 180, 140),
                pad: (40, 30),
                grid: 8,
                step: (0, 8),
                texels,
            },
        );
        assert_eq!(
            built.edge_builds,
            [2, 2, 2],
            "{name}: each shifted alpha, Feather's cells and finished alpha are built ahead"
        );
    }
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
    check_overlay_at(idle, (SIZE * 55 / 100, SIZE * 45 / 100));
}

/// [`check_overlay`] of a mask that covers pixel `inside` (column, row) but
/// not the first pixel of the photo.
fn check_overlay_at(idle: Mask, inside: (u32, u32)) {
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
    edit.masks = vec![exposure_mask("Linear", linear_source()), idle];

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
    assert_the_overlay_matches("overlay", &photo, &edit, 1, &overlaid);

    // Red where the mask is, the plain picture where it is not.
    let centre = (SIZE * inside.1 + inside.0) as usize;
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

/// Holds `overlaid`, a render of `edit` on `photo` with the overlay of mask
/// `shown` on, to the twin within the golden tolerances. The reference is the
/// developed picture through the output transform, with the overlay of that
/// mask between the clip and the curve. The overlay shows the alpha itself,
/// and near black one code of alpha is five or six of the output, so the
/// reference takes each path of [`stored_alpha_paths`] and holds each pixel
/// to the path closest to it.
fn assert_the_overlay_matches(
    name: &str,
    photo: &Photo,
    edit: &PhotoEdit,
    shown: usize,
    overlaid: &[[u8; 3]],
) {
    let linear: Vec<[f32; 3]> = photo
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| basic::decode_rgb8([px[0], px[1], px[2]], photo.source))
        .collect();
    let geometry = Geometry::full((SIZE, SIZE), (SIZE, SIZE));
    let references_at = |rounding: Rounding| -> Vec<Vec<[u8; 3]>> {
        let stored: Vec<[f32; 3]> = linear
            .iter()
            .map(|px| px.map(|c| half(c, rounding)))
            .collect();
        let store = |v: f32| half(v, rounding);
        let paths = stored_alpha_paths(&edit.masks[shown], &stored, &geometry, &store);
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
        let srgb: Vec<[f32; 3]> =
            mask_twin::develop_image_with(&image, edit, [1.0; 3], &store, &store)
                .into_iter()
                .map(|px| {
                    matrices::rec2020_to_srgb()
                        .apply(px)
                        .map(|c| c.clamp(0.0, 1.0))
                })
                .collect();
        paths
            .iter()
            .map(|alphas| {
                srgb.iter()
                    .zip(alphas)
                    .map(|(px, alpha)| {
                        mask_twin::overlay(*px, *alpha).map(transfer::linear_to_srgb8)
                    })
                    .collect()
            })
            .collect()
    };
    let references: Vec<Vec<[u8; 3]>> = [Rounding::Nearest, Rounding::TowardZero]
        .into_iter()
        .flat_map(references_at)
        .collect();
    let mut max = 0;
    let mut sum = 0u64;
    let mut worst = (0usize, [0u8; 3], [0u8; 3]);
    for (i, g) in overlaid.iter().enumerate() {
        // One path per pixel: the one whose largest channel difference is the
        // smallest, then whose sum is.
        let (d, closest) = references
            .iter()
            .map(|r| {
                let d = [0, 1, 2].map(|k| (i32::from(r[i][k]) - i32::from(g[k])).abs());
                (d, r[i])
            })
            .min_by_key(|(d, _)| (d.iter().max().copied(), d.iter().sum::<i32>()))
            .expect("at least six references");
        let largest = d.iter().max().copied().unwrap_or(0);
        if largest > max {
            max = largest;
            worst = (i, closest, *g);
        }
        sum += d.iter().sum::<i32>() as u64;
    }
    let mean = sum as f64 / (overlaid.len() * 3) as f64;
    println!(
        "{name}: max {max} at pixel {} (cpu {:?}, gpu {:?}), mean {mean:.3}, {} references",
        worst.0,
        worst.1,
        worst.2,
        references.len()
    );
    assert!(max <= MAX_DIFFERENCE, "{name}: max difference {max}");
    assert!(mean <= MEAN_DIFFERENCE, "{name}: mean difference {mean}");
}

/// The alpha of `mask` over every pixel as the GPU may hold it, one path for
/// each step a GPU may take at each r8unorm store on the mask path: the alpha
/// of the components (the "mask alpha" target of develop.rs), the refined
/// alpha while Refine edges is on (the refined target of refine.rs), and the
/// finished alpha while an edge control is on (the finished alpha of edge.rs,
/// or the shifted alpha while Shift edge alone is on). Each store is taken at
/// -0.1, 0 and +0.1 of a code, the whole of
/// [`mask_twin::UNORM_STEP_TOLERANCE`]: an alpha beside a half code may land
/// on either code at each store, and Contrast multiplies a code taken before
/// it by its gain. The arithmetic between the stores is the twin's, in the
/// order of [`mask_twin::alpha_image_before_the_store`]. The path with no
/// step, the middle one, equals the twin's stored alpha. Shift edge moves
/// whole codes only, so the shifted alpha under Feather or Contrast takes no
/// step of its own.
fn stored_alpha_paths(
    mask: &Mask,
    pixels: &[[f32; 3]],
    geometry: &Geometry,
    layer_store: &dyn Fn(f32) -> f32,
) -> Vec<Vec<f32>> {
    let tolerance = mask_twin::UNORM_STEP_TOLERANCE;
    let steps = [-tolerance, 0.0, tolerance];
    let stepped = |alpha: &[f32], step: f32| -> Vec<f32> {
        alpha
            .iter()
            .map(|alpha| mask_twin::stored_alpha_stepping(*alpha, step))
            .collect()
    };
    let mask = mask.sanitised();
    let mut drawn = mask.clone();
    drawn.refine = Refine::default();
    drawn.edge = Edge::default();
    let components =
        mask_twin::alpha_image_before_the_store(&drawn, pixels, geometry, None, layer_store);
    let mut paths = Vec::new();
    for components_step in steps {
        let alpha = stepped(&components, components_step);
        if mask.refine.is_off() && mask.edge.is_off() {
            paths.push(alpha);
            continue;
        }
        let refined = if mask.refine.is_off() {
            vec![alpha]
        } else {
            let refined =
                gamut_color::refine::refined(&alpha, pixels, geometry, &mask.refine, &|m| m);
            if mask.edge.is_off() {
                paths.extend(steps.map(|step| stepped(&refined, step)));
                continue;
            }
            steps.map(|step| stepped(&refined, step)).to_vec()
        };
        for held in refined {
            let finished = gamut_color::edge::finished(&held, geometry, &mask.edge, &|c| c);
            paths.extend(steps.map(|step| stepped(&finished, step)));
        }
    }
    let twin: Vec<f32> =
        mask_twin::alpha_image_before_the_store(&mask, pixels, geometry, None, layer_store)
            .into_iter()
            .map(mask_twin::stored_alpha)
            .collect();
    assert!(
        paths[paths.len() / 2] == twin,
        "the path with no step is the twin's stored alpha"
    );
    paths
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

/// A full stroke down the colours beside the grey ramp, then a stroke at a
/// flow of 45 along the most saturated row of the photo.
fn saturated_row_strokes(flow: f32) -> Vec<Stroke> {
    vec![
        stroke(&[[0.17, 0.1], [0.17, 0.8]], 0.12, 60.0, 100.0),
        stroke(&[[0.25, 0.82], [0.9, 0.82]], 0.1, 30.0, flow),
    ]
}

/// A part alpha over a saturated pixel lifted past 1: a step of the half
/// float developed texture in a bright channel is an output code in the near
/// black one. The mask pass makes the mix itself, so every GPU stores the
/// mix the twin makes; the blender of one GPU, left to it, cut the colour
/// and the alpha to half floats first and read 3 here.
#[test]
fn a_low_flow_brush_built_up_along_a_saturated_row_matches() {
    check_masks(
        "low flow along a saturated row",
        &masked(vec![exposure_mask(
            "Brush",
            brush_source(&saturated_row_strokes(45.0)),
        )]),
    );
}

/// The layer is still built by the blender, which one GPU runs at half
/// precision: it lands up to two steps of the layer, a quarter of an alpha
/// code, from the twin. That shows only where the alpha lies that near to a
/// half code, so this flow puts many pixels of the saturated row there.
#[test]
fn a_layer_alpha_beside_a_half_code_on_a_saturated_row_matches() {
    let strokes = saturated_row_strokes(HALF_CODE_FLOW);
    let mask = exposure_mask("Beside a half code", brush_source(&strokes));
    let photo = synthetic_photo();
    let linear: Vec<[f32; 3]> = photo
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| basic::decode_rgb8([px[0], px[1], px[2]], photo.source))
        .collect();
    let geometry = Geometry::full((SIZE, SIZE), (SIZE, SIZE));
    let store = |v: f32| half(v, Rounding::Nearest);
    let alphas = mask_twin::alpha_image_before_the_store(&mask, &linear, &geometry, None, &store);
    let beside = alphas
        .iter()
        .filter(|alpha| {
            let codes = **alpha * 255.0;
            codes > 1.0 && codes < 254.0 && (codes.fract() - 0.5).abs() < 0.25
        })
        .count();
    assert!(
        beside >= HALF_CODE_PIXELS,
        "{beside} pixels lie beside a half code"
    );
    check_masks("layer alpha beside a half code", &masked(vec![mask]));
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
    auto_strokes_at(sensitivity, 100.0)
}

/// [`auto_strokes`] with the flow of the stroke along the bottom named.
fn auto_strokes_at(sensitivity: f32, flow: f32) -> Vec<Stroke> {
    saturated_row_strokes(flow)
        .into_iter()
        .map(|stroke| auto(stroke, sensitivity))
        .collect()
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
fn an_auto_brush_at_a_low_flow_along_a_saturated_row_matches() {
    let edit = masked(vec![exposure_mask(
        "Auto",
        brush_source(&auto_strokes_at(70.0, 45.0)),
    )]);
    assert_the_gate_shows(&edit);
    check_masks("auto brush at a low flow", &edit);
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

/// `mask` with Refine edges on. On the 64 pixel photos a radius of 0.05 is a
/// box of two pixels either side, in cells of one pixel.
fn refined_at(mut mask: Mask, amount: f32, radius: f32, sensitivity: f32) -> Mask {
    mask.refine = Refine {
        amount,
        radius,
        sensitivity,
    };
    mask
}

fn refined(mask: Mask) -> Mask {
    refined_at(mask, 100.0, 0.05, 50.0)
}

/// A radial gradient drawn loosely over the colours, whose rim spills two
/// rows into the near-black band at the bottom of the synthetic photo (the
/// band begins at row 58): centre at row 50, full to 8 pixels out and gone at
/// 10. Refine edges takes the spill off the band.
fn radial_over_the_band() -> MaskSource {
    MaskSource::Radial(RadialGradient {
        centre: [0.5, 0.78],
        radius: [0.3, 0.16],
        rotation: 0.0,
        feather: 20.0,
    })
}

/// A stroke down the photo with its centre on the first column of colour
/// (column 7), a radius of 3 pixels and a hard rim, so it spills two columns
/// into the grey ramp of the first six columns.
fn stroke_along_the_grey_columns() -> MaskSource {
    brush_source(&[stroke(&[[0.11, 0.08], [0.11, 0.85]], 0.05, 20.0, 100.0)])
}

/// Refine edges has to move the picture for its golden test to mean anything.
/// How many pixels of the twin it moves.
fn assert_the_refine_shows(photo: &Photo, edit: &PhotoEdit) -> usize {
    let mut plain = edit.clone();
    for mask in &mut plain.masks {
        mask.refine = Refine::default();
    }
    let with = cpu_reference(photo, edit, Rounding::Nearest, 0);
    let without = cpu_reference(photo, &plain, Rounding::Nearest, 0);
    let moved = with.iter().zip(&without).filter(|(a, b)| a != b).count();
    assert!(
        moved * 100 > with.len(),
        "refine edges moves only {moved} of {} pixels",
        with.len()
    );
    moved
}

fn check_refined(name: &str, edit: &PhotoEdit) {
    let moved = assert_the_refine_shows(&synthetic_photo(), edit);
    println!("{name}: refine edges moves {moved} pixels of the twin");
    check_masks(name, edit);
}

#[test]
fn a_refined_radial_gradient_over_an_edge_matches() {
    check_refined(
        "refined radial gradient",
        &masked(vec![refined(exposure_mask(
            "Radial",
            radial_over_the_band(),
        ))]),
    );
}

#[test]
fn a_refined_brush_mask_along_the_grey_columns_matches() {
    check_refined(
        "refined brush mask",
        &masked(vec![refined(exposure_mask(
            "Brush",
            stroke_along_the_grey_columns(),
        ))]),
    );
}

#[test]
fn refine_edges_at_an_amount_of_50_matches() {
    let half = masked(vec![refined_at(
        exposure_mask("Radial", radial_over_the_band()),
        50.0,
        0.05,
        50.0,
    )]);
    check_refined("refine edges at an amount of 50", &half);
    let photo = synthetic_photo();
    let whole = masked(vec![refined(exposure_mask(
        "Radial",
        radial_over_the_band(),
    ))]);
    assert_ne!(
        cpu_reference(&photo, &half, Rounding::Nearest, 0),
        cpu_reference(&photo, &whole, Rounding::Nearest, 0)
    );
}

#[test]
fn refine_edges_at_a_sensitivity_of_0_and_of_100_matches() {
    let at = |sensitivity: f32| {
        masked(vec![refined_at(
            exposure_mask("Radial", radial_over_the_band()),
            100.0,
            0.05,
            sensitivity,
        )])
    };
    check_refined("refine edges at a sensitivity of 0", &at(0.0));
    check_refined("refine edges at a sensitivity of 100", &at(100.0));
    let photo = synthetic_photo();
    assert_ne!(
        cpu_reference(&photo, &at(0.0), Rounding::Nearest, 0),
        cpu_reference(&photo, &at(100.0), Rounding::Nearest, 0)
    );
}

#[test]
fn refine_edges_at_each_end_of_its_radius_matches() {
    for radius in [0.001, 0.02, 0.05] {
        check_masks(
            &format!("refine edges at a radius of {radius}"),
            &masked(vec![refined_at(
                exposure_mask("Radial", radial_over_the_band()),
                100.0,
                radius,
                50.0,
            )]),
        );
    }
}

#[test]
fn a_refined_auto_brush_matches() {
    let edit = masked(vec![refined(exposure_mask(
        "Auto",
        brush_source(&auto_strokes(70.0)),
    ))]);
    assert_the_gate_shows(&edit);
    check_refined("refined auto brush", &edit);
}

#[test]
fn a_refined_mask_that_is_inverted_and_one_at_half_opacity_match() {
    let mut inverted = refined(exposure_mask("Outside", radial_over_the_band()));
    inverted.invert = true;
    check_refined("refined inverted mask", &masked(vec![inverted]));
    let mut half = refined(exposure_mask("Half", radial_over_the_band()));
    half.opacity = 50.0;
    check_refined("refined mask at opacity 50", &masked(vec![half]));
}

/// Every mask lifts the exposure and nothing else. A refined alpha lies
/// within a tenth of a half code on many pixels of an edge, where a GPU may
/// store either code ([`mask_twin::UNORM_STEP_TOLERANCE`]); under an exposure
/// that is one output code at most, while a mask that also pulls the
/// saturation down makes it three on the near-black blue of a saturated row.
#[test]
fn two_refined_masks_and_a_plain_one_blend_in_list_order() {
    let brush = refined(exposure_mask("Brush", stroke_along_the_grey_columns()));
    let edit = masked(vec![
        refined_at(
            exposure_mask("Radial", radial_over_the_band()),
            80.0,
            0.03,
            70.0,
        ),
        exposure_mask("Linear", linear_source()),
        brush,
    ]);
    check_refined("two refined masks and a plain one", &edit);
}

#[test]
fn a_refined_mask_carrying_everything_over_a_global_edit_with_everything_matches() {
    let mut edit = everything_global();
    let mut mask = refined(Mask::new("Everything", radial_over_the_band()));
    mask.adjust = everything_in_a_mask();
    edit.masks = vec![mask];
    check("refined mask carrying everything", &edit);
}

/// The overlay shows the refined alpha, and an export never shows it. The
/// idle mask covers the middle of the photo and not its first pixel, refined
/// or not.
#[test]
fn the_overlay_of_a_refined_mask_matches_and_stays_out_of_an_export() {
    check_overlay(refined(Mask::new("Idle", radial_source())));
}

/// A mask with Refine edges at 0 draws byte for byte what the mask with no
/// refine draws, whatever its radius says, and runs no refine pass.
#[test]
fn refine_edges_at_0_draws_what_no_refine_draws_and_runs_no_pass() {
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
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let mut render = |edit: &PhotoEdit| -> Vec<u8> {
        let view = develop
            .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
            .expect("a source is set");
        readback.read(&gpu.device, &gpu.queue, view, SIZE, SIZE)
    };
    let mut edit = everything_global();
    edit.masks = vec![
        exposure_mask("Radial", radial_over_the_band()),
        exposure_mask("Brush", stroke_along_the_grey_columns()),
    ];
    let plain = render(&edit);
    for radius in [0.001, 0.05] {
        for mask in &mut edit.masks {
            mask.refine = Refine {
                amount: 0.0,
                radius,
                sensitivity: 90.0,
            };
        }
        assert_eq!(render(&edit), plain, "radius {radius}");
    }
    assert_eq!(develop.refine_builds(), (0, 0));
    assert_eq!(
        develop.mask_alpha_builds(),
        2,
        "and no alpha was drawn again"
    );
}

/// The refined alpha is cached like the alpha under it: a develop slider draws
/// neither, a Refine slider filters the alpha again and does not redraw it, a
/// component redraws the alpha and so the filter, and a new source draws both.
#[test]
fn a_develop_slider_draws_no_refine_pass_and_a_refine_slider_draws_no_alpha() {
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
    // The alphas drawn, the refined alphas drawn, and how many times the
    // moments of the source were taken. A refine of the 64 pixel photo is
    // one tile: 13 passes with the solve in one pass of four targets, 16
    // with it in two passes of two, and 5 more when it takes the moments of
    // the source in one pass of three targets, 6 with them in two passes. A
    // Radius step that keeps the side of a cell takes no moments of the
    // source and draws their 4 box means again. `boxed` counts those steps.
    // The reached field adds the seed, a pass a doubling step and the pass
    // that writes it at each pixel: at Radius 0.05 the radius is 3.2 pixels
    // in cells of one, a flood of 3 cells, steps 1 and 2, 1 + 2 + 1 = 4
    // passes; at 0.02 it is 1.28 pixels, a flood of 1 cell, step 1,
    // 1 + 1 + 1 = 3 passes. The first three refines below are at 0.05 and
    // every later one at 0.02, so `refines` refines take
    // 4 min(refines, 3) + 3 (refines - 3) such passes.
    let (refine, source) = if develop.refine_fused() {
        (13, 5)
    } else {
        (16, 6)
    };
    let flooded = |refines: u64| 4 * refines.min(3) + 3 * refines.saturating_sub(3);
    let builds_after = |develop: &mut Develop, edit: &PhotoEdit, boxed: u64| -> (u64, u64, u64) {
        develop
            .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
            .expect("a source is set");
        let (refines, sources) = (develop.refine_builds().0, develop.refine_source_builds());
        assert_eq!(
            u64::from(develop.refine_tiles()),
            refines,
            "one tile a refine"
        );
        assert_eq!(
            develop.refine_source_strips(),
            0,
            "a refine of the whole photo takes no strips"
        );
        assert_eq!(
            u64::from(develop.refine_passes()),
            refine * refines + source * sources + 4 * boxed + flooded(refines),
            "{refine} passes a refine, {source} a source, 4 a box of the source and the flood"
        );
        (develop.mask_alpha_builds(), refines, sources)
    };
    let mut edit = masked(vec![
        refined(exposure_mask("Radial", radial_over_the_band())),
        exposure_mask("Luminance", luminance_source()),
    ]);
    assert_eq!(
        builds_after(&mut develop, &edit, 0),
        (2, 1, 1),
        "two alphas, one of them refined"
    );
    assert_eq!(
        builds_after(&mut develop, &edit, 0),
        (2, 1, 1),
        "the same edit again"
    );

    let (passes, tiles) = (develop.refine_passes(), develop.refine_tiles());
    edit.exposure = 0.7;
    edit.masks[0].adjust.exposure = -0.5;
    edit.masks[0].adjust.look.curves.master = s_curve();
    edit.masks[0].opacity = 35.0;
    assert_eq!(
        builds_after(&mut develop, &edit, 0),
        (2, 1, 1),
        "sliders of the develop chain, global and of the mask"
    );
    assert_eq!(
        (develop.refine_passes(), develop.refine_tiles()),
        (passes, tiles),
        "a develop slider draws no refine pass"
    );

    edit.masks[0].refine.amount = 60.0;
    assert_eq!(
        builds_after(&mut develop, &edit, 0),
        (2, 2, 1),
        "the Amount slider gathers again over the moments of the source it holds"
    );
    edit.masks[0].refine.sensitivity = 80.0;
    assert_eq!(
        builds_after(&mut develop, &edit, 0),
        (2, 3, 1),
        "the Edge sensitivity slider too"
    );
    // On the 64 pixel photo a Radius of 0.05 and one of 0.02 both take
    // cells of one pixel, with a box of 2 cells and of 1.
    let geometry = Geometry::full((SIZE, SIZE), (SIZE, SIZE));
    for (radius, cells, flood) in [(0.05, 2, 3), (0.02, 1, 1)] {
        let mut refine = edit.masks[0].refine;
        refine.radius = radius;
        let plan = Plan::for_geometry(&refine, &geometry);
        assert_eq!((plan.step, plan.cells, plan.flood), (1, cells, flood));
    }
    edit.masks[0].refine.radius = 0.02;
    assert_eq!(
        builds_after(&mut develop, &edit, 1),
        (2, 4, 1),
        "the Radius slider keeps the moments of the source of the same side of a cell and draws their box means again"
    );

    edit.masks[0].components[0].invert = true;
    assert_eq!(
        builds_after(&mut develop, &edit, 1),
        (3, 5, 1),
        "a component redraws the alpha, and the filter reads it"
    );
    // The second mask takes the radius of the first, so the moments of the
    // source the frame holds serve it.
    edit.masks[1].refine.radius = 0.02;
    edit.masks[1].refine.amount = 100.0;
    assert_eq!(
        builds_after(&mut develop, &edit, 1),
        (3, 6, 1),
        "refine switched on for the second mask filters the alpha it holds"
    );
    edit.masks[1].refine.amount = 0.0;
    assert_eq!(
        builds_after(&mut develop, &edit, 1),
        (3, 6, 1),
        "and off again"
    );

    develop.set_source(&hazy_photo());
    assert_eq!(
        builds_after(&mut develop, &edit, 1),
        (5, 7, 2),
        "a new source content draws all three"
    );
    assert_eq!(develop.refine_builds().1, 0, "never a part");
    // Seven refines, three at 0.05 with 4 passes of the reached field each
    // and four at 0.02 with 3: 3 x 4 + 4 x 3 = 24. Fused, 7 x 13 + 2 x 5 + 4
    // + 24 = 129 passes.
    assert_eq!(flooded(7), 24);
    assert_eq!(
        (develop.refine_passes(), develop.refine_tiles()),
        (7 * refine as u32 + 2 * source as u32 + 4 + 24, 7),
        "seven refines of one tile, two of them with the moments of the source and one with their box means alone"
    );
}

/// A photo with hard edges, large enough that a zoomed window is a real part
/// of it: blocks of colour 96 pixels wide under a brightness that steps every
/// 60 rows, with a little grain so no two cells are alike.
fn blocky_photo(width: u32, height: u32) -> Photo {
    const COLOURS: [[f32; 3]; 5] = [
        [0.9, 0.3, 0.15],
        [0.15, 0.2, 0.5],
        [0.85, 0.8, 0.3],
        [0.1, 0.45, 0.2],
        [0.6, 0.6, 0.65],
    ];
    let mut rgba8 = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let colour = COLOURS[((x / 96 + 2 * (y / 60)) % 5) as usize];
            let level = if (y / 60) % 2 == 0 { 1.0 } else { 0.45 };
            let grain = 0.94 + 0.06 * ((x * 7 + y * 13) % 11) as f32 / 10.0;
            for c in colour {
                rgba8.push((c * level * grain * 255.0).round() as u8);
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

/// On a photo wider than 1024 pixels a radius of 0.015 is 16 pixels and one of
/// 0.03 is 33, so the moments are taken in cells of two and of four pixels:
/// the cell grid, the bilinear step back up and the odd last column are real.
#[test]
fn a_refined_mask_on_a_photo_wider_than_1024_pixels_matches() {
    let photo = blocky_photo(1101, 90);
    for (radius, cells) in [(0.015, (2, 6)), (0.03, (4, 6))] {
        let plan = Plan::for_geometry(
            &refined_at(Mask::default(), 100.0, radius, 50.0).refine,
            &Geometry::full((1101, 90), (1101, 90)),
        );
        assert_eq!((plan.step, plan.cells), cells);
        // A radial gradient with a hard rim 8 pixels over the block edge at
        // column 576, across the row edge at 60, and a stroke with a hard rim
        // 5 pixels over the block edge at column 288.
        let radial = MaskSource::Radial(RadialGradient {
            centre: [0.4995, 0.5],
            radius: [0.0309, 0.05],
            rotation: 0.0,
            feather: 15.0,
        });
        let brush = brush_source(&[stroke(&[[0.2543, 0.1], [0.2543, 0.9]], 0.012, 20.0, 100.0)]);
        let edit = masked(vec![
            refined_at(exposure_mask("Radial", radial), 100.0, radius, 50.0),
            refined_at(exposure_mask("Brush", brush), 100.0, radius, 60.0),
        ]);
        let moved = assert_the_refine_shows(&photo, &edit);
        println!(
            "refined masks on a wide photo, cells of {}: refine edges moves {moved} pixels of the twin",
            plan.step
        );
        check_on(
            &format!(
                "refined masks on a photo wider than 1024 pixels, cells of {}",
                plan.step
            ),
            &photo,
            &edit,
        );
    }
}

/// A scratch budget under the bytes of the grid of cells cuts a refine into
/// tiles, and the tiles draw byte for byte what one tile draws. On the 1101
/// by 90 photo of the test above, the grid is 551 by 45 cells of two pixels
/// at a radius of 0.015 (24,795 cells of 232 bytes) and 276 by 23 cells of
/// four at 0.03 (6,348 cells of 232 bytes), each with a margin of 29 cells:
/// three boxes of 6, the 2 cells beside of the gathers, the cell beside of
/// the reached field and its flood of 8 cells (16.5 pixels of radius in cells
/// of 2, 33 in cells of 4), 18 + 2 + 1 + 8 = 29. A budget of 157 by 157 cells
/// at a step of 2 gives tiles of 157 cells a side, which write
/// (157 - 2 x 29 - 3) x 2 = 192 pixels a row: 6 tiles. One of 79 by 79 cells
/// at a step of 4 gives tiles of 122 cells, the floor of two margins and 64,
/// which write (122 - 61) x 4 = 244: 5 tiles. The default budget holds either
/// grid in one. A stroke along the whole photo, with a hard rim over the row
/// edge at 60, crosses every cut between the tiles.
#[test]
fn the_tiles_of_a_small_budget_give_what_one_tile_gives() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = blocky_photo(1101, 90);
    let size = (photo.width, photo.height);
    let readback = Readback::new(&gpu.device);
    // The radius, the step, the cells a side of the budget, the bytes of a
    // cell, the side of a tile, and the tiles a mask.
    for (radius, step, budget_side, cell_bytes, side, cut) in [
        (0.015, 2, 157u64, 232u64, 157u32, 6u32),
        (0.03, 4, 79, 232, 122, 5),
    ] {
        let plan = Plan::for_geometry(
            &refined_at(Mask::default(), 100.0, radius, 50.0).refine,
            &Geometry::full(size, size),
        );
        assert_eq!(plan.flood, 8);
        assert_eq!(plan.margin(), 3 * 6 + 2 + 1 + 8);
        assert_eq!((plan.step, plan.cells, plan.margin()), (step, 6, 29));
        assert_eq!(side.max(2 * 29 + 64), side);
        assert_eq!(1101u32.div_ceil((side - 2 * 29 - 3) * step), cut);
        let budget = budget_side * budget_side * cell_bytes;
        let (_, grid) = plan.grid();
        assert!(
            u64::from(grid.0 * grid.1) * cell_bytes > budget,
            "the budget is under the bytes of the grid"
        );
        // A stroke across the photo over the darker rows from 60 down, its
        // centre on row 76 and its hard rim 5 pixels over the row edge at 60
        // into the bright rows, refined alone; then with the radial gradient
        // and the stroke down the photo of the test above, all three
        // refined. Under the connectivity prior, ruled 2026-09-24, a band
        // whose inside the bright rows reach through like colour holds
        // still: the band of this test before T-19, centred on row 52 over
        // the bright rows, moved no pixel at Radius 0.03. Over the darker
        // rows the inside is out of the outside's reach and the spill
        // leaves: the twin moves 5238 pixels alone and 5541 with the other
        // two masks in cells of 2, 5363 and 5348 in cells of 4.
        let band = brush_source(&[stroke(
            &[[0.02, 76.0 / 90.0], [0.98, 76.0 / 90.0]],
            21.0 / 1101.0,
            20.0,
            100.0,
        )]);
        let radial = MaskSource::Radial(RadialGradient {
            centre: [0.4995, 0.5],
            radius: [0.0309, 0.05],
            rotation: 0.0,
            feather: 15.0,
        });
        let brush = brush_source(&[stroke(&[[0.2543, 0.1], [0.2543, 0.9]], 0.012, 20.0, 100.0)]);
        let one = masked(vec![refined_at(
            exposure_mask("Band", band.clone()),
            100.0,
            radius,
            60.0,
        )]);
        let three = masked(vec![
            refined_at(exposure_mask("Band", band), 100.0, radius, 60.0),
            refined_at(exposure_mask("Radial", radial), 100.0, radius, 50.0),
            refined_at(exposure_mask("Brush", brush), 100.0, radius, 60.0),
        ]);
        for (edit, refined_masks) in [(one, 1u32), (three, 3)] {
            let name = format!(
                "{refined_masks} refined masks on the 1101 pixel photo, cells of {step}, in tiles of {side} cells"
            );
            let moved = assert_the_refine_shows(&photo, &edit);
            // A graph of its own for each budget, so its counters read the
            // tiles of this render alone.
            let render = |budget: Option<u64>| -> (Vec<u8>, u32) {
                let develop = Develop::new(&gpu.device, &gpu.queue);
                let mut develop = match budget {
                    Some(bytes) => develop.with_refine_scratch_budget(bytes),
                    None => develop,
                };
                develop.set_source(&photo);
                let view = develop
                    .render(&edit, CropRect::FULL, size, size)
                    .expect("a source is set");
                let pixels = readback.read(&gpu.device, &gpu.queue, view, size.0, size.1);
                (pixels, develop.refine_tiles())
            };
            let (tiled, tiles) = render(Some(budget));
            let (whole, one_tile) = render(None);
            println!("{name}: refine_tiles {tiles} then {one_tile}; the twin moves {moved} pixels");
            assert_eq!(
                (tiles, one_tile),
                (cut * refined_masks, refined_masks),
                "{name}: {cut} tiles a mask at {budget} bytes, then 1"
            );
            let differing = tiled.iter().zip(&whole).filter(|(a, b)| a != b).count();
            assert_eq!(tiled.len(), whole.len());
            assert_eq!(
                differing, 0,
                "{name}: {differing} bytes of the tiles differ from one tile"
            );
            let pixels: Vec<[u8; 3]> = tiled
                .as_chunks::<4>()
                .0
                .iter()
                .map(|px| [px[0], px[1], px[2]])
                .collect();
            assert_matches_the_twin(&name, &photo, &edit, &pixels);
        }
    }
}

/// Reads the bytes of `view`, a view of an R8Unorm texture of `size` pixels,
/// one a pixel, row by row from the top left. The texture cannot be copied
/// from, so a pass loads each texel into a target of the same format that
/// can; a unorm code loaded and stored again is the same code.
fn read_r8(gpu: &Headless, view: &wgpu::TextureView, size: (u32, u32)) -> Vec<u8> {
    const SHADER: &str = "
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
            let x = f32(i32(index & 1u) * 4 - 1);
            let y = f32(i32(index >> 1u) * 4 - 1);
            return vec4<f32>(x, y, 0.0, 1.0);
        }
        @group(0) @binding(0) var source: texture_2d<f32>;
        @fragment
        fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
            return textureLoad(source, vec2<i32>(position.xy), 0);
        }
    ";
    let device = &gpu.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("r8 copy"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("r8 copy"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("r8 copy"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("r8 copy"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let extent = wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("r8 copy"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("r8 copy"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(view),
        }],
    });
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = size.0.div_ceil(alignment) * alignment;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("r8 copy"),
        size: u64::from(padded) * u64::from(size.1),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("r8 copy"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("r8 copy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size.1),
            },
        },
        extent,
    );
    let submission = gpu.queue.submit(Some(encoder.finish()));
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("wait for the copy");
    receiver
        .recv()
        .expect("map callback ran")
        .expect("map the copy");
    let mut bytes = Vec::with_capacity((size.0 * size.1) as usize);
    {
        let mapped = buffer
            .get_mapped_range(..)
            .expect("mapped range of the copy");
        for row in mapped.chunks_exact(padded as usize) {
            bytes.extend_from_slice(&row[..size.0 as usize]);
        }
    }
    buffer.unmap();
    bytes
}

/// The budget of the scratch in the test below: 112,560 cells of 232 bytes
/// (CELL_BYTES), 335 by 336, 26,113,920 bytes. Its square side is 335, about
/// a quarter of the default budget's 1317, and it holds one column more than
/// that square. At 224 bytes a cell it was 25,213,440 bytes, the same cells.
const SMALL_SCRATCH_BUDGET: u64 = 335 * 336 * CELL_BYTES;

/// The bytes one cell of a tile costs the scratch, as `CELL_BYTES` in
/// refine.rs: fourteen Rgba32Float targets of 16 bytes and two R32Float of 4.
const CELL_BYTES: u64 = 14 * 16 + 2 * 4;

/// Two refined masks of one frame at different steps share its scratch and
/// keep it inside the budget, and each draws the refined alpha it draws
/// alone. The 24 megapixel photo of the timing test at the default budget
/// ran 212 s on the fallback adapter, so this is its case at a quarter of
/// each side: a 1500 by 1000 photo under 26,113,920 bytes
/// ([`SMALL_SCRATCH_BUDGET`]). A Radius of 0.05 is 75 pixels there: cells of
/// 4, a grid of 375 by 250 (93,750 cells, one tile of 375). One of 0.0012 is
/// 1.8 pixels: cells of one, a grid of 1500 by 1000, tiled at 335 alone.
/// After 0.05, the step 1 mask would want 375 by 335 (125,625 cells), so it
/// cuts its tiles at 300, the most rows 375 columns leave: 375 by 300 is
/// 112,500 cells, and 375 by 301 would be 112,875. After 0.0012, the step 4
/// mask would want 375 by 335, so it cuts at 336: 336 by 335 is 112,560,
/// and 337 by 335 would be 112,895. The floors, 184 cells at step 4 (margin
/// 60) and 78 at step 1 (margin 7), lie under both cuts. The first is the
/// cut at 1157 of the 24 megapixel photo at the default budget. The default
/// budget holds no column beside its square of 1317 (1318 by 1317 is
/// 1,735,806 cells, over its 1,735,574), so there the second order keeps
/// 1317 by 1317; this budget keeps the column, and the cut, of 224 bytes.
#[test]
fn two_masks_at_different_steps_hold_their_scratch_inside_the_budget() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let started = std::time::Instant::now();
    let photo = blocky_photo(1500, 1000);
    let size = (photo.width, photo.height);
    // Two radial gradients with a firm rim, one over the other's edge.
    let radial = |centre: [f32; 2], radius: [f32; 2]| {
        MaskSource::Radial(RadialGradient {
            centre,
            radius,
            rotation: 0.0,
            feather: 10.0,
        })
    };
    let coarse = refined_at(
        exposure_mask("Coarse", radial([0.45, 0.5], [0.25, 0.3])),
        100.0,
        0.05,
        50.0,
    );
    let fine = refined_at(
        exposure_mask("Fine", radial([0.6, 0.45], [0.2, 0.25])),
        100.0,
        0.0012,
        50.0,
    );
    for (mask, step, grid) in [(&coarse, 4, (375, 250)), (&fine, 1, (1500, 1000))] {
        let plan = Plan::for_geometry(&mask.refine, &Geometry::full(size, size));
        assert_eq!((plan.step, plan.grid().1), (step, grid));
    }
    // The refined alpha of each mask of `masks` on a graph of its own, and
    // the scratch the last of them drew with.
    let render = |masks: &[&Mask]| -> (Vec<Vec<u8>>, gamut_gpu::develop::RefineHold) {
        let mut develop =
            Develop::new(&gpu.device, &gpu.queue).with_refine_scratch_budget(SMALL_SCRATCH_BUDGET);
        develop.set_source(&photo);
        let edit = masked(masks.iter().map(|&mask| mask.clone()).collect());
        develop
            .render(&edit, CropRect::FULL, size, size)
            .expect("a source is set");
        let alphas = (0..masks.len())
            .map(|index| {
                let (view, frame) = develop.refined_alpha(index).expect("the mask is refined");
                assert_eq!(frame, size);
                read_r8(&gpu, view, size)
            })
            .collect();
        (alphas, develop.refine_last_hold().expect("a refine ran"))
    };
    let said = |hold: &gamut_gpu::develop::RefineHold| {
        format!(
            "held {}x{}, side {}, made {}, branch {}",
            hold.size.0, hold.size.1, hold.side, hold.made, hold.branch
        )
    };
    let mut alone = Vec::new();
    for (mask, held, side) in [(&coarse, (375, 250), 375), (&fine, (335, 335), 335)] {
        let (mut alphas, hold) = render(&[mask]);
        assert_eq!(
            (hold.size, hold.side, hold.made, hold.branch),
            (held, side, true, "First"),
            "{}: the scratch alone",
            mask.name
        );
        let alpha = alphas.remove(0);
        let inside = alpha.iter().filter(|&&a| a == 255).count();
        let partial = alpha.iter().filter(|&&a| a > 0 && a < 255).count();
        println!(
            "{} alone: {}; {inside} pixels at 255, {partial} between 0 and 255",
            mask.name,
            said(&hold)
        );
        assert!(
            inside > 0 && partial > 0,
            "{}: the refined alpha has an edge",
            mask.name
        );
        alone.push((mask.name.clone(), alpha));
    }
    let alone_of = |mask: &Mask| -> &Vec<u8> {
        &alone
            .iter()
            .find(|(name, _)| *name == mask.name)
            .expect("rendered alone")
            .1
    };
    // The order of the list, then the size held and the side of the tiles
    // after the second mask.
    for (order, masks, held, side) in [
        ("0.05 then 0.0012", [&coarse, &fine], (375, 300), 300),
        ("0.0012 then 0.05", [&fine, &coarse], (336, 335), 336),
    ] {
        let (alphas, hold) = render(&masks);
        println!("{order}: {}", said(&hold));
        assert_eq!(
            (hold.size, hold.side, hold.made, hold.branch),
            (held, side, true, "Cut"),
            "{order}: the scratch of the second mask"
        );
        let bytes = u64::from(hold.size.0) * u64::from(hold.size.1) * CELL_BYTES;
        assert!(bytes <= SMALL_SCRATCH_BUDGET, "{order}: {bytes} bytes");
        for (index, mask) in masks.into_iter().enumerate() {
            let differing = alphas[index]
                .iter()
                .zip(alone_of(mask))
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "{order}: mask {} ({}) differs from itself alone in {differing} bytes",
                index + 1,
                mask.name
            );
            assert_eq!(
                differing,
                0,
                "{order}: mask {} beside the other differs from it alone",
                index + 1
            );
        }
    }
    println!("{:.1} s", started.elapsed().as_secs_f64());
}

/// The photo, the mask and the render of the two tests of a Radius step:
/// blocks of colour 1420 pixels wide and 160 rows, where a Radius of 0.05
/// and one of 0.049 both take cells of 4 pixels, with a box of 13 cells
/// either side and of 12, each summed in 3 blocks of 8 and the cells left
/// over, and a Radius of 0.01 takes cells of one pixel. The mask is a radial
/// gradient 160 pixels across either side with a hard rim 8 pixels over the
/// block edge at column 576.
fn radius_step_photo_and_mask(radius: f32) -> (Photo, PhotoEdit) {
    let photo = blocky_photo(1420, 160);
    let w = photo.width as f32;
    let radial = MaskSource::Radial(RadialGradient {
        centre: [(576.0 + 8.0 - 160.0) / w, 0.5],
        radius: [160.0 / w, 70.0 / w],
        rotation: 0.0,
        feather: 15.0,
    });
    let edit = masked(vec![refined_at(
        exposure_mask("Radial", radial),
        100.0,
        radius,
        50.0,
    )]);
    (photo, edit)
}

/// The edit of `radius_step_photo_and_mask` at another Radius.
fn at_radius(edit: &PhotoEdit, radius: f32) -> PhotoEdit {
    let mut edit = edit.clone();
    edit.masks[0].refine.radius = radius;
    edit
}

/// A Radius step that keeps the side of a cell keeps the moments of the
/// source and draws only their box means again. On one photo and one mask,
/// Radius 0.05 rendered fresh, then 0.049, then 0.05 again on the same graph:
/// the moments of the source are taken once, the render at 0.049 is byte for
/// byte a fresh render at 0.049, and the third render is byte for byte the
/// first. Then a zoomed window, whose work at 0.049 is 3 cells a side
/// narrower than at 0.05 inside the grid: 0.049 and then 0.05 on one graph
/// takes the moments over the strips the wider work adds, and gives byte for
/// byte a fresh render at 0.05.
#[test]
fn a_radius_step_of_the_same_cell_size_holds_the_moments_of_the_source() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let (photo, wide) = radius_step_photo_and_mask(0.05);
    let narrow = at_radius(&wide, 0.049);
    let size = (photo.width, photo.height);
    for (edit, cells) in [(&wide, 13), (&narrow, 12)] {
        let plan = Plan::for_geometry(&edit.masks[0].refine, &Geometry::full(size, size));
        assert_eq!((plan.step, plan.cells), (4, cells));
    }
    let moved = assert_the_refine_shows(&photo, &wide);
    let readback = Readback::new(&gpu.device);
    let render = |develop: &mut Develop, edit: &PhotoEdit| -> Vec<u8> {
        let view = develop
            .render(edit, CropRect::FULL, size, size)
            .expect("a source is set");
        readback.read(&gpu.device, &gpu.queue, view, size.0, size.1)
    };
    let fresh = |edit: &PhotoEdit| -> Vec<u8> {
        let mut develop = Develop::new(&gpu.device, &gpu.queue);
        develop.set_source(&photo);
        render(&mut develop, edit)
    };
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let first = render(&mut develop, &wide);
    let second = render(&mut develop, &narrow);
    let third = render(&mut develop, &wide);
    let (builds, strips) = (
        develop.refine_source_builds(),
        develop.refine_source_strips(),
    );
    println!(
        "a Radius step of the same cell size, the whole photo: refine_source_builds {builds}, refine_source_strips {strips}, refine_passes {}; the twin moves {moved} pixels",
        develop.refine_passes()
    );
    assert!(first != second, "the Radius step shows on the picture");
    assert_eq!(
        second,
        fresh(&narrow),
        "the box means drawn again over the moments held give a fresh render"
    );
    assert!(
        third == first,
        "the third render is byte for byte the first"
    );
    assert_eq!(builds, 1, "the moments of the source taken whole once");
    assert!(strips <= 4, "at most four strips on the grow back");

    // A zoomed window: the refine works over the cells of the window and the
    // margin around them, inside a grid padded by the reach of Refine edges.
    let view = ViewWindow {
        full: size,
        window: (400, 0, 480, 160),
        visible: (400, 0, 480, 160),
    };
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    view_render_of(&mut develop, &gpu, &readback, &narrow, &view);
    let grown = view_render_of(&mut develop, &gpu, &readback, &wide, &view);
    let (builds, strips) = (
        develop.refine_source_builds(),
        develop.refine_source_strips(),
    );
    println!(
        "a Radius step of the same cell size, a zoomed window: refine_source_builds {builds}, refine_source_strips {strips}"
    );
    let mut alone = Develop::new(&gpu.device, &gpu.queue);
    alone.set_source(&photo);
    let fresh_view = view_render_of(&mut alone, &gpu, &readback, &wide, &view);
    assert!(
        grown == fresh_view,
        "the moments taken over the strips give a fresh render byte for byte"
    );
    assert_eq!(builds, 1, "the moments of the source taken whole once");
    assert!(
        (1..=4).contains(&strips),
        "the wider work takes the moments over 1 to 4 strips, not {strips}"
    );
}

/// A Radius step that changes the side of a cell takes the moments of the
/// source again: on the photo of the test above, Radius 0.05 takes cells of
/// 4 pixels and Radius 0.01 cells of one, and the render at 0.01 after 0.05
/// is byte for byte a fresh render at 0.01.
#[test]
fn a_radius_step_to_another_cell_size_takes_the_moments_again() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let (photo, wide) = radius_step_photo_and_mask(0.05);
    let fine = at_radius(&wide, 0.01);
    let size = (photo.width, photo.height);
    for (edit, step) in [(&wide, 4), (&fine, 1)] {
        let plan = Plan::for_geometry(&edit.masks[0].refine, &Geometry::full(size, size));
        assert_eq!(plan.step, step);
    }
    let readback = Readback::new(&gpu.device);
    let render = |develop: &mut Develop, edit: &PhotoEdit| -> Vec<u8> {
        let view = develop
            .render(edit, CropRect::FULL, size, size)
            .expect("a source is set");
        readback.read(&gpu.device, &gpu.queue, view, size.0, size.1)
    };
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    render(&mut develop, &wide);
    let stepped = render(&mut develop, &fine);
    let (builds, strips) = (
        develop.refine_source_builds(),
        develop.refine_source_strips(),
    );
    println!(
        "a Radius step to another cell size: refine_source_builds {builds}, refine_source_strips {strips}"
    );
    let mut alone = Develop::new(&gpu.device, &gpu.queue);
    alone.set_source(&photo);
    assert!(
        stepped == render(&mut alone, &fine),
        "the render at 0.01 after 0.05 is byte for byte a fresh one"
    );
    assert_eq!(
        builds, 2,
        "the moments of the source taken again at the new step"
    );
    assert_eq!(strips, 0, "and never over strips");
}

/// A box of 24 cells or more adds the sums of 8 cells a block pass wrote,
/// and the cells left over. At Radius 0.05 a photo 1344 pixels wide has
/// cells of 4 pixels and a box of 12 cells either side, 25 cells: 3 blocks
/// and 1 cell. One 2048 pixels wide has a box of 18, 37 cells: 4 blocks and
/// 5. One 1024 pixels wide has a box of 9, 19 cells, under 24: it sums every
/// cell in the direct loop, draws no block pass, and gives the direct loop's
/// render byte for byte. At 160 rows the grid is 40 cells down, so a box down
/// is held at the edges of the grid in some cells and at neither edge in
/// others. Each render matches the twin and is set beside the same render
/// with every box summed in the direct loop.
///
/// A refine of one tile is 13 passes, and 5 more when it takes the moments
/// of the source (16 and 6 with the solve and the source each drawn in two
/// passes); with blocks, each of the 6 boxes of the three gathers and of the
/// 4 boxes of the source has a block pass before it. The reached field adds
/// its seed, k passes of the flood and the pass that writes it at each pixel
/// to every refine, k the doubling steps whose sum stays at most the radius
/// in cells: 51.2 pixels in cells of 4 at 1024 pixels wide, 12 cells,
/// 1 + 2 + 4 = 7, k = 3; 67.2 pixels at 1344, 16 cells, 1 + 2 + 4 + 8 = 15,
/// k = 4; 102.4 pixels at 2048, 25 cells, 15 and 16 more is 31, k = 4. Of the
/// two masks here the first takes the moments of the source and the second
/// holds them: at 2048 pixels 28 + 6 and 19 + 6 passes, 59 in all, against
/// 18 + 6 and 13 + 6, 43, in the direct loop.
#[test]
fn a_refined_mask_whose_box_spans_blocks_matches() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let readback = Readback::new(&gpu.device);
    // The width, the box in cells either side, its blocks and the cells left
    // over, the block edge the masks lie over, and the passes of the flood.
    for (width, cells, blocks, singles, edge, floods) in [
        (1024u32, 9u32, 0u32, 19u32, 576f32, 3u32),
        (1344, 12, 3, 1, 768.0, 4),
        (2048, 18, 4, 5, 1152.0, 4),
    ] {
        let photo = blocky_photo(width, 160);
        let size = (photo.width, photo.height);
        let plan = Plan::for_geometry(
            &refined_at(Mask::default(), 100.0, 0.05, 50.0).refine,
            &Geometry::full(size, size),
        );
        assert_eq!((plan.step, plan.cells), (4, cells));
        assert_eq!(plan.grid().1, (width / 4, 40));
        assert_eq!(plan.flood, width * 5 / 100 / 4);
        let steps = (0..)
            .map(|k| (1u32 << k) * 2 - 1)
            .take_while(|sum| *sum <= plan.flood)
            .count() as u32;
        assert_eq!(steps, floods);
        let taps = 2 * cells + 1;
        assert_eq!(
            if taps < 24 {
                (0, taps)
            } else {
                (taps / 8, taps % 8)
            },
            (blocks, singles)
        );
        let w = width as f32;
        // A radial gradient 160 pixels across either side with a hard rim 8
        // pixels over the block edge and across the row edges at 60 and 120,
        // and a stroke with a hard rim 5 pixels over the block edge at
        // column 288.
        let radial = MaskSource::Radial(RadialGradient {
            centre: [(edge + 8.0 - 160.0) / w, 0.5],
            radius: [160.0 / w, 70.0 / w],
            rotation: 0.0,
            feather: 15.0,
        });
        let brush = brush_source(&[stroke(
            &[[280.0 / w, 0.1], [280.0 / w, 0.9]],
            13.0 / w,
            20.0,
            100.0,
        )]);
        let edit = masked(vec![
            refined_at(exposure_mask("Radial", radial), 100.0, 0.05, 50.0),
            refined_at(exposure_mask("Brush", brush), 100.0, 0.05, 60.0),
        ]);
        let name = if blocks > 0 {
            format!(
                "refined masks on a photo {width} pixels wide, a box of {taps} cells in {blocks} blocks and {singles} cells"
            )
        } else {
            format!(
                "refined masks on a photo {width} pixels wide, a box of {taps} cells in the direct loop"
            )
        };
        let moved = assert_the_refine_shows(&photo, &edit);
        // A graph of its own for each render, so its counters read that
        // render alone: the passes, the refines and the moments of the
        // source taken.
        let render = |direct: bool| -> (Vec<u8>, u32, u64, u64, bool) {
            let mut develop = Develop::new(&gpu.device, &gpu.queue).with_refine_direct_box(direct);
            develop.set_source(&photo);
            let view = develop
                .render(&edit, CropRect::FULL, size, size)
                .expect("a source is set");
            let pixels = readback.read(&gpu.device, &gpu.queue, view, size.0, size.1);
            (
                pixels,
                develop.refine_passes(),
                develop.refine_builds().0,
                develop.refine_source_builds(),
                develop.refine_fused(),
            )
        };
        let (summed, passes, refines, sources, fused) = render(false);
        let (direct, direct_passes, direct_refines, direct_sources, _) = render(true);
        assert_eq!(
            (refines, sources, direct_refines, direct_sources),
            (2, 1, 2, 1),
            "{name}: two refines of one tile, the moments of the source taken once"
        );
        println!(
            "{name}: refine_passes {passes} with blocks, {direct_passes} in the direct loop, the source and the solve fused {fused}; the twin moves {moved} pixels"
        );
        // A refine without blocks: 13 passes with the solve in one pass of
        // four targets, 16 with it in two passes of two, and the seed and the
        // passes of the flood. The moments of the source without blocks: 5
        // passes with them in one pass of three targets, 6 with them in a
        // pass of two and a pass of one.
        let (refine, source) = if fused { (13, 5) } else { (16, 6) };
        // The seed, the passes of the flood, and the pass that writes the
        // reached field at each pixel.
        let refine = refine + 1 + floods + 1;
        if blocks > 0 {
            assert_eq!(
                passes,
                (refine + 6) * 2 + (source + 4),
                "{name}: with blocks"
            );
        } else {
            assert_eq!(passes, refine * 2 + source, "{name}: no block pass");
        }
        assert_eq!(
            direct_passes,
            refine * 2 + source,
            "{name}: in the direct loop"
        );
        assert_eq!(summed.len(), direct.len());
        let largest = summed
            .iter()
            .zip(&direct)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        let differing = summed.iter().zip(&direct).filter(|(a, b)| a != b).count();
        println!(
            "{name}: the blocks against the direct loop: largest byte difference {largest}, {differing} of {} bytes differ",
            summed.len()
        );
        // The block sums move each box sum by about 1e-7 relative, which
        // moves an output byte by 1 at most.
        assert!(
            largest <= 1,
            "{name}: the blocks differ from the direct loop by {largest} in a byte, over 1"
        );
        if blocks == 0 {
            assert_eq!(
                differing, 0,
                "{name}: a box under 24 cells draws the direct loop"
            );
        }
        let pixels: Vec<[u8; 3]> = summed
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| [px[0], px[1], px[2]])
            .collect();
        assert_matches_the_twin(&name, &photo, &edit, &pixels);
    }
}

/// The same with Refine edges at 100 on the growing mask: a stroke with a hard
/// rim, down beside the grey columns and along the near-black band so it
/// spills over both edges, painted in ten appended pieces, and an erase
/// stroke in five after it, are the strokes drawn whole. New dabs change the
/// refined alpha the filter's reach further out than they change the alpha
/// and no further, so the alpha, the refined alpha and the develop passes
/// are drawn over parts only, and the refined alpha whole once.
#[test]
fn a_stroke_appended_into_a_refined_mask_equals_the_stroke_drawn_whole() {
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
    // Down column 7, two columns over the grey ramp, then across row 56, two
    // rows over the band.
    let path: Vec<[f32; 2]> = (0..=40)
        .map(|i| {
            if i <= 20 {
                [0.11, 0.1 + i as f32 / 20.0 * 0.775]
            } else {
                [0.11 + (i - 20) as f32 / 20.0 * 0.7, 0.875]
            }
        })
        .collect();
    let eraser: Vec<[f32; 2]> = (0..=20)
        .map(|i| [0.5, 0.7 + i as f32 / 20.0 * 0.25])
        .collect();
    let edit_at = |painted: usize, erased: usize, refine: bool| -> PhotoEdit {
        let mut strokes = vec![stroke(&path[..painted], 0.05, 20.0, 100.0)];
        if erased > 0 {
            strokes.push(Stroke {
                erase: true,
                ..stroke(&eraser[..erased], 0.04, 20.0, 100.0)
            });
        }
        let mut edit = everything_global();
        let mut growing = exposure_mask("Growing", brush_source(&strokes));
        growing.adjust.clarity = 25.0;
        if refine {
            growing = refined(growing);
        }
        edit.masks = vec![growing, exposure_mask("Radial", radial_source())];
        edit
    };
    // At 100 percent, the lower left of the photo, where the stroke turns:
    // both of its edges and the eraser lie in what is seen.
    let view = ViewWindow {
        full: (SIZE, SIZE),
        window: (0, 30, 52, 34),
        visible: (0, 40, 40, 24),
    };
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
            pieces.push(render(&mut develop, &edit_at(piece * 4 + 1, 0, true)));
        }
        for piece in 1..=5 {
            pieces.push(render(&mut develop, &edit_at(41, piece * 4 + 1, true)));
        }
        assert!(pieces[0] != pieces[9], "the stroke grew on the picture");
        assert!(
            pieces[9] != pieces[14],
            "and the eraser took some of it away"
        );
        assert_eq!(
            develop.brush_layer_builds(),
            1,
            "the layer stamped whole once"
        );
        assert_eq!(develop.brush_layer_appends(), 14, "then new dabs only");
        let (whole_refines, part_refines) = develop.refine_builds();
        let (alphas, develops) = develop.brush_patches();
        println!(
            "zoomed {zoomed}: the refined alpha whole {whole_refines} times and over a part {part_refines}, the alpha over a part {alphas}, the develop {develops}"
        );
        assert_eq!(whole_refines, 1, "no whole refine pass after the first");
        assert_eq!(develop.refine_source_builds(), 1, "the source taken once");
        if zoomed {
            assert!(alphas > 0 && alphas <= 14 && develops > 0 && develops <= alphas);
            assert!(part_refines > 0 && part_refines <= 14);
        } else {
            assert_eq!((alphas, develops, part_refines), (14, 14, 14), "parts only");
        }

        let mut fresh = Develop::new(&gpu.device, &gpu.queue);
        fresh.set_source(&photo);
        let whole = render(&mut fresh, &edit_at(41, 21, true));
        assert_eq!(fresh.brush_layer_appends(), 0);
        assert_eq!(fresh.refine_builds(), (1, 0));
        let max = max_difference(&pieces[14], &whole);
        println!(
            "a stroke appended into a refined mask against the stroke whole, zoomed {zoomed}: max difference {max}"
        );
        assert!(max <= 1, "max difference {max}, zoomed {zoomed}");

        // Refine edges is in what was compared: without it the picture is
        // another.
        let unrefined = render(&mut fresh, &edit_at(41, 21, false));
        let moved = whole
            .as_chunks::<4>()
            .0
            .iter()
            .zip(unrefined.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count();
        println!("refine edges moves {moved} pixels, zoomed {zoomed}");
        assert!(moved > 40, "refine edges moves {moved} pixels");
    }
}

/// The synthetic photo drawn `by` times over on each side: each of its pixels
/// a block of `by` by `by`.
fn synthetic_photo_times(by: u32) -> Photo {
    let small = synthetic_photo();
    let side = SIZE * by;
    let mut rgba8 = Vec::with_capacity((side * side * 4) as usize);
    for y in 0..side {
        for x in 0..side {
            let at = (((y / by) * SIZE + x / by) * 4) as usize;
            rgba8.extend_from_slice(&small.rgba8[at..at + 4]);
        }
    }
    Photo {
        width: side,
        height: side,
        rgba8,
        ..small
    }
}

/// `rect` with `by` more pixels on every side, inside a frame of `size`, as
/// the render grows the reach of new dabs.
fn grown_by(rect: (u32, u32, u32, u32), by: u32, size: (u32, u32)) -> (u32, u32, u32, u32) {
    let (x0, y0) = (rect.0.saturating_sub(by), rect.1.saturating_sub(by));
    let x1 = (rect.0 + rect.2 + by).min(size.0).max(x0);
    let y1 = (rect.1 + rect.3 + by).min(size.1).max(y0);
    (x0, y0, x1 - x0, y1 - y0)
}

fn overlap(a: (u32, u32, u32, u32), b: (u32, u32, u32, u32)) -> Option<(u32, u32, u32, u32)> {
    let (x0, y0) = (a.0.max(b.0), a.1.max(b.1));
    let (x1, y1) = ((a.0 + a.2).min(b.0 + b.2), (a.1 + a.3).min(b.1 + b.3));
    (x1 > x0 && y1 > y0).then(|| (x0, y0, x1 - x0, y1 - y0))
}

/// A stroke appended into a mask with Refine edges and the three edge
/// controls on, then erased from, equals the strokes drawn whole: each stage
/// of the shape is drawn again over the reach of the stage before it grown
/// by its own reach, and none of them whole after the first render. On the
/// 64 pixel photo a Shift edge of 1 percent lies under the octagon's first
/// pixel and runs no pass, so this test draws the synthetic photo four times
/// over, 256 pixels a side, where it grows the octagon one pixel along each
/// axis and each diagonal.
#[test]
fn a_stroke_appended_into_an_edged_refined_mask_equals_the_stroke_drawn_whole() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    const BY: u32 = 4;
    let side = SIZE * BY;
    let photo = synthetic_photo_times(BY);
    let readback = Readback::new(&gpu.device);
    let edge = Edge {
        shift: -0.01,
        feather: 0.01,
        contrast: 50.0,
    };
    let at = |full: u32| gamut_color::edge::Plan::new(&edge, (full, full), (0, 0), (full, full));
    assert!(
        !at(SIZE).shifts(),
        "no Shift edge pass on the 64 pixel photo"
    );
    let plan = at(side);
    assert!(plan.shifts() && plan.feathers() && plan.contrasts());
    let refine = Refine {
        amount: 100.0,
        radius: 0.05,
        sensitivity: 50.0,
    };
    let refine_reach = Plan::new(&refine, (side, side), (0, 0), (side, side)).reach();
    let (shift_reach, feather_reach) = (plan.shift_reach(), plan.feather_reach());
    println!(
        "reach in pixels: refine {refine_reach}, shift {shift_reach}, feather {feather_reach}"
    );
    assert!(shift_reach > 0 && feather_reach > 0);

    // Down column 7 of the small photo, two of its columns over the grey
    // ramp, then across its row 56, two of its rows over the band.
    let path: Vec<[f32; 2]> = (0..=40)
        .map(|i| {
            if i <= 20 {
                [0.11, 0.1 + i as f32 / 20.0 * 0.775]
            } else {
                [0.11 + (i - 20) as f32 / 20.0 * 0.7, 0.875]
            }
        })
        .collect();
    let eraser: Vec<[f32; 2]> = (0..=20)
        .map(|i| [0.5, 0.7 + i as f32 / 20.0 * 0.25])
        .collect();
    let edit_at = |painted: usize, erased: usize, edged: bool| -> PhotoEdit {
        let mut strokes = vec![stroke(&path[..painted], 0.05, 20.0, 100.0)];
        if erased > 0 {
            strokes.push(Stroke {
                erase: true,
                ..stroke(&eraser[..erased], 0.04, 20.0, 100.0)
            });
        }
        let mut edit = everything_global();
        let mut growing = exposure_mask("Growing", brush_source(&strokes));
        growing.adjust.clarity = 25.0;
        growing.refine = refine;
        if edged {
            growing.edge = edge;
        }
        edit.masks = vec![growing, exposure_mask("Radial", radial_source())];
        edit
    };
    // At 100 percent, the lower left of the photo, where the stroke turns:
    // both of its edges and the eraser lie in what is seen.
    let view = ViewWindow {
        full: (side, side),
        window: (0, 30 * BY, 52 * BY, 34 * BY),
        visible: (0, 40 * BY, 40 * BY, 24 * BY),
    };
    for zoomed in [false, true] {
        let render = |develop: &mut Develop, edit: &PhotoEdit| -> Vec<u8> {
            if zoomed {
                view_render_of(develop, &gpu, &readback, edit, &view)
            } else {
                let drawn = develop
                    .render(edit, CropRect::FULL, (side, side), (side, side))
                    .expect("a source is set");
                readback.read(&gpu.device, &gpu.queue, drawn, side, side)
            }
        };
        let mut develop = Develop::new(&gpu.device, &gpu.queue);
        develop.set_source(&photo);
        let mut pieces = Vec::new();
        let mut traces = Vec::new();
        for piece in 1..=10 {
            pieces.push(render(&mut develop, &edit_at(piece * 4 + 1, 0, true)));
            traces.push(develop.shape_redrawn(0));
        }
        for piece in 1..=5 {
            pieces.push(render(&mut develop, &edit_at(41, piece * 4 + 1, true)));
            traces.push(develop.shape_redrawn(0));
        }
        assert!(pieces[0] != pieces[9], "the stroke grew on the picture");
        assert!(
            pieces[9] != pieces[14],
            "and the eraser took some of it away"
        );
        assert_eq!(
            develop.brush_layer_builds(),
            1,
            "the layer stamped whole once"
        );
        assert_eq!(develop.brush_layer_appends(), 14, "then new dabs only");
        let (whole_refines, part_refines) = develop.refine_builds();
        let (alphas, develops) = develop.brush_patches();
        let (whole_edges, part_edges) = develop.edge_builds();
        let passes = develop.edge_passes();
        println!(
            "zoomed {zoomed}: the alpha over a part {alphas}, the develop {develops}; the refined alpha whole {whole_refines} and over a part {part_refines}; shift, feather and finished alpha whole {whole_edges:?} and over a part {part_edges:?}; passes shift {}, feather {}, finish {}",
            passes.shift, passes.feather, passes.finish
        );
        assert_eq!(whole_refines, 1, "no whole refine pass after the first");
        assert_eq!(whole_edges, [1, 1, 1], "no whole edge pass after the first");
        assert_eq!(develop.refine_source_builds(), 1, "the source taken once");
        if zoomed {
            assert!(alphas > 0 && alphas <= 14 && develops > 0 && develops <= alphas);
            assert!(part_refines > 0 && part_refines <= alphas);
            assert_eq!(part_edges[0], part_refines, "a shift after each refine");
            assert!(part_edges[1] > 0 && part_edges[1] <= part_edges[0]);
            assert_eq!(
                part_edges[2], part_edges[1],
                "the finished alpha after each"
            );
        } else {
            assert_eq!((alphas, develops, part_refines), (14, 14, 14), "parts only");
            assert_eq!(part_edges, [14, 14, 14], "parts only");
        }

        // Each stage drew again over the stage before it grown by its own
        // reach, and nothing where the stage before drew nothing.
        for (piece, trace) in traces.iter().enumerate().skip(1) {
            let frame = (0, 0, trace.frame.0, trace.frame.1);
            let Some(alpha) = trace.alpha else {
                assert_eq!(
                    (trace.refine, trace.shift, trace.feather, trace.finish),
                    (None, None, None, None),
                    "piece {piece}, zoomed {zoomed}"
                );
                continue;
            };
            assert_ne!(alpha, frame, "piece {piece}: the alpha over a part");
            let refined = overlap(grown_by(alpha, refine_reach, trace.frame), frame);
            assert_eq!(trace.refine, refined, "piece {piece}, zoomed {zoomed}");
            let refined = refined.expect("the refine reach holds the new dabs");
            assert_ne!(refined, frame, "piece {piece}: the refine over a part");
            let shifted = overlap(grown_by(refined, shift_reach, trace.frame), frame);
            assert_eq!(trace.shift, shifted, "piece {piece}, zoomed {zoomed}");
            let shifted = shifted.expect("the shift reach holds the refine's");
            assert_ne!(shifted, frame, "piece {piece}: the shift over a part");
            let feathered = overlap(grown_by(shifted, feather_reach, trace.frame), trace.region);
            assert_eq!(trace.feather, feathered, "piece {piece}, zoomed {zoomed}");
            assert_eq!(trace.finish, feathered, "piece {piece}, zoomed {zoomed}");
            assert_ne!(
                feathered,
                Some(trace.region),
                "piece {piece}: the feather over a part"
            );
        }

        let mut fresh = Develop::new(&gpu.device, &gpu.queue);
        fresh.set_source(&photo);
        let whole = render(&mut fresh, &edit_at(41, 21, true));
        assert_eq!(fresh.brush_layer_appends(), 0);
        assert_eq!(fresh.refine_builds(), (1, 0));
        assert_eq!(fresh.edge_builds(), ([1, 1, 1], [0, 0, 0]));
        let max = max_difference(&pieces[14], &whole);
        println!(
            "a stroke appended into an edged refined mask against the stroke whole, zoomed {zoomed}: max difference {max}"
        );
        assert!(max <= 1, "max difference {max}, zoomed {zoomed}");

        // The edge controls are in what was compared: without them the
        // picture is another.
        let plain = render(&mut fresh, &edit_at(41, 21, false));
        let moved = whole
            .as_chunks::<4>()
            .0
            .iter()
            .zip(plain.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count();
        let pixels = whole.len() / 4;
        println!("the edge controls move {moved} of {pixels} pixels, zoomed {zoomed}");
        assert!(
            moved * 100 > pixels,
            "the edge controls move {moved} pixels"
        );
    }
}

/// A tower of two tones against a sky: dark slate on its left half and pale
/// stone on its right, columns 24 to 39 from row 10 down, with a little grain
/// so no two cells are alike.
fn tower_photo() -> Photo {
    let mut rgba8 = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let colour = if y < 10 || !(24..40).contains(&x) {
                [0.56, 0.76, 0.96]
            } else if x < 32 {
                [0.16, 0.2, 0.27]
            } else {
                [0.78, 0.74, 0.62]
            };
            let grain = 0.96 + 0.04 * ((x * 7 + y * 13) % 11) as f32 / 10.0;
            for c in colour {
                rgba8.push((c * grain * 255.0).round() as u8);
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

/// A stroke with a hard rim down the tower of [`tower_photo`], 10 pixels
/// either side of its middle: it spills two columns into the sky on each
/// side and over the top.
fn stroke_down_the_tower() -> MaskSource {
    brush_source(&[stroke(&[[0.5, 0.3], [0.5, 1.0]], 0.156, 10.0, 100.0)])
}

/// The largest difference between two pictures over the pixels `inside`
/// names, by column and row.
fn most_over(a: &[[u8; 3]], b: &[[u8; 3]], inside: impl Fn(u32, u32) -> bool) -> i32 {
    let mut most = 0;
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        if inside(i as u32 % SIZE, i as u32 / SIZE) {
            for k in 0..3 {
                most = most.max((i32::from(a[k]) - i32::from(b[k])).abs());
            }
        }
    }
    most
}

/// An object of two tones under a loose mask is kept whole: the spill leaves
/// the sky on both sides, and neither the slate, which is far from the sky,
/// nor the stone, which is as bright as the sky, loses its mask.
#[test]
fn a_two_tone_object_under_a_loose_refined_mask_is_kept_whole() {
    let photo = tower_photo();
    let mask = exposure_mask("Tower", stroke_down_the_tower());
    let loose = masked(vec![mask.clone()]);
    let snapped = masked(vec![refined(mask)]);
    let reference = |edit: &PhotoEdit| cpu_reference(&photo, edit, Rounding::Nearest, 0);
    let (none, loose, snapped_twin) = (
        reference(&PhotoEdit::default()),
        reference(&loose),
        reference(&snapped),
    );
    let tower = |x: u32, y: u32| (24..40).contains(&x) && (20..SIZE).contains(&y);
    let spill =
        |x: u32, y: u32| (x == 22 || x == 23 || x == 40 || x == 41) && (24..56).contains(&y);
    assert!(
        most_over(&loose, &none, spill) > 30,
        "the loose mask lifts the sky beside the tower"
    );
    let left = most_over(&snapped_twin, &none, spill);
    let lost = most_over(&snapped_twin, &loose, tower);
    println!("two-tone object: the spill keeps {left} codes, the tower loses {lost}");
    assert!(left <= 1, "the spill keeps {left} codes of its lift");
    assert!(lost <= 1, "the tower loses {lost} codes of its lift");
    check_on("refined mask over a two-tone object", &photo, &snapped);
}

/// The change of a refined alpha from the alpha as drawn over the pixels the
/// mask covers, mean and max in codes of alpha, on the twin at Amount 100.
fn soft_change(photo: &Photo, mask: &Mask) -> (f32, f32) {
    let stored: Vec<[f32; 3]> = photo
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| {
            basic::decode_rgb8([px[0], px[1], px[2]], photo.source)
                .map(|c| half(c, Rounding::Nearest))
        })
        .collect();
    let geometry = Geometry::full((photo.width, photo.height), (photo.width, photo.height));
    let store = |v: f32| half(v, Rounding::Nearest);
    let mut drawn = mask.clone();
    drawn.refine = Refine::default();
    let p: Vec<f32> =
        mask_twin::alpha_image_before_the_store(&drawn, &stored, &geometry, None, &store)
            .into_iter()
            .map(mask_twin::stored_alpha)
            .collect();
    let q = gamut_color::refine::refined(&p, &stored, &geometry, &mask.refine, &|m| m);
    let changes: Vec<f32> = p
        .iter()
        .zip(&q)
        .filter(|(p, _)| **p > 0.0)
        .map(|(p, q)| (q - p).abs() * 255.0)
        .collect();
    let mean = changes.iter().sum::<f32>() / changes.len() as f32;
    (mean, changes.iter().copied().fold(0.0, f32::max))
}

/// A mask that is soft at the scale of the box changes no more than the soft
/// bar allows, and the GPU matches the twin. The bar, ruled 2026-09-21: where
/// the mask's gradient is wider than the radius, the refined change is at most
/// x1.00 of the baseline guided filter's change on the same input (the ruled
/// colour guided filter at the eps of sensitivity 50, t18-design-pass
/// summary2.py), on the mean and on the max.
///
/// At Radius 0.02 (1.28 pixels) the twin returns the soft radial as drawn. At
/// Radius 0.05 (3.2 pixels) its gradient is wider than the radius: 3.46 pixels
/// from alpha 0.9 to 0.1 on the ray through the worst pixel (a median of 3.98
/// over 72 rays). Measured 2026-09-25 under the connectivity prior
/// (tools-2026-09-24-t19/soft_radial_bar.txt), over the 883 pixels the mask
/// covers: the baseline changes the alpha by 23.509 codes on the mean and
/// 63.611 at most, the ruled design by 0.678 and 25.722, x0.03 and x0.40.
/// The render moves by 8 codes at most, at pixel (34, 40), 67 pixels in all.
/// The snap before T-19 left this mask unchanged, which this test asserted
/// until the bar was measured.
#[test]
fn a_soft_refined_mask_stays_inside_the_soft_bar_and_matches() {
    const BASELINE_MEAN: f32 = 23.509;
    const BASELINE_MAX: f32 = 63.611;
    let photo = synthetic_photo();
    let soft = exposure_mask("Soft", radial_source());
    let plain = cpu_reference(&photo, &masked(vec![soft.clone()]), Rounding::Nearest, 0);
    for radius in [0.02, 0.05] {
        let refined = refined_at(soft.clone(), 100.0, radius, 50.0);
        let edit = masked(vec![refined.clone()]);
        let with = cpu_reference(&photo, &edit, Rounding::Nearest, 0);
        let most = most_over(&with, &plain, |_, _| true);
        let (mean, max) = soft_change(&photo, &refined);
        println!(
            "soft mask at a radius of {radius}: refine edges moves it {most} codes at most; the alpha changes by {mean:.3} codes on the mean and {max:.3} at most"
        );
        if radius == 0.02 {
            assert!(most <= 1, "radius {radius}: moved by {most} codes");
        } else {
            assert!(
                mean <= BASELINE_MEAN && max <= BASELINE_MAX,
                "radius {radius}: the change {mean:.3}/{max:.3} codes passes the baseline's {BASELINE_MEAN}/{BASELINE_MAX}"
            );
            println!(
                "soft mask at a radius of {radius}: x{:.2} on the mean, x{:.2} on the max",
                mean / BASELINE_MEAN,
                max / BASELINE_MAX
            );
        }
        check_masks(&format!("soft refined mask at a radius of {radius}"), &edit);
    }
}

/// A radial gradient with a hard rim that covers the middle of the synthetic
/// photo and spills two rows into the near-black band, as
/// [`radial_over_the_band`] does.
fn radial_over_the_middle_and_the_band() -> MaskSource {
    MaskSource::Radial(RadialGradient {
        centre: [0.5, 0.6],
        radius: [0.3, 0.34],
        rotation: 0.0,
        feather: 10.0,
    })
}

/// How many refined alphas must lie within a tenth of a half code.
const REFINED_HALF_CODE_PIXELS: usize = 30;

/// At an amount of 50 every pixel of the spill that leaves the mask lands on
/// 127.5 codes, within a rounding of the half code, where a GPU may store 127
/// or 128 ([`mask_twin::UNORM_STEP_TOLERANCE`]). Under an exposure that is one
/// output code at most, inside the unchanged tolerance; the overlay shows the
/// alpha itself, and its reference alone accepts either code.
#[test]
fn a_refined_alpha_beside_a_half_code_matches() {
    let mask = refined_at(
        exposure_mask("Beside a half code", radial_over_the_middle_and_the_band()),
        50.0,
        0.05,
        50.0,
    );
    let photo = synthetic_photo();
    let linear: Vec<[f32; 3]> = photo
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| basic::decode_rgb8([px[0], px[1], px[2]], photo.source))
        .collect();
    let geometry = Geometry::full((SIZE, SIZE), (SIZE, SIZE));
    let store = |v: f32| half(v, Rounding::Nearest);
    let alphas = mask_twin::alpha_image_before_the_store(&mask, &linear, &geometry, None, &store);
    let beside = alphas
        .iter()
        .filter(|alpha| {
            let codes = **alpha * 255.0;
            codes > 1.0
                && codes < 254.0
                && (codes.fract() - 0.5).abs() < mask_twin::UNORM_STEP_TOLERANCE
        })
        .count();
    println!("{beside} refined alphas lie within a tenth of a half code");
    assert!(
        beside >= REFINED_HALF_CODE_PIXELS,
        "{beside} refined alphas lie beside a half code"
    );
    check_refined(
        "refined alpha beside a half code",
        &masked(vec![mask.clone()]),
    );
    let mut idle = mask;
    idle.adjust = Adjustments::default();
    check_overlay(idle);
}

/// A refined mask under a crop window and under a zoomed window equals the
/// same region of the full render: the cells lie on the pixels of the whole
/// picture and the window holds the filter's reach. The masks have an edge
/// that crosses the border of each window, and the windows begin on odd
/// pixels, off the cell grid.
#[test]
fn a_refined_mask_under_a_crop_window_and_a_zoomed_window_matches_the_full_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let (width, height) = (1100, 600);
    let photo = blocky_photo(width, height);
    // A radial gradient with a hard rim 6 pixels over the block edge at
    // column 480 and over the row edge at 180, which crosses the left border
    // of what the windows below show, and a stroke with a hard rim 5 pixels
    // over the same block edge, which crosses their top and bottom borders.
    // At 100 percent the radial is refined in cells of two pixels and at 200
    // percent in cells of four.
    let radial = MaskSource::Radial(RadialGradient {
        centre: [0.393, 0.25],
        radius: [0.049, 0.0327],
        rotation: 0.0,
        feather: 12.0,
    });
    let brush = brush_source(&[stroke(&[[0.425, 0.05], [0.425, 0.5]], 0.016, 20.0, 100.0)]);
    let masks = vec![
        refined_at(exposure_mask("Radial", radial), 100.0, 0.03, 50.0),
        refined_at(exposure_mask("Brush", brush), 90.0, 0.02, 70.0),
    ];
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);

    // The crop window carries the masks alone, as the crop tests of the other
    // sources do.
    let crop = CropRect {
        x: 0.3,
        y: 0.2,
        width: 0.3,
        height: 0.4,
    };
    let output = (330, 240);
    let masks_only = masked(masks.clone());
    let full = develop
        .render(&masks_only, crop, (width, height), output)
        .expect("a source is set");
    let full = readback.read(&gpu.device, &gpu.queue, full, output.0, output.1);
    let windowed = develop
        .render_crop(&masks_only, crop, output)
        .expect("a source is set");
    let windowed = readback.read(&gpu.device, &gpu.queue, windowed, output.0, output.1);
    let max = max_difference(&full, &windowed);
    println!("refined masks under a crop window against the full render: max difference {max}");
    assert!(max <= 1, "max difference {max}");

    let mut edit = everything_global();
    edit.masks = masks;
    let mut unrefined = edit.clone();
    for mask in &mut unrefined.masks {
        mask.refine = Refine::default();
    }
    // 100 and 200 percent of a fit of 550 by 300.
    let views = [
        ViewWindow {
            full: (550, 300),
            window: (181, 41, 130, 100),
            visible: (201, 57, 90, 70),
        },
        ViewWindow {
            full: (width, height),
            window: (361, 81, 260, 200),
            visible: (401, 113, 180, 140),
        },
    ];
    for view in views {
        let full = full_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let zoomed = view_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let without = view_render_of(&mut develop, &gpu, &readback, &unrefined, &view);
        let max = max_difference(&full, &zoomed);
        println!(
            "refined masks under a zoomed window at {:?} against the full render: max difference {max}",
            view.full
        );
        assert!(max <= 1, "max difference {max} at {:?}", view.full);
        let moved = zoomed
            .as_chunks::<4>()
            .0
            .iter()
            .zip(without.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count();
        println!("refine edges moves {moved} pixels of the window");
        assert!(moved > 200, "refine edges shows inside the window");
    }
}

/// example.heic, 1280 by 854 pixels: the church tower of case 9 of the T-19
/// design pass stands about column 822 from row 85 to 225, dark slate over
/// pale stone, with sky on both sides.
fn tower_heic() -> Photo {
    let photo = open_photo(&fixtures::require("example.heic")).expect("open the fixture");
    assert_eq!((photo.width, photo.height), (1280, 854));
    photo
}

/// Case 9's stroke: one stroke of radius 70 pixels and feather 50 down the
/// tower, from (820, 100) to (822, 235) in pixels of the photo, as the T-19
/// design pass places it (t18-design-pass/cases.py), refined at `radius`.
fn stroke_down_the_heic_tower(radius: f32) -> PhotoEdit {
    let (w, h) = (1280.0, 854.0);
    let tower = brush_source(&[stroke(
        &[[820.0 / w, 100.0 / h], [822.0 / w, 235.0 / h]],
        70.0 / w,
        50.0,
        100.0,
    )]);
    masked(vec![refined_at(
        exposure_mask("Tower", tower),
        100.0,
        radius,
        50.0,
    )])
}

/// Case 9 of the T-19 design pass: a stroke drawn much wider than the tower
/// it covers, with sky on both sides, refined at Radius 0.01, 0.03 and 0.05.
/// The reached field lets the sky the outside reaches leave the mask; the
/// GPU matches the twin at each radius.
#[test]
fn case_9s_stroke_down_the_tower_refined_at_three_radii_matches() {
    let photo = tower_heic();
    let size = (photo.width, photo.height);
    // Radius, then the side of a cell, the box and the flood in cells: 12.8
    // pixels in cells of 1; 38.4 in cells of 4; 64 in cells of 4.
    for (radius, plan_of) in [(0.01, (1, 9, 12)), (0.03, (4, 7, 9)), (0.05, (4, 11, 16))] {
        let edit = stroke_down_the_heic_tower(radius);
        let plan = Plan::for_geometry(&edit.masks[0].refine, &Geometry::full(size, size));
        assert_eq!((plan.step, plan.cells, plan.flood), plan_of);
        // The spill beside the tower is a small part of the whole photo, so
        // the share assert_the_refine_shows asks of a render is not asked
        // here: the twin has to move some pixels at each radius.
        let mut plain = edit.clone();
        plain.masks[0].refine = Refine::default();
        let with = cpu_reference(&photo, &edit, Rounding::Nearest, 0);
        let without = cpu_reference(&photo, &plain, Rounding::Nearest, 0);
        let moved = with.iter().zip(&without).filter(|(a, b)| a != b).count();
        let name = format!("case 9's stroke down the tower at Radius {radius}");
        println!("{name}: refine edges moves {moved} pixels of the twin");
        assert!(moved > 0, "{name}: refine edges moves no pixel");
        check_on(&name, &photo, &edit);
    }
}

/// Case 9's stroke at Radius 0.05 under a scratch budget of 150 by 150 cells
/// is drawn in tiles, and the tiles give byte for byte what one tile gives.
/// The grid is 320 by 214 cells of 4 pixels, 68,480 cells of 232 bytes, over
/// the budget. The margin is 3 boxes of 11, the 2 cells beside of the
/// gathers, the cell beside of the reached field and its flood of 16 cells,
/// 33 + 2 + 1 + 16 = 52, so a tile is the floor of 2 x 52 + 64 = 168 cells a
/// side and writes (168 - 2 x 52 - 3) x 4 = 244 pixels a row and a column:
/// 6 tiles across 1280 pixels and 4 down 854, 24 in all.
#[test]
fn case_9s_stroke_in_tiles_of_a_small_budget_gives_what_one_tile_gives() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = tower_heic();
    let size = (photo.width, photo.height);
    let edit = stroke_down_the_heic_tower(0.05);
    let plan = Plan::for_geometry(&edit.masks[0].refine, &Geometry::full(size, size));
    assert_eq!((plan.step, plan.cells, plan.flood), (4, 11, 16));
    assert_eq!(plan.margin(), 3 * 11 + 2 + 1 + 16);
    assert_eq!(plan.grid().1, (320, 214));
    let cell_bytes = 232u64;
    let budget = 150 * 150 * cell_bytes;
    assert!(
        320 * 214 * cell_bytes > budget,
        "the budget is under the grid"
    );
    let side = 150u32.max(2 * plan.margin() + 64);
    assert_eq!(side, 168);
    let span = (side - 2 * plan.margin() - 3) * plan.step;
    assert_eq!(span, 244);
    let cut = 1280u32.div_ceil(span) * 854u32.div_ceil(span);
    assert_eq!(cut, 24);
    let readback = Readback::new(&gpu.device);
    // A graph of its own for each budget, so its counters read the tiles of
    // this render alone.
    let render = |budget: Option<u64>| -> (Vec<u8>, u32) {
        let develop = Develop::new(&gpu.device, &gpu.queue);
        let mut develop = match budget {
            Some(bytes) => develop.with_refine_scratch_budget(bytes),
            None => develop,
        };
        develop.set_source(&photo);
        let view = develop
            .render(&edit, CropRect::FULL, size, size)
            .expect("a source is set");
        let pixels = readback.read(&gpu.device, &gpu.queue, view, size.0, size.1);
        (pixels, develop.refine_tiles())
    };
    let (tiled, tiles) = render(Some(budget));
    let (whole, one_tile) = render(None);
    let differing = tiled.iter().zip(&whole).filter(|(a, b)| a != b).count();
    println!(
        "case 9's stroke at Radius 0.05: refine_tiles {tiles} then {one_tile}; {differing} bytes of the tiles differ from one tile"
    );
    assert_eq!((tiles, one_tile), (cut, 1));
    assert_eq!(tiled.len(), whole.len());
    assert_eq!(differing, 0, "the tiles differ from one tile");
}

/// A zoomed viewer on case 9's tower at Radius 0.01, 0.03 and 0.05: a
/// window of the tower padded by the reach of Refine edges, which holds the
/// flood of the reached field, gives what the full render gives on the
/// pixels it shows.
#[test]
fn a_zoomed_window_on_case_9s_tower_matches_the_full_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = tower_heic();
    let full = (photo.width, photo.height);
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    // The tower and the sky beside it, on odd pixels off the cell grid.
    let visible = (761, 81, 121, 161);
    for radius in [0.01, 0.03, 0.05] {
        let edit = stroke_down_the_heic_tower(radius);
        let mut unrefined = edit.clone();
        unrefined.masks[0].refine = Refine::default();
        let reach = gamut_color::refine::reach(&edit.masks[0].refine.sanitised(), full);
        let view = ViewWindow {
            full,
            window: gamut_gpu::develop::padded_window(full, visible, (reach, reach), 1),
            visible,
        };
        let whole = full_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let zoomed = view_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let without = view_render_of(&mut develop, &gpu, &readback, &unrefined, &view);
        let max = max_difference(&whole, &zoomed);
        let mean = whole
            .as_chunks::<4>()
            .0
            .iter()
            .zip(zoomed.as_chunks::<4>().0)
            .flat_map(|(a, b)| (0..3).map(move |k| f64::from(a[k].abs_diff(b[k]))))
            .sum::<f64>()
            / (whole.len() / 4 * 3) as f64;
        let moved = zoomed
            .as_chunks::<4>()
            .0
            .iter()
            .zip(without.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "case 9's tower in a zoomed window at Radius {radius}, a reach of {reach} pixels: max difference {max}, mean {mean:.4}; refine edges moves {moved} pixels of the window"
        );
        assert!(
            max <= MAX_DIFFERENCE,
            "Radius {radius}: max difference {max}"
        );
        assert!(
            mean <= MEAN_DIFFERENCE,
            "Radius {radius}: mean difference {mean}"
        );
        assert!(
            moved > 0,
            "Radius {radius}: refine edges shows inside the window"
        );
    }
}

/// Portrait_8.jpg, 1200 by 1800 pixels once turned upright: a field under a
/// sky, with the horizon about row 1030 over columns 1032 to 1127, where
/// cases 1a and 8 of the T-18 benchmark place their radial.
fn portrait_8() -> Photo {
    let photo = open_photo(&fixtures::require("Portrait_8.jpg")).expect("open the fixture");
    assert_eq!((photo.width, photo.height), (1200, 1800));
    photo
}

/// The radial of cases 1a and 8 (t18-design-pass/benchlib.py `ellipse`):
/// centre (1080, 1130), 360 pixels across and 100 + `rim_over` down, feather
/// 10, lifting the exposure one stop, refined at `radius` with an amount of
/// 100 and a sensitivity of 50.
fn radial_at_the_horizon(rim_over: f32, radius: f32) -> PhotoEdit {
    let long = 1800.0;
    let mut mask = Mask::new(
        "Radial",
        MaskSource::Radial(RadialGradient {
            centre: [1080.0 / 1200.0, 1130.0 / 1800.0],
            radius: [360.0 / long, (100.0 + rim_over) / long],
            rotation: 0.0,
            feather: 10.0,
        }),
    );
    mask.adjust.exposure = 1.0;
    masked(vec![refined_at(mask, 100.0, radius, 50.0)])
}

/// The rows and columns of Portrait_8.jpg benchlib reads a case at: the
/// columns 1032 to 1127 of `c1_regions`, and the gap of case 8, 4 to 16 rows
/// under the horizon, or the sky row of case 1, 10 to 26 rows over it.
const HORIZON_COLUMNS: std::ops::Range<u32> = 1032..1128;
const GAP_ROWS: std::ops::Range<u32> = 1034..1046;
const RIM_ROWS: std::ops::Range<u32> = 1004..1020;

/// The indices of the pixels of a region of a photo `width` pixels wide.
fn region(width: u32, rows: std::ops::Range<u32>) -> Vec<usize> {
    rows.flat_map(|y| HORIZON_COLUMNS.map(move |x| (y * width + x) as usize))
        .collect()
}

/// Reads an R8Unorm alpha of `size` one byte a pixel, code for code: the
/// readback writes 8-bit sRGB, which folds the codes over 75 together.
fn read_alpha(gpu: &Headless, source: &wgpu::TextureView, size: (u32, u32)) -> Vec<u8> {
    const SHADER: &str = "
        @group(0) @binding(0) var source: texture_2d<f32>;
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
            let x = f32(i32(index & 1u) * 4 - 1);
            let y = f32(i32(index & 2u) * 2 - 1);
            return vec4<f32>(x, y, 0.0, 1.0);
        }
        @fragment
        fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(textureLoad(source, vec2<i32>(position.xy), 0).r, 0.0, 0.0, 1.0);
        }
    ";
    let device = &gpu.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("alpha reader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("alpha reader layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("alpha reader"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("alpha reader"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            }),
        ),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let extent = wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("alpha reader target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("alpha reader"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(source),
        }],
    });
    let row =
        size.0.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alpha reader buffer"),
        size: u64::from(row) * u64::from(size.1),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("alpha reader"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("alpha reader"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(size.1),
            },
        },
        extent,
    );
    let submission = gpu.queue.submit(Some(encoder.finish()));
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("wait for the alpha");
    receiver
        .recv()
        .expect("map callback ran")
        .expect("map the alpha");
    let alpha = {
        let mapped = buffer.get_mapped_range(..).expect("mapped alpha");
        mapped
            .chunks_exact(row as usize)
            .flat_map(|line| line[..size.0 as usize].to_vec())
            .collect()
    };
    buffer.unmap();
    alpha
}

/// What the GPU draws of an edit with one refined mask.
struct RefineReadings {
    /// The render.
    refined: Vec<[u8; 3]>,
    /// The render with Refine edges off.
    unrefined: Vec<[u8; 3]>,
    /// The refined alpha of mask 0, one byte a pixel.
    alpha: Vec<u8>,
}

/// [`RefineReadings`] of `edit` on `photo`, or `None` when the machine has no
/// adapter.
fn gpu_refine_readings(photo: &Photo, edit: &PhotoEdit) -> Option<RefineReadings> {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = Headless::new()?;
    let size = (photo.width, photo.height);
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(photo);
    let render = |develop: &mut Develop, edit: &PhotoEdit| -> Vec<[u8; 3]> {
        let view = develop
            .render(edit, CropRect::FULL, size, size)
            .expect("a source is set");
        readback
            .read(&gpu.device, &gpu.queue, view, size.0, size.1)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| [px[0], px[1], px[2]])
            .collect()
    };
    let refined = render(&mut develop, edit);
    let (view, frame) = develop.mask_product(0).expect("mask 0 is drawn");
    assert_eq!(frame, size);
    let alpha = read_alpha(&gpu, view, size);
    let mut plain = edit.clone();
    plain.masks[0].refine = Refine::default();
    let unrefined = render(&mut develop, &plain);
    Some(RefineReadings {
        refined,
        unrefined,
        alpha,
    })
}

/// The twin's refined alpha of mask 0 of `edit` on `photo`, as the GPU
/// stores it, one byte a pixel: the working pixels in half floats rounded
/// as `rounding` rounds, as `cpu_reference` takes them.
fn twin_refined_alpha(photo: &Photo, edit: &PhotoEdit, rounding: Rounding) -> Vec<u8> {
    let size = (photo.width, photo.height);
    let store = |v: f32| half(v, rounding);
    let stored: Vec<[f32; 3]> = photo
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| basic::decode_rgb8([px[0], px[1], px[2]], photo.source).map(store))
        .collect();
    mask_twin::alpha_image_before_the_store(
        &edit.masks[0],
        &stored,
        &Geometry::full(size, size),
        None,
        &store,
    )
    .into_iter()
    .map(|alpha| (mask_twin::stored_alpha(alpha) * 255.0).round() as u8)
    .collect()
}

/// The mean of an alpha over `pixels`, 0 to 1.
fn mean_alpha(alpha: &[u8], pixels: &[usize]) -> f64 {
    pixels.iter().map(|i| f64::from(alpha[*i])).sum::<f64>() / pixels.len() as f64 / 255.0
}

/// The largest difference over `pixels` of the GPU's alpha from the closer
/// of the twin's two, in codes: the twin takes the working pixels rounded
/// either way, as the golden tests take the GPU's half float stores.
fn most_apart(gpu: &[u8], twins: &[Vec<u8>; 2], pixels: &[usize]) -> i32 {
    pixels
        .iter()
        .map(|i| {
            twins
                .iter()
                .map(|twin| (i32::from(gpu[*i]) - i32::from(twin[*i])).abs())
                .min()
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

/// benchlib's rim row: the move of a render, the largest channel difference
/// from the photo in codes, against the move of the unrefined render, over
/// `pixels`. The largest, and how many pixels differ by more than 3 codes.
fn rim_moves(
    photo: &Photo,
    refined: &[[u8; 3]],
    unrefined: &[[u8; 3]],
    pixels: &[usize],
) -> (i32, usize) {
    let source = photo.rgba8.as_chunks::<4>().0;
    let move_of = |render: &[[u8; 3]], i: usize| -> i32 {
        (0..3)
            .map(|k| (i32::from(render[i][k]) - i32::from(source[i][k])).abs())
            .max()
            .unwrap_or(0)
    };
    let gaps: Vec<i32> = pixels
        .iter()
        .map(|i| (move_of(refined, *i) - move_of(unrefined, *i)).abs())
        .collect();
    (
        gaps.iter().copied().max().unwrap_or(0),
        gaps.iter().filter(|gap| **gap > 3).count(),
    )
}

/// Case 8 of the T-18 benchmark: the radial of layout a drawn 20 pixels short
/// of the horizon (ry 80, feather 10), lifting the exposure a stop, refined
/// at Radius 0.01, 0.03 and 0.05. The move grows the mask down into the field
/// between its rim and the horizon: the twin fills the gap to a mean alpha of
/// 0.000, 0.731 and 0.706 on the guide of the app, ACEScct of the working
/// pixels in Rec.2020. The benchmark's 0.256 and 0.209 come from its own
/// guide, ACEScct of the pixels in sRGB primaries. The GPU matches the twin
/// at each radius, and the mean alpha of the gap is printed for both.
#[test]
fn case_8s_radial_short_of_the_horizon_refined_at_three_radii_matches() {
    let photo = portrait_8();
    let size = (photo.width, photo.height);
    let gap = region(photo.width, GAP_ROWS);
    assert_eq!(gap.len(), 1152);
    // Radius, then the side of a cell, the box and the flood in cells: 18
    // pixels in cells of 2; 54 in cells of 4; 90 in cells of 4.
    for (radius, plan_of) in [(0.01, (2, 6, 9)), (0.03, (4, 10, 13)), (0.05, (4, 16, 22))] {
        let edit = radial_at_the_horizon(-20.0, radius);
        let plan = Plan::for_geometry(&edit.masks[0].refine, &Geometry::full(size, size));
        assert_eq!((plan.step, plan.cells, plan.flood), plan_of);
        let name = format!("case 8's radial short of the horizon at Radius {radius}");
        let twins = [Rounding::Nearest, Rounding::TowardZero]
            .map(|rounding| twin_refined_alpha(&photo, &edit, rounding));
        if let Some(RefineReadings { alpha: gpu, .. }) = gpu_refine_readings(&photo, &edit) {
            println!(
                "{name}: gap mean alpha gpu {:.3}, twin {:.3}; the alphas apart by at most {} codes on the gap",
                mean_alpha(&gpu, &gap),
                mean_alpha(&twins[0], &gap),
                most_apart(&gpu, &twins, &gap)
            );
        }
        check_on(&name, &photo, &edit);
    }
}

/// Layout a of case 1 of the T-18 benchmark: the radial over the horizon (ry
/// 134, feather 10), lifting the exposure a stop, refined at Radius 0.01. The
/// rim row, the sky 10 to 26 rows over the horizon, keeps some of the move;
/// benchlib reads the twin's worst pixel there at 28 codes and 64 of 1536
/// pixels over 3 codes on its own guide and move model. The GPU matches the
/// twin, and the rim row's largest move and count over 3 are printed for
/// both, with how far their alphas lie apart there.
#[test]
fn layout_as_radial_over_the_horizon_at_radius_0_01_matches() {
    let photo = portrait_8();
    let size = (photo.width, photo.height);
    let rim = region(photo.width, RIM_ROWS);
    assert_eq!(rim.len(), 1536);
    let edit = radial_at_the_horizon(34.0, 0.01);
    let plan = Plan::for_geometry(&edit.masks[0].refine, &Geometry::full(size, size));
    assert_eq!((plan.step, plan.cells, plan.flood), (2, 6, 9));
    let name = "layout a's radial over the horizon at Radius 0.01";
    let mut plain = edit.clone();
    plain.masks[0].refine = Refine::default();
    let twin_refined = cpu_reference(&photo, &edit, Rounding::Nearest, 0);
    let twin_unrefined = cpu_reference(&photo, &plain, Rounding::Nearest, 0);
    let (twin_most, twin_over) = rim_moves(&photo, &twin_refined, &twin_unrefined, &rim);
    let twins = [Rounding::Nearest, Rounding::TowardZero]
        .map(|rounding| twin_refined_alpha(&photo, &edit, rounding));
    if let Some(RefineReadings {
        refined,
        unrefined,
        alpha: gpu,
    }) = gpu_refine_readings(&photo, &edit)
    {
        let (most, over) = rim_moves(&photo, &refined, &unrefined, &rim);
        println!(
            "{name}: rim row gpu worst pixel {most} codes, {over} of {} px over 3; twin worst pixel {twin_most} codes, {twin_over} px over 3; rim mean alpha gpu {:.3}, twin {:.3}, apart by at most {} codes",
            rim.len(),
            mean_alpha(&gpu, &rim),
            mean_alpha(&twins[0], &rim),
            most_apart(&gpu, &twins, &rim)
        );
    }
    check_on(name, &photo, &edit);
}

/// `mask` with Shift edge, Feather and Contrast at these values: shares of the
/// longer side of the photo for the first two. On the 64 pixel photos a shift
/// of 0.02 is one pixel a side along each axis and a feather of 0.01 a sigma
/// of 0.64 pixels in cells of one pixel.
fn edged_at(mut mask: Mask, shift: f32, feather: f32, contrast: f32) -> Mask {
    mask.edge = Edge {
        shift,
        feather,
        contrast,
    };
    mask
}

/// The edge controls have to move the picture for their golden test to mean
/// anything. How many pixels of the twin they move.
fn assert_the_edge_shows(photo: &Photo, edit: &PhotoEdit) -> usize {
    let mut plain = edit.clone();
    for mask in &mut plain.masks {
        mask.edge = Edge::default();
    }
    let with = cpu_reference(photo, edit, Rounding::Nearest, 0);
    let without = cpu_reference(photo, &plain, Rounding::Nearest, 0);
    let moved = with.iter().zip(&without).filter(|(a, b)| a != b).count();
    assert!(
        moved * 100 > with.len(),
        "the edge controls move only {moved} of {} pixels",
        with.len()
    );
    moved
}

fn check_edged(name: &str, edit: &PhotoEdit) {
    let moved = assert_the_edge_shows(&synthetic_photo(), edit);
    println!("{name}: the edge controls move {moved} pixels of the twin");
    check_masks(name, edit);
}

fn refined_radial_edged(shift: f32, feather: f32, contrast: f32) -> PhotoEdit {
    masked(vec![edged_at(
        refined(exposure_mask("Radial", radial_over_the_band())),
        shift,
        feather,
        contrast,
    )])
}

#[test]
fn a_refined_mask_grown_by_shift_edge_matches() {
    check_edged(
        "refined mask, Shift edge 2 percent",
        &refined_radial_edged(0.02, 0.0, 0.0),
    );
}

#[test]
fn a_refined_mask_shrunk_by_shift_edge_matches() {
    check_edged(
        "refined mask, Shift edge -2 percent",
        &refined_radial_edged(-0.02, 0.0, 0.0),
    );
}

/// A render of `edit` on `photo` with the work textures of Shift edge held to
/// `side` pixels a side, or to the device's limit for `None`, and the pad
/// those textures took.
fn gpu_render_at_pad_limit(
    gpu: &Headless,
    photo: &Photo,
    edit: &PhotoEdit,
    side: Option<u32>,
) -> (Vec<[u8; 3]>, Option<u32>) {
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    if let Some(side) = side {
        develop = develop.with_edge_pad_limit(side);
    }
    develop.set_source(photo);
    let size = (photo.width, photo.height);
    let view = develop
        .render(edit, CropRect::FULL, size, size)
        .expect("a source is set");
    let pixels = Readback::new(&gpu.device)
        .read(&gpu.device, &gpu.queue, view, size.0, size.1)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| [px[0], px[1], px[2]])
        .collect();
    (pixels, develop.edge_pad())
}

/// A radial gradient over the top left corner whose soft rim crosses the top
/// and the left edge of the frame at a slant, so the runs near those edges
/// read an alpha that changes along them and across them.
fn radial_across_a_corner() -> MaskSource {
    MaskSource::Radial(RadialGradient {
        centre: [0.12, 0.18],
        radius: [0.35, 0.3],
        rotation: 0.4,
        feather: 60.0,
    })
}

/// Where the frame and two pads would pass the device's limit on a side, the
/// pad of the work textures is cut to what fits, and a run whose half is
/// wider takes its samples one by one at the pixels whose reads of the table
/// would leave the padded texture. Each frame, with the pad cut to 0, to half
/// the widest half and to one less than it, gives the render at the full pad
/// byte for byte and matches the twin.
///
/// The frames: those of the grown and the shrunk golden (Shift edge 2
/// percent of the 64 pixel photos, axis runs of half 1, so half of it is 0
/// too), the same mask at 5 percent of the synthetic photo drawn 4 times
/// over (256 pixels, halves of 5 and 4), and a mask across a corner at both
/// sizes, whose rim meets the edges where the loop runs.
#[test]
fn a_shifted_mask_at_a_cut_pad_matches() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let corner = |shift: f32| {
        masked(vec![edged_at(
            refined(exposure_mask("Corner", radial_across_a_corner())),
            shift,
            0.0,
            0.0,
        )])
    };
    let mut cases: Vec<(String, Photo, PhotoEdit)> = Vec::new();
    for shift in [0.02, -0.02] {
        let edit = refined_radial_edged(shift, 0.0, 0.0);
        cases.push((
            format!("golden frame {shift}"),
            synthetic_photo(),
            edit.clone(),
        ));
        cases.push((
            format!("golden frame {shift}, hazy photo"),
            hazy_photo(),
            edit,
        ));
    }
    for shift in [0.05, -0.05] {
        cases.push((
            format!("golden mask {shift}, 256 pixels"),
            synthetic_photo_times(4),
            refined_radial_edged(shift, 0.0, 0.0),
        ));
        cases.push((format!("corner {shift}"), synthetic_photo(), corner(shift)));
        cases.push((
            format!("corner {shift}, 256 pixels"),
            synthetic_photo_times(4),
            corner(shift),
        ));
    }
    for (case, photo, edit) in &cases {
        let size = (photo.width, photo.height);
        let moved = assert_the_edge_shows(photo, edit);
        let plan = gamut_color::edge::Plan::new(&edit.masks[0].edge, size, (0, 0), size);
        let widest = plan.axis.max(plan.diagonal);
        assert!(widest > 0, "{case}: Shift edge takes a step");
        let (full, full_pad) = gpu_render_at_pad_limit(&gpu, photo, edit, None);
        assert_eq!(
            full_pad,
            Some(widest),
            "{case}: the full pad is the widest half"
        );
        let mut pads = vec![0, widest / 2, widest - 1];
        pads.dedup();
        for pad in pads {
            let name = format!(
                "{case}, runs of {} and {}, pad cut to {pad} of {widest}",
                plan.axis, plan.diagonal
            );
            let side = size.0.max(size.1) + 2 * pad;
            let (cut, took) = gpu_render_at_pad_limit(&gpu, photo, edit, Some(side));
            assert_eq!(took, Some(pad), "{name}: the pad is cut");
            let differing = full.iter().zip(&cut).filter(|(a, b)| a != b).count();
            println!(
                "{name}: the edge controls move {moved} pixels of the twin; {differing} of {} pixels differ from the full pad",
                full.len()
            );
            assert_eq!(differing, 0, "{name}: the render at the full pad");
            assert_matches_the_twin(&name, photo, edit, &cut);
        }
    }
}

#[test]
fn a_refined_mask_under_feather_matches() {
    check_edged(
        "refined mask, Feather 1 percent",
        &refined_radial_edged(0.0, 0.01, 0.0),
    );
}

#[test]
fn a_refined_mask_under_contrast_matches() {
    check_edged(
        "refined mask, Contrast 80",
        &refined_radial_edged(0.0, 0.0, 80.0),
    );
}

#[test]
fn a_refined_mask_under_all_three_edge_controls_matches() {
    check_edged(
        "refined mask, the three together",
        &refined_radial_edged(-0.02, 0.01, 50.0),
    );
    check_edged(
        "refined mask, the three together, grown",
        &refined_radial_edged(0.03, 0.02, 90.0),
    );
}

/// With Refine edges off the chain reads the alpha of the components.
#[test]
fn the_edge_controls_on_an_unrefined_brush_mask_match() {
    check_edged(
        "unrefined brush mask, the three together",
        &masked(vec![edged_at(
            exposure_mask("Brush", stroke_along_the_grey_columns()),
            0.02,
            0.01,
            50.0,
        )]),
    );
}

#[test]
fn an_edged_mask_that_is_inverted_and_one_at_half_opacity_match() {
    let mut inverted = edged_at(
        refined(exposure_mask("Outside", radial_over_the_band())),
        0.02,
        0.01,
        60.0,
    );
    inverted.invert = true;
    check_edged("edged inverted mask", &masked(vec![inverted]));
    let mut half = edged_at(
        refined(exposure_mask("Half", radial_over_the_band())),
        -0.02,
        0.01,
        60.0,
    );
    half.opacity = 50.0;
    check_edged("edged mask at opacity 50", &masked(vec![half]));
}

#[test]
fn an_edged_refined_auto_brush_matches() {
    let auto = refined(exposure_mask("Auto", brush_source(&auto_strokes(70.0))));
    // The gate of the auto brush shows under Refine edges, as it does in
    // a_refined_auto_brush_matches, before the edge controls act on it.
    assert_the_gate_shows(&masked(vec![auto.clone()]));
    let edit = masked(vec![edged_at(auto, 0.02, 0.01, 60.0)]);
    check_edged("edged refined auto brush", &edit);
}

#[test]
fn two_edged_masks_and_a_plain_one_blend_in_list_order() {
    let edit = masked(vec![
        edged_at(
            refined_at(
                exposure_mask("Radial", radial_over_the_band()),
                80.0,
                0.03,
                70.0,
            ),
            -0.02,
            0.01,
            40.0,
        ),
        exposure_mask("Linear", linear_source()),
        edged_at(
            exposure_mask("Brush", stroke_along_the_grey_columns()),
            0.03,
            0.0,
            70.0,
        ),
    ]);
    check_edged("two edged masks and a plain one", &edit);
}

/// The overlay shows the finished alpha, and an export never shows it.
#[test]
fn the_overlay_of_an_edged_mask_matches_and_stays_out_of_an_export() {
    check_overlay(edged_at(
        refined(Mask::new("Idle", radial_source())),
        0.02,
        0.01,
        50.0,
    ));
    check_overlay(edged_at(
        Mask::new("Idle", radial_source()),
        -0.03,
        0.0,
        0.0,
    ));
}

/// The overlay of an edged refined mask whose rim spills into the near-black
/// band, where one code of alpha is five or six codes of the overlay. The
/// raw alpha, the refined alpha and the finished alpha are each an r8unorm
/// store, and an alpha beside a half code may land on either code at each.
#[test]
fn the_overlay_of_an_edged_refined_mask_over_the_band_matches() {
    check_overlay_at(
        edged_at(
            refined(Mask::new("Idle", radial_over_the_band())),
            0.02,
            0.01,
            50.0,
        ),
        (SIZE / 2, SIZE * 78 / 100),
    );
}

/// A mask whose three edge keys are in the file at 0, or at values sanitising
/// takes to 0, draws byte for byte what the mask with no edge key draws, with
/// Refine edges off and on, and runs no edge pass and holds no edge texture.
#[test]
fn edge_controls_at_0_draw_what_no_edge_draws_and_run_no_pass() {
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
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let render = |develop: &mut Develop, edit: &PhotoEdit| -> Vec<u8> {
        let view = develop
            .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
            .expect("a source is set");
        readback.read(&gpu.device, &gpu.queue, view, SIZE, SIZE)
    };
    for refine in [false, true] {
        let mut edit = everything_global();
        edit.masks = vec![
            exposure_mask("Radial", radial_over_the_band()),
            exposure_mask("Brush", stroke_along_the_grey_columns()),
        ];
        if refine {
            edit.masks = edit.masks.into_iter().map(refined).collect();
        }
        let plain = render(&mut develop, &edit);
        // The same masks read back from a file that holds the three keys.
        let text = gamut_core::Sidecar::new(edit.clone(), Default::default())
            .to_json()
            .replace(
                "\"adjust\": {",
                "\"edge\": {\"shift\": 0, \"feather\": 0, \"contrast\": 0},\n\"adjust\": {",
            );
        let keyed = gamut_core::Sidecar::from_json(&text).expect("parse").edit;
        assert!(text.contains("\"edge\""));
        assert_eq!(keyed, edit, "the keys at 0 are the default");
        assert_eq!(render(&mut develop, &keyed), plain, "refine {refine}");
        let mut sanitised = edit.clone();
        for mask in &mut sanitised.masks {
            mask.edge = Edge {
                shift: -0.0,
                feather: -0.3,
                contrast: f32::NAN,
            };
        }
        assert_eq!(render(&mut develop, &sanitised), plain, "refine {refine}");
        assert_eq!(develop.edge_passes(), EdgePasses::default());
        assert_eq!(develop.edge_textures(), 0);
    }
}

/// The edge products are cached like the refined alpha under them: a develop
/// slider draws none of their passes; a Shift edge slider draws Shift edge,
/// Feather and the finished alpha of that mask and no alpha and no refine; a
/// Feather slider draws Feather and the finished alpha; a Contrast slider the
/// finished alpha alone; a new source content draws all of them.
#[test]
fn a_develop_slider_draws_no_edge_pass_and_each_edge_slider_draws_its_own() {
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
    // The alphas drawn, the refined alphas drawn, and the passes of Shift
    // edge, of Feather and of the finished alpha, since the render before.
    let mut last = (0u64, 0u64, EdgePasses::default());
    let mut drawn = |develop: &mut Develop, edit: &PhotoEdit| -> (u64, u64, [u64; 3]) {
        develop
            .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
            .expect("a source is set");
        let now = (
            develop.mask_alpha_builds(),
            develop.refine_builds().0,
            develop.edge_passes(),
        );
        let step = (
            now.0 - last.0,
            now.1 - last.1,
            [
                now.2.shift - last.2.shift,
                now.2.feather - last.2.feather,
                now.2.finish - last.2.finish,
            ],
        );
        last = now;
        step
    };
    let mut edit = masked(vec![
        edged_at(
            refined(exposure_mask("Radial", radial_over_the_band())),
            0.02,
            0.01,
            50.0,
        ),
        exposure_mask("Luminance", luminance_source()),
    ]);
    // Shift edge of one pixel: a run along x and one along y, each one pass
    // forward and one back.
    assert_eq!(
        drawn(&mut develop, &edit),
        (2, 1, [4, 3, 1]),
        "the first render"
    );
    assert_eq!(
        drawn(&mut develop, &edit),
        (0, 0, [0, 0, 0]),
        "the same edit again"
    );

    edit.exposure = 0.7;
    edit.masks[0].adjust.exposure = -0.5;
    edit.masks[0].adjust.look.curves.master = s_curve();
    edit.masks[0].opacity = 35.0;
    assert_eq!(
        drawn(&mut develop, &edit),
        (0, 0, [0, 0, 0]),
        "sliders of the develop chain, global and of the mask"
    );

    edit.masks[0].edge.shift = -0.02;
    assert_eq!(
        drawn(&mut develop, &edit),
        (0, 0, [4, 3, 1]),
        "a Shift edge slider: Shift edge, Feather and Contrast, no alpha and no refine"
    );
    edit.masks[0].edge.shift = -0.05;
    let deeper = drawn(&mut develop, &edit);
    assert_eq!((deeper.0, deeper.1, &deeper.2[1..]), (0, 0, &[3, 1][..]));
    assert!(deeper.2[0] > 4, "a wider shift takes more doubling passes");
    edit.masks[0].edge.feather = 0.03;
    assert_eq!(
        drawn(&mut develop, &edit),
        (0, 0, [0, 3, 1]),
        "a Feather slider: Feather and Contrast"
    );
    edit.masks[0].edge.contrast = 80.0;
    assert_eq!(
        drawn(&mut develop, &edit),
        (0, 0, [0, 0, 1]),
        "a Contrast slider: the finished alpha alone"
    );
    edit.masks[0].refine.amount = 70.0;
    assert_eq!(
        drawn(&mut develop, &edit),
        (0, 1, [deeper.2[0], 3, 1]),
        "a Refine slider refines again and runs the chain after it"
    );

    develop.set_source(&hazy_photo());
    let fresh = drawn(&mut develop, &edit);
    assert_eq!(
        (fresh.0, fresh.1),
        (2, 1),
        "a new source content draws the alphas"
    );
    assert_eq!(fresh.2, [deeper.2[0], 3, 1], "and every edge pass");

    // All three at rest again: the passes stop and the textures go.
    edit.masks[0].edge = Edge::default();
    drawn(&mut develop, &edit);
    assert_eq!(develop.edge_textures(), 0);
    assert_eq!(drawn(&mut develop, &edit).2, [0, 0, 0]);
}

/// A Feather slider whose sigma crosses a cell step makes the cells of
/// Feather again at the size of the new grid, and still draws Feather and the
/// finished alpha of that mask alone: the shifted alpha does not depend on
/// the cell grid and is kept. So does a Feather slider that comes on from 0
/// and makes the cells again. On the photo 1101 pixels wide a feather of
/// 0.015 is a sigma of 16.5 pixels in cells of 4, and one of 0.012 a sigma
/// of 13.2 in cells of 3, a larger grid. Every render matches the twin.
#[test]
fn a_feather_slider_across_a_cell_step_draws_no_shift_pass_and_matches() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = blocky_photo(1101, 90);
    let size = (photo.width, photo.height);
    let edge_at = |feather: f32, contrast: f32| Edge {
        shift: 0.005,
        feather,
        contrast,
    };
    let plan = |edge: Edge| gamut_color::edge::Plan::new(&edge, size, (0, 0), size);
    let (four, three) = (plan(edge_at(0.015, 30.0)), plan(edge_at(0.012, 30.0)));
    assert!(four.shifts() && three.shifts());
    assert_eq!((four.step, three.step), (4, 3));
    let (grid_four, grid_three) = (four.grid().1, three.grid().1);
    assert!(
        grid_three.0 > grid_four.0 && grid_three.1 > grid_four.1,
        "cells of 3 take a larger cell texture: {grid_four:?} against {grid_three:?}"
    );

    let radial = MaskSource::Radial(RadialGradient {
        centre: [0.4995, 0.5],
        radius: [0.0309, 0.05],
        rotation: 0.0,
        feather: 15.0,
    });
    let brush = brush_source(&[stroke(&[[0.2543, 0.1], [0.2543, 0.9]], 0.012, 20.0, 100.0)]);
    let mut radial = refined_at(exposure_mask("Radial", radial), 100.0, 0.015, 50.0);
    radial.edge = edge_at(0.015, 30.0);
    let mut brush = exposure_mask("Brush", brush);
    brush.edge = edge_at(0.015, 30.0);
    let mut edit = masked(vec![radial, brush]);
    let moved = assert_the_edge_shows(&photo, &edit);
    println!("the edge controls move {moved} pixels of the twin");

    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    // What one render drew: the mask alphas whole and in part, the refined
    // alphas whole and in part, the passes of Shift edge, of Feather's cells
    // and of the finished alpha, and each of those three stages drawn whole
    // and in part. The render is held to the twin.
    type Drawn = ([u64; 2], [u64; 2], [u64; 3], [u64; 3], [u64; 3]);
    let counts = |develop: &Develop| -> Drawn {
        let passes = develop.edge_passes();
        let (whole, parts) = develop.edge_builds();
        let (refined, refine_parts) = develop.refine_builds();
        (
            [develop.mask_alpha_builds(), develop.brush_patches().0],
            [refined, refine_parts],
            [passes.shift, passes.feather, passes.finish],
            whole,
            parts,
        )
    };
    let render = |develop: &mut Develop, edit: &PhotoEdit, name: &str| -> Drawn {
        let before = counts(develop);
        let view = develop
            .render(edit, CropRect::FULL, size, size)
            .expect("a source is set");
        let pixels: Vec<[u8; 3]> = readback
            .read(&gpu.device, &gpu.queue, view, size.0, size.1)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| [px[0], px[1], px[2]])
            .collect();
        assert_matches_the_twin(name, &photo, edit, &pixels);
        let after = counts(develop);
        let since = |now: &[u64], then: &[u64]| -> Vec<u64> {
            now.iter().zip(then).map(|(now, then)| now - then).collect()
        };
        (
            since(&after.0, &before.0).try_into().expect("two"),
            since(&after.1, &before.1).try_into().expect("two"),
            since(&after.2, &before.2).try_into().expect("three"),
            since(&after.3, &before.3).try_into().expect("three"),
            since(&after.4, &before.4).try_into().expect("three"),
        )
    };

    let first = render(&mut develop, &edit, "the first render, cells of 4");
    assert!(first.2[0] > 0, "Shift edge draws on the first render");
    assert_eq!(first.3, [2, 2, 2], "each stage of both masks drawn whole");

    // Only Feather and the finished alpha of the radial mask, whole, and no
    // other pass: no alpha, no refine and no Shift edge.
    let feather_alone = ([0, 0], [0, 0], [0, 3, 1], [0, 1, 1], [0, 0, 0]);
    edit.masks[0].edge.feather = 0.012;
    assert_eq!(
        render(&mut develop, &edit, "Feather from cells of 4 to cells of 3"),
        feather_alone,
        "a Feather slider from cells of 4 to cells of 3"
    );
    edit.masks[0].edge.feather = 0.015;
    assert_eq!(
        render(&mut develop, &edit, "Feather from cells of 3 to cells of 4"),
        feather_alone,
        "a Feather slider from cells of 3 to cells of 4"
    );

    // Feather of the radial mask to 0 drops its cells and draws the finished
    // alpha alone; back on, it makes the cells again.
    edit.masks[0].edge.feather = 0.0;
    assert_eq!(
        render(&mut develop, &edit, "Feather to 0"),
        ([0, 0], [0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]),
        "a Feather slider to 0"
    );
    edit.masks[0].edge.feather = 0.012;
    assert_eq!(
        render(&mut develop, &edit, "Feather from 0 to cells of 3"),
        feather_alone,
        "a Feather slider from 0 makes the cells again"
    );
}

/// The blend and the overlay read the alpha the edge controls hand on, and
/// follow it when a control comes on or goes to 0: the finished alpha, the
/// shifted alpha, the finished alpha made again, and the alpha under the
/// three. One Develop renders the steps in turn, first the picture and then
/// the overlay of the mask held on, with Refine edges off and then on. Every
/// render matches the twin.
#[test]
fn the_blend_and_the_overlay_follow_each_edge_control_on_and_off() {
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
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);
    let render = |develop: &mut Develop, edit: &PhotoEdit| -> Vec<[u8; 3]> {
        let view = develop
            .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
            .expect("a source is set");
        readback
            .read(&gpu.device, &gpu.queue, view, SIZE, SIZE)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| [px[0], px[1], px[2]])
            .collect()
    };
    // Shift edge, Feather and Contrast at each step, and the alpha the blend
    // reads after it.
    let steps = [
        (0.02, 0.01, 50.0, "the three on, the finished alpha"),
        (
            0.02,
            0.0,
            0.0,
            "Feather and Contrast to 0, the shifted alpha",
        ),
        (
            0.02,
            0.0,
            50.0,
            "Contrast on alone, the finished alpha made again",
        ),
        (
            0.02,
            0.01,
            0.0,
            "Feather on and Contrast to 0, the finished alpha",
        ),
        (0.0, 0.0, 0.0, "the three to 0, the alpha under them"),
        (0.02, 0.0, 0.0, "Shift edge on again, the shifted alpha"),
    ];
    // A mask Refine edges moves, clear of the near-black band, where one code
    // of alpha is six of the overlay.
    for refine in [false, true] {
        let mask = exposure_mask("Brush", stroke_along_the_grey_columns());
        let mask = if refine { refined(mask) } else { mask };
        if refine {
            let moved = assert_the_refine_shows(&photo, &masked(vec![mask.clone()]));
            println!("Refine edges moves {moved} pixels of the twin");
        }
        let edits: Vec<(String, PhotoEdit)> = steps
            .iter()
            .map(|&(shift, feather, contrast, step)| {
                let edit = masked(vec![edged_at(mask.clone(), shift, feather, contrast)]);
                (format!("refine {refine}, {step}"), edit)
            })
            .collect();
        for (name, edit) in &edits {
            if edit.masks[0].edge != Edge::default() {
                let moved = assert_the_edge_shows(&photo, edit);
                println!("{name}: the edge controls move {moved} pixels of the twin");
            }
        }
        // The overlay stays on through its steps, so each change of the alpha
        // reaches a bind made before it, as it does for the blend.
        develop.set_overlay(None);
        for (name, edit) in &edits {
            let pixels = render(&mut develop, edit);
            assert_matches_the_twin(name, &photo, edit, &pixels);
        }
        develop.set_overlay(Some(0));
        for (name, edit) in &edits {
            let overlaid = render(&mut develop, edit);
            assert_the_overlay_matches(&format!("{name}, overlay"), &photo, edit, 0, &overlaid);
        }
    }
}

/// An edged mask under a crop window and under a zoomed window equals the
/// same region of the full render: the feather cells lie on the pixels of the
/// whole picture and the window holds the reach of Refine edges and of the
/// edge controls. The masks have an edge that crosses the border of each
/// window, and the windows begin on odd pixels, off the cell grid.
#[test]
fn an_edged_mask_under_a_crop_window_and_a_zoomed_window_matches_the_full_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let (width, height) = (1100, 600);
    let photo = blocky_photo(width, height);
    // The masks of the refined window test: a radial gradient whose rim
    // crosses the left border of what the windows show, and a stroke that
    // crosses their top and bottom borders.
    let radial = MaskSource::Radial(RadialGradient {
        centre: [0.393, 0.25],
        radius: [0.049, 0.0327],
        rotation: 0.0,
        feather: 12.0,
    });
    let brush = brush_source(&[stroke(&[[0.425, 0.05], [0.425, 0.5]], 0.016, 20.0, 100.0)]);
    let masks = vec![
        edged_at(
            refined_at(exposure_mask("Radial", radial), 100.0, 0.03, 50.0),
            0.03,
            0.02,
            40.0,
        ),
        edged_at(exposure_mask("Brush", brush), -0.03, 0.02, 0.0),
    ];
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);

    let crop = CropRect {
        x: 0.3,
        y: 0.2,
        width: 0.3,
        height: 0.4,
    };
    let output = (330, 240);
    let masks_only = masked(masks.clone());
    let full = develop
        .render(&masks_only, crop, (width, height), output)
        .expect("a source is set");
    let full = readback.read(&gpu.device, &gpu.queue, full, output.0, output.1);
    let windowed = develop
        .render_crop(&masks_only, crop, output)
        .expect("a source is set");
    let windowed = readback.read(&gpu.device, &gpu.queue, windowed, output.0, output.1);
    let max = max_difference(&full, &windowed);
    println!("edged masks under a crop window against the full render: max difference {max}");
    assert!(max <= 1, "max difference {max}");

    let mut edit = everything_global();
    edit.masks = masks;
    let mut plain = edit.clone();
    for mask in &mut plain.masks {
        mask.edge = Edge::default();
    }
    // 100 and 200 percent of a fit of 550 by 300: feather cells of two and
    // of five pixels.
    let views = [
        ViewWindow {
            full: (550, 300),
            window: (181, 41, 130, 100),
            visible: (201, 57, 90, 70),
        },
        ViewWindow {
            full: (width, height),
            window: (361, 81, 260, 200),
            visible: (401, 113, 180, 140),
        },
    ];
    for view in views {
        let plan = gamut_color::edge::Plan::new(&edit.masks[0].edge, view.full, (0, 0), view.full);
        println!("feather cells of {} pixels at {:?}", plan.step, view.full);
        assert!(plan.step > 1);
        let full = full_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let zoomed = view_render_of(&mut develop, &gpu, &readback, &edit, &view);
        let without = view_render_of(&mut develop, &gpu, &readback, &plain, &view);
        let max = max_difference(&full, &zoomed);
        println!(
            "edged masks under a zoomed window at {:?} against the full render: max difference {max}",
            view.full
        );
        assert!(max <= 1, "max difference {max} at {:?}", view.full);
        let moved = zoomed
            .as_chunks::<4>()
            .0
            .iter()
            .zip(without.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count();
        println!("the edge controls move {moved} pixels of the window");
        assert!(moved > 200, "the edge controls show inside the window");
    }
}

/// On a photo wider than 1024 pixels a feather of 0.01 is a sigma of 11
/// pixels in cells of two and one of 0.02 a sigma of 22 in cells of five: the
/// cell grid, the bilinear read and the odd last column are real.
#[test]
fn an_edged_mask_on_a_photo_wider_than_1024_pixels_matches() {
    let photo = blocky_photo(1101, 90);
    for (feather, step) in [(0.01, 2), (0.02, 5)] {
        let edge = Edge {
            shift: 0.005,
            feather,
            contrast: 30.0,
        };
        let plan = gamut_color::edge::Plan::new(&edge, (1101, 90), (0, 0), (1101, 90));
        assert_eq!(plan.step, step);
        let radial = MaskSource::Radial(RadialGradient {
            centre: [0.4995, 0.5],
            radius: [0.0309, 0.05],
            rotation: 0.0,
            feather: 15.0,
        });
        let brush = brush_source(&[stroke(&[[0.2543, 0.1], [0.2543, 0.9]], 0.012, 20.0, 100.0)]);
        let mut radial = refined_at(exposure_mask("Radial", radial), 100.0, 0.015, 50.0);
        radial.edge = edge;
        let mut brush = exposure_mask("Brush", brush);
        brush.edge = edge;
        let edit = masked(vec![radial, brush]);
        let moved = assert_the_edge_shows(&photo, &edit);
        println!(
            "edged masks on a wide photo, cells of {step}: the edge controls move {moved} pixels"
        );
        check_on(
            &format!("edged masks on a photo wider than 1024 pixels, cells of {step}"),
            &photo,
            &edit,
        );
    }
}

/// A feathered alpha takes every value between two codes, so many land within
/// a tenth of a half code, where a GPU may store either code. Under an
/// exposure that is one output code at most; the overlay's reference alone
/// accepts either code.
#[test]
fn an_edged_alpha_beside_a_half_code_matches() {
    let mask = edged_at(
        refined_at(
            exposure_mask("Beside a half code", radial_over_the_middle_and_the_band()),
            50.0,
            0.05,
            50.0,
        ),
        0.02,
        0.02,
        30.0,
    );
    let photo = synthetic_photo();
    let linear: Vec<[f32; 3]> = photo
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| basic::decode_rgb8([px[0], px[1], px[2]], photo.source))
        .collect();
    let geometry = Geometry::full((SIZE, SIZE), (SIZE, SIZE));
    let store = |v: f32| half(v, Rounding::Nearest);
    let alphas = mask_twin::alpha_image_before_the_store(&mask, &linear, &geometry, None, &store);
    let beside = alphas
        .iter()
        .filter(|alpha| {
            let codes = **alpha * 255.0;
            codes > 1.0
                && codes < 254.0
                && (codes.fract() - 0.5).abs() < mask_twin::UNORM_STEP_TOLERANCE
        })
        .count();
    println!("{beside} edged alphas lie within a tenth of a half code");
    assert!(
        beside >= REFINED_HALF_CODE_PIXELS,
        "{beside} edged alphas lie beside a half code"
    );
    check_edged(
        "edged alpha beside a half code",
        &masked(vec![mask.clone()]),
    );
    let mut idle = mask;
    idle.adjust = Adjustments::default();
    check_overlay(idle);
}
