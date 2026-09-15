//! Golden tests: every GPU operator against its CPU twin in slate-color,
//! through the same input and output transforms, on a 64 by 64 synthetic
//! photo that covers hue, brightness and neutral ramps. The tests skip,
//! and say so, when the machine has no adapter.

use std::sync::Mutex;

use half::f16;
use slate_color::SourceSpace;
use slate_color::basic;
use slate_core::{CropRect, PhotoEdit};
use slate_gpu::{Develop, Headless, Readback};
use slate_media::Photo;

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

fn cpu_reference(photo: &Photo, edit: &PhotoEdit, rounding: Rounding) -> Vec<[u8; 3]> {
    let linear: Vec<[f32; 3]> = photo
        .rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| {
            basic::decode_rgb8([px[0], px[1], px[2]], photo.source).map(|c| half(c, rounding))
        })
        .collect();
    let base = basic::base_layer(&linear, photo.width, photo.height);
    let wb = basic::white_balance_matrix(edit.white_balance_temperature, edit.white_balance_tint);
    linear
        .iter()
        .zip(&base)
        .map(|(px, b)| {
            let developed = basic::develop_pixel_with(*px, half(*b, rounding), edit, &wb)
                .map(|c| half(c, rounding));
            basic::output_srgb8(developed)
        })
        .collect()
}

fn gpu_render(gpu: &Headless, photo: &Photo, edit: &PhotoEdit) -> Vec<[u8; 3]> {
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(photo);
    let view = develop
        .render(edit, CropRect::FULL, (SIZE, SIZE), (SIZE, SIZE))
        .expect("a source is set");
    Readback::new(&gpu.device)
        .read(&gpu.device, &gpu.queue, view, SIZE, SIZE)
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
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = synthetic_photo();
    let nearest = cpu_reference(&photo, edit, Rounding::Nearest);
    let toward_zero = cpu_reference(&photo, edit, Rounding::TowardZero);
    let gpu_pixels = gpu_render(&gpu, &photo, edit);
    assert_eq!(nearest.len(), gpu_pixels.len());
    let mut max = 0;
    let mut sum = 0u64;
    let mut worst = (0usize, [0u8; 3], [0u8; 3]);
    for (i, ((n, z), g)) in nearest
        .iter()
        .zip(&toward_zero)
        .zip(&gpu_pixels)
        .enumerate()
    {
        for k in 0..3 {
            let d = (i32::from(n[k]) - i32::from(g[k]))
                .abs()
                .min((i32::from(z[k]) - i32::from(g[k])).abs());
            sum += d as u64;
            if d > max {
                max = d;
                worst = (i, *z, *g);
            }
        }
    }
    let mean = sum as f64 / (nearest.len() * 3) as f64;
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
        &PhotoEdit {
            white_balance_temperature: 40.0,
            white_balance_tint: -20.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn exposure_matches() {
    check(
        "exposure",
        &PhotoEdit {
            exposure: 1.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn contrast_matches() {
    check(
        "contrast",
        &PhotoEdit {
            contrast: 60.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn highlights_match() {
    check(
        "highlights",
        &PhotoEdit {
            highlights: -70.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn shadows_match() {
    check(
        "shadows",
        &PhotoEdit {
            shadows: 60.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn whites_match() {
    check(
        "whites",
        &PhotoEdit {
            whites: 40.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn blacks_match() {
    check(
        "blacks",
        &PhotoEdit {
            blacks: -40.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn vibrance_matches() {
    check(
        "vibrance",
        &PhotoEdit {
            vibrance: 60.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn saturation_matches() {
    check(
        "saturation",
        &PhotoEdit {
            saturation: -50.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn every_slider_together_matches_develop_pixel() {
    check(
        "all sliders",
        &PhotoEdit {
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
        },
    );
}
