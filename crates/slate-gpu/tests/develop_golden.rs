//! Golden tests: every GPU operator against its CPU twin in slate-color,
//! through the same input and output transforms, on a 64 by 64 synthetic
//! photo that covers hue, brightness and neutral ramps. The tests skip,
//! and say so, when the machine has no adapter.

use std::sync::Mutex;

use half::f16;
use slate_color::SourceSpace;
use slate_color::basic::{self, Neighbourhood, Prepared};
use slate_color::{dehaze, local};
use slate_core::look::{Curve, HslRange, Wheel};
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
    if edit.dehaze != 0.0 {
        &[0, -1, 1]
    } else {
        &[0]
    }
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
    let transmission = dehaze::transmission_stored(&linear, width, height, atmosphere, &store);
    let prepared = Prepared::new(edit, atmosphere);
    (0..linear.len())
        .map(|i| {
            let around = Neighbourhood {
                base_luma: base[i],
                texture_luma: texture[i],
                transmission: half_steps(transmission[i], transmission_step),
            };
            let developed = basic::develop_pixel_with(linear[i], &around, edit, &prepared)
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
            ..PhotoEdit::default()
        },
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
            &PhotoEdit {
                texture,
                ..PhotoEdit::default()
            },
        );
    }
}

#[test]
fn clarity_matches() {
    for clarity in [60.0, -60.0] {
        check(
            "clarity",
            &PhotoEdit {
                clarity,
                ..PhotoEdit::default()
            },
        );
    }
}

#[test]
fn positive_dehaze_matches() {
    check(
        "dehaze +",
        &PhotoEdit {
            dehaze: 70.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn negative_dehaze_matches() {
    check(
        "dehaze -",
        &PhotoEdit {
            dehaze: -50.0,
            ..PhotoEdit::default()
        },
    );
}

#[test]
fn every_operator_together_matches_develop_pixel() {
    let mut edit = PhotoEdit {
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
        ..PhotoEdit::default()
    };
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
    let on = PhotoEdit {
        texture: 50.0,
        dehaze: 60.0,
        ..PhotoEdit::default()
    };
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
