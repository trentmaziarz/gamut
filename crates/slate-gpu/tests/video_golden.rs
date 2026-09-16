//! Golden tests for the video pass: a synthetic NV12 frame and a P010
//! frame covering the code range, drawn through the YUV pass, the develop
//! pass with the neutral edit and the output pass, against the CPU twin in
//! slate-color through the same chain. The tests skip, and say so, when
//! the machine has no adapter, and the P010 test skips when the device has
//! no 16 bit normalised formats.

use std::sync::Mutex;

use half::f16;
use slate_color::basic;
use slate_color::video::{PlaneFormat, Transfer, VideoColour, YuvSpace, decode_video_pixel};
use slate_core::{CropRect, PhotoEdit};
use slate_gpu::video::P010_FEATURE;
use slate_gpu::{Develop, Headless, Readback};
use slate_media::VideoFrame;

const SIZE: u32 = 64;

/// The largest difference allowed in any channel of any pixel, in 8-bit codes.
const MAX_DIFFERENCE: i32 = 2;

/// The mean absolute difference allowed over all channels, in 8-bit codes.
const MEAN_DIFFERENCE: f64 = 0.5;

/// Luma across from below black to above white, Cb across the chroma
/// plane, Cr down it, so every code range and both clips are covered.
fn synthetic_frame(format: PlaneFormat) -> VideoFrame {
    let scale = match format {
        PlaneFormat::Nv12 => 1u32,
        PlaneFormat::P010 => 4,
    };
    let code = |v: f32| (v.round() as u32 * scale).min(format.max_code());
    let store = |code: u32, out: &mut Vec<u8>| match format {
        PlaneFormat::Nv12 => out.push(code as u8),
        PlaneFormat::P010 => out.extend_from_slice(&((code as u16) << 6).to_le_bytes()),
    };
    let mut y = Vec::new();
    for _row in 0..SIZE {
        for x in 0..SIZE {
            let luma = 8.0 + 237.0 * x as f32 / (SIZE - 1) as f32;
            store(code(luma), &mut y);
        }
    }
    let half = SIZE / 2;
    let mut uv = Vec::new();
    for row in 0..half {
        for x in 0..half {
            let cb = 16.0 + 224.0 * x as f32 / (half - 1) as f32;
            let cr = 240.0 - 224.0 * row as f32 / (half - 1) as f32;
            store(code(cb), &mut uv);
            store(code(cr), &mut uv);
        }
    }
    let bytes = format.bytes_per_sample();
    VideoFrame {
        pts_seconds: 0.0,
        format,
        width: SIZE,
        height: SIZE,
        y,
        uv,
        y_stride: SIZE as usize * bytes,
        uv_stride: SIZE as usize * bytes,
    }
}

#[derive(Clone, Copy)]
enum Rounding {
    Nearest,
    TowardZero,
}

/// Rounds a value the way the rgba16float and r16float textures store it;
/// see develop_golden.rs for why both modes are accepted.
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

/// The normalised luma at a texel.
fn luma_at(frame: &VideoFrame, x: u32, y: u32) -> f32 {
    frame.format.normalised(frame.luma_code(x, y))
}

/// The chroma plane sampled with bilinear filtering and clamp to edge at
/// continuous texel coordinates, as the GPU sampler does.
fn chroma_bilinear(frame: &VideoFrame, cx: f32, cy: f32) -> (f32, f32) {
    let half = (SIZE / 2) as i32;
    let x0 = cx.floor();
    let y0 = cy.floor();
    let fx = cx - x0;
    let fy = cy - y0;
    let at = |x: i32, y: i32| {
        let (cb, cr) = frame.chroma_code(x.clamp(0, half - 1) as u32, y.clamp(0, half - 1) as u32);
        (frame.format.normalised(cb), frame.format.normalised(cr))
    };
    let (x0, y0) = (x0 as i32, y0 as i32);
    let mix = |a: (f32, f32), b: (f32, f32), t: f32| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
    let top = mix(at(x0, y0), at(x0 + 1, y0), fx);
    let bottom = mix(at(x0, y0 + 1), at(x0 + 1, y0 + 1), fx);
    mix(top, bottom, fy)
}

/// The linear Rec.2020 pixel the pass writes at display pixel `dx`, `dy`
/// of a frame turned by `rotation`.
fn linear_at(frame: &VideoFrame, colour: VideoColour, rotation: u32, dx: u32, dy: u32) -> [f32; 3] {
    let u = (dx as f32 + 0.5) / SIZE as f32;
    let v = (dy as f32 + 0.5) / SIZE as f32;
    let (su, sv) = match rotation {
        90 => (v, 1.0 - u),
        180 => (1.0 - u, 1.0 - v),
        270 => (1.0 - v, u),
        _ => (u, v),
    };
    let sx = ((su * SIZE as f32) as u32).min(SIZE - 1);
    let sy = ((sv * SIZE as f32) as u32).min(SIZE - 1);
    let y = luma_at(frame, sx, sy);
    let half = (SIZE / 2) as f32;
    let (cb, cr) = chroma_bilinear(frame, su * half - 0.5, sv * half - 0.5);
    decode_video_pixel([y, cb, cr], frame.format, colour)
}

fn cpu_reference(
    frame: &VideoFrame,
    colour: VideoColour,
    rotation: u32,
    rounding: Rounding,
) -> Vec<[u8; 3]> {
    let linear: Vec<[f32; 3]> = (0..SIZE)
        .flat_map(|dy| (0..SIZE).map(move |dx| (dx, dy)))
        .map(|(dx, dy)| linear_at(frame, colour, rotation, dx, dy).map(|c| half(c, rounding)))
        .collect();
    let base = basic::base_layer(&linear, SIZE, SIZE);
    let edit = PhotoEdit::default();
    let wb = basic::white_balance_matrix(0.0, 0.0);
    linear
        .iter()
        .zip(&base)
        .map(|(px, b)| {
            let developed = basic::develop_pixel_with(*px, half(*b, rounding), &edit, &wb)
                .map(|c| half(c, rounding));
            basic::output_srgb8(developed)
        })
        .collect()
}

fn gpu_render(
    gpu: &Headless,
    frame: &VideoFrame,
    colour: VideoColour,
    rotation: u32,
) -> Vec<[u8; 3]> {
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_video_frame(frame, colour, rotation);
    let view = develop
        .render(
            &PhotoEdit::default(),
            CropRect::FULL,
            (SIZE, SIZE),
            (SIZE, SIZE),
        )
        .expect("a source is set");
    Readback::new(&gpu.device)
        .read(&gpu.device, &gpu.queue, view, SIZE, SIZE)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| [px[0], px[1], px[2]])
        .collect()
}

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn check(name: &str, format: PlaneFormat, colour: VideoColour, rotation: u32) {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    if format == PlaneFormat::P010 && !gpu.device.features().contains(P010_FEATURE) {
        println!("{name}: the device has no 16 bit normalised formats, skipped");
        return;
    }
    let frame = synthetic_frame(format);
    let nearest = cpu_reference(&frame, colour, rotation, Rounding::Nearest);
    let toward_zero = cpu_reference(&frame, colour, rotation, Rounding::TowardZero);
    let gpu_pixels = gpu_render(&gpu, &frame, colour, rotation);
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

const SDR: VideoColour = VideoColour {
    space: YuvSpace::Bt709,
    transfer: Transfer::Sdr,
    full_range: false,
};

const HLG: VideoColour = VideoColour {
    space: YuvSpace::Bt2020,
    transfer: Transfer::Hlg,
    full_range: false,
};

#[test]
fn an_nv12_bt709_frame_matches_the_twin() {
    check("nv12 bt709", PlaneFormat::Nv12, SDR, 0);
}

#[test]
fn a_full_range_nv12_frame_matches_the_twin() {
    check(
        "nv12 full range",
        PlaneFormat::Nv12,
        VideoColour {
            full_range: true,
            ..SDR
        },
        0,
    );
}

#[test]
fn a_p010_hlg_frame_matches_the_twin() {
    check("p010 hlg", PlaneFormat::P010, HLG, 0);
}

/// The crop rendered through the window path against the same crop cut
/// out of the full render: the two must agree, since the window holds the
/// blur's whole reach.
#[test]
fn a_windowed_render_matches_the_full_render() {
    let _turn = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let frame = synthetic_frame(PlaneFormat::Nv12);
    let crop = CropRect {
        x: 0.25,
        y: 0.0,
        width: 0.5,
        height: 1.0,
    };
    let output = (32, 64);
    let edit = PhotoEdit {
        highlights: -60.0,
        shadows: 40.0,
        ..PhotoEdit::default()
    };
    let readback = Readback::new(&gpu.device);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_video_frame(&frame, SDR, 0);
    let full = develop
        .render(&edit, crop, (SIZE, SIZE), output)
        .expect("a source is set");
    let full = readback.read(&gpu.device, &gpu.queue, full, output.0, output.1);
    let windowed = develop
        .render_crop(&edit, crop, output)
        .expect("a source is set");
    let windowed = readback.read(&gpu.device, &gpu.queue, windowed, output.0, output.1);
    let max = full
        .iter()
        .zip(&windowed)
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
        .max()
        .unwrap_or(0);
    println!("windowed render against the full render: max difference {max}");
    assert!(max <= 1, "max difference {max}");
}

#[test]
fn a_turned_frame_matches_the_turned_twin() {
    check("nv12 rotated 90", PlaneFormat::Nv12, SDR, 90);
    check("nv12 rotated 270", PlaneFormat::Nv12, SDR, 270);
}
