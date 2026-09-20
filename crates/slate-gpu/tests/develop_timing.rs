//! Times the develop render of the 24 megapixel fixture at the viewer's
//! pixel size: 100 renders with the exposure slider stepping, a device poll
//! after each submit, and the median, 95th percentile and maximum printed.
//! It is measured three times: with every other slider at rest, with every
//! operator of the develop chain on, and with four masks on top of that, one
//! of each source, each carrying every adjustment. Each p95 must be under
//! 16 ms only when SLATE_TIMING_GATE=1 is set, so CI on WARP prints and never
//! fails. A fourth run steps a slider of a mask instead of the global
//! exposure and holds that no mask alpha is drawn again. The one-off cost of
//! the texture and dehaze head passes, of the four mask alphas and of one
//! full-resolution render are timed as well, for the record.

use std::time::Instant;

use slate_core::look::{Curve, Wheel};
use slate_core::mask::{ColourRange, LinearGradient, LuminanceRange, MaskSource, RadialGradient};
use slate_core::{Adjustments, CropRect, Mask, PhotoEdit};
use slate_gpu::{Develop, Headless};
use slate_media::{fixtures, open_photo};

/// The 4:5 fit of the viewer on the reference laptop, in pixels.
const VIEWER_SIZE: (u32, u32) = (1280, 1600);

/// How many slider changes are timed.
const RENDERS: usize = 100;

/// The p95 the gate demands, in milliseconds.
const GATE_MS: f64 = 16.0;

/// The environment variable that turns the print into an assertion.
const GATE: &str = "SLATE_TIMING_GATE";

fn timed_render(gpu: &Headless, develop: &mut Develop, edit: &PhotoEdit, size: (u32, u32)) -> f64 {
    let start = Instant::now();
    develop
        .render(edit, CropRect::FULL, size, size)
        .expect("the source is set");
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("wait for the render");
    start.elapsed().as_secs_f64() * 1000.0
}

/// Every operator of the develop chain away from its neutral value.
fn everything_on() -> PhotoEdit {
    let mut edit = PhotoEdit::from(Adjustments {
        white_balance_temperature: -20.0,
        white_balance_tint: 10.0,
        exposure: 0.0,
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
    edit.look.curves.master = Curve {
        points: vec![[0.0, 0.0], [0.25, 0.17], [0.75, 0.85], [1.0, 1.0]],
    };
    edit.look.curves.red = Curve {
        points: vec![[0.0, 0.03], [0.5, 0.54], [1.0, 1.0]],
    };
    edit.look.hsl[1].saturation = -40.0;
    edit.look.hsl[4].hue = 50.0;
    edit.look.hsl[5].luminance = -30.0;
    edit.look.wheels.shadows = Wheel {
        x: -0.4,
        y: -0.3,
        luminance: -5.0,
    };
    edit.look.wheels.midtones = Wheel {
        x: 0.1,
        y: 0.2,
        luminance: 10.0,
    };
    edit.look.wheels.highlights = Wheel {
        x: 0.4,
        y: 0.15,
        luminance: 5.0,
    };
    edit
}

/// [`everything_on`] with four enabled masks over it, one of each source,
/// each carrying every adjustment: Basic, Presence, curves, mixer, wheels.
fn everything_on_with_four_masks() -> PhotoEdit {
    let mut edit = everything_on();
    let sources = [
        MaskSource::Linear(LinearGradient {
            start: [0.5, 0.7],
            end: [0.45, 0.2],
        }),
        MaskSource::Radial(RadialGradient {
            centre: [0.55, 0.45],
            radius: [0.35, 0.25],
            rotation: 20.0,
            feather: 60.0,
        }),
        MaskSource::Luminance(LuminanceRange {
            low: 0.35,
            high: 0.9,
            falloff: 0.1,
        }),
        MaskSource::Colour(ColourRange {
            hue: 250.0,
            hue_width: 120.0,
            chroma_low: 0.01,
            falloff: 30.0,
        }),
    ];
    for (k, source) in sources.into_iter().enumerate() {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let mut mask = Mask::new(&format!("Mask {k}"), source);
        mask.opacity = 90.0 - 10.0 * k as f32;
        mask.adjust = Adjustments {
            white_balance_temperature: 15.0 * sign,
            white_balance_tint: -8.0 * sign,
            exposure: 0.5 * sign,
            contrast: 12.0,
            highlights: 20.0 * sign,
            shadows: -15.0 * sign,
            whites: 8.0,
            blacks: -6.0,
            vibrance: 18.0 * sign,
            saturation: -12.0 * sign,
            texture: 20.0 * sign,
            clarity: 15.0,
            dehaze: 10.0 * sign,
            ..Adjustments::default()
        };
        mask.adjust.look.curves.master = Curve {
            points: vec![[0.0, 0.02], [0.4, 0.45 + 0.02 * k as f32], [1.0, 0.98]],
        };
        mask.adjust.look.curves.blue = Curve {
            points: vec![[0.0, 0.0], [0.5, 0.46], [1.0, 1.0]],
        };
        mask.adjust.look.hsl[k].saturation = 30.0 * sign;
        mask.adjust.look.hsl[(k + 3) % 8].hue = -25.0;
        mask.adjust.look.wheels.shadows = Wheel {
            x: 0.2 * sign,
            y: 0.1,
            luminance: 5.0,
        };
        mask.adjust.look.wheels.highlights = Wheel {
            x: -0.15,
            y: 0.25 * sign,
            luminance: -5.0,
        };
        edit.masks.push(mask);
    }
    edit
}

/// The median, the 95th percentile and the maximum of `RENDERS` renders of
/// `base` with the exposure slider stepping from -1 to 1.
fn stepped(gpu: &Headless, develop: &mut Develop, base: &PhotoEdit) -> (f64, f64, f64) {
    stepped_with(gpu, develop, base, |edit, value| edit.exposure = value)
}

/// The same with `step` putting a value that runs from -1 to 1 into the
/// edit before each render.
fn stepped_with(
    gpu: &Headless,
    develop: &mut Develop,
    base: &PhotoEdit,
    step: impl Fn(&mut PhotoEdit, f32),
) -> (f64, f64, f64) {
    let mut times: Vec<f64> = (0..RENDERS)
        .map(|i| {
            let mut edit = base.clone();
            step(&mut edit, -1.0 + 2.0 * i as f32 / (RENDERS - 1) as f32);
            timed_render(gpu, develop, &edit, VIEWER_SIZE)
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    (
        times[RENDERS / 2],
        times[(RENDERS * 95 / 100).min(RENDERS - 1)],
        times[RENDERS - 1],
    )
}

#[test]
fn develop_at_viewer_size_is_fast_enough() {
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let photo = open_photo(&fixtures::require("timing_24mp.jpg")).expect("open the fixture");
    println!("photo: {}x{}", photo.width, photo.height);

    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&photo);

    // The first render builds the working and base textures; it is timed
    // separately because a slider change never repeats it.
    let first = timed_render(&gpu, &mut develop, &PhotoEdit::default(), VIEWER_SIZE);
    println!("first render at {VIEWER_SIZE:?} (input transform and blur): {first:.2} ms");

    let (p50, p95, max) = stepped(&gpu, &mut develop, &PhotoEdit::default());
    println!(
        "develop at {VIEWER_SIZE:?} over {RENDERS} renders: p50 {p50:.2} ms, p95 {p95:.2} ms, max {max:.2} ms"
    );

    // Texture and dehaze leaving 0 draw their head-pass products once, and
    // the tone curves upload their table once; no later render repeats it.
    let everything = everything_on();
    let switched_on = timed_render(&gpu, &mut develop, &everything, VIEWER_SIZE);
    println!(
        "first render with every operator on (texture blur, transmission map, curve table): {switched_on:.2} ms"
    );
    let (all_p50, all_p95, all_max) = stepped(&gpu, &mut develop, &everything);
    println!(
        "develop with every operator on at {VIEWER_SIZE:?} over {RENDERS} renders: p50 {all_p50:.2} ms, p95 {all_p95:.2} ms, max {all_max:.2} ms"
    );

    // Four masks of every source over that, each carrying every adjustment:
    // the first render draws the four alphas and uploads four table rows,
    // and no slider change after it repeats either.
    let masked = everything_on_with_four_masks();
    let builds_before = develop.mask_alpha_builds();
    let masks_on = timed_render(&gpu, &mut develop, &masked, VIEWER_SIZE);
    assert_eq!(develop.mask_alpha_builds() - builds_before, 4);
    println!("first render with four masks on (four alphas, four table rows): {masks_on:.2} ms");
    let (masks_p50, masks_p95, masks_max) = stepped(&gpu, &mut develop, &masked);
    println!(
        "develop with every operator and four full masks on at {VIEWER_SIZE:?} over {RENDERS} renders: p50 {masks_p50:.2} ms, p95 {masks_p95:.2} ms, max {masks_max:.2} ms"
    );
    assert_eq!(
        develop.mask_alpha_builds() - builds_before,
        4,
        "a global slider drew a mask alpha again"
    );

    // Informational: the same with a slider of one mask stepping.
    let (slider_p50, slider_p95, slider_max) =
        stepped_with(&gpu, &mut develop, &masked, |edit, value| {
            edit.masks[1].adjust.exposure = value;
        });
    println!(
        "the same with a mask slider stepping: p50 {slider_p50:.2} ms, p95 {slider_p95:.2} ms, max {slider_max:.2} ms"
    );
    assert_eq!(
        develop.mask_alpha_builds() - builds_before,
        4,
        "a slider of a mask drew its alpha again"
    );

    let info = gpu.adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        println!("full-resolution render skipped on a CPU adapter");
    } else {
        let full = (photo.width, photo.height);
        let time = timed_render(&gpu, &mut develop, &PhotoEdit::default(), full);
        println!("full-resolution render at {full:?}: {time:.2} ms");
    }

    if std::env::var(GATE).as_deref() == Ok("1") {
        assert!(p95 < GATE_MS, "p95 {p95:.2} ms is not under {GATE_MS} ms");
        assert!(
            all_p95 < GATE_MS,
            "p95 with every operator on, {all_p95:.2} ms, is not under {GATE_MS} ms"
        );
        assert!(
            masks_p95 < GATE_MS,
            "p95 with four full masks on, {masks_p95:.2} ms, is not under {GATE_MS} ms"
        );
    } else {
        println!("{GATE} is not set, so the {GATE_MS} ms gate is printed and not asserted");
    }
}
