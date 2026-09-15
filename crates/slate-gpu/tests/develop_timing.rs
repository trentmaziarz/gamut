//! Times the develop render of the 24 megapixel fixture at the viewer's
//! pixel size: 100 renders with the exposure slider stepping, a device poll
//! after each submit, and the median, 95th percentile and maximum printed.
//! The p95 must be under 16 ms only when SLATE_TIMING_GATE=1 is set, so CI
//! on WARP prints and never fails. One full-resolution render is timed as
//! well on hardware adapters, for the record.

use std::time::Instant;

use slate_core::{CropRect, PhotoEdit};
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

    let mut times: Vec<f64> = (0..RENDERS)
        .map(|i| {
            let edit = PhotoEdit {
                exposure: -1.0 + 2.0 * i as f32 / (RENDERS - 1) as f32,
                ..PhotoEdit::default()
            };
            timed_render(&gpu, &mut develop, &edit, VIEWER_SIZE)
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let p50 = times[RENDERS / 2];
    let p95 = times[(RENDERS * 95 / 100).min(RENDERS - 1)];
    let max = times[RENDERS - 1];
    println!(
        "develop at {VIEWER_SIZE:?} over {RENDERS} renders: p50 {p50:.2} ms, p95 {p95:.2} ms, max {max:.2} ms"
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
    } else {
        println!("{GATE} is not set, so the {GATE_MS} ms gate is printed and not asserted");
    }
}
