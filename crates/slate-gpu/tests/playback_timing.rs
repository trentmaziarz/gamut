//! Times the playback path on the 4K30 timing clip: 300 frames through
//! NVDEC, the transfer into system memory, the upload into the plane
//! textures and the video, blur, develop and output passes at 1080x1920,
//! with a device poll after each submit. The median, 95th percentile and
//! maximum per frame are printed with the decode, copy and upload shares.
//! The p95 must be under 33.3 ms and the mean under 25 ms only when
//! SLATE_TIMING_GATE=1 is set; the test skips when the clip is not on the
//! machine or no CUDA device answers.

use std::time::Instant;

use slate_core::{CropAspect, CropRect, PhotoEdit};
use slate_gpu::develop::render_size_for_crop;
use slate_gpu::{Develop, Headless};
use slate_media::hwaccel::nvdec_available;
use slate_media::{Decoder, VideoSource, fixtures};

/// The Reel output size.
const OUTPUT: (u32, u32) = (1080, 1920);

/// How many frames are timed.
const FRAMES: usize = 300;

/// The p95 the gate demands, in milliseconds: one frame at 30 fps.
const GATE_P95_MS: f64 = 1000.0 / 30.0;

/// The mean the gate demands, in milliseconds.
const GATE_MEAN_MS: f64 = 25.0;

const GATE: &str = "SLATE_TIMING_GATE";

#[test]
fn playback_at_4k30_is_fast_enough() {
    let Some(path) = fixtures::local("timing_4k30_60s.mp4") else {
        println!("timing_4k30_60s.mp4 is not on this machine, skipped");
        return;
    };
    if !nvdec_available() {
        println!("no CUDA device, skipped");
        return;
    }
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());
    let mut source = VideoSource::open(&path).expect("open the timing clip");
    println!(
        "source: {}x{} {:?} decoder {:?}",
        source.width, source.height, source.colour, source.decoder_kind
    );
    assert_eq!(source.decoder_kind, Decoder::Nvdec);
    let crop = CropRect::fitted(CropAspect::Story9x16, source.width, source.height);
    let render_size = render_size_for_crop(crop, OUTPUT);
    println!("render size {render_size:?} for the crop {crop:?}");

    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    let edit = PhotoEdit::default();
    let colour = source.colour;
    let rotation = source.rotation;

    let mut totals = Vec::with_capacity(FRAMES);
    let mut decode = 0.0;
    let mut copy = 0.0;
    let mut upload = 0.0;
    for i in 0..FRAMES {
        let started = Instant::now();
        let frame = source
            .next_frame()
            .expect("decode")
            .unwrap_or_else(|| panic!("the clip ended at frame {i}"));
        let decoded = Instant::now();
        develop.set_video_frame(&frame, colour, rotation);
        develop
            .render(&edit, crop, render_size, OUTPUT)
            .expect("the source is set");
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the render");
        let done = Instant::now();
        let total = done.duration_since(started).as_secs_f64() * 1000.0;
        // The first frame builds the textures and pipelines; it is printed
        // and left out of the statistics.
        if i == 0 {
            println!("first frame (textures built): {total:.2} ms");
            continue;
        }
        totals.push(total);
        decode += source.last_timing.decode_ms;
        copy += source.last_timing.copy_ms;
        upload += done.duration_since(decoded).as_secs_f64() * 1000.0;
    }
    let n = totals.len();
    let mean = totals.iter().sum::<f64>() / n as f64;
    totals.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let p50 = totals[n / 2];
    let p95 = totals[(n * 95 / 100).min(n - 1)];
    let max = totals[n - 1];
    println!(
        "playback of {n} frames at 4K30 to {OUTPUT:?}: p50 {p50:.2} ms, p95 {p95:.2} ms, max {max:.2} ms, mean {mean:.2} ms"
    );
    println!(
        "shares per frame: decode {:.2} ms, copy {:.2} ms, upload and render {:.2} ms",
        decode / n as f64,
        copy / n as f64,
        upload / n as f64
    );
    if std::env::var(GATE).as_deref() == Ok("1") {
        assert!(
            p95 < GATE_P95_MS,
            "p95 {p95:.2} ms is not under {GATE_P95_MS:.1} ms"
        );
        assert!(
            mean < GATE_MEAN_MS,
            "mean {mean:.2} ms is not under {GATE_MEAN_MS} ms"
        );
    } else {
        println!("{GATE} is not set, so the 4K30 gate is printed and not asserted");
    }
}
