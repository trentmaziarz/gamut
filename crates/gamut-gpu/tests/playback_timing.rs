//! Times the playback path on the 4K30 timing clip: 300 frames through
//! NVDEC, the transfer into system memory, the upload into the plane
//! textures and the video, blur, develop and output passes at 1080x1920,
//! with a device poll after each submit. The median, 95th percentile and
//! maximum per frame are printed with the decode, copy and upload shares.
//! The p95 must be under 33.3 ms and the mean under 25 ms only when
//! GAMUT_TIMING_GATE=1 is set; the test skips when the clip is not on the
//! machine or no CUDA device answers. For the record, a second run of 100
//! frames renders what a Viewer zoomed to 100 percent shows of the frame,
//! through the padded window; nothing is asserted on it. A third run of 100
//! frames plays the clip under an auto brush mask of 200 strokes of 50
//! points: its layer reads the frame, so every frame builds the proxy of the
//! source and stamps the layer again, and its p95 must stay inside a frame
//! at 30 fps as well. So must two more runs of 100 frames: the same mask
//! with Refine edges at 100, and with Shift edge, Feather and Contrast on
//! over that.

use std::time::Instant;

use gamut_core::brush::{Brush, SharedStroke, Stroke};
use gamut_core::mask::MaskSource;
use gamut_core::{CropAspect, CropRect, Mask, PhotoEdit};
use gamut_gpu::develop::{padded_window, render_size_for_crop};
use gamut_gpu::{Develop, Headless, ViewWindow};
use gamut_media::hwaccel::nvdec_available;
use gamut_media::{Decoder, VideoSource, fixtures};

/// The Reel output size.
const OUTPUT: (u32, u32) = (1080, 1920);

/// How many frames are timed.
const FRAMES: usize = 300;

/// The p95 the gate demands, in milliseconds: one frame at 30 fps.
const GATE_P95_MS: f64 = 1000.0 / 30.0;

/// The mean the gate demands, in milliseconds.
const GATE_MEAN_MS: f64 = 25.0;

const GATE: &str = "GAMUT_TIMING_GATE";

/// How many strokes the auto brush of the third run holds, and how many
/// points each.
const AUTO_STROKES: usize = 200;
const STROKE_POINTS: usize = 50;

/// A number from 0 to 1 that is the same on every run.
fn next(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

/// The default edit under one mask: an auto brush of [`AUTO_STROKES`] strokes
/// of [`STROKE_POINTS`] points wandering over the frame, every third one a
/// pen's, lifting the exposure.
fn auto_brush_edit() -> PhotoEdit {
    let mut seed = 20_260_920;
    let strokes = (0..AUTO_STROKES)
        .map(|k| {
            let size = 0.004 + 0.03 * next(&mut seed);
            let mut at = [next(&mut seed), next(&mut seed)];
            let mut heading = next(&mut seed) * std::f32::consts::TAU;
            let points: Vec<[f32; 2]> = (0..STROKE_POINTS)
                .map(|_| {
                    heading += (next(&mut seed) - 0.5) * 0.8;
                    at[0] = (at[0] + heading.cos() * size * 0.25).clamp(0.0, 1.0);
                    at[1] = (at[1] + heading.sin() * size * 0.44).clamp(0.0, 1.0);
                    at
                })
                .collect();
            let pen = k % 3 == 0;
            SharedStroke::new(&Stroke {
                pressure: if pen {
                    (0..points.len())
                        .map(|_| 0.2 + 0.8 * next(&mut seed))
                        .collect()
                } else {
                    Vec::new()
                },
                points,
                size,
                feather: 100.0 * next(&mut seed),
                flow: 20.0 + 80.0 * next(&mut seed),
                erase: k % 10 == 9,
                auto: true,
                sensitivity: 100.0 * next(&mut seed),
                pressure_size: pen,
                pressure_flow: pen,
            })
        })
        .collect();
    let mut mask = Mask::new("Auto brush", MaskSource::Brush(Brush { strokes }));
    mask.adjust.exposure = 0.6;
    PhotoEdit {
        masks: vec![mask],
        ..PhotoEdit::default()
    }
}

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
    // For the record: the same clip with the view at 100 percent in a tab
    // of the output size. Every frame is new content, so the head passes run
    // over the padded window on each one.
    let full = (source.width, source.height);
    let seen = ((OUTPUT.0 + 1).min(full.0), (OUTPUT.1 + 1).min(full.1));
    let visible = ((full.0 - seen.0) / 2, (full.1 - seen.1) / 2, seen.0, seen.1);
    let view = ViewWindow {
        full,
        window: padded_window(full, visible, (OUTPUT.0 / 2, OUTPUT.1 / 2), 64),
        visible,
    };
    let mut zoomed = Vec::with_capacity(100);
    for i in 0..=100 {
        let started = Instant::now();
        let Some(frame) = source.next_frame().expect("decode") else {
            break;
        };
        develop.set_video_frame(&frame, colour, rotation);
        develop
            .render_view(&edit, &view)
            .expect("the source is set");
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the render");
        if i > 0 {
            zoomed.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    if !zoomed.is_empty() {
        zoomed.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let n = zoomed.len();
        println!(
            "playback with the view at 100 percent, window {:?} of {full:?}, {n} frames (not asserted): p50 {:.2} ms, p95 {:.2} ms, max {:.2} ms",
            (view.window.2, view.window.3),
            zoomed[n / 2],
            zoomed[(n * 95 / 100).min(n - 1)],
            zoomed[n - 1]
        );
    }

    // The same playback under an auto brush mask. Every frame is another
    // source content: the proxy is built and the layer stamped on each one.
    let gated = auto_brush_edit();
    let (layers, proxies) = (develop.brush_layer_builds(), develop.proxy_builds());
    let mut auto = Vec::with_capacity(100);
    for i in 0..=100 {
        let started = Instant::now();
        let Some(frame) = source.next_frame().expect("decode") else {
            break;
        };
        develop.set_video_frame(&frame, colour, rotation);
        develop
            .render(&gated, crop, render_size, OUTPUT)
            .expect("the source is set");
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the render");
        if i > 0 {
            auto.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    assert!(auto.len() >= 50, "the clip is long enough for the auto run");
    let frames = auto.len() as u64 + 1;
    assert_eq!(
        (
            develop.brush_layer_builds() - layers,
            develop.proxy_builds() - proxies
        ),
        (frames, frames),
        "every frame stamps the auto layer and builds the proxy once"
    );
    auto.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let auto_p95 = auto[(auto.len() * 95 / 100).min(auto.len() - 1)];
    println!(
        "playback under an auto brush mask of {AUTO_STROKES} strokes of {STROKE_POINTS} points, {} frames: p50 {:.2} ms, p95 {auto_p95:.2} ms, max {:.2} ms (plain playback p95 {p95:.2} ms)",
        auto.len(),
        auto[auto.len() / 2],
        auto[auto.len() - 1]
    );

    // The same with Refine edges at 100 on that mask: its alpha is drawn
    // again on every frame, so it is refined again on every frame, the
    // moments of the source with it.
    let mut refined_edit = gated.clone();
    for mask in &mut refined_edit.masks {
        mask.refine = gamut_core::mask::Refine {
            amount: 100.0,
            radius: 0.01,
            sensitivity: 50.0,
        };
    }
    let (refines, sources) = (develop.refine_builds().0, develop.refine_source_builds());
    let mut refined = Vec::with_capacity(100);
    for i in 0..=100 {
        let started = Instant::now();
        let Some(frame) = source.next_frame().expect("decode") else {
            break;
        };
        develop.set_video_frame(&frame, colour, rotation);
        develop
            .render(&refined_edit, crop, render_size, OUTPUT)
            .expect("the source is set");
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the render");
        if i > 0 {
            refined.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    assert!(
        refined.len() >= 50,
        "the clip is long enough for the refined run"
    );
    let frames = refined.len() as u64 + 1;
    assert_eq!(
        (
            develop.refine_builds().0 - refines,
            develop.refine_source_builds() - sources
        ),
        (frames, frames),
        "every frame refines the mask again, from the moments of its own source"
    );
    refined.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let refined_p95 = refined[(refined.len() * 95 / 100).min(refined.len() - 1)];
    println!(
        "playback under a refined auto brush mask of {AUTO_STROKES} strokes, Radius 0.01, {} frames: p50 {:.2} ms, p95 {refined_p95:.2} ms, max {:.2} ms (under the auto mask alone p95 {auto_p95:.2} ms)",
        refined.len(),
        refined[refined.len() / 2],
        refined[refined.len() - 1]
    );

    // The same with Shift edge -1 percent, Feather 1 percent and Contrast 50
    // on that mask: its refined alpha is new on every frame, so each edge
    // stage is drawn again whole on every frame. On a CPU adapter that is
    // seconds a frame and says nothing about a GPU, so it runs on a GPU alone.
    let mut edged_p95 = None;
    if gpu.adapter.get_info().device_type == wgpu::DeviceType::Cpu {
        println!("the edged playback line skipped on a CPU adapter");
    } else {
        let mut edged_edit = refined_edit.clone();
        for mask in &mut edged_edit.masks {
            mask.edge = gamut_core::mask::Edge {
                shift: -0.01,
                feather: 0.01,
                contrast: 50.0,
            };
        }
        let (stages, _) = develop.edge_builds();
        let mut edged = Vec::with_capacity(100);
        for i in 0..=100 {
            let started = Instant::now();
            let Some(frame) = source.next_frame().expect("decode") else {
                break;
            };
            develop.set_video_frame(&frame, colour, rotation);
            develop
                .render(&edged_edit, crop, render_size, OUTPUT)
                .expect("the source is set");
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("wait for the render");
            if i > 0 {
                edged.push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        assert!(
            edged.len() >= 50,
            "the clip is long enough for the edged run"
        );
        let frames = edged.len() as u64 + 1;
        assert_eq!(
            develop.edge_builds().0,
            stages.map(|stage| stage + frames),
            "every frame draws Shift edge, Feather and the finished alpha again whole"
        );
        edged.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let p95 = edged[(edged.len() * 95 / 100).min(edged.len() - 1)];
        println!(
            "playback under a refined auto brush mask of {AUTO_STROKES} strokes, Radius 0.01, with Shift edge -1 percent, Feather 1 percent and Contrast 50, {} frames: p50 {:.2} ms, p95 {p95:.2} ms, max {:.2} ms (under the refined mask alone p95 {refined_p95:.2} ms)",
            edged.len(),
            edged[edged.len() / 2],
            edged[edged.len() - 1]
        );
        edged_p95 = Some(p95);
    }

    if std::env::var(GATE).as_deref() == Ok("1") {
        if let Some(edged_p95) = edged_p95 {
            assert!(
                edged_p95 < GATE_P95_MS,
                "p95 under a refined auto brush mask with the three edge controls on, {edged_p95:.2} ms, is not under {GATE_P95_MS:.1} ms"
            );
        }
        assert!(
            refined_p95 < GATE_P95_MS,
            "p95 under a refined auto brush mask, {refined_p95:.2} ms, is not under {GATE_P95_MS:.1} ms"
        );
        assert!(
            auto_p95 < GATE_P95_MS,
            "p95 under an auto brush mask, {auto_p95:.2} ms, is not under {GATE_P95_MS:.1} ms"
        );
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
