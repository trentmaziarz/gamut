//! Times the develop render of the 24 megapixel fixture at the viewer's
//! pixel size: 100 renders with the exposure slider stepping, a device poll
//! after each submit, and the median, 95th percentile and maximum printed.
//! It is measured three times: with every other slider at rest, with every
//! operator of the develop chain on, and with four masks on top of that, one
//! of each source, each carrying every adjustment. Each p95 must be under
//! 16 ms only when GAMUT_TIMING_GATE=1 is set, so CI on WARP prints and never
//! fails. A fourth run steps a slider of a mask instead of the global
//! exposure and holds that no mask alpha is drawn again. The one-off cost of
//! the texture and dehaze head passes, of the four mask alphas and of one
//! full-resolution render are timed as well, for the record.
//!
//! The zoomed viewer is measured the same way, with every operator and the
//! four masks on: a slider step at 100 percent, which develops what is seen
//! and not the whole photo; a pan of 32 source pixels a step inside the
//! padded window, which develops what came into view and runs no head pass;
//! the same slider step at 400 percent; and, for the record, what replacing
//! the padded window costs and how often a steady pan pays it. The two
//! slider steps print the T-12 figures beside them.
//!
//! The brush is measured last, over all of the above: a fifth mask that is a
//! brush of 500 strokes of 50 points carrying exposure, clarity and a curve.
//! A slider step with the five masks at the viewer size and at 100 percent,
//! and painting at 100 and at 400 percent, one appended point a frame for
//! 100 frames, are held to the gate. What stamping the whole layer over the
//! padded window costs, and what replacing the window costs with the brush
//! in it, are printed for the record. Then a steady pan of 32 source pixels
//! a frame at 100 percent with the five masks, right, down, left and up,
//! crosses the edge of the window at least three times, with the window
//! ahead of it built as the viewer builds it, one slice a frame. A warm-up
//! leg of one crossing comes before it and is not timed, so the pan holds
//! two frames when the measurement starts. It holds that no crossing
//! replaces the window in one submit and that no frame of the measured pan
//! makes textures of its own, and its p95 is held to the gate. At most one
//! of its frames may reach the gate, and its maximum is printed: the frames
//! over it are device waits on random frames, attributed 2026-09-26 (T-26)
//! and ruled by Trent (D-10). The frames it leaves are dropped after it, so
//! the lines after it time renders that hold one frame.
//!
//! The auto brush comes after it: a sixth mask that is a brush of 500 auto
//! strokes of 50 points, a third of them painted with a pen whose pressure
//! scales the size and the flow. A slider step with the six masks at the
//! viewer size and at 100 percent, and painting an auto stroke with a pen at
//! 100 and at 400 percent, are held to the gate. What the proxy of the source
//! costs, built once, and what stamping the whole auto layer over the padded
//! window costs are printed for the record.
//!
//! Refine edges at 100 on two of the six masks follows. A step of each
//! Refine edges slider at 100 percent, Amount, Radius and Sensitivity at
//! Radius 0.05 and at 0.01, the mask refined whole each time, is held to the
//! gate. The whole refine of one mask with the moments of the source held
//! and taken again is printed for the record.
//!
//! The edge controls come last: Shift edge -1 percent, Feather 1 percent and
//! Contrast 50 on the two refined masks. A develop slider step, a Shift edge,
//! a Feather and a Contrast slider step at the viewer size and at 100
//! percent, and painting an auto stroke with a pen into such a mask at 100
//! and at 400 percent, are held to the gate. A Shift edge, a Feather and a
//! Contrast slider step at 100 percent with the three at 5 percent are held
//! to the gate too. What the shift passes alone and the feather passes alone
//! cost over the padded window at 100 percent at 1 and at 5 percent is
//! printed for the record.

use std::time::Instant;

use gamut_core::brush::{Brush, SharedStroke, Stroke};
use gamut_core::look::{Curve, Wheel};
use gamut_core::mask::{
    ColourRange, Edge, LinearGradient, LuminanceRange, MaskSource, RadialGradient, Refine,
};
use gamut_core::{Adjustments, CropRect, Mask, PhotoEdit};
use gamut_gpu::develop::{PixelRect, holds, padded_window, pan_exit, window_ahead};
use gamut_gpu::{Develop, EdgePasses, Headless, ViewWindow};
use gamut_media::{fixtures, open_photo};

/// The 4:5 fit of the viewer on the reference laptop, in pixels.
const VIEWER_SIZE: (u32, u32) = (1280, 1600);

/// How many slider changes are timed.
const RENDERS: usize = 100;

/// The p95 the gate demands, in milliseconds.
const GATE_MS: f64 = 16.0;

/// The environment variable that turns the print into an assertion.
const GATE: &str = "GAMUT_TIMING_GATE";

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

/// The grid the viewer snaps its padded window to, in source pixels, and the
/// pan of one step.
const WINDOW_GRID: u32 = 64;
const PAN_STEP: u32 = 32;

/// What the viewer renders of a photo of `full` pixels at `zoom` (1 is 100
/// percent) with the middle of the photo in the middle of a tab of
/// [`VIEWER_SIZE`] pixels: what is seen, padded by half a tab and snapped.
fn zoomed_view(full: (u32, u32), zoom: u32) -> (ViewWindow, (u32, u32)) {
    let seen = (
        (VIEWER_SIZE.0.div_ceil(zoom) + 1).min(full.0),
        (VIEWER_SIZE.1.div_ceil(zoom) + 1).min(full.1),
    );
    let visible = ((full.0 - seen.0) / 2, (full.1 - seen.1) / 2, seen.0, seen.1);
    let pad = (
        VIEWER_SIZE.0.div_ceil(2 * zoom),
        VIEWER_SIZE.1.div_ceil(2 * zoom),
    );
    let view = ViewWindow {
        full,
        window: padded_window(full, visible, pad, WINDOW_GRID),
        visible,
    };
    (view, pad)
}

fn timed_view(gpu: &Headless, develop: &mut Develop, edit: &PhotoEdit, view: &ViewWindow) -> f64 {
    let start = Instant::now();
    develop.render_view(edit, view).expect("the source is set");
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("wait for the render");
    start.elapsed().as_secs_f64() * 1000.0
}

/// The median, the 95th percentile and the maximum of timed runs, `RENDERS`
/// of them but for the pan across the window edge.
fn percentiles(mut times: Vec<f64>) -> (f64, f64, f64) {
    times.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let n = times.len();
    (times[n / 2], times[(n * 95 / 100).min(n - 1)], times[n - 1])
}

/// The legs of the steady pan across the window edge: a direction on each
/// axis and a count of [`PAN_STEP`] steps. Right and left 64 steps, down and
/// up 36, so the pan ends where it began after 200 frames, and every
/// rectangle seen of timing_24mp.jpg at 100 percent stays inside the photo.
const EDGE_PAN_LEGS: [((i64, i64), u32); 4] =
    [((1, 0), 64), ((0, 1), 36), ((-1, 0), 64), ((0, -1), 36)];

/// The steps of the warm-up leg before the pan across the window edge:
/// rightward to where the measured pan begins, across the edge of the window
/// once. The first render of a pan and its first build ahead make the
/// textures of their frames, and a steady pan is what follows them: once a
/// swap has left a frame, each window built ahead is drawn into the textures
/// of the frame the swap before it left. So the warm-up crosses once,
/// untimed, and the pan is measured from there.
const WARM_UP_STEPS: u32 = 24;

/// The rectangles seen on each frame of a steady pan from `start` through
/// `legs`, one [`PAN_STEP`] a frame.
fn pan_legs(start: PixelRect, legs: &[((i64, i64), u32)]) -> Vec<PixelRect> {
    let step = i64::from(PAN_STEP);
    let mut at = start;
    let mut path = Vec::new();
    for &((dx, dy), steps) in legs {
        for _ in 0..steps {
            at.0 = u32::try_from(i64::from(at.0) + dx * step).expect("the pan stays in the photo");
            at.1 = u32::try_from(i64::from(at.1) + dy * step).expect("the pan stays in the photo");
            path.push(at);
        }
    }
    path
}

/// What a pan across the edge of the window drew, beside its frame times,
/// counted from the end of its warm-up.
struct EdgePan {
    times: Vec<f64>,
    /// The frames of the warm-up, and how many of them crossed the edge of
    /// the window.
    warm_up_frames: usize,
    warm_up_crossings: u64,
    /// Frames whose window was not the one of the frame before.
    crossings: u64,
    /// Renders that replaced the frame and ran the head passes in one submit.
    replaces: u64,
    /// Renders that took the frame built ahead.
    swaps: u64,
    /// Slices of frames built ahead submitted.
    slices: u64,
    /// Frames made with textures of their own.
    frame_makes: u64,
}

/// A pan through `warm_up` and then `path` as the zoomed viewer draws it,
/// one UI frame a rectangle seen, after an untimed render of `start`. Each
/// frame asks for the window the viewer asks for: the one rendered while it
/// holds what is seen, else the one built ahead when it holds it, else a new
/// padded one. It renders that and then drives the build ahead as the viewer
/// does, once a frame while the pan moves. Each frame of `path` is timed
/// from the render to the device poll after both, and the counts are taken
/// over `path` alone; the frames of `warm_up` are drawn the same way and
/// neither timed nor counted.
fn pan_across_edge(
    gpu: &Headless,
    develop: &mut Develop,
    edit: &PhotoEdit,
    start: &ViewWindow,
    pad: (u32, u32),
    warm_up: &[PixelRect],
    path: &[PixelRect],
) -> EdgePan {
    let full = start.full;
    timed_view(gpu, develop, edit, start);
    let (mut window, mut previous) = (start.window, start.visible);
    let mut crossings = 0;
    for &visible in warm_up {
        let asked = if holds(window, visible) {
            window
        } else {
            crossings += 1;
            develop
                .window_ahead()
                .filter(|&(held, ahead)| held == full && holds(ahead, visible))
                .map_or_else(
                    || padded_window(full, visible, pad, WINDOW_GRID),
                    |(_, ahead)| ahead,
                )
        };
        let view = ViewWindow {
            full,
            window: asked,
            visible,
        };
        develop.render_view(edit, &view).expect("the source is set");
        drive_ahead(develop, edit, &view, previous, pad);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the render");
        (window, previous) = (asked, visible);
    }
    let warm_up_crossings = crossings;
    let (replaces, swaps, slices, makes) = (
        develop.window_replaces(),
        develop.window_swaps(),
        develop.ahead_slices(),
        develop.frame_makes(),
    );
    let mut crossings = 0;
    let mut times = Vec::with_capacity(path.len());
    for &visible in path {
        let asked = if holds(window, visible) {
            window
        } else {
            crossings += 1;
            develop
                .window_ahead()
                .filter(|&(held, ahead)| held == full && holds(ahead, visible))
                .map_or_else(
                    || padded_window(full, visible, pad, WINDOW_GRID),
                    |(_, ahead)| ahead,
                )
        };
        let started = Instant::now();
        let view = ViewWindow {
            full,
            window: asked,
            visible,
        };
        develop.render_view(edit, &view).expect("the source is set");
        drive_ahead(develop, edit, &view, previous, pad);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the render");
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        (window, previous) = (asked, visible);
    }
    EdgePan {
        times,
        warm_up_frames: warm_up.len(),
        warm_up_crossings,
        crossings,
        replaces: develop.window_replaces() - replaces,
        swaps: develop.window_swaps() - swaps,
        slices: develop.ahead_slices() - slices,
        frame_makes: develop.frame_makes() - makes,
    }
}

/// The build ahead as the viewer drives it once a UI frame, after a render
/// of `view` where the frame before showed `previous`: nothing on a jump;
/// the frame still building on a pause; else the window held ahead while it
/// holds where the pan leaves the window rendered, a new one otherwise.
fn drive_ahead(
    develop: &mut Develop,
    edit: &PhotoEdit,
    view: &ViewWindow,
    previous: PixelRect,
    pad: (u32, u32),
) {
    let (full, window, visible) = (view.full, view.window, view.visible);
    // A step of more than the pad is a jump, not a pan.
    if previous.0.abs_diff(visible.0) > pad.0 || previous.1.abs_diff(visible.1) > pad.1 {
        return;
    }
    let built = develop
        .window_ahead()
        .filter(|&(held, _)| held == full)
        .map(|(_, ahead)| ahead);
    let wanted = if previous == visible {
        built.filter(|_| develop.ahead_building())
    } else {
        pan_exit(full, window, previous, visible).and_then(|exit| match built {
            Some(built) if holds(built, exit) => Some(built),
            _ => window_ahead(full, window, previous, visible, pad, WINDOW_GRID),
        })
    };
    if let Some(ahead) = wanted {
        develop.build_ahead(edit, full, ahead);
    }
}

/// `RENDERS` renders of a zoomed view with the exposure slider stepping.
fn stepped_view(
    gpu: &Headless,
    develop: &mut Develop,
    base: &PhotoEdit,
    view: &ViewWindow,
) -> (f64, f64, f64) {
    percentiles(
        (0..RENDERS)
            .map(|i| {
                let mut edit = base.clone();
                edit.exposure = -1.0 + 2.0 * i as f32 / (RENDERS - 1) as f32;
                timed_view(gpu, develop, &edit, view)
            })
            .collect(),
    )
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

/// How many strokes the timed brush holds, and how many points each.
const BRUSH_STROKES: usize = 500;
const STROKE_POINTS: usize = 50;

/// A number from 0 to 1 that is the same on every run.
fn next(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

/// `base` with a fifth mask: a brush of [`BRUSH_STROKES`] strokes of
/// [`STROKE_POINTS`] points wandering over the whole photo, every tenth an
/// erase, with brushes from small and hard to large and soft, carrying
/// exposure, clarity and a curve.
fn with_brush_mask(base: &PhotoEdit) -> PhotoEdit {
    let mut seed = 20_260_920;
    let strokes = (0..BRUSH_STROKES)
        .map(|k| {
            let size = 0.004 + 0.03 * next(&mut seed);
            let mut at = [next(&mut seed), next(&mut seed)];
            let mut heading = next(&mut seed) * std::f32::consts::TAU;
            let points = (0..STROKE_POINTS)
                .map(|_| {
                    // A point every quarter radius, as the window adds them.
                    heading += (next(&mut seed) - 0.5) * 0.8;
                    at[0] = (at[0] + heading.cos() * size * 0.25).clamp(0.0, 1.0);
                    at[1] = (at[1] + heading.sin() * size * 0.375).clamp(0.0, 1.0);
                    at
                })
                .collect();
            SharedStroke::new(&Stroke {
                points,
                size,
                feather: 100.0 * next(&mut seed),
                flow: 20.0 + 80.0 * next(&mut seed),
                erase: k % 10 == 9,
                ..Stroke::default()
            })
        })
        .collect();
    let mut mask = Mask::new("Brush", MaskSource::Brush(Brush { strokes }));
    mask.adjust.exposure = 0.6;
    mask.adjust.clarity = 25.0;
    mask.adjust.look.curves.master = Curve {
        points: vec![[0.0, 0.0], [0.3, 0.26], [0.7, 0.76], [1.0, 1.0]],
    };
    let mut edit = base.clone();
    edit.masks.push(mask);
    edit
}

/// The fifth mask is the brush of 4c, the sixth the auto brush.
const PLAIN_BRUSH: usize = 4;
const AUTO_BRUSH: usize = 5;

/// `base` with a sixth mask: the strokes of [`with_brush_mask`] as auto
/// strokes at sensitivities from loose to strict, every third one painted
/// with a pen whose pressure wanders and scales the size and the flow.
fn with_auto_mask(base: &PhotoEdit) -> PhotoEdit {
    let mut seed = 20_260_921;
    let plain = with_brush_mask(&PhotoEdit::default());
    let MaskSource::Brush(plain) = &plain.masks[0].components[0].source else {
        panic!("a brush");
    };
    let strokes = plain
        .strokes
        .iter()
        .enumerate()
        .map(|(k, stroke)| {
            let pen = k % 3 == 0;
            let mut pressure = next(&mut seed);
            SharedStroke::new(&Stroke {
                auto: true,
                sensitivity: 100.0 * next(&mut seed),
                pressure: if pen {
                    (0..stroke.points.len())
                        .map(|_| {
                            pressure = (pressure + (next(&mut seed) - 0.5) * 0.2).clamp(0.05, 1.0);
                            pressure
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                pressure_size: pen,
                pressure_flow: pen,
                ..(**stroke).clone()
            })
        })
        .collect();
    let mut mask = Mask::new("Auto brush", MaskSource::Brush(Brush { strokes }));
    mask.adjust.exposure = -0.4;
    mask.adjust.saturation = 20.0;
    let mut edit = base.clone();
    edit.masks.push(mask);
    edit
}

/// The brush of a mask of the timed edit.
fn timed_brush(edit: &mut PhotoEdit, mask: usize) -> &mut Brush {
    match &mut edit.masks[mask].components[0].source {
        MaskSource::Brush(brush) => brush,
        _ => panic!("mask {mask} is a brush"),
    }
}

/// `RENDERS` frames of painting inside what `view` shows: a press in the
/// middle, then one appended point a frame, a quarter radius on, along a
/// path that turns back before it leaves what is seen.
fn painted_view(
    gpu: &Headless,
    develop: &mut Develop,
    base: &PhotoEdit,
    view: &ViewWindow,
) -> (f64, f64, f64) {
    painted_view_with(gpu, develop, base, view, PLAIN_BRUSH, &Stroke::default())
}

/// The same into the brush of mask `mask`, with a stroke that takes `like`
/// for what a press stores beside the size, the feather and the flow. An
/// auto stroke is painted with a pen whose pressure rises and falls.
fn painted_view_with(
    gpu: &Headless,
    develop: &mut Develop,
    base: &PhotoEdit,
    view: &ViewWindow,
    mask: usize,
    like: &Stroke,
) -> (f64, f64, f64) {
    let pen = |i: usize| like.auto.then(|| 0.55 + 0.45 * (i as f32 * 0.21).sin());
    let (full_w, full_h) = (view.full.0 as f32, view.full.1 as f32);
    let (x, y, w, h) = view.visible;
    let centre = [
        (x as f32 + w as f32 / 2.0) / full_w,
        (y as f32 + h as f32 / 2.0) / full_h,
    ];
    // A brush a twelfth of what is seen across, and a loop that stays in it.
    let longer = full_w.max(full_h);
    let size = w.min(h) as f32 / 12.0 / longer;
    let orbit = [w as f32 * 0.3 / full_w, h as f32 * 0.3 / full_h];
    let mut edit = base.clone();
    timed_brush(&mut edit, mask)
        .strokes
        .push(SharedStroke::new(&Stroke {
            points: vec![[centre[0] + orbit[0], centre[1]]],
            pressure: pen(0).into_iter().collect(),
            size,
            feather: 50.0,
            flow: 60.0,
            ..like.clone()
        }));
    timed_view(gpu, develop, &edit, view);
    // A quarter radius of arc a frame.
    let step = size * 0.25 * longer / (w as f32 * 0.3);
    percentiles(
        (1..=RENDERS)
            .map(|i| {
                let angle = step * i as f32;
                let at = [
                    centre[0] + orbit[0] * angle.cos(),
                    centre[1] + orbit[1] * angle.sin(),
                ];
                let stroke = timed_brush(&mut edit, mask)
                    .strokes
                    .last_mut()
                    .expect("pressed");
                assert!(stroke.push(at, pen(i)));
                timed_view(gpu, develop, &edit, view)
            })
            .collect(),
    )
}

/// The radial mask of [`everything_on_with_four_masks`].
const RADIAL: usize = 1;

/// The six masks with Refine edges at 100 on two of them: the widest box, a
/// Radius of 0.05, on the auto brush mask, and 0.01 on the radial mask.
fn with_refined_masks(base: &PhotoEdit) -> PhotoEdit {
    let mut edit = base.clone();
    for (mask, radius) in [(AUTO_BRUSH, 0.05), (RADIAL, 0.01)] {
        edit.masks[mask].refine = Refine {
            amount: 100.0,
            radius,
            sensitivity: 50.0,
        };
    }
    edit
}

/// `RENDERS` renders of a zoomed view with `step` putting a value that runs
/// from -1 to 1 into the edit before each.
fn stepped_view_with(
    gpu: &Headless,
    develop: &mut Develop,
    base: &PhotoEdit,
    view: &ViewWindow,
    step: impl Fn(&mut PhotoEdit, f32),
) -> (f64, f64, f64) {
    percentiles(
        (0..RENDERS)
            .map(|i| {
                let mut edit = base.clone();
                step(&mut edit, -1.0 + 2.0 * i as f32 / (RENDERS - 1) as f32);
                timed_view(gpu, develop, &edit, view)
            })
            .collect(),
    )
}

/// The edge controls of the timed lines, and what each slider line moves
/// through on the auto brush mask. Shift edge moves in whole pixels, so its
/// line alternates with a value at least a pixel away at the viewer size and
/// at 100 percent; Feather runs through a range. Both keep the pad of the
/// window at 100 percent, so no step replaces it.
struct EdgeLine {
    edge: Edge,
    shift_to: f32,
    feather: (f32, f32),
    name: &'static str,
}

/// Shift edge -1 percent, Feather 1 percent and Contrast 50.
const EDGED: EdgeLine = EdgeLine {
    edge: Edge {
        shift: -0.01,
        feather: 0.01,
        contrast: 50.0,
    },
    shift_to: -0.011,
    feather: (0.009, 0.011),
    name: "1 percent",
};

/// The same at 5 percent, the widest Shift edge and Feather.
const EDGED_WIDE: EdgeLine = EdgeLine {
    edge: Edge {
        shift: -0.05,
        feather: 0.05,
        contrast: 50.0,
    },
    shift_to: -0.049,
    feather: (0.049, 0.05),
    name: "5 percent",
};

/// `base` with `edge` on its two refined masks.
fn with_edged_masks(base: &PhotoEdit, edge: Edge) -> PhotoEdit {
    let mut edit = base.clone();
    for mask in [AUTO_BRUSH, RADIAL] {
        edit.masks[mask].edge = edge;
    }
    edit
}

/// The value of a slider line at step `value`, which runs from -1 to 1:
/// `from` on the even steps and `to` on the odd ones.
fn alternating(value: f32, from: f32, to: f32) -> f32 {
    let odd = ((value + 1.0) * (RENDERS - 1) as f32 / 2.0).round() as u32 % 2 == 1;
    if odd { to } else { from }
}

/// What the develop graph has drawn so far, to hold which passes a run of
/// renders drew.
struct Drawn {
    alphas: u64,
    refines: (u64, u64),
    stages: [u64; 3],
    passes: EdgePasses,
}

impl Drawn {
    fn of(develop: &Develop) -> Self {
        let (whole, parts) = develop.edge_builds();
        Drawn {
            alphas: develop.mask_alpha_builds(),
            refines: develop.refine_builds(),
            stages: [0, 1, 2].map(|stage| whole[stage] + parts[stage]),
            passes: develop.edge_passes(),
        }
    }

    /// Holds that no mask alpha and no Refine edges pass was drawn since
    /// `self`, and no edge stage before `first`: 0 Shift edge, 1 Feather's
    /// cells, 2 the finished alpha, 3 none of them. A Feather slider whose
    /// cell grid grows past the held cell texture makes the cells again and
    /// still draws no Shift edge pass.
    fn hold(&self, develop: &Develop, first: usize, what: &str) {
        let now = Drawn::of(develop);
        assert_eq!(now.alphas, self.alphas, "{what} drew a mask alpha again");
        assert_eq!(now.refines, self.refines, "{what} drew a Refine edges pass");
        for stage in 0..first.min(3) {
            assert_eq!(
                now.stages[stage],
                self.stages[stage],
                "{what} drew edge stage {stage} {} times",
                now.stages[stage] - self.stages[stage]
            );
        }
        let passes = [
            (now.passes.shift, self.passes.shift),
            (now.passes.feather, self.passes.feather),
            (now.passes.finish, self.passes.finish),
        ];
        for (stage, (now, then)) in passes.into_iter().enumerate().take(first) {
            assert_eq!(
                now,
                then,
                "{what} drew {} passes of edge stage {stage}",
                now - then
            );
        }
    }

    /// The edge passes drawn since `self`, on average a render, and how many
    /// times each stage was drawn.
    fn passes_since(&self, develop: &Develop, renders: usize) -> String {
        let now = Drawn::of(develop);
        let per = |now: u64, then: u64| (now - then) as f64 / renders as f64;
        format!(
            "{:.1} shift, {:.1} feather cell and {:.1} finishing passes a render; the stages drawn {}, {} and {} times",
            per(now.passes.shift, self.passes.shift),
            per(now.passes.feather, self.passes.feather),
            per(now.passes.finish, self.passes.finish),
            now.stages[0] - self.stages[0],
            now.stages[1] - self.stages[1],
            now.stages[2] - self.stages[2]
        )
    }
}

/// What a slider line puts into the edit at a value that runs from -1 to 1.
type Step<'a> = &'a dyn Fn(&mut PhotoEdit, f32);

/// `RENDERS` renders of `base` with `step` putting a value that runs from -1
/// to 1 into the edit before each: at the viewer size when `view` is `None`,
/// else through `view`.
fn stepped_at(
    gpu: &Headless,
    develop: &mut Develop,
    base: &PhotoEdit,
    view: Option<&ViewWindow>,
    step: impl Fn(&mut PhotoEdit, f32),
) -> (f64, f64, f64) {
    match view {
        None => stepped_with(gpu, develop, base, step),
        Some(view) => stepped_view_with(gpu, develop, base, view, step),
    }
}

/// A Shift edge, a Feather and a Contrast slider step on the auto brush mask
/// of `base`, whose edge controls are `line.edge`: at the viewer size when
/// `view` is `None`, else through `view`, which `at` names. Each is printed
/// with the edge passes a render draws, and held to drawing no mask alpha,
/// no Refine edges pass and no edge stage before its own. Returns the three
/// p95s in that order.
fn edge_slider_steps(
    gpu: &Headless,
    develop: &mut Develop,
    base: &PhotoEdit,
    view: Option<&ViewWindow>,
    line: &EdgeLine,
    at: &str,
    asserted: bool,
) -> [f64; 3] {
    let note = if asserted { "" } else { " (not asserted)" };
    let (from, to) = (line.edge.shift, line.shift_to);
    let (low, high) = line.feather;
    let shift = |edit: &mut PhotoEdit, value: f32| {
        edit.masks[AUTO_BRUSH].edge.shift = alternating(value, from, to);
    };
    let feather = |edit: &mut PhotoEdit, value: f32| {
        edit.masks[AUTO_BRUSH].edge.feather = low + (high - low) * (value + 1.0) / 2.0;
    };
    let contrast = |edit: &mut PhotoEdit, value: f32| {
        edit.masks[AUTO_BRUSH].edge.contrast = 50.0 + 49.0 * value;
    };
    let sliders: [(&str, usize, Step); 3] = [
        ("Shift edge", 0, &shift),
        ("Feather", 1, &feather),
        ("Contrast", 2, &contrast),
    ];
    sliders.map(|(name, first, step)| {
        // Back to `base` first: the line before left its own slider moved.
        match view {
            None => timed_render(gpu, develop, base, VIEWER_SIZE),
            Some(view) => timed_view(gpu, develop, base, view),
        };
        let drawn = Drawn::of(develop);
        let (p50, p95, max) = stepped_at(gpu, develop, base, view, step);
        let now = Drawn::of(develop);
        println!(
            "{name} slider step {at}, the three at {}{note}, {RENDERS} renders: p50 {p50:.2} ms, p95 {p95:.2} ms, max {max:.2} ms; {}; mask_alpha_builds {} then {}, refine_builds {:?} then {:?}",
            line.name,
            drawn.passes_since(develop, RENDERS),
            drawn.alphas,
            now.alphas,
            drawn.refines,
            now.refines
        );
        drawn.hold(develop, first, &format!("a {name} slider"));
        p95
    })
}

/// What the shift passes alone and the feather passes alone cost over the
/// padded window of `view` at 100 percent, printed and not asserted. Each is
/// the slider step of its control, with the other two at rest on the auto
/// brush mask of `base`, against a develop slider step on the same edit,
/// which draws no edge pass.
fn edge_passes_alone(
    gpu: &Headless,
    develop: &mut Develop,
    base: &PhotoEdit,
    view: &ViewWindow,
    line: &EdgeLine,
) {
    let (from, to) = (line.edge.shift, line.shift_to);
    let (low, high) = line.feather;
    let shift = |edit: &mut PhotoEdit, value: f32| {
        edit.masks[AUTO_BRUSH].edge.shift = alternating(value, from, to);
    };
    let feather = |edit: &mut PhotoEdit, value: f32| {
        edit.masks[AUTO_BRUSH].edge.feather = low + (high - low) * (value + 1.0) / 2.0;
    };
    let alone: [(&str, usize, Edge, Step); 2] = [
        (
            "shift",
            0,
            Edge {
                shift: from,
                ..Edge::default()
            },
            &shift,
        ),
        (
            "feather",
            1,
            Edge {
                feather: line.edge.feather,
                ..Edge::default()
            },
            &feather,
        ),
    ];
    for (name, first, edge, step) in alone {
        let mut one = base.clone();
        one.masks[AUTO_BRUSH].edge = edge;
        timed_view(gpu, develop, &one, view);
        let drawn = Drawn::of(develop);
        let (held_p50, held_p95, _) = stepped_view(gpu, develop, &one, view);
        drawn.hold(develop, 3, "a slider of the develop chain");
        let drawn = Drawn::of(develop);
        let (p50, p95, _) = stepped_view_with(gpu, develop, &one, view, step);
        drawn.hold(develop, first, &format!("a {name} slider"));
        // The count of Shift edge passes, whole, over the renders that drew
        // Shift edge: the first step puts the value the edit holds and draws
        // none, every other step of a Shift edge line draws the four runs,
        // and a Feather step draws none.
        let now = Drawn::of(develop);
        let shift_passes = now.passes.shift - drawn.passes.shift;
        let shift_renders = now.stages[0] - drawn.stages[0];
        println!(
            "the {name} passes alone at 100 percent, {} (not asserted): develop slider step p50 {held_p50:.2} ms, p95 {held_p95:.2} ms; {name} slider step p50 {p50:.2} ms, p95 {p95:.2} ms; so the {name} passes about {:.2} ms; edge_passes.shift {shift_passes} in the {shift_renders} renders that drew Shift edge, {} a render; {}",
            line.name,
            (p50 - held_p50).max(0.0),
            shift_passes.checked_div(shift_renders).unwrap_or(0),
            drawn.passes_since(develop, RENDERS)
        );
    }
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

    // The zoomed viewer. On a CPU adapter a window of several megapixels
    // takes minutes and proves nothing, so the lines are skipped there.
    let info = gpu.adapter.get_info();
    let mut zoomed_p95 = None;
    if info.device_type == wgpu::DeviceType::Cpu {
        println!("zoomed viewer lines skipped on a CPU adapter");
    } else {
        let full = (photo.width, photo.height);
        let (actual, pad) = zoomed_view(full, 1);
        let builds = develop.mask_alpha_builds();
        let first = timed_view(&gpu, &mut develop, &masked, &actual);
        println!(
            "first render of the padded window at 100 percent, window {:?} of {full:?}, seen {:?}: {first:.2} ms",
            (actual.window.2, actual.window.3),
            (actual.visible.2, actual.visible.3)
        );
        assert_eq!(develop.mask_alpha_builds() - builds, 4);
        let (step_p50, step_p95, step_max) = stepped_view(&gpu, &mut develop, &masked, &actual);
        println!(
            "slider step at 100 percent, {RENDERS} renders: p50 {step_p50:.2} ms, p95 {step_p95:.2} ms, max {step_max:.2} ms; T-12 read p95 1.44 to 1.63 ms (not asserted against it)"
        );

        // A pan of 32 source pixels a step, there and back inside the pad:
        // the window stays, so no head pass and no mask alpha runs again.
        let reach = pad.0 / PAN_STEP;
        let pans: Vec<PixelRect> = (0..RENDERS as u32)
            .map(|i| {
                let leg = i % (2 * reach);
                let offset = PAN_STEP * if leg < reach { leg } else { 2 * reach - leg };
                let (x, y, w, h) = actual.visible;
                (x + offset, y, w, h)
            })
            .collect();
        assert!(pans.iter().all(|visible| holds(actual.window, *visible)));
        let (pan_p50, pan_p95, pan_max) = percentiles(
            pans.iter()
                .map(|visible| {
                    let view = ViewWindow {
                        visible: *visible,
                        ..actual
                    };
                    timed_view(&gpu, &mut develop, &masked, &view)
                })
                .collect(),
        );
        println!(
            "pan of {PAN_STEP} source pixels a step inside the padded window, {RENDERS} renders: p50 {pan_p50:.2} ms, p95 {pan_p95:.2} ms, max {pan_max:.2} ms"
        );
        assert_eq!(
            develop.mask_alpha_builds() - builds,
            4,
            "a slider step or a pan inside the window drew a mask alpha again"
        );

        // For the record: a pan that leaves the window replaces it, which
        // runs the head passes and the four alphas over the new one.
        let mut replaced = Vec::new();
        for k in 1..=5u32 {
            let (x, y, w, h) = actual.visible;
            let visible = (x + k * (pad.0 + PAN_STEP), y, w, h);
            if visible.0 + w > full.0 {
                break;
            }
            let view = ViewWindow {
                full,
                window: padded_window(full, visible, pad, WINDOW_GRID),
                visible,
            };
            replaced.push(timed_view(&gpu, &mut develop, &masked, &view));
        }
        replaced.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        println!(
            "replacing the padded window, {} times: least {:.2} ms, most {:.2} ms; a steady pan of {PAN_STEP} pixels a step pays it every {} steps across and every {} down",
            replaced.len(),
            replaced.first().copied().unwrap_or(0.0),
            replaced.last().copied().unwrap_or(0.0),
            pad.0 / PAN_STEP,
            pad.1 / PAN_STEP
        );

        let (deep, _) = zoomed_view(full, 4);
        let first = timed_view(&gpu, &mut develop, &masked, &deep);
        println!(
            "first render of the padded window at 400 percent, window {:?}, seen {:?}: {first:.2} ms",
            (deep.window.2, deep.window.3),
            (deep.visible.2, deep.visible.3)
        );
        let (deep_p50, deep_p95, deep_max) = stepped_view(&gpu, &mut develop, &masked, &deep);
        println!(
            "slider step at 400 percent, {RENDERS} renders: p50 {deep_p50:.2} ms, p95 {deep_p95:.2} ms, max {deep_max:.2} ms; T-12 read p95 0.29 to 0.38 ms (not asserted against it)"
        );
        zoomed_p95 = Some((step_p95, pan_p95, deep_p95));
    }

    if info.device_type == wgpu::DeviceType::Cpu {
        println!("full-resolution render skipped on a CPU adapter");
    } else {
        let full = (photo.width, photo.height);
        let time = timed_render(&gpu, &mut develop, &PhotoEdit::default(), full);
        println!("full-resolution render at {full:?}: {time:.2} ms");
    }

    // The brush, over every operator and the four masks: a fifth mask of
    // 500 strokes. Its layer is stamped once at each size and kept.
    let painted = with_brush_mask(&masked);
    let layers = develop.brush_layer_builds();
    let brush_on = timed_render(&gpu, &mut develop, &painted, VIEWER_SIZE);
    println!(
        "first render with the brush mask on ({BRUSH_STROKES} strokes of {STROKE_POINTS} points stamped at {VIEWER_SIZE:?}): {brush_on:.2} ms"
    );
    let (five_p50, five_p95, five_max) = stepped(&gpu, &mut develop, &painted);
    println!(
        "slider step with the five masks at {VIEWER_SIZE:?} over {RENDERS} renders: p50 {five_p50:.2} ms, p95 {five_p95:.2} ms, max {five_max:.2} ms"
    );
    assert_eq!(
        develop.brush_layer_builds() - layers,
        1,
        "a slider stamped the brush layer again"
    );
    let mut brush_p95 = None;
    let mut edge_pan = None;
    if info.device_type == wgpu::DeviceType::Cpu {
        println!("zoomed brush lines skipped on a CPU adapter");
    } else {
        let full = (photo.width, photo.height);
        let (actual, pad) = zoomed_view(full, 1);
        let first = timed_view(&gpu, &mut develop, &painted, &actual);
        println!(
            "first render of the padded window at 100 percent with the five masks: {first:.2} ms"
        );
        let (step_p50, step_p95, step_max) = stepped_view(&gpu, &mut develop, &painted, &actual);
        println!(
            "slider step at 100 percent with the five masks, {RENDERS} renders: p50 {step_p50:.2} ms, p95 {step_p95:.2} ms, max {step_max:.2} ms"
        );
        let (layers, appends) = (develop.brush_layer_builds(), develop.brush_layer_appends());
        let (paint_p50, paint_p95, paint_max) = painted_view(&gpu, &mut develop, &painted, &actual);
        println!(
            "painting at 100 percent, one appended point a frame, {RENDERS} frames: p50 {paint_p50:.2} ms, p95 {paint_p95:.2} ms, max {paint_max:.2} ms"
        );
        assert_eq!(
            (
                develop.brush_layer_builds() - layers,
                develop.brush_layer_appends() - appends
            ),
            (0, RENDERS as u64 + 1),
            "painting stamped the whole layer again"
        );

        // For the record: the whole layer stamped again over the padded
        // window, which an undo or a changed earlier stroke asks for, and the
        // window replaced with the brush in it.
        timed_view(&gpu, &mut develop, &painted, &actual);
        let mut undone = painted.clone();
        timed_brush(&mut undone, PLAIN_BRUSH).strokes.pop();
        let whole = timed_view(&gpu, &mut develop, &undone, &actual);
        println!(
            "the brush layer stamped whole over the padded window {:?}, its alpha and what is seen developed: {whole:.2} ms",
            (actual.window.2, actual.window.3)
        );
        let mut replaced = Vec::new();
        for k in 1..=5u32 {
            let (x, y, w, h) = actual.visible;
            let visible = (x + k * (pad.0 + PAN_STEP), y, w, h);
            if visible.0 + w > full.0 {
                break;
            }
            let view = ViewWindow {
                full,
                window: padded_window(full, visible, pad, WINDOW_GRID),
                visible,
            };
            replaced.push(timed_view(&gpu, &mut develop, &painted, &view));
        }
        replaced.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        println!(
            "replacing the padded window with the five masks, {} times: least {:.2} ms, most {:.2} ms",
            replaced.len(),
            replaced.first().copied().unwrap_or(0.0),
            replaced.last().copied().unwrap_or(0.0)
        );

        let (deep, _) = zoomed_view(full, 4);
        timed_view(&gpu, &mut develop, &painted, &deep);
        let (deep_p50, deep_p95, deep_max) = painted_view(&gpu, &mut develop, &painted, &deep);
        println!(
            "painting at 400 percent, one appended point a frame, {RENDERS} frames: p50 {deep_p50:.2} ms, p95 {deep_p95:.2} ms, max {deep_max:.2} ms"
        );
        brush_p95 = Some((step_p95, paint_p95, deep_p95));

        // A steady pan at 100 percent with the five masks, right, down, left
        // and up, across the edge of the window, with the window ahead of it
        // built one slice a frame as the viewer builds it. No crossing
        // replaces the window in one submit: each takes the frame built
        // ahead. The first render of a pan and its first build ahead are not
        // part of a steady pan: they make the textures of the two frames a
        // steady pan draws into. A warm-up leg rightward to where the pan
        // begins crosses the edge once before the measurement, untimed, so
        // the pan holds two frames when the measurement starts, and no
        // measured frame makes frame textures of its own.
        let path = pan_legs(actual.visible, &EDGE_PAN_LEGS);
        assert!(
            path.iter()
                .all(|&(x, y, w, h)| x + w <= full.0 && y + h <= full.1),
            "the pan stays inside the photo"
        );
        let (x, y, w, h) = actual.visible;
        let warm_start = (x - WARM_UP_STEPS * PAN_STEP, y, w, h);
        let warm_up = pan_legs(warm_start, &[((1, 0), WARM_UP_STEPS)]);
        assert_eq!(
            warm_up.last(),
            Some(&actual.visible),
            "the warm-up ends where the pan begins"
        );
        let warm_view = ViewWindow {
            full,
            window: padded_window(full, warm_start, pad, WINDOW_GRID),
            visible: warm_start,
        };
        let pan = pan_across_edge(
            &gpu,
            &mut develop,
            &painted,
            &warm_view,
            pad,
            &warm_up,
            &path,
        );
        println!(
            "warm-up before the steady pan, not timed and not counted: {} frames rightward to where the pan begins, {} crossings",
            pan.warm_up_frames, pan.warm_up_crossings
        );
        assert_eq!(
            pan.warm_up_crossings, 1,
            "the warm-up crossed the window edge {} times, not once",
            pan.warm_up_crossings
        );
        let frames = pan.times.len();
        let over = pan.times.iter().filter(|t| **t >= GATE_MS).count();
        let (p50, p95, max) = percentiles(pan.times);
        println!(
            "steady pan of {PAN_STEP} source pixels a frame across the window edge at 100 percent with the five masks, right, down, left and up, {frames} frames after the warm-up: p50 {p50:.2} ms, p95 {p95:.2} ms, max {max:.2} ms, {over} of {frames} frames at or over {GATE_MS} ms; {} crossings, {} swaps, {} replaces, {} slices run, {} frame makes",
            pan.crossings, pan.swaps, pan.replaces, pan.slices, pan.frame_makes
        );
        assert!(
            pan.crossings >= 3,
            "the pan crossed the window edge {} times, not 3 or more",
            pan.crossings
        );
        assert_eq!(
            pan.replaces, 0,
            "a pan across the window edge replaced the window in one submit"
        );
        assert!(
            pan.swaps >= pan.crossings,
            "{} swaps for {} crossings: a crossing did not take the frame built ahead",
            pan.swaps,
            pan.crossings
        );
        assert_eq!(
            pan.frame_makes, 0,
            "the steady pan after the warm-up made {} frames with textures of their own",
            pan.frame_makes
        );
        edge_pan = Some((p95, max, over, frames));

        // The frame the pan left built ahead, and the one kept for its
        // textures, are dropped, as a render at another zoom drops them, so
        // the lines after this one time renders that hold one frame, as they
        // did before the build ahead.
        let dropped = develop.drop_ahead();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the drop");
        println!(
            "the frame built ahead after the pan across the window edge: dropped {dropped}, held now {}",
            develop.window_ahead().is_some()
        );
        assert!(
            develop.window_ahead().is_none(),
            "no frame built ahead is held after the pan"
        );
    }

    // The auto brush, over all of that: a sixth mask of 500 auto strokes, a
    // third of them a pen's. First a single auto dab, so the proxy of the
    // source is what that render pays for: it is built once and kept.
    let proxies = develop.proxy_builds();
    let mut one_dab = painted.clone();
    one_dab.masks.push(Mask::new(
        "Auto brush",
        MaskSource::Brush(Brush {
            strokes: vec![SharedStroke::new(&Stroke {
                points: vec![[0.5, 0.5]],
                size: 0.001,
                auto: true,
                ..Stroke::default()
            })],
        }),
    ));
    one_dab.masks[AUTO_BRUSH].adjust.exposure = -0.4;
    let proxy_ms = timed_render(&gpu, &mut develop, &one_dab, VIEWER_SIZE);
    println!(
        "first render with one auto dab (the proxy of {:?} built, {} by {}): {proxy_ms:.2} ms",
        (photo.width, photo.height),
        gamut_color::brush::Proxy::size_for((photo.width, photo.height)).0,
        gamut_color::brush::Proxy::size_for((photo.width, photo.height)).1,
    );
    let gated = with_auto_mask(&painted);
    let layers = develop.brush_layer_builds();
    let auto_on = timed_render(&gpu, &mut develop, &gated, VIEWER_SIZE);
    println!(
        "first render with the auto brush mask on ({BRUSH_STROKES} auto strokes stamped at {VIEWER_SIZE:?}): {auto_on:.2} ms"
    );
    let (six_p50, six_p95, six_max) = stepped(&gpu, &mut develop, &gated);
    println!(
        "slider step with the six masks at {VIEWER_SIZE:?} over {RENDERS} renders: p50 {six_p50:.2} ms, p95 {six_p95:.2} ms, max {six_max:.2} ms"
    );
    assert_eq!(
        develop.brush_layer_builds() - layers,
        1,
        "a slider stamped the auto layer again"
    );
    let mut auto_p95 = None;
    if info.device_type == wgpu::DeviceType::Cpu {
        println!("zoomed auto brush lines skipped on a CPU adapter");
    } else {
        let full = (photo.width, photo.height);
        let (actual, _) = zoomed_view(full, 1);
        let first = timed_view(&gpu, &mut develop, &gated, &actual);
        println!(
            "first render of the padded window at 100 percent with the six masks: {first:.2} ms"
        );
        let (step_p50, step_p95, step_max) = stepped_view(&gpu, &mut develop, &gated, &actual);
        println!(
            "slider step at 100 percent with the six masks, {RENDERS} renders: p50 {step_p50:.2} ms, p95 {step_p95:.2} ms, max {step_max:.2} ms"
        );
        let pen = Stroke {
            auto: true,
            sensitivity: 60.0,
            pressure_size: true,
            pressure_flow: true,
            ..Stroke::default()
        };
        let (layers, appends) = (develop.brush_layer_builds(), develop.brush_layer_appends());
        let (paint_p50, paint_p95, paint_max) =
            painted_view_with(&gpu, &mut develop, &gated, &actual, AUTO_BRUSH, &pen);
        println!(
            "painting an auto stroke with a pen at 100 percent, one appended point a frame, {RENDERS} frames: p50 {paint_p50:.2} ms, p95 {paint_p95:.2} ms, max {paint_max:.2} ms"
        );
        assert_eq!(
            (
                develop.brush_layer_builds() - layers,
                develop.brush_layer_appends() - appends
            ),
            (0, RENDERS as u64 + 1),
            "painting stamped the whole auto layer again"
        );

        // For the record: the whole auto layer stamped again over the padded
        // window, which an undo asks for and a new frame of a video.
        timed_view(&gpu, &mut develop, &gated, &actual);
        let mut undone = gated.clone();
        timed_brush(&mut undone, AUTO_BRUSH).strokes.pop();
        let whole = timed_view(&gpu, &mut develop, &undone, &actual);
        println!(
            "the auto brush layer stamped whole over the padded window {:?}, its alpha and what is seen developed: {whole:.2} ms",
            (actual.window.2, actual.window.3)
        );

        let (deep, _) = zoomed_view(full, 4);
        timed_view(&gpu, &mut develop, &gated, &deep);
        let (deep_p50, deep_p95, deep_max) =
            painted_view_with(&gpu, &mut develop, &gated, &deep, AUTO_BRUSH, &pen);
        println!(
            "painting an auto stroke with a pen at 400 percent, one appended point a frame, {RENDERS} frames: p50 {deep_p50:.2} ms, p95 {deep_p95:.2} ms, max {deep_max:.2} ms"
        );
        auto_p95 = Some((step_p95, paint_p95, deep_p95));
    }
    assert_eq!(
        develop.proxy_builds() - proxies,
        1,
        "one photo, one proxy: no window, slider or stroke built it again"
    );

    // Refine edges at 100 on the auto brush mask and on the radial mask.
    let refined = with_refined_masks(&gated);
    let (refines, sources) = (develop.refine_builds().0, develop.refine_source_builds());
    let refine_on = timed_render(&gpu, &mut develop, &refined, VIEWER_SIZE);
    println!(
        "first render with two refined masks at {VIEWER_SIZE:?} (Radius 0.05 and 0.01, each refined whole): {refine_on:.2} ms"
    );
    let (refined_p50, refined_p95, refined_max) = stepped(&gpu, &mut develop, &refined);
    println!(
        "slider step with the six masks, two of them refined, at {VIEWER_SIZE:?} over {RENDERS} renders: p50 {refined_p50:.2} ms, p95 {refined_p95:.2} ms, max {refined_max:.2} ms"
    );
    // Each mask refined once, and the moments of the source taken once a
    // tile of each: a Radius of 0.01 is more than one tile at this size.
    let source_tiles = develop.refine_source_builds() - sources;
    println!("the moments of the source were taken over {source_tiles} tiles for the two masks");
    assert_eq!(
        develop.refine_builds().0 - refines,
        2,
        "a slider of the develop chain refined a mask again"
    );
    // Every render from here to the zoomed lines refines a mask whole. On a
    // CPU adapter that is seconds a render and says nothing about a GPU, so
    // those lines are run on a GPU alone.
    let on_gpu = info.device_type != wgpu::DeviceType::Cpu;
    if !on_gpu {
        println!("whole refine lines skipped on a CPU adapter");
    }
    let mut amount_p95 = None;
    if on_gpu {
        let (p50, p95, max) = stepped_with(&gpu, &mut develop, &refined, |edit, value| {
            edit.masks[AUTO_BRUSH].refine.amount = 50.0 + 49.0 * value;
        });
        println!(
            "Refine edges slider step at {VIEWER_SIZE:?}, Radius 0.05, the mask refined whole each time, {RENDERS} renders: p50 {p50:.2} ms, p95 {p95:.2} ms, max {max:.2} ms"
        );
        amount_p95 = Some(p95);
    }
    // Not asserted: the whole refine of one mask with the moments of the
    // source held (a step of Edge sensitivity) and with Radius stepping
    // between two boxes, at each end of the radius. At the viewer size both
    // Radius pairs keep the side of a cell, 2 pixels at 0.01 and 0.012 and 4
    // at 0.05 and 0.04, so a Radius step takes no moments of the source and
    // draws their four box means again. Held: 13 passes at 0.01, and 19 at
    // 0.05, whose boxes of 14 cells draw a block pass each. Stepped: 17 at
    // 0.01, 0.012 and 0.04 (a box of 11 cells, no block pass), 27 at 0.05,
    // each count before the flood. The flood adds its seed and 6 passes at
    // 0.05, so the timing line, a Radius step from 0.05 to 0.049, draws
    // 27 + 1 + 6 = 34 passes.
    // "Taken again" below is that step, and its difference from the held
    // line is the four box means of the source.
    let ends: &[(f32, f32)] = if on_gpu {
        &[(0.01, 0.012), (0.05, 0.04)]
    } else {
        &[]
    };
    for &(radius, other) in ends {
        let mut one = gated.clone();
        one.masks[AUTO_BRUSH].refine = Refine {
            amount: 100.0,
            radius,
            sensitivity: 50.0,
        };
        timed_render(&gpu, &mut develop, &one, VIEWER_SIZE);
        let (held_p50, held_p95, _) = stepped_with(&gpu, &mut develop, &one, |edit, value| {
            edit.masks[AUTO_BRUSH].refine.sensitivity = 50.0 + 40.0 * value;
        });
        let (taken_p50, taken_p95, _) = stepped_with(&gpu, &mut develop, &one, |edit, value| {
            let odd = ((value + 1.0) * (RENDERS - 1) as f32 / 2.0).round() as u32 % 2 == 1;
            edit.masks[AUTO_BRUSH].refine.radius = if odd { other } else { radius };
        });
        println!(
            "the refine of one mask at {VIEWER_SIZE:?}, Radius {radius} (not asserted): source moments held p50 {held_p50:.2} ms, p95 {held_p95:.2} ms; taken again p50 {taken_p50:.2} ms, p95 {taken_p95:.2} ms; so the source moments about {:.2} ms and one gather about {:.2} ms",
            (taken_p50 - held_p50).max(0.0),
            held_p50 / 3.0
        );
    }
    let mut refined_view_p95 = None;
    // The p95 of each Refine edges slider step at 100 percent, with its
    // slider and its radius.
    let mut refine_step_p95: Vec<(&str, f32, f64)> = Vec::new();
    if info.device_type == wgpu::DeviceType::Cpu {
        println!("zoomed refine lines skipped on a CPU adapter");
    } else {
        let full = (photo.width, photo.height);
        let (actual, _) = zoomed_view(full, 1);
        let first = timed_view(&gpu, &mut develop, &refined, &actual);
        println!(
            "first render of the padded window at 100 percent with two refined masks (window {:?}): {first:.2} ms",
            (actual.window.2, actual.window.3)
        );
        let (step_p50, step_p95, step_max) = stepped_view(&gpu, &mut develop, &refined, &actual);
        println!(
            "slider step at 100 percent with two refined masks, {RENDERS} renders: p50 {step_p50:.2} ms, p95 {step_p95:.2} ms, max {step_max:.2} ms"
        );
        // Each pair keeps the pad of the window, so no step replaces it.
        for (radius, other) in [(0.01, 0.011), (0.05, 0.049)] {
            let mut one = gated.clone();
            one.masks[AUTO_BRUSH].refine = Refine {
                amount: 100.0,
                radius,
                sensitivity: 50.0,
            };
            timed_view(&gpu, &mut develop, &one, &actual);
            let (tiles, builds) = (develop.refine_tiles(), develop.refine_builds());
            let (held_p50, held_p95, _) =
                stepped_view_with(&gpu, &mut develop, &one, &actual, |edit, value| {
                    edit.masks[AUTO_BRUSH].refine.sensitivity = 50.0 + 40.0 * value;
                });
            let (taken_p50, taken_p95, _) =
                stepped_view_with(&gpu, &mut develop, &one, &actual, |edit, value| {
                    let odd = ((value + 1.0) * (RENDERS - 1) as f32 / 2.0).round() as u32 % 2 == 1;
                    edit.masks[AUTO_BRUSH].refine.radius = if odd { other } else { radius };
                });
            // The tiles of a refine, over every refine of the two runs.
            let refines =
                develop.refine_builds().0 - builds.0 + develop.refine_builds().1 - builds.1;
            let tiles_a_refine = f64::from(develop.refine_tiles() - tiles) / refines.max(1) as f64;
            println!(
                "the refine of one mask at 100 percent, Radius {radius} (not asserted): source moments held p50 {held_p50:.2} ms, p95 {held_p95:.2} ms; taken again p50 {taken_p50:.2} ms, p95 {taken_p95:.2} ms; so the source moments about {:.2} ms and one gather about {:.2} ms; refine_tiles {tiles_a_refine:.2} a refine over {refines} refines",
                (taken_p50 - held_p50).max(0.0),
                held_p50 / 3.0
            );
        }
        // A step of each Refine edges slider at 100 percent, the mask refined
        // whole each time, at each end of the radius, held to the gate. Each
        // Radius pair keeps the pad of the window.
        for (radius, other) in [(0.05, 0.049), (0.01, 0.011)] {
            let mut one = gated.clone();
            one.masks[AUTO_BRUSH].refine = Refine {
                amount: 100.0,
                radius,
                sensitivity: 50.0,
            };
            let amount = |edit: &mut PhotoEdit, value: f32| {
                edit.masks[AUTO_BRUSH].refine.amount = 50.0 + 49.0 * value;
            };
            let radius_step = |edit: &mut PhotoEdit, value: f32| {
                let odd = ((value + 1.0) * (RENDERS - 1) as f32 / 2.0).round() as u32 % 2 == 1;
                edit.masks[AUTO_BRUSH].refine.radius = if odd { other } else { radius };
            };
            let sensitivity = |edit: &mut PhotoEdit, value: f32| {
                edit.masks[AUTO_BRUSH].refine.sensitivity = 50.0 + 40.0 * value;
            };
            // What a slider line puts into the edit for a value from -1 to 1.
            type Step<'a> = &'a dyn Fn(&mut PhotoEdit, f32);
            let sliders: [(&str, Step); 3] = [
                ("Amount", &amount),
                ("Radius", &radius_step),
                ("Sensitivity", &sensitivity),
            ];
            for (slider, step) in sliders {
                timed_view(&gpu, &mut develop, &one, &actual);
                let (tiles, builds) = (develop.refine_tiles(), develop.refine_builds());
                let (p50, p95, max) = stepped_view_with(&gpu, &mut develop, &one, &actual, step);
                let refines =
                    develop.refine_builds().0 - builds.0 + develop.refine_builds().1 - builds.1;
                let tiles_a_refine =
                    f64::from(develop.refine_tiles() - tiles) / refines.max(1) as f64;
                println!(
                    "Refine edges {slider} slider step at 100 percent, Radius {radius}, the mask refined whole each time, {RENDERS} renders: p50 {p50:.2} ms, p95 {p95:.2} ms, max {max:.2} ms; refine_tiles {tiles_a_refine:.2} a refine over {refines} refines"
                );
                refine_step_p95.push((slider, radius, p95));
            }
        }
        let pen = Stroke {
            auto: true,
            sensitivity: 60.0,
            pressure_size: true,
            pressure_flow: true,
            ..Stroke::default()
        };
        timed_view(&gpu, &mut develop, &refined, &actual);
        let (whole, parts) = develop.refine_builds();
        let (paint_p50, paint_p95, paint_max) =
            painted_view_with(&gpu, &mut develop, &refined, &actual, AUTO_BRUSH, &pen);
        println!(
            "painting an auto stroke with a pen into a refined mask at 100 percent, one appended point a frame, {RENDERS} frames: p50 {paint_p50:.2} ms, p95 {paint_p95:.2} ms, max {paint_max:.2} ms"
        );
        assert_eq!(
            (
                develop.refine_builds().0 - whole,
                develop.refine_builds().1 - parts
            ),
            (0, RENDERS as u64 + 1),
            "painting refined the whole mask again"
        );
        let (deep, _) = zoomed_view(full, 4);
        timed_view(&gpu, &mut develop, &refined, &deep);
        let (deep_p50, deep_p95, deep_max) =
            painted_view_with(&gpu, &mut develop, &refined, &deep, AUTO_BRUSH, &pen);
        println!(
            "painting an auto stroke with a pen into a refined mask at 400 percent, one appended point a frame, {RENDERS} frames: p50 {deep_p50:.2} ms, p95 {deep_p95:.2} ms, max {deep_max:.2} ms"
        );
        refined_view_p95 = Some((step_p95, paint_p95, deep_p95));
    }

    // Shift edge -1 percent, Feather 1 percent and Contrast 50 on the two
    // refined masks. An edge slider draws an edge stage over the whole frame
    // on every render, so the edge lines run on a GPU alone.
    let edged = with_edged_masks(&refined, EDGED.edge);
    let mut edged_p95 = None;
    if on_gpu {
        let first = timed_render(&gpu, &mut develop, &edged, VIEWER_SIZE);
        println!(
            "first render with the three edge controls on the two refined masks at {VIEWER_SIZE:?} (Shift edge -1 percent, Feather 1 percent, Contrast 50): {first:.2} ms"
        );
        let drawn = Drawn::of(&develop);
        let (p50, p95, max) = stepped(&gpu, &mut develop, &edged);
        drawn.hold(&develop, 3, "a slider of the develop chain");
        println!(
            "slider step with two refined masks and the three edge controls on at {VIEWER_SIZE:?} over {RENDERS} renders: p50 {p50:.2} ms, p95 {p95:.2} ms, max {max:.2} ms"
        );
        let [shift_p95, feather_p95, contrast_p95] = edge_slider_steps(
            &gpu,
            &mut develop,
            &edged,
            None,
            &EDGED,
            &format!("at {VIEWER_SIZE:?}"),
            true,
        );
        edged_p95 = Some([p95, shift_p95, feather_p95, contrast_p95]);
    } else {
        println!("edge lines at the viewer size skipped on a CPU adapter");
    }
    let mut edged_view_p95 = None;
    let mut wide_view_p95 = None;
    if info.device_type == wgpu::DeviceType::Cpu {
        println!("zoomed edge lines skipped on a CPU adapter");
    } else {
        let full = (photo.width, photo.height);
        let (actual, _) = zoomed_view(full, 1);
        let first = timed_view(&gpu, &mut develop, &edged, &actual);
        println!(
            "first render of the padded window at 100 percent with the three edge controls on the two refined masks: {first:.2} ms"
        );
        let drawn = Drawn::of(&develop);
        let (step_p50, step_p95, step_max) = stepped_view(&gpu, &mut develop, &edged, &actual);
        drawn.hold(&develop, 3, "a slider of the develop chain");
        println!(
            "slider step at 100 percent with two refined masks and the three edge controls on, {RENDERS} renders: p50 {step_p50:.2} ms, p95 {step_p95:.2} ms, max {step_max:.2} ms"
        );
        let [shift_p95, feather_p95, contrast_p95] = edge_slider_steps(
            &gpu,
            &mut develop,
            &edged,
            Some(&actual),
            &EDGED,
            "at 100 percent",
            true,
        );
        let pen = Stroke {
            auto: true,
            sensitivity: 60.0,
            pressure_size: true,
            pressure_flow: true,
            ..Stroke::default()
        };
        let mut paint = |view: &ViewWindow, zoom: &str| {
            timed_view(&gpu, &mut develop, &edged, view);
            let drawn = Drawn::of(&develop);
            let shifted = develop.edge_builds().0[0];
            let (p50, p95, max) =
                painted_view_with(&gpu, &mut develop, &edged, view, AUTO_BRUSH, &pen);
            println!(
                "painting an auto stroke with a pen into a refined mask with the three edge controls on at {zoom} percent, one appended point a frame, {RENDERS} frames: p50 {p50:.2} ms, p95 {p95:.2} ms, max {max:.2} ms; {}",
                drawn.passes_since(&develop, RENDERS + 1)
            );
            assert_eq!(
                (
                    develop.refine_builds().0 - drawn.refines.0,
                    develop.refine_builds().1 - drawn.refines.1
                ),
                (0, RENDERS as u64 + 1),
                "painting refined the whole edged mask again"
            );
            assert_eq!(
                develop.edge_builds().0[0],
                shifted,
                "painting shifted the whole edged mask again"
            );
            p95
        };
        let paint_p95 = paint(&actual, "100");
        let (deep, _) = zoomed_view(full, 4);
        let deep_p95 = paint(&deep, "400");

        // For the record: each control's passes alone. Then a Shift edge, a
        // Feather and a Contrast slider step with the three at 5 percent, held
        // to the gate.
        edge_passes_alone(&gpu, &mut develop, &refined, &actual, &EDGED);
        edge_passes_alone(&gpu, &mut develop, &refined, &actual, &EDGED_WIDE);
        let wide = with_edged_masks(&refined, EDGED_WIDE.edge);
        let first = timed_view(&gpu, &mut develop, &wide, &actual);
        println!(
            "first render of the padded window at 100 percent with the three edge controls at 5 percent on the two refined masks: {first:.2} ms"
        );
        wide_view_p95 = Some(edge_slider_steps(
            &gpu,
            &mut develop,
            &wide,
            Some(&actual),
            &EDGED_WIDE,
            "at 100 percent",
            true,
        ));
        edged_view_p95 = Some([
            step_p95,
            shift_p95,
            feather_p95,
            contrast_p95,
            paint_p95,
            deep_p95,
        ]);
    }

    if std::env::var(GATE).as_deref() == Ok("1") {
        if let Some((p95, max, over, frames)) = edge_pan {
            assert!(
                p95 < GATE_MS,
                "p95 of a steady pan across the window edge, {p95:.2} ms, is not under {GATE_MS} ms"
            );
            // At most one frame of the pan at or over the gate, not its max under it:
            // the spikes are device waits on random frames (T-26, 2026-09-26; D-10, Trent).
            assert!(
                over <= 1,
                "frames over {GATE_MS} ms of a steady pan across the window edge: {over} of {frames}, not at most 1; max {max:.2} ms"
            );
        }
        if let Some([develop_p95, shift_p95, feather_p95, contrast_p95]) = edged_p95 {
            for (p95, what) in [
                (develop_p95, "a slider step with two edged masks"),
                (shift_p95, "a Shift edge slider step"),
                (feather_p95, "a Feather slider step"),
                (contrast_p95, "a Contrast slider step"),
            ] {
                assert!(
                    p95 < GATE_MS,
                    "p95 of {what} at {VIEWER_SIZE:?}, {p95:.2} ms, is not under {GATE_MS} ms"
                );
            }
        }
        if let Some(
            [
                step_p95,
                shift_p95,
                feather_p95,
                contrast_p95,
                paint_p95,
                deep_p95,
            ],
        ) = edged_view_p95
        {
            for (p95, what) in [
                (
                    step_p95,
                    "a slider step at 100 percent with two edged masks",
                ),
                (shift_p95, "a Shift edge slider step at 100 percent"),
                (feather_p95, "a Feather slider step at 100 percent"),
                (contrast_p95, "a Contrast slider step at 100 percent"),
                (paint_p95, "painting into an edged mask at 100 percent"),
                (deep_p95, "painting into an edged mask at 400 percent"),
            ] {
                assert!(
                    p95 < GATE_MS,
                    "p95 of {what}, {p95:.2} ms, is not under {GATE_MS} ms"
                );
            }
        }
        if let Some([shift_p95, feather_p95, contrast_p95]) = wide_view_p95 {
            for (p95, what) in [
                (shift_p95, "Shift edge"),
                (feather_p95, "Feather"),
                (contrast_p95, "Contrast"),
            ] {
                assert!(
                    p95 < GATE_MS,
                    "p95 of a {what} slider step at 100 percent with the three at 5 percent, {p95:.2} ms, is not under {GATE_MS} ms"
                );
            }
        }
        for (slider, radius, p95) in &refine_step_p95 {
            assert!(
                *p95 < GATE_MS,
                "p95 of a Refine edges {slider} slider step at 100 percent, Radius {radius}, {p95:.2} ms, is not under {GATE_MS} ms"
            );
        }
        assert!(
            refined_p95 < GATE_MS,
            "p95 of a slider step with two refined masks, {refined_p95:.2} ms, is not under {GATE_MS} ms"
        );
        if let Some(amount_p95) = amount_p95 {
            assert!(
                amount_p95 < GATE_MS,
                "p95 of a Refine edges slider step, {amount_p95:.2} ms, is not under {GATE_MS} ms"
            );
        }
        if let Some((step_p95, paint_p95, deep_p95)) = refined_view_p95 {
            assert!(
                step_p95 < GATE_MS,
                "p95 of a slider step at 100 percent with two refined masks, {step_p95:.2} ms, is not under {GATE_MS} ms"
            );
            assert!(
                paint_p95 < GATE_MS,
                "p95 of painting into a refined mask at 100 percent, {paint_p95:.2} ms, is not under {GATE_MS} ms"
            );
            assert!(
                deep_p95 < GATE_MS,
                "p95 of painting into a refined mask at 400 percent, {deep_p95:.2} ms, is not under {GATE_MS} ms"
            );
        }
        assert!(
            six_p95 < GATE_MS,
            "p95 of a slider step with the six masks, {six_p95:.2} ms, is not under {GATE_MS} ms"
        );
        if let Some((step_p95, paint_p95, deep_p95)) = auto_p95 {
            assert!(
                step_p95 < GATE_MS,
                "p95 of a slider step at 100 percent with the six masks, {step_p95:.2} ms, is not under {GATE_MS} ms"
            );
            assert!(
                paint_p95 < GATE_MS,
                "p95 of painting an auto stroke at 100 percent, {paint_p95:.2} ms, is not under {GATE_MS} ms"
            );
            assert!(
                deep_p95 < GATE_MS,
                "p95 of painting an auto stroke at 400 percent, {deep_p95:.2} ms, is not under {GATE_MS} ms"
            );
        }
        assert!(
            five_p95 < GATE_MS,
            "p95 of a slider step with the five masks, {five_p95:.2} ms, is not under {GATE_MS} ms"
        );
        if let Some((step_p95, paint_p95, deep_p95)) = brush_p95 {
            assert!(
                step_p95 < GATE_MS,
                "p95 of a slider step at 100 percent with the five masks, {step_p95:.2} ms, is not under {GATE_MS} ms"
            );
            assert!(
                paint_p95 < GATE_MS,
                "p95 of painting at 100 percent, {paint_p95:.2} ms, is not under {GATE_MS} ms"
            );
            assert!(
                deep_p95 < GATE_MS,
                "p95 of painting at 400 percent, {deep_p95:.2} ms, is not under {GATE_MS} ms"
            );
        }
        assert!(p95 < GATE_MS, "p95 {p95:.2} ms is not under {GATE_MS} ms");
        assert!(
            all_p95 < GATE_MS,
            "p95 with every operator on, {all_p95:.2} ms, is not under {GATE_MS} ms"
        );
        assert!(
            masks_p95 < GATE_MS,
            "p95 with four full masks on, {masks_p95:.2} ms, is not under {GATE_MS} ms"
        );
        if let Some((step_p95, pan_p95, deep_p95)) = zoomed_p95 {
            assert!(
                step_p95 < GATE_MS,
                "p95 of a slider step at 100 percent, {step_p95:.2} ms, is not under {GATE_MS} ms"
            );
            assert!(
                pan_p95 < GATE_MS,
                "p95 of a pan inside the padded window, {pan_p95:.2} ms, is not under {GATE_MS} ms"
            );
            assert!(
                deep_p95 < GATE_MS,
                "p95 of a slider step at 400 percent, {deep_p95:.2} ms, is not under {GATE_MS} ms"
            );
        }
    } else {
        println!("{GATE} is not set, so the {GATE_MS} ms gate is printed and not asserted");
    }
}
