//! The Reel export: runs the timeline through the video pass and the
//! develop graph at 1080x1920 and hands each frame to the writer. Three
//! threads overlap: one decodes the clips in timeline order and keeps a
//! bounded queue of frames, the caller's thread uploads, renders and reads
//! back, and one encodes and muxes. Audio follows the same clip map and is
//! written first so the muxer interleaves it.

use std::path::Path;
use std::sync::mpsc::{self, SyncSender};
use std::time::Instant;

use gamut_core::{Crop, CropAspect, CropRect, Project, Track};
use gamut_gpu::{Develop, Headless, PendingReadback, Readback};
use gamut_media::audio::AudioSource;
use gamut_media::ffmpeg::Rational;
use gamut_media::ffmpeg::frame;
use gamut_media::video_export::{AUDIO_RATE, ReelWriter};
use gamut_media::{DecodedFrame, FramePlanes, VideoError, VideoSource};

use crate::headless::HeadlessError;
use crate::player::MediaInfo;
use crate::project::{for_video, load};

/// The Reel size.
pub const SIZE: (u32, u32) = (1080, 1920);

/// How many decoded frames wait for the render.
const DECODE_QUEUE: usize = 3;

/// How many rendered frames wait for the encoder.
const ENCODE_QUEUE: usize = 4;

/// How many transfer buffers the decode thread keeps in flight, so no
/// frame pays for fresh pages once the pool is warm.
const TRANSFER_BUFFERS: usize = DECODE_QUEUE + 3;

/// The output frame rate for a source rate: 24, 25, 30 or 60 when the
/// source is one of those, else 30.
pub fn output_rate(source_rate: f64) -> u32 {
    let rounded = source_rate.round();
    if [24.0, 25.0, 30.0, 60.0].contains(&rounded) {
        rounded as u32
    } else {
        30
    }
}

enum ToWriter {
    Video(Vec<u8>, i64),
    Audio(Vec<f32>),
}

/// A source frame for one output frame, or `None` when the previous one
/// serves again.
type Decoded = Option<Box<DecodedFrame>>;

/// Exports a video or a project file to `out`. Returns the wall time in
/// seconds.
pub fn export_file(input: &Path, out: &Path) -> Result<f64, HeadlessError> {
    let loaded = if Project::is_project_path(input) {
        load(input).map_err(HeadlessError::Edit)?
    } else {
        let info = MediaInfo::probe(input).map_err(HeadlessError::Video)?;
        for_video(input, &info)
    };
    export(&loaded.project, &loaded.dir, out)
}

/// Exports `project`, whose media paths are relative to `dir`, to `out`.
/// Returns the wall time in seconds.
pub fn export(project: &Project, dir: &Path, out: &Path) -> Result<f64, HeadlessError> {
    let started = Instant::now();
    let media: Vec<MediaInfo> = (0..project.media.len())
        .map(|index| {
            let path = project.media_path(dir, index).expect("index in range");
            MediaInfo::probe(&path).map_err(HeadlessError::Video)
        })
        .collect::<Result<_, _>>()?;
    let track = project.track.clone();
    let Some(first) = track.clips.first() else {
        return Err(HeadlessError::Edit("the track is empty".to_string()));
    };
    let first_media = media
        .get(first.media)
        .ok_or_else(|| HeadlessError::Edit("a clip names no media".to_string()))?;
    let fps = output_rate(first_media.frame_rate);
    let frame_rate = Rational::new(fps as i32, 1);
    let duration = track.duration();
    let frames = ((duration * f64::from(fps)) - 1e-6).ceil().max(1.0) as i64;
    let crop = if project.crop.rect == CropRect::FULL {
        Crop::fitted(CropAspect::Story9x16, first_media.width, first_media.height)
    } else {
        project.crop
    };
    log::info!(
        "exporting {frames} frames at {fps} fps ({duration:.3} s) to {} with crop {:?}",
        out.display(),
        crop.rect
    );

    let gpu = Headless::new().ok_or(HeadlessError::NoAdapter)?;
    log::info!("adapter: {}", gpu.describe());
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    let readback = Readback::new(&gpu.device);

    // The writer thread.
    let (to_writer, from_render) = mpsc::sync_channel::<ToWriter>(ENCODE_QUEUE);
    let out_path = out.to_path_buf();
    let writer = std::thread::spawn(move || -> Result<(), VideoError> {
        let mut writer = ReelWriter::create(&out_path, SIZE.0, SIZE.1, frame_rate, AUDIO_RATE)?;
        for message in from_render {
            match message {
                ToWriter::Video(rgba, pts) => writer.write_video(&rgba, pts)?,
                ToWriter::Audio(samples) => writer.write_audio(&samples)?,
            }
        }
        writer.finish()
    });

    // The decode thread, started first so its queue fills while the audio
    // is written. Uploaded frames go back to it through `recycle` so their
    // buffers serve the next transfer.
    let (to_render, from_decode) = mpsc::sync_channel::<Decoded>(DECODE_QUEUE);
    let (recycle, recycled) = mpsc::channel::<frame::Video>();
    let decode_track = track.clone();
    let decode_media = media.clone();
    let decoder = std::thread::spawn(move || -> Result<(), VideoError> {
        decode(
            &decode_track,
            &decode_media,
            fps,
            frames,
            &to_render,
            &recycled,
        )
    });

    // Audio next, so the muxer has it to interleave.
    let audio_result = write_audio(&track, &media, &to_writer);

    log::info!("setup took {:.2} s", started.elapsed().as_secs_f64());
    let render_result = (|| -> Result<(), HeadlessError> {
        audio_result?;
        let mut render_ms = 0.0;
        let mut upload_ms = 0.0;
        let mut wait_ms = 0.0;
        let mut pending: Option<(PendingReadback, i64)> = None;
        for i in 0..frames {
            let Ok(decoded) = from_decode.recv() else {
                break;
            };
            let clip = track.clip_at(i as f64 / f64::from(fps)).map(|(c, _)| c);
            let info = clip.and_then(|c| media.get(track.clips[c].media));
            let t0 = Instant::now();
            if let (Some(frame), Some(info)) = (decoded, info) {
                develop.set_video_frame(frame.planes(), info.colour, info.rotation);
                if let DecodedFrame::Raw(raw) = *frame {
                    let _ = recycle.send(raw.into_buffer());
                }
            }
            let t1 = Instant::now();
            if !develop.has_video() {
                return Err(HeadlessError::Edit(format!(
                    "no frame decoded for frame {i}"
                )));
            }
            let view = develop
                .render_crop(&project.edit, crop.rect, SIZE)
                .expect("a frame is set");
            let started = readback.start(&gpu.device, &gpu.queue, view, SIZE.0, SIZE.1);
            let t2 = Instant::now();
            // The previous frame's bytes are ready once this frame is
            // queued behind it, so the wait overlaps the GPU's work.
            if let Some((previous, pts)) = pending.take() {
                let rgba = previous.wait(&gpu.device);
                if to_writer.send(ToWriter::Video(rgba, pts)).is_err() {
                    break;
                }
            }
            pending = Some((started, i));
            upload_ms += t1.duration_since(t0).as_secs_f64() * 1000.0;
            render_ms += t2.duration_since(t1).as_secs_f64() * 1000.0;
            wait_ms += t2.elapsed().as_secs_f64() * 1000.0;
        }
        if let Some((previous, pts)) = pending.take() {
            let rgba = previous.wait(&gpu.device);
            let _ = to_writer.send(ToWriter::Video(rgba, pts));
        }
        log::info!(
            "render thread: upload {:.2} ms, render {:.2} ms, readback wait {:.2} ms per frame",
            upload_ms / frames as f64,
            render_ms / frames as f64,
            wait_ms / frames as f64
        );
        Ok(())
    })();
    log::info!(
        "render loop done at {:.2} s",
        started.elapsed().as_secs_f64()
    );
    drop(to_writer);
    drop(from_decode);
    drop(recycle);
    let decode_result = decoder.join().unwrap_or_else(|_| {
        Err(VideoError::Encode {
            path: out.to_path_buf(),
            source: gamut_media::ffmpeg::Error::Unknown,
        })
    });
    let writer_result = writer.join().unwrap_or_else(|_| {
        Err(VideoError::Encode {
            path: out.to_path_buf(),
            source: gamut_media::ffmpeg::Error::Unknown,
        })
    });
    render_result?;
    decode_result.map_err(HeadlessError::Video)?;
    writer_result.map_err(HeadlessError::Video)?;
    let seconds = started.elapsed().as_secs_f64();
    log::info!("exported {} in {seconds:.2} s", out.display());
    Ok(seconds)
}

/// Decodes the source frame for each output frame in order and sends it,
/// or `None` when the previous frame serves again.
fn decode(
    track: &Track,
    media: &[MediaInfo],
    fps: u32,
    frames: i64,
    to_render: &SyncSender<Decoded>,
    recycled: &mpsc::Receiver<frame::Video>,
) -> Result<(), VideoError> {
    let mut source: Option<(usize, VideoSource)> = None;
    let mut current_clip = None;
    let mut sent_pts: Option<f64> = None;
    let mut lookahead: Option<DecodedFrame> = None;
    let mut unsent: Option<DecodedFrame> = None;
    let started = Instant::now();
    let mut decode_ms = 0.0;
    let mut copy_ms = 0.0;
    let mut send_ms = 0.0;
    let mut decoded_frames = 0u64;
    let mut pool: Vec<frame::Video> = Vec::new();
    let mut allocated = 0usize;
    let mut layout: Option<(gamut_media::ffmpeg::format::Pixel, u32, u32)> = None;
    for i in 0..frames {
        while let Ok(buffer) = recycled.try_recv() {
            pool.push(buffer);
        }
        let t = i as f64 / f64::from(fps);
        let Some((clip_index, offset)) = track.clip_at(t) else {
            if to_render.send(None).is_err() {
                return Ok(());
            }
            continue;
        };
        let clip = track.clips[clip_index];
        let source_t = clip.source_in + offset;
        if current_clip != Some(clip_index) {
            let path = &media[clip.media].path;
            let reopen = source.as_ref().is_none_or(|(m, _)| *m != clip.media);
            if reopen {
                let opened = VideoSource::open(path)?;
                log::info!(
                    "reel decoder for {}: {:?}",
                    path.display(),
                    opened.decoder_kind
                );
                source = Some((clip.media, opened));
            }
            let (_, video) = source.as_mut().expect("opened above");
            video.seek(source_t)?;
            current_clip = Some(clip_index);
            sent_pts = None;
            lookahead = None;
            unsent = None;
        }
        let (_, video) = source.as_mut().expect("opened above");
        let half = video.frame_seconds() * 0.5;
        // Advance until the lookahead frame is past the wanted time.
        loop {
            match &lookahead {
                Some(frame) if frame.pts_seconds() <= source_t + half => {
                    // A frame that was never sent is dropped; the pool
                    // refills from the render thread.
                    unsent = lookahead.take();
                }
                Some(_) => break,
                None => {
                    let reuse = pool.pop().or_else(|| {
                        // A fresh buffer of the known layout, up to the cap.
                        let (format, w, h) = layout?;
                        (allocated < TRANSFER_BUFFERS).then(|| {
                            allocated += 1;
                            frame::Video::new(format, w, h)
                        })
                    });
                    match video.next_frame_raw(reuse)? {
                        Some(frame) => {
                            decode_ms += video.last_timing.decode_ms;
                            copy_ms += video.last_timing.copy_ms;
                            decoded_frames += 1;
                            if let (None, DecodedFrame::Raw(raw)) = (layout, &frame) {
                                layout = Some((raw.pixel(), raw.width(), raw.height()));
                                allocated = 1;
                            }
                            lookahead = Some(frame);
                        }
                        None => break,
                    }
                }
            }
        }
        if unsent.is_none() && sent_pts.is_none() {
            // The seek landed after the wanted time: use the first frame.
            unsent = lookahead.take();
        }
        let message = match unsent.take() {
            Some(frame) => {
                sent_pts = Some(frame.pts_seconds());
                Some(Box::new(frame))
            }
            None => None,
        };
        let sending = Instant::now();
        if to_render.send(message).is_err() {
            return Ok(());
        }
        send_ms += sending.elapsed().as_secs_f64() * 1000.0;
    }
    let n = decoded_frames.max(1) as f64;
    log::info!(
        "decode thread: {decoded_frames} frames in {:.2} s, decode {:.2} ms, copy {:.2} ms per frame, waiting on the render {:.2} ms per frame",
        started.elapsed().as_secs_f64(),
        decode_ms / n,
        copy_ms / n,
        send_ms / frames.max(1) as f64
    );
    Ok(())
}

/// Writes the audio of every clip in order, silence where a media file has
/// none, padded or cut to each clip's exact length.
fn write_audio(
    track: &Track,
    media: &[MediaInfo],
    to_writer: &SyncSender<ToWriter>,
) -> Result<(), HeadlessError> {
    let rate = f64::from(AUDIO_RATE);
    for clip in &track.clips {
        let info = &media[clip.media];
        let wanted = (clip.duration() * rate).round() as usize;
        let mut written = 0usize;
        if info.has_audio {
            let mut audio =
                AudioSource::open(&info.path, AUDIO_RATE, 2).map_err(HeadlessError::Video)?;
            audio.seek(clip.source_in).map_err(HeadlessError::Video)?;
            while written < wanted {
                let Some(chunk) = audio.next_samples().map_err(HeadlessError::Video)? else {
                    break;
                };
                let frames = chunk.samples.len() / 2;
                let keep = frames.min(wanted - written);
                let samples = chunk.samples[..keep * 2].to_vec();
                written += keep;
                if to_writer.send(ToWriter::Audio(samples)).is_err() {
                    return Ok(());
                }
            }
        }
        if written < wanted {
            let silence = vec![0.0f32; (wanted - written) * 2];
            if to_writer.send(ToWriter::Audio(silence)).is_err() {
                return Ok(());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::output_rate;

    #[test]
    fn the_output_rate_keeps_the_common_rates_and_defaults_to_30() {
        assert_eq!(output_rate(23.976), 24);
        assert_eq!(output_rate(25.0), 25);
        assert_eq!(output_rate(29.97), 30);
        assert_eq!(output_rate(59.94), 60);
        assert_eq!(output_rate(120.0), 30);
        assert_eq!(output_rate(0.0), 30);
    }
}
