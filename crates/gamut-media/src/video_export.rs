//! Writing the Reel: an MP4 with the moov atom first, H.264 High from
//! h264_nvenc at 1080x1920 in yuv420p with a closed one second GOP, and
//! AAC at 128 kbps, 48 kHz, stereo. Video comes in as sRGB RGBA bytes
//! from the GPU readback and is converted to yuv420p with the BT.709
//! coefficients in limited range on every core (swscale does this on one
//! thread and was the slowest link at 4K); audio comes in as interleaved
//! f32 and goes through swresample into planar frames of the encoder's
//! size.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffmpeg::codec::encoder::{audio, video};
use ffmpeg::format::sample::Type as SampleType;
use ffmpeg::format::{Pixel, Sample};
use ffmpeg::software::resampling;
use ffmpeg::util::color;
use ffmpeg::{
    ChannelLayout, ChannelLayoutMask, Dictionary, Error, Packet, Rational, codec, format, frame,
};
use ffmpeg_the_third as ffmpeg;

use crate::video::VideoError;

/// The video bit rate the rate control aims at.
pub const VIDEO_BIT_RATE: usize = 12_000_000;

/// The video bit rate ceiling: Instagram's VBR cap.
pub const VIDEO_MAX_BIT_RATE: usize = 25_000_000;

/// The AAC bit rate.
pub const AUDIO_BIT_RATE: usize = 128_000;

/// The audio sample rate.
pub const AUDIO_RATE: u32 = 48_000;

/// The encoder name.
pub const VIDEO_ENCODER: &str = "h264_nvenc";

/// Whether h264_nvenc opens on this machine. The encoder is in every
/// build; opening it needs an NVIDIA GPU, which CI does not have.
pub fn nvenc_available() -> bool {
    crate::init_ffmpeg();
    let Some(codec) = ffmpeg::encoder::find_by_name(VIDEO_ENCODER) else {
        return false;
    };
    let Ok(mut context) = codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()
    else {
        return false;
    };
    context.set_width(256);
    context.set_height(256);
    context.set_format(Pixel::YUV420P);
    context.set_time_base(Rational::new(1, 30));
    context.set_frame_rate(Some(Rational::new(30, 1)));
    match context.open_with(Dictionary::new()) {
        Ok(_) => true,
        Err(error) => {
            log::info!("{VIDEO_ENCODER} does not open here: {error}");
            false
        }
    }
}

/// The MP4 being written.
pub struct ReelWriter {
    path: PathBuf,
    output: format::context::Output,
    video: video::Encoder,
    audio: audio::Encoder,
    video_stream: usize,
    audio_stream: usize,
    /// The time bases of the streams in the file.
    video_time_base: Rational,
    audio_time_base: Rational,
    /// The time bases the encoders stamp their packets in.
    encoder_video_time_base: Rational,
    encoder_audio_time_base: Rational,
    yuv: frame::Video,
    planes: Planes,
    convert_ms: f64,
    encode_ms: f64,
    video_frames: u64,
    resampler: resampling::Context,
    /// Interleaved samples waiting for a full encoder frame.
    fifo: Vec<f32>,
    audio_written: i64,
    frame_size: usize,
    width: u32,
    height: u32,
    finished: bool,
}

impl ReelWriter {
    /// Opens `path` for writing at `width` by `height` and `frame_rate`,
    /// with the audio at `audio_rate`.
    pub fn create(
        path: &Path,
        width: u32,
        height: u32,
        frame_rate: Rational,
        audio_rate: u32,
    ) -> Result<Self, VideoError> {
        crate::init_ffmpeg();
        let at = |source| VideoError::Encode {
            path: path.to_path_buf(),
            source,
        };
        let mut output = format::output(path).map_err(at)?;
        let global_header = output
            .format()
            .flags()
            .contains(format::Flags::GLOBAL_HEADER);

        let video_codec = ffmpeg::encoder::find_by_name(VIDEO_ENCODER)
            .ok_or_else(|| at(Error::EncoderNotFound))?;
        let video_time_base = frame_rate.invert();
        let mut video = codec::context::Context::new_with_codec(video_codec)
            .encoder()
            .video()
            .map_err(at)?;
        video.set_width(width);
        video.set_height(height);
        video.set_format(Pixel::YUV420P);
        video.set_time_base(video_time_base);
        video.set_frame_rate(Some(frame_rate));
        video
            .set_gop(frame_rate.numerator().max(1) as u32 / frame_rate.denominator().max(1) as u32);
        video.set_max_b_frames(2);
        video.set_bit_rate(VIDEO_BIT_RATE);
        video.set_max_bit_rate(VIDEO_MAX_BIT_RATE);
        video.set_colorspace(color::Space::BT709);
        video.set_color_range(color::Range::MPEG);
        if global_header {
            video.set_flags(codec::Flags::GLOBAL_HEADER);
        }
        let mut options = Dictionary::new();
        options.set("preset", "p4");
        options.set("profile", "high");
        options.set("rc", "vbr");
        options.set("forced-idr", "1");
        options.set("no-scenecut", "1");
        let video = video.open_with(options).map_err(at)?;
        let video_stream = {
            let mut stream = output.add_stream(video_codec).map_err(at)?;
            stream.copy_parameters_from_context(&video);
            stream.set_time_base(video_time_base);
            stream.set_avg_frame_rate(frame_rate);
            stream.index()
        };

        let audio_codec =
            ffmpeg::encoder::find_by_name("aac").ok_or_else(|| at(Error::EncoderNotFound))?;
        let audio_time_base = Rational::new(1, audio_rate as i32);
        let mut audio = codec::context::Context::new_with_codec(audio_codec)
            .encoder()
            .audio()
            .map_err(at)?;
        audio.set_rate(audio_rate as i32);
        audio.set_format(Sample::F32(SampleType::Planar));
        audio.set_ch_layout(ChannelLayout::STEREO);
        audio.set_bit_rate(AUDIO_BIT_RATE);
        audio.set_time_base(audio_time_base);
        if global_header {
            audio.set_flags(codec::Flags::GLOBAL_HEADER);
        }
        let audio = audio.open_with(Dictionary::new()).map_err(at)?;
        let frame_size = audio.frame_size().max(1) as usize;
        let audio_stream = {
            let mut stream = output.add_stream(audio_codec).map_err(at)?;
            stream.copy_parameters_from_context(&audio);
            stream.set_time_base(audio_time_base);
            stream.index()
        };

        let mut header = Dictionary::new();
        header.set("movflags", "+faststart");
        output.write_header_with(header).map_err(at)?;
        // The muxer may have chosen its own time bases.
        let encoder_video_time_base = video_time_base;
        let encoder_audio_time_base = audio_time_base;
        let video_time_base = output
            .stream(video_stream)
            .map(|s| s.time_base())
            .unwrap_or(video_time_base);
        let audio_time_base = output
            .stream(audio_stream)
            .map(|s| s.time_base())
            .unwrap_or(audio_time_base);

        let resampler = resampling::Context::get2(
            Sample::F32(SampleType::Packed),
            ChannelLayout::STEREO,
            audio_rate,
            Sample::F32(SampleType::Planar),
            ChannelLayout::STEREO,
            audio_rate,
        )
        .map_err(at)?;
        log::info!(
            "writing {} at {width}x{height}, {} fps, audio {audio_rate} Hz, frame size {frame_size}",
            path.display(),
            f64::from(frame_rate)
        );
        Ok(Self {
            path: path.to_path_buf(),
            output,
            video,
            audio,
            video_stream,
            audio_stream,
            video_time_base,
            audio_time_base,
            encoder_video_time_base,
            encoder_audio_time_base,
            yuv: frame::Video::new(Pixel::YUV420P, width, height),
            planes: Planes::new(width as usize, height as usize),
            convert_ms: 0.0,
            encode_ms: 0.0,
            video_frames: 0,
            resampler,
            fifo: Vec::new(),
            audio_written: 0,
            frame_size,
            width,
            height,
            finished: false,
        })
    }

    fn error(&self, source: Error) -> VideoError {
        VideoError::Encode {
            path: self.path.clone(),
            source,
        }
    }

    /// Encodes one frame of sRGB RGBA bytes (width times height times
    /// four) at frame number `pts_frame`.
    pub fn write_video(&mut self, rgba: &[u8], pts_frame: i64) -> Result<(), VideoError> {
        let (width, height) = (self.width as usize, self.height as usize);
        assert_eq!(
            rgba.len(),
            width * height * 4,
            "rgba is width times height times four bytes"
        );
        let started = Instant::now();
        rgba_to_yuv420p(rgba, width, height, &mut self.planes);
        for (index, plane) in [&self.planes.y, &self.planes.u, &self.planes.v]
            .into_iter()
            .enumerate()
        {
            let stride = self.yuv.stride(index);
            let row = if index == 0 { width } else { width.div_ceil(2) };
            let data = self.yuv.data_mut(index);
            for (y, source) in plane.chunks_exact(row).enumerate() {
                data[y * stride..y * stride + row].copy_from_slice(source);
            }
        }
        let converted = Instant::now();
        self.yuv.set_color_space(color::Space::BT709);
        self.yuv.set_color_range(color::Range::MPEG);
        self.yuv.set_pts(Some(pts_frame));
        self.video
            .send_frame(&self.yuv)
            .map_err(|e| self.error(e))?;
        self.drain_video()?;
        self.convert_ms += converted.duration_since(started).as_secs_f64() * 1000.0;
        self.encode_ms += converted.elapsed().as_secs_f64() * 1000.0;
        self.video_frames += 1;
        Ok(())
    }

    /// Queues interleaved stereo f32 samples at the audio rate and encodes
    /// every full frame they complete.
    pub fn write_audio(&mut self, samples: &[f32]) -> Result<(), VideoError> {
        self.fifo.extend_from_slice(samples);
        while self.fifo.len() >= self.frame_size * 2 {
            let chunk: Vec<f32> = self.fifo.drain(..self.frame_size * 2).collect();
            self.encode_audio(&chunk)?;
        }
        Ok(())
    }

    fn encode_audio(&mut self, interleaved: &[f32]) -> Result<(), VideoError> {
        let samples = interleaved.len() / 2;
        let mut packed = frame::Audio::new(
            Sample::F32(SampleType::Packed),
            samples,
            ChannelLayoutMask::STEREO,
        );
        packed.set_rate(AUDIO_RATE);
        packed.data_mut(0)[..interleaved.len() * 4]
            .copy_from_slice(bytemuck::cast_slice(interleaved));
        let mut planar = frame::Audio::empty();
        self.resampler
            .run(&packed, &mut planar)
            .map_err(|e| self.error(e))?;
        planar.set_pts(Some(self.audio_written));
        self.audio.send_frame(&planar).map_err(|e| self.error(e))?;
        self.audio_written += samples as i64;
        self.drain_audio()
    }

    fn drain_video(&mut self) -> Result<(), VideoError> {
        let mut packet = Packet::empty();
        loop {
            match self.video.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(self.video_stream);
                    packet.rescale_ts(self.encoder_video_time_base, self.video_time_base);
                    packet
                        .write_interleaved(&mut self.output)
                        .map_err(|e| self.error(e))?;
                }
                Err(Error::Eof) => return Ok(()),
                Err(Error::Other { errno }) if errno == libc::EAGAIN => return Ok(()),
                Err(e) => return Err(self.error(e)),
            }
        }
    }

    fn drain_audio(&mut self) -> Result<(), VideoError> {
        let mut packet = Packet::empty();
        loop {
            match self.audio.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(self.audio_stream);
                    packet.rescale_ts(self.encoder_audio_time_base, self.audio_time_base);
                    packet
                        .write_interleaved(&mut self.output)
                        .map_err(|e| self.error(e))?;
                }
                Err(Error::Eof) => return Ok(()),
                Err(Error::Other { errno }) if errno == libc::EAGAIN => return Ok(()),
                Err(e) => return Err(self.error(e)),
            }
        }
    }

    /// Encodes what audio is left, flushes both encoders and writes the
    /// trailer.
    pub fn finish(mut self) -> Result<(), VideoError> {
        if !self.fifo.is_empty() {
            let rest = std::mem::take(&mut self.fifo);
            self.encode_audio(&rest)?;
        }
        self.video.send_eof().map_err(|e| self.error(e))?;
        self.drain_video()?;
        self.audio.send_eof().map_err(|e| self.error(e))?;
        self.drain_audio()?;
        self.output.write_trailer().map_err(|e| self.error(e))?;
        self.finished = true;
        let frames = self.video_frames.max(1) as f64;
        log::info!(
            "wrote {}: {} video frames (convert {:.2} ms, encode {:.2} ms per frame), {} audio samples",
            self.path.display(),
            self.video_frames,
            self.convert_ms / frames,
            self.encode_ms / frames,
            self.audio_written
        );
        Ok(())
    }

    /// The samples of audio written so far, for keeping the sound in step
    /// with the picture.
    pub fn audio_samples_queued(&self) -> i64 {
        self.audio_written + (self.fifo.len() / 2) as i64
    }
}

/// The three tightly packed planes of a yuv420p frame.
pub struct Planes {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    width: usize,
    height: usize,
}

impl Planes {
    pub fn new(width: usize, height: usize) -> Self {
        let chroma = width.div_ceil(2) * height.div_ceil(2);
        Planes {
            y: vec![0; width * height],
            u: vec![0; chroma],
            v: vec![0; chroma],
            width,
            height,
        }
    }
}

/// The BT.709 limited range coefficients in 16.16 fixed point, on 8 bit
/// RGB: Y from 16 to 235, Cb and Cr from 16 to 240 around 128.
const Y_R: i32 = 11_966;
const Y_G: i32 = 40_254;
const Y_B: i32 = 4_064;
const CB_R: i32 = -6_596;
const CB_G: i32 = -22_189;
const CB_B: i32 = 28_784;
const CR_R: i32 = 28_784;
const CR_G: i32 = -26_145;
const CR_B: i32 = -2_639;

/// Converts sRGB RGBA bytes to yuv420p with the BT.709 matrix, chroma from
/// the mean of each 2 by 2 block, in bands of rows on every core.
pub fn rgba_to_yuv420p(rgba: &[u8], width: usize, height: usize, planes: &mut Planes) {
    assert_eq!((planes.width, planes.height), (width, height));
    let chroma_width = width.div_ceil(2);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8);
    let rows_per_band = height.div_ceil(2).div_ceil(threads).max(1) * 2;
    std::thread::scope(|scope| {
        let mut y_rest = planes.y.as_mut_slice();
        let mut u_rest = planes.u.as_mut_slice();
        let mut v_rest = planes.v.as_mut_slice();
        let mut row = 0;
        while row < height {
            let rows = rows_per_band.min(height - row);
            let chroma_rows = rows.div_ceil(2);
            let (y_band, y_next) = y_rest.split_at_mut(rows * width);
            let (u_band, u_next) = u_rest.split_at_mut(chroma_rows * chroma_width);
            let (v_band, v_next) = v_rest.split_at_mut(chroma_rows * chroma_width);
            y_rest = y_next;
            u_rest = u_next;
            v_rest = v_next;
            let source = &rgba[row * width * 4..(row + rows) * width * 4];
            scope.spawn(move || convert_band(source, width, rows, y_band, u_band, v_band));
            row += rows;
        }
    });
}

fn convert_band(rgba: &[u8], width: usize, rows: usize, y: &mut [u8], u: &mut [u8], v: &mut [u8]) {
    let chroma_width = width.div_ceil(2);
    let half = 1 << 15;
    for row in 0..rows {
        let line = &rgba[row * width * 4..(row + 1) * width * 4];
        let out = &mut y[row * width..(row + 1) * width];
        for (px, out) in line.as_chunks::<4>().0.iter().zip(out.iter_mut()) {
            let (r, g, b) = (i32::from(px[0]), i32::from(px[1]), i32::from(px[2]));
            *out = ((Y_R * r + Y_G * g + Y_B * b + half) >> 16).clamp(0, 255) as u8 + 16;
        }
    }
    for cy in 0..rows.div_ceil(2) {
        let top = cy * 2;
        let bottom = (top + 1).min(rows - 1);
        for cx in 0..chroma_width {
            let left = cx * 2;
            let right = (left + 1).min(width - 1);
            let (mut r, mut g, mut b) = (0i32, 0i32, 0i32);
            for (yy, xx) in [(top, left), (top, right), (bottom, left), (bottom, right)] {
                let at = (yy * width + xx) * 4;
                r += i32::from(rgba[at]);
                g += i32::from(rgba[at + 1]);
                b += i32::from(rgba[at + 2]);
            }
            // The block mean, in quarter units, folded into the fixed point.
            let cb = (CB_R * r + CB_G * g + CB_B * b + 4 * half) >> 18;
            let cr = (CR_R * r + CR_G * g + CR_B * b + 4 * half) >> 18;
            u[cy * chroma_width + cx] = (cb + 128).clamp(0, 255) as u8;
            v[cy * chroma_width + cx] = (cr + 128).clamp(0, 255) as u8;
        }
    }
}

impl Drop for ReelWriter {
    fn drop(&mut self) {
        if !self.finished {
            log::warn!("{} was dropped before finish", self.path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Planes, rgba_to_yuv420p};

    #[test]
    fn flat_colours_land_on_the_bt709_codes() {
        let (w, h) = (4, 4);
        let mut planes = Planes::new(w, h);
        for (rgb, expected) in [
            ([0u8, 0, 0], (16u8, 128u8, 128u8)),
            ([255, 255, 255], (235, 128, 128)),
            ([255, 0, 0], (63, 102, 240)),
            ([0, 255, 0], (173, 42, 26)),
            ([0, 0, 255], (32, 240, 118)),
            ([128, 128, 128], (126, 128, 128)),
        ] {
            let rgba: Vec<u8> = (0..w * h)
                .flat_map(|_| [rgb[0], rgb[1], rgb[2], 255])
                .collect();
            rgba_to_yuv420p(&rgba, w, h, &mut planes);
            assert_eq!((planes.y[5], planes.u[1], planes.v[1]), expected, "{rgb:?}");
        }
    }
}
