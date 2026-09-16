//! Opening a video: H.264 and HEVC in MP4 and MOV through ffmpeg, decoded
//! by NVDEC when a CUDA device answers and by the software decoder
//! otherwise. Every frame comes out as NV12 (8 bit) or P010 (10 bit, the ten
//! bits in the top of each 16 bit word) in system memory, tightly packed,
//! with its presentation time in seconds and the colour tags the GPU pass
//! needs. The display matrix of a phone clip is read from the stream side
//! data and carried as a rotation; the planes stay in their stored
//! orientation and the GPU pass turns them.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ffmpeg::format::Pixel;
use ffmpeg::util::color;
use ffmpeg::{Error, Packet, Rational, codec, format, frame, media};
use ffmpeg_the_third as ffmpeg;
use thiserror::Error;

use crate::hwaccel::{self, HwDevice};

/// Which decoder produced the frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoder {
    /// NVIDIA's hardware decoder through the CUDA hwaccel.
    Nvdec,
    /// ffmpeg's software decoder.
    Software,
}

pub use slate_color::video::{PlaneFormat, Transfer, VideoColour, YuvSpace};

/// One decoded frame in system memory, planes tightly packed.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoFrame {
    pub pts_seconds: f64,
    pub format: PlaneFormat,
    /// The stored width, before the rotation.
    pub width: u32,
    /// The stored height, before the rotation.
    pub height: u32,
    /// The luma plane, `height` rows of `y_stride` bytes.
    pub y: Vec<u8>,
    /// The chroma plane, `height / 2` rows of `uv_stride` bytes.
    pub uv: Vec<u8>,
    pub y_stride: usize,
    pub uv_stride: usize,
}

impl VideoFrame {
    /// The luma sample at `x`, `y` as a code in the plane's bit depth (0 to
    /// 255 for NV12, the 16 bit word for P010).
    pub fn luma(&self, x: u32, y: u32) -> u16 {
        self.sample(&self.y, self.y_stride, x, y)
    }

    /// The Cb and Cr samples of the chroma block at `x`, `y` (in chroma
    /// plane coordinates, half the frame size).
    pub fn chroma(&self, x: u32, y: u32) -> (u16, u16) {
        (
            self.sample(&self.uv, self.uv_stride, 2 * x, y),
            self.sample(&self.uv, self.uv_stride, 2 * x + 1, y),
        )
    }

    /// The luma code at `x`, `y` in the plane's bit depth: 0 to 255 for
    /// NV12, 0 to 1023 for P010.
    pub fn luma_code(&self, x: u32, y: u32) -> u32 {
        self.code(self.luma(x, y))
    }

    /// The Cb and Cr codes of the chroma block at `x`, `y` in the plane's
    /// bit depth.
    pub fn chroma_code(&self, x: u32, y: u32) -> (u32, u32) {
        let (cb, cr) = self.chroma(x, y);
        (self.code(cb), self.code(cr))
    }

    fn code(&self, word: u16) -> u32 {
        match self.format {
            PlaneFormat::Nv12 => u32::from(word),
            PlaneFormat::P010 => u32::from(word >> 6),
        }
    }

    fn sample(&self, plane: &[u8], stride: usize, x: u32, y: u32) -> u16 {
        let bytes = self.format.bytes_per_sample();
        let at = y as usize * stride + x as usize * bytes;
        match self.format {
            PlaneFormat::Nv12 => u16::from(plane[at]),
            PlaneFormat::P010 => u16::from_le_bytes([plane[at], plane[at + 1]]),
        }
    }
}

/// How long the parts of the last `next_frame` took, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DecodeTiming {
    /// Reading packets and receiving the frame from the decoder.
    pub decode_ms: f64,
    /// The transfer from CUDA memory and the copy into the packed planes.
    pub copy_ms: f64,
}

/// Why a video could not be opened or decoded.
#[derive(Debug, Error)]
pub enum VideoError {
    #[error("could not open {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: Error,
    },
    #[error("{path} has no video stream")]
    NoVideoStream { path: PathBuf },
    #[error("{path} has no audio stream")]
    NoAudioStream { path: PathBuf },
    #[error("could not open the decoder for {path}: {source}")]
    Decoder {
        path: PathBuf,
        #[source]
        source: Error,
    },
    #[error("decoding {path} failed: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: Error,
    },
    #[error("{path} decodes to {format:?}, which slate does not draw")]
    UnsupportedFormat { path: PathBuf, format: Pixel },
    #[error("{path} is not a video format slate opens (mp4, mov, m4v)")]
    Unsupported { path: PathBuf },
}

/// The extensions the open path accepts.
pub const VIDEO_EXTENSIONS: [&str; 3] = ["mp4", "mov", "m4v"];

/// Whether a path has one of the video extensions.
pub fn is_video_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| VIDEO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// An open video file and its decoder.
pub struct VideoSource {
    path: PathBuf,
    input: format::context::Input,
    stream_index: usize,
    decoder: codec::decoder::Video,
    /// Keeps the device alive for as long as the decoder runs.
    _device: Option<HwDevice>,
    time_base: Rational,
    start_time: i64,
    hardware: frame::Video,
    system: frame::Video,
    sent_eof: bool,
    finished: bool,
    /// Frames before this time are dropped after a seek.
    skip_until: Option<f64>,
    /// The stored size, before the rotation.
    pub coded_width: u32,
    pub coded_height: u32,
    /// The display size, after the rotation.
    pub width: u32,
    pub height: u32,
    /// Degrees clockwise the stored frame turns to be displayed: 0, 90, 180
    /// or 270.
    pub rotation: u32,
    pub frame_rate: Rational,
    pub duration_seconds: f64,
    pub colour: VideoColour,
    pub decoder_kind: Decoder,
    pub has_audio: bool,
    pub last_timing: DecodeTiming,
}

impl VideoSource {
    /// Opens `path` and its first video stream, through NVDEC when a CUDA
    /// device answers and the software decoder otherwise.
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        Self::open_with(path, true)
    }

    /// Opens `path` with the software decoder even when NVDEC is there.
    pub fn open_software(path: &Path) -> Result<Self, VideoError> {
        Self::open_with(path, false)
    }

    fn open_with(path: &Path, try_hardware: bool) -> Result<Self, VideoError> {
        crate::init_ffmpeg();
        if !is_video_path(path) {
            return Err(VideoError::Unsupported {
                path: path.to_path_buf(),
            });
        }
        let at = |source| VideoError::Open {
            path: path.to_path_buf(),
            source,
        };
        let input = format::input(path).map_err(at)?;
        let stream =
            input
                .streams()
                .best(media::Type::Video)
                .ok_or_else(|| VideoError::NoVideoStream {
                    path: path.to_path_buf(),
                })?;
        let stream_index = stream.index();
        let time_base = stream.time_base();
        let start_time = match stream.start_time() {
            ffmpeg::ffi::AV_NOPTS_VALUE => 0,
            t => t,
        };
        let has_audio = input.streams().best(media::Type::Audio).is_some();
        let rotation = hwaccel::display_matrix(&stream.parameters())
            .map(|matrix| rotation_from_display_matrix(&matrix))
            .unwrap_or(0);
        let frame_rate = {
            let avg = stream.avg_frame_rate();
            if avg.numerator() > 0 && avg.denominator() > 0 {
                avg
            } else {
                stream.rate()
            }
        };
        let duration_seconds = if stream.duration() > 0 {
            stream.duration() as f64 * f64::from(time_base)
        } else if input.duration() > 0 {
            input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
        } else {
            0.0
        };

        let mut context =
            codec::context::Context::from_parameters(stream.parameters()).map_err(|source| {
                VideoError::Decoder {
                    path: path.to_path_buf(),
                    source,
                }
            })?;
        let device = if try_hardware { HwDevice::cuda() } else { None };
        let decoder_kind = match &device {
            Some(device) => {
                device.attach(&mut context);
                Decoder::Nvdec
            }
            None => {
                context.set_threading(codec::threading::Config {
                    kind: codec::threading::Type::Frame,
                    count: 0,
                });
                Decoder::Software
            }
        };
        let decoder = context
            .decoder()
            .video()
            .map_err(|source| VideoError::Decoder {
                path: path.to_path_buf(),
                source,
            })?;
        let coded_width = decoder.width();
        let coded_height = decoder.height();
        let (width, height) = if rotation == 90 || rotation == 270 {
            (coded_height, coded_width)
        } else {
            (coded_width, coded_height)
        };
        let colour = colour_of(
            decoder.color_space(),
            decoder.color_transfer_characteristic(),
            decoder.color_range(),
        );
        log::info!(
            "opened {}: {coded_width}x{coded_height} rotation {rotation} {:?} at {} fps, {duration_seconds:.3} s, {colour:?}, decoder {decoder_kind:?}",
            path.display(),
            decoder.id(),
            f64::from(frame_rate)
        );
        Ok(Self {
            path: path.to_path_buf(),
            input,
            stream_index,
            decoder,
            _device: device,
            time_base,
            start_time,
            hardware: frame::Video::empty(),
            system: frame::Video::empty(),
            sent_eof: false,
            finished: false,
            skip_until: None,
            coded_width,
            coded_height,
            width,
            height,
            rotation,
            frame_rate,
            duration_seconds,
            colour,
            decoder_kind,
            has_audio,
            last_timing: DecodeTiming::default(),
        })
    }

    /// The path the source was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The codec of the video stream.
    pub fn codec(&self) -> codec::Id {
        self.decoder.id()
    }

    /// The length of one frame in seconds.
    pub fn frame_seconds(&self) -> f64 {
        let rate = f64::from(self.frame_rate);
        if rate > 0.0 { 1.0 / rate } else { 1.0 / 30.0 }
    }

    /// The next frame in presentation order, or `None` at the end of the
    /// stream. After a [`seek`](Self::seek) the frames before the target
    /// are decoded and dropped, so the first frame returned is the one at
    /// or just after it.
    pub fn next_frame(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        loop {
            let started = Instant::now();
            if !self.receive()? {
                return Ok(None);
            }
            let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
            let pts = self.pts_seconds(&self.hardware);
            if let Some(target) = self.skip_until
                && pts + self.frame_seconds() * 0.5 < target
            {
                continue;
            }
            self.skip_until = None;
            let started = Instant::now();
            let frame = self.pack(pts)?;
            self.last_timing = DecodeTiming {
                decode_ms,
                copy_ms: started.elapsed().as_secs_f64() * 1000.0,
            };
            return Ok(Some(frame));
        }
    }

    /// Seeks to the key frame at or before `seconds`; the next
    /// [`next_frame`](Self::next_frame) decodes forward to the frame at or
    /// just after it.
    pub fn seek(&mut self, seconds: f64) -> Result<(), VideoError> {
        let seconds = seconds.max(0.0);
        let ts = (seconds * f64::from(ffmpeg::ffi::AV_TIME_BASE)).round() as i64;
        self.input
            .seek(ts, ..=ts)
            .map_err(|source| VideoError::Decode {
                path: self.path.clone(),
                source,
            })?;
        self.decoder.flush();
        self.sent_eof = false;
        self.finished = false;
        self.skip_until = Some(seconds);
        Ok(())
    }

    /// Receives one frame into `self.hardware`, feeding packets until the
    /// decoder gives one. `false` at the end of the stream.
    fn receive(&mut self) -> Result<bool, VideoError> {
        if self.finished {
            return Ok(false);
        }
        loop {
            match self.decoder.receive_frame(&mut self.hardware) {
                Ok(()) => return Ok(true),
                Err(Error::Eof) => {
                    self.finished = true;
                    return Ok(false);
                }
                Err(Error::Other { errno }) if errno == libc::EAGAIN => {}
                Err(source) => {
                    return Err(VideoError::Decode {
                        path: self.path.clone(),
                        source,
                    });
                }
            }
            if self.sent_eof {
                self.finished = true;
                return Ok(false);
            }
            let mut packet = Packet::empty();
            loop {
                match packet.read(&mut self.input) {
                    Ok(()) if packet.stream() == self.stream_index => {
                        self.send(&packet)?;
                        break;
                    }
                    Ok(()) => continue,
                    Err(Error::Eof) => {
                        self.decoder
                            .send_eof()
                            .map_err(|source| VideoError::Decode {
                                path: self.path.clone(),
                                source,
                            })?;
                        self.sent_eof = true;
                        break;
                    }
                    Err(source) => {
                        return Err(VideoError::Decode {
                            path: self.path.clone(),
                            source,
                        });
                    }
                }
            }
        }
    }

    fn send(&mut self, packet: &Packet) -> Result<(), VideoError> {
        match self.decoder.send_packet(packet) {
            Ok(()) => Ok(()),
            // A corrupt packet is skipped; the decoder conceals it.
            Err(Error::InvalidData) => Ok(()),
            Err(source) => Err(VideoError::Decode {
                path: self.path.clone(),
                source,
            }),
        }
    }

    fn pts_seconds(&self, frame: &frame::Video) -> f64 {
        let ts = frame.timestamp().or(frame.pts()).unwrap_or(self.start_time);
        (ts - self.start_time) as f64 * f64::from(self.time_base)
    }

    /// Moves the received frame into system memory when it is a CUDA frame
    /// and packs its planes.
    fn pack(&mut self, pts: f64) -> Result<VideoFrame, VideoError> {
        let source: &frame::Video = if hwaccel::is_hardware_frame(&self.hardware) {
            hwaccel::transfer(&self.hardware, &mut self.system).map_err(|source| {
                VideoError::Decode {
                    path: self.path.clone(),
                    source,
                }
            })?;
            &self.system
        } else {
            &self.hardware
        };
        let width = source.width();
        let height = source.height();
        let unsupported = |format| VideoError::UnsupportedFormat {
            path: self.path.clone(),
            format,
        };
        let (format, y, uv, y_stride, uv_stride) = match source.format() {
            Pixel::NV12 => {
                let y = pack_plane(source.data(0), source.stride(0), width as usize, height);
                let uv = pack_plane(
                    source.data(1),
                    source.stride(1),
                    width as usize,
                    height.div_ceil(2),
                );
                (PlaneFormat::Nv12, y, uv, width as usize, width as usize)
            }
            Pixel::P010LE => {
                let y = pack_plane(source.data(0), source.stride(0), width as usize * 2, height);
                let uv = pack_plane(
                    source.data(1),
                    source.stride(1),
                    width as usize * 2,
                    height.div_ceil(2),
                );
                (
                    PlaneFormat::P010,
                    y,
                    uv,
                    width as usize * 2,
                    width as usize * 2,
                )
            }
            Pixel::YUV420P | Pixel::YUVJ420P => {
                let y = pack_plane(source.data(0), source.stride(0), width as usize, height);
                let uv = interleave_8(
                    source.data(1),
                    source.stride(1),
                    source.data(2),
                    source.stride(2),
                    width.div_ceil(2) as usize,
                    height.div_ceil(2),
                );
                (PlaneFormat::Nv12, y, uv, width as usize, width as usize)
            }
            Pixel::YUV420P10LE => {
                let y = shift_10_to_p010(source.data(0), source.stride(0), width as usize, height);
                let uv = interleave_10(
                    source.data(1),
                    source.stride(1),
                    source.data(2),
                    source.stride(2),
                    width.div_ceil(2) as usize,
                    height.div_ceil(2),
                );
                (
                    PlaneFormat::P010,
                    y,
                    uv,
                    width as usize * 2,
                    width as usize * 2,
                )
            }
            other => return Err(unsupported(other)),
        };
        Ok(VideoFrame {
            pts_seconds: pts,
            format,
            width,
            height,
            y,
            uv,
            y_stride,
            uv_stride,
        })
    }
}

/// Copies `rows` rows of `row_bytes` bytes out of a strided plane.
fn pack_plane(data: &[u8], stride: usize, row_bytes: usize, rows: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(row_bytes * rows as usize);
    for row in 0..rows as usize {
        out.extend_from_slice(&data[row * stride..row * stride + row_bytes]);
    }
    out
}

/// Interleaves two 8 bit chroma planes into one NV12 chroma plane.
fn interleave_8(
    u: &[u8],
    u_stride: usize,
    v: &[u8],
    v_stride: usize,
    chroma_width: usize,
    rows: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(chroma_width * 2 * rows as usize);
    for row in 0..rows as usize {
        let u_row = &u[row * u_stride..row * u_stride + chroma_width];
        let v_row = &v[row * v_stride..row * v_stride + chroma_width];
        for (a, b) in u_row.iter().zip(v_row) {
            out.push(*a);
            out.push(*b);
        }
    }
    out
}

/// Moves 10 bit samples stored in the low bits of 16 bit words to the top
/// of the word, as P010 stores them.
fn shift_10_to_p010(data: &[u8], stride: usize, width: usize, rows: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * 2 * rows as usize);
    for row in 0..rows as usize {
        for px in data[row * stride..row * stride + width * 2]
            .as_chunks::<2>()
            .0
        {
            let code = u16::from_le_bytes(*px) << 6;
            out.extend_from_slice(&code.to_le_bytes());
        }
    }
    out
}

/// Interleaves two 10 bit chroma planes into one P010 chroma plane.
fn interleave_10(
    u: &[u8],
    u_stride: usize,
    v: &[u8],
    v_stride: usize,
    chroma_width: usize,
    rows: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(chroma_width * 4 * rows as usize);
    for row in 0..rows as usize {
        let u_row = u[row * u_stride..row * u_stride + chroma_width * 2]
            .as_chunks::<2>()
            .0;
        let v_row = v[row * v_stride..row * v_stride + chroma_width * 2]
            .as_chunks::<2>()
            .0;
        for (a, b) in u_row.iter().zip(v_row) {
            out.extend_from_slice(&(u16::from_le_bytes(*a) << 6).to_le_bytes());
            out.extend_from_slice(&(u16::from_le_bytes(*b) << 6).to_le_bytes());
        }
    }
    out
}

/// The colour tags of a stream, with the SDR defaults for what is unset.
pub fn colour_of(
    space: color::Space,
    transfer: color::TransferCharacteristic,
    range: color::Range,
) -> VideoColour {
    let transfer = match transfer {
        color::TransferCharacteristic::ARIB_STD_B67 => Transfer::Hlg,
        color::TransferCharacteristic::SMPTE2084 => {
            log::warn!("PQ (SMPTE ST 2084) is decoded as HLG until the tone map lands");
            Transfer::Pq
        }
        _ => Transfer::Sdr,
    };
    let space = match space {
        color::Space::BT2020NCL | color::Space::BT2020CL => YuvSpace::Bt2020,
        color::Space::BT709 => YuvSpace::Bt709,
        _ if transfer != Transfer::Sdr => YuvSpace::Bt2020,
        _ => YuvSpace::Bt709,
    };
    VideoColour {
        space,
        transfer,
        full_range: range == color::Range::JPEG,
    }
}

/// The clockwise display rotation a QuickTime display matrix asks for, in
/// degrees, rounded to a quarter turn. The matrix is nine 16.16 fixed
/// point values in the order ffmpeg stores them; the rotation is the angle
/// of its first column, negated as `av_display_rotation_get` does.
pub fn rotation_from_display_matrix(matrix: &[i32; 9]) -> u32 {
    let value = |i: usize| f64::from(matrix[i]) / 65536.0;
    let (m0, m1, m3, m4) = (value(0), value(1), value(3), value(4));
    let scale_x = m0.hypot(m3);
    let scale_y = m1.hypot(m4);
    if scale_x == 0.0 || scale_y == 0.0 {
        return 0;
    }
    let rotation = -(m1 / scale_y).atan2(m0 / scale_x).to_degrees();
    let quarter = (rotation / 90.0).round() as i32;
    // A negative value from ffmpeg means the frame turns clockwise to be
    // displayed; the display matrix rotates counterclockwise.
    ((-quarter).rem_euclid(4) * 90) as u32
}

#[cfg(test)]
mod tests {
    use super::{PlaneFormat, VideoFrame, is_video_path, rotation_from_display_matrix};
    use std::path::Path;

    fn matrix(values: [i32; 9]) -> [i32; 9] {
        values
    }

    #[test]
    fn an_iphone_portrait_matrix_turns_clockwise() {
        // The matrix of a portrait iPhone clip: ffprobe prints rotation -90.
        let data = matrix([0, 65536, 0, -65536, 0, 0, 141557760, 0, 1073741824]);
        assert_eq!(rotation_from_display_matrix(&data), 90);
        let identity = matrix([65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824]);
        assert_eq!(rotation_from_display_matrix(&identity), 0);
        let upside_down = matrix([-65536, 0, 0, 0, -65536, 0, 0, 0, 1073741824]);
        assert_eq!(rotation_from_display_matrix(&upside_down), 180);
        let other_way = matrix([0, -65536, 0, 65536, 0, 0, 0, 0, 1073741824]);
        assert_eq!(rotation_from_display_matrix(&other_way), 270);
        assert_eq!(rotation_from_display_matrix(&[0; 9]), 0);
    }

    #[test]
    fn the_video_extensions_are_recognised_in_any_case() {
        assert!(is_video_path(Path::new("a.MOV")));
        assert!(is_video_path(Path::new("a.mp4")));
        assert!(!is_video_path(Path::new("a.jpg")));
    }

    #[test]
    fn samples_read_back_from_both_plane_formats() {
        let nv12 = VideoFrame {
            pts_seconds: 0.0,
            format: PlaneFormat::Nv12,
            width: 2,
            height: 2,
            y: vec![10, 20, 30, 40],
            uv: vec![100, 200],
            y_stride: 2,
            uv_stride: 2,
        };
        assert_eq!(nv12.luma(1, 1), 40);
        assert_eq!(nv12.chroma(0, 0), (100, 200));
        let p010 = VideoFrame {
            pts_seconds: 0.0,
            format: PlaneFormat::P010,
            width: 2,
            height: 2,
            y: [1u16 << 6, 2 << 6, 3 << 6, 1023 << 6]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            uv: [512u16 << 6, 64 << 6]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            y_stride: 4,
            uv_stride: 4,
        };
        assert_eq!(p010.luma(1, 1), 1023 << 6);
        assert_eq!(p010.chroma(0, 0), (512 << 6, 64 << 6));
    }
}
