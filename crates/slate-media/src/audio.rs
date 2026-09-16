//! Opening the audio of a video: the first audio stream decoded through
//! ffmpeg and converted by swresample to interleaved f32 at the rate and
//! channel count the caller asks for, which is the playback device's rate
//! for the player and 48 kHz stereo for the Reel.

use std::path::{Path, PathBuf};

use ffmpeg::format::Sample;
use ffmpeg::format::sample::Type as SampleType;
use ffmpeg::software::resampling;
use ffmpeg::{ChannelLayout, Error, Packet, Rational, codec, format, frame, media};
use ffmpeg_the_third as ffmpeg;

use crate::video::VideoError;

/// A run of interleaved f32 samples and the time of the first one.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioChunk {
    pub pts_seconds: f64,
    /// Interleaved, `channels` values per frame.
    pub samples: Vec<f32>,
}

/// An open audio stream and its decoder and resampler.
pub struct AudioSource {
    path: PathBuf,
    input: format::context::Input,
    stream_index: usize,
    decoder: codec::decoder::Audio,
    resampler: resampling::Context,
    time_base: Rational,
    start_time: i64,
    decoded: frame::Audio,
    sent_eof: bool,
    finished: bool,
    skip_until: Option<f64>,
    pub out_rate: u32,
    pub channels: u16,
    pub duration_seconds: f64,
}

impl AudioSource {
    /// Opens the first audio stream of `path`, converting to `out_rate` and
    /// `channels` (1 or 2). [`VideoError::NoAudioStream`] when the file has
    /// none.
    pub fn open(path: &Path, out_rate: u32, channels: u16) -> Result<Self, VideoError> {
        crate::init_ffmpeg();
        let at = |source| VideoError::Open {
            path: path.to_path_buf(),
            source,
        };
        let input = format::input(path).map_err(at)?;
        let stream =
            input
                .streams()
                .best(media::Type::Audio)
                .ok_or_else(|| VideoError::NoAudioStream {
                    path: path.to_path_buf(),
                })?;
        let stream_index = stream.index();
        let time_base = stream.time_base();
        let start_time = match stream.start_time() {
            ffmpeg::ffi::AV_NOPTS_VALUE => 0,
            t => t,
        };
        let duration_seconds = if stream.duration() > 0 {
            stream.duration() as f64 * f64::from(time_base)
        } else if input.duration() > 0 {
            input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
        } else {
            0.0
        };
        let decoder_error = |source| VideoError::Decoder {
            path: path.to_path_buf(),
            source,
        };
        let context =
            codec::context::Context::from_parameters(stream.parameters()).map_err(decoder_error)?;
        let decoder = context.decoder().audio().map_err(decoder_error)?;
        let in_layout = decoder.ch_layout();
        let in_layout = if in_layout.mask().is_some() {
            in_layout
        } else {
            ChannelLayout::default_for_channels(in_layout.channels())
        };
        let out_layout = if channels == 1 {
            ChannelLayout::MONO
        } else {
            ChannelLayout::STEREO
        };
        let resampler = resampling::Context::get2(
            decoder.format(),
            in_layout,
            decoder.rate(),
            Sample::F32(SampleType::Packed),
            out_layout,
            out_rate,
        )
        .map_err(decoder_error)?;
        log::info!(
            "opened audio of {}: {:?} {} Hz {} channels to {out_rate} Hz {channels} channels",
            path.display(),
            decoder.id(),
            decoder.rate(),
            decoder.ch_layout().channels()
        );
        Ok(Self {
            path: path.to_path_buf(),
            input,
            stream_index,
            decoder,
            resampler,
            time_base,
            start_time,
            decoded: frame::Audio::empty(),
            sent_eof: false,
            finished: false,
            skip_until: None,
            out_rate,
            channels,
            duration_seconds,
        })
    }

    /// The next run of samples, or `None` at the end of the stream. After a
    /// [`seek`](Self::seek) the samples before the target are dropped.
    pub fn next_samples(&mut self) -> Result<Option<AudioChunk>, VideoError> {
        loop {
            if !self.receive()? {
                return Ok(None);
            }
            let pts = self.pts_seconds();
            let mut chunk = self.convert(pts)?;
            if let Some(target) = self.skip_until {
                let per_frame = self.channels as usize;
                let frames = chunk.samples.len() / per_frame;
                let end = pts + frames as f64 / f64::from(self.out_rate);
                if end <= target {
                    continue;
                }
                let drop = ((target - pts) * f64::from(self.out_rate)).round().max(0.0) as usize;
                let drop = drop.min(frames);
                chunk.samples.drain(..drop * per_frame);
                chunk.pts_seconds = pts + drop as f64 / f64::from(self.out_rate);
                self.skip_until = None;
            }
            return Ok(Some(chunk));
        }
    }

    /// Seeks so that the next samples start at `seconds`.
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

    fn receive(&mut self) -> Result<bool, VideoError> {
        if self.finished {
            return Ok(false);
        }
        loop {
            match self.decoder.receive_frame(&mut self.decoded) {
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
                        match self.decoder.send_packet(&packet) {
                            Ok(()) | Err(Error::InvalidData) => {}
                            Err(source) => {
                                return Err(VideoError::Decode {
                                    path: self.path.clone(),
                                    source,
                                });
                            }
                        }
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

    fn pts_seconds(&self) -> f64 {
        let ts = self
            .decoded
            .timestamp()
            .or(self.decoded.pts())
            .unwrap_or(self.start_time);
        (ts - self.start_time) as f64 * f64::from(self.time_base)
    }

    /// Runs the decoded frame through the resampler into an f32 chunk.
    fn convert(&mut self, pts: f64) -> Result<AudioChunk, VideoError> {
        let in_rate = self.decoder.rate().max(1) as usize;
        let wanted = self.decoded.samples() * self.out_rate as usize / in_rate + 256;
        let layout = if self.channels == 1 {
            ffmpeg::ChannelLayoutMask::MONO
        } else {
            ffmpeg::ChannelLayoutMask::STEREO
        };
        let mut out = frame::Audio::new(Sample::F32(SampleType::Packed), wanted, layout);
        self.resampler
            .run(&self.decoded, &mut out)
            .map_err(|source| VideoError::Decode {
                path: self.path.clone(),
                source,
            })?;
        let count = out.samples() * self.channels as usize;
        let bytes = &out.data(0)[..count * 4];
        let samples: Vec<f32> = bytemuck::pod_collect_to_vec(bytes);
        Ok(AudioChunk {
            pts_seconds: pts,
            samples,
        })
    }
}
