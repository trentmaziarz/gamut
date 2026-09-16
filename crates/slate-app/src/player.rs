//! Playback of the one track: a video decode thread that walks the clips
//! and fills a bounded frame queue, an audio decode thread that fills a
//! ring buffer the cpal output stream drains, and a clock. With audio the
//! clock is the count of samples the device has consumed, so the picture
//! follows the sound; without audio (no stream, or no output device) it is
//! wall time. The window thread asks for the frame whose time is at or
//! before the clock and drops older ones rather than falling behind. A
//! seek decodes the frame at the target even while paused, so a scrub
//! shows the frame under the playhead.
//!
//! The decode threads own no wgpu objects; frames cross to the window as
//! CPU planes and the Viewer uploads them.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use slate_color::video::VideoColour;
use slate_core::Track;
use slate_media::audio::AudioSource;
use slate_media::{VideoError, VideoFrame, VideoSource};

/// How many decoded frames wait for the window.
const QUEUE_FRAMES: usize = 4;

/// How much audio the ring buffer holds ahead of the device, in seconds.
const RING_SECONDS: f64 = 0.5;

/// What the decode threads need to know about one media file.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaInfo {
    pub path: PathBuf,
    pub duration: f64,
    /// The display size, after the rotation.
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub colour: VideoColour,
    pub rotation: u32,
    pub has_audio: bool,
}

impl MediaInfo {
    /// Opens the file once to read its properties.
    pub fn probe(path: &Path) -> Result<Self, VideoError> {
        let source = VideoSource::open_software(path)?;
        Ok(MediaInfo {
            path: path.to_path_buf(),
            duration: source.duration_seconds,
            width: source.width,
            height: source.height,
            frame_rate: f64::from(source.frame_rate),
            colour: source.colour,
            rotation: source.rotation,
            has_audio: source.has_audio,
        })
    }

    /// The length of one frame in seconds.
    pub fn frame_seconds(&self) -> f64 {
        if self.frame_rate > 0.0 {
            1.0 / self.frame_rate
        } else {
            1.0 / 30.0
        }
    }
}

/// A decoded frame placed on the timeline.
pub struct TimelineFrame {
    /// Where the frame sits on the timeline, in seconds.
    pub time: f64,
    pub frame: VideoFrame,
    pub colour: VideoColour,
    pub rotation: u32,
    /// The clip the frame came from.
    pub clip: usize,
}

enum Command {
    Seek(f64),
    Play,
    Pause,
    SetTrack(Track),
    Stop,
}

/// The playback position. With audio it advances as the device consumes
/// samples; without, as wall time passes.
pub struct Clock {
    state: Mutex<ClockState>,
    audio_driven: AtomicBool,
}

struct ClockState {
    anchor: f64,
    samples: u64,
    rate: u32,
    started: Option<Instant>,
}

impl Clock {
    fn new(rate: u32) -> Self {
        Clock {
            state: Mutex::new(ClockState {
                anchor: 0.0,
                samples: 0,
                rate: rate.max(1),
                started: None,
            }),
            audio_driven: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ClockState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn set_rate(&self, rate: u32) {
        self.lock().rate = rate.max(1);
    }

    /// The position in seconds.
    pub fn position(&self) -> f64 {
        let state = self.lock();
        match state.started {
            None => state.anchor,
            Some(_) if self.audio_driven.load(Ordering::Relaxed) => {
                state.anchor + state.samples as f64 / f64::from(state.rate)
            }
            Some(started) => state.anchor + started.elapsed().as_secs_f64(),
        }
    }

    fn seek(&self, seconds: f64) {
        let mut state = self.lock();
        state.anchor = seconds;
        state.samples = 0;
        if state.started.is_some() {
            state.started = Some(Instant::now());
        }
    }

    fn play(&self) {
        let mut state = self.lock();
        if state.started.is_none() {
            state.started = Some(Instant::now());
        }
    }

    fn pause(&self) {
        let position = self.position();
        let mut state = self.lock();
        state.anchor = position;
        state.samples = 0;
        state.started = None;
    }

    fn is_playing(&self) -> bool {
        self.lock().started.is_some()
    }

    /// The audio callback consumed `frames` sample frames.
    fn advance(&self, frames: u64) {
        let mut state = self.lock();
        if state.started.is_some() {
            state.samples += frames;
        }
    }
}

/// The samples waiting for the output device, interleaved stereo at the
/// device rate.
type Ring = Arc<Mutex<VecDeque<f32>>>;

/// The player of one project: the threads, the queue and the clock.
pub struct Player {
    video_commands: Sender<Command>,
    audio_commands: Option<Sender<Command>>,
    frames: Receiver<TimelineFrame>,
    clock: Arc<Clock>,
    current: Option<TimelineFrame>,
    pending: Option<TimelineFrame>,
    serial: u64,
    duration: f64,
    frame_seconds: f64,
    _stream: Option<cpal::Stream>,
    threads: Vec<JoinHandle<()>>,
    /// The last error a decode thread reported.
    errors: Receiver<String>,
}

impl Player {
    /// Starts the threads for `track` over `media`, paused at zero.
    pub fn new(media: Vec<MediaInfo>, track: Track) -> Self {
        let has_audio = media.iter().any(|m| m.has_audio);
        let frame_seconds = media
            .first()
            .map(MediaInfo::frame_seconds)
            .unwrap_or(1.0 / 30.0);
        let clock = Arc::new(Clock::new(48_000));
        let audio_output = if has_audio {
            AudioOutput::open(clock.clone())
        } else {
            None
        };
        if let Some(output) = &audio_output {
            clock.set_rate(output.rate);
            clock.audio_driven.store(true, Ordering::Relaxed);
        }
        let (error_tx, errors) = mpsc::channel();
        let (video_tx, video_rx) = mpsc::channel();
        let (frame_tx, frames) = mpsc::sync_channel(QUEUE_FRAMES);
        let duration = track.duration();
        let mut threads = Vec::new();
        {
            let media = media.clone();
            let track = track.clone();
            let error_tx = error_tx.clone();
            threads.push(std::thread::spawn(move || {
                VideoThread {
                    media,
                    track,
                    sources: HashMap::new(),
                    commands: video_rx,
                    frames: frame_tx,
                    errors: error_tx,
                    position: 0.0,
                    playing: false,
                    want_frame: true,
                    need_seek: true,
                    current_clip: None,
                }
                .run();
            }));
        }
        let (audio_commands, stream) = match audio_output {
            Some(output) => {
                let (audio_tx, audio_rx) = mpsc::channel();
                let ring = output.ring.clone();
                let rate = output.rate;
                let media = media.clone();
                let track = track.clone();
                let error_tx = error_tx.clone();
                threads.push(std::thread::spawn(move || {
                    AudioThread {
                        media,
                        track,
                        sources: HashMap::new(),
                        commands: audio_rx,
                        ring,
                        rate,
                        errors: error_tx,
                        position: 0.0,
                        playing: false,
                        need_seek: true,
                        current_clip: None,
                        silence_left: 0,
                    }
                    .run();
                }));
                (Some(audio_tx), Some(output.stream))
            }
            None => (None, None),
        };
        Player {
            video_commands: video_tx,
            audio_commands,
            frames,
            clock,
            current: None,
            pending: None,
            serial: 0,
            duration,
            frame_seconds,
            _stream: stream,
            threads,
            errors,
        }
    }

    fn send(&self, command: impl Fn() -> Command) {
        let _ = self.video_commands.send(command());
        if let Some(audio) = &self.audio_commands {
            let _ = audio.send(command());
        }
    }

    /// The length of the track.
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// The length of one frame of the first media.
    pub fn frame_seconds(&self) -> f64 {
        self.frame_seconds
    }

    pub fn is_playing(&self) -> bool {
        self.clock.is_playing()
    }

    /// The playhead in seconds.
    pub fn position(&self) -> f64 {
        self.clock.position().clamp(0.0, self.duration)
    }

    pub fn play(&mut self) {
        if self.duration <= 0.0 {
            return;
        }
        if self.position() >= self.duration {
            self.seek(0.0);
        }
        self.clock.play();
        self.send(|| Command::Play);
    }

    pub fn pause(&mut self) {
        self.clock.pause();
        self.send(|| Command::Pause);
    }

    pub fn toggle(&mut self) {
        if self.is_playing() {
            self.pause();
        } else {
            self.play();
        }
    }

    /// Moves the playhead; the frame under it arrives on the next
    /// [`current_frame`](Self::current_frame) calls.
    pub fn seek(&mut self, seconds: f64) {
        let seconds = seconds.clamp(0.0, self.duration);
        self.clock.seek(seconds);
        self.pending = None;
        // Frames still queued belong to the old position.
        while self.frames.try_recv().is_ok() {}
        self.send(|| Command::Seek(seconds));
    }

    /// Replaces the track after an edit; playback continues from the same
    /// time.
    pub fn set_track(&mut self, track: Track) {
        self.duration = track.duration();
        let position = self.position();
        self.pending = None;
        while self.frames.try_recv().is_ok() {}
        self.send(|| Command::SetTrack(track.clone()));
        self.clock.seek(position);
        self.send(|| Command::Seek(position));
    }

    /// Steps one frame forward or back while paused.
    pub fn step(&mut self, frames: i32) {
        self.pause();
        let at = self.position() + f64::from(frames) * self.frame_seconds;
        self.seek(at);
    }

    /// The frame to draw now with its serial: the newest queued frame
    /// whose time is at or before the clock, else the last one shown.
    /// Stops at the end.
    pub fn current_frame(&mut self) -> Option<(u64, &TimelineFrame)> {
        let playing = self.is_playing();
        let position = self.clock.position();
        let slack = self.frame_seconds * 0.5;
        if let Some(pending) = self.pending.take() {
            if !playing || pending.time <= position + slack {
                self.show(pending);
            } else {
                self.pending = Some(pending);
            }
        }
        if self.pending.is_none() {
            while let Ok(frame) = self.frames.try_recv() {
                if !playing || frame.time <= position + slack {
                    self.show(frame);
                } else {
                    self.pending = Some(frame);
                    break;
                }
            }
        }
        if playing && position >= self.duration {
            self.pause();
            self.clock.seek(self.duration);
        }
        let serial = self.serial;
        self.current.as_ref().map(|frame| (serial, frame))
    }

    fn show(&mut self, frame: TimelineFrame) {
        self.current = Some(frame);
        self.serial += 1;
    }

    /// Counts up every time the current frame changes.
    pub fn serial(&self) -> u64 {
        self.serial
    }

    /// The last error a decode thread reported, once.
    pub fn take_error(&self) -> Option<String> {
        self.errors.try_recv().ok()
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.send(|| Command::Stop);
        while self.frames.try_recv().is_ok() {}
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

/// The cpal output stream and the ring it drains.
struct AudioOutput {
    stream: cpal::Stream,
    ring: Ring,
    rate: u32,
}

impl AudioOutput {
    /// Opens the default output device as stereo f32 at its default rate,
    /// advancing `clock` as it consumes samples. `None`, with a log line,
    /// when there is no device or it refuses.
    fn open(clock: Arc<Clock>) -> Option<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        let default = device
            .default_output_config()
            .inspect_err(|error| log::warn!("no default output config: {error}"))
            .ok()?;
        let rate = default.sample_rate();
        let config = cpal::StreamConfig {
            channels: 2,
            sample_rate: rate,
            buffer_size: cpal::BufferSize::Default,
        };
        let ring: Ring = Arc::new(Mutex::new(VecDeque::new()));
        let callback_ring = ring.clone();
        let stream = device
            .build_output_stream(
                config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut ring = callback_ring
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let mut consumed = 0u64;
                    for pair in data.chunks_mut(2) {
                        if ring.len() >= 2 {
                            for sample in pair.iter_mut() {
                                *sample = ring.pop_front().unwrap_or(0.0);
                            }
                            consumed += 1;
                        } else {
                            pair.fill(0.0);
                        }
                    }
                    drop(ring);
                    clock.advance(consumed);
                },
                |error| log::error!("audio stream: {error}"),
                None,
            )
            .inspect_err(|error| log::warn!("no audio output stream: {error}"))
            .ok()?;
        stream
            .play()
            .inspect_err(|error| log::warn!("audio stream would not start: {error}"))
            .ok()?;
        log::info!("audio output: the default device at {rate} Hz stereo");
        Some(AudioOutput { stream, ring, rate })
    }
}

struct VideoThread {
    media: Vec<MediaInfo>,
    track: Track,
    sources: HashMap<usize, VideoSource>,
    commands: Receiver<Command>,
    frames: SyncSender<TimelineFrame>,
    errors: Sender<String>,
    /// The next timeline time to decode.
    position: f64,
    playing: bool,
    /// A frame is wanted even while paused (after a seek).
    want_frame: bool,
    need_seek: bool,
    current_clip: Option<usize>,
}

impl VideoThread {
    fn run(mut self) {
        loop {
            if !self.playing && !self.want_frame {
                match self.commands.recv() {
                    Ok(command) => {
                        if !self.handle(command) {
                            return;
                        }
                    }
                    Err(_) => return,
                }
                continue;
            }
            while let Ok(command) = self.commands.try_recv() {
                if !self.handle(command) {
                    return;
                }
            }
            if !self.playing && !self.want_frame {
                continue;
            }
            match self.decode_one() {
                Ok(Some(frame)) => {
                    self.want_frame = false;
                    if !self.deliver(frame) {
                        return;
                    }
                }
                Ok(None) => {
                    // The end of the track.
                    self.playing = false;
                    self.want_frame = false;
                }
                Err(error) => {
                    let _ = self.errors.send(error.to_string());
                    self.playing = false;
                    self.want_frame = false;
                }
            }
        }
    }

    /// Applies a command; `false` means stop.
    fn handle(&mut self, command: Command) -> bool {
        match command {
            Command::Seek(seconds) => {
                self.position = seconds;
                self.need_seek = true;
                self.want_frame = true;
            }
            Command::Play => self.playing = true,
            Command::Pause => self.playing = false,
            Command::SetTrack(track) => {
                self.track = track;
                self.need_seek = true;
            }
            Command::Stop => return false,
        }
        true
    }

    /// Sends a frame, handling commands while the queue is full; `false`
    /// means stop.
    fn deliver(&mut self, mut frame: TimelineFrame) -> bool {
        loop {
            match self.frames.try_send(frame) {
                Ok(()) => return true,
                Err(TrySendError::Disconnected(_)) => return false,
                Err(TrySendError::Full(back)) => {
                    frame = back;
                    match self.commands.recv_timeout(Duration::from_millis(10)) {
                        Ok(command) => {
                            let drop_frame =
                                matches!(command, Command::Seek(_) | Command::SetTrack(_));
                            if !self.handle(command) {
                                return false;
                            }
                            if drop_frame {
                                return true;
                            }
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => return false,
                    }
                }
            }
        }
    }

    /// Decodes the frame at `position` and moves `position` past it.
    fn decode_one(&mut self) -> Result<Option<TimelineFrame>, VideoError> {
        loop {
            let Some((index, offset)) = self.track.clip_at(self.position) else {
                return Ok(None);
            };
            let clip = self.track.clips[index];
            let info = self.media[clip.media].clone();
            let clip_start = self.track.start_of(index);
            let source = match self.sources.entry(clip.media) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let source = VideoSource::open(&info.path)?;
                    log::info!(
                        "player decoder for {}: {:?}",
                        info.path.display(),
                        source.decoder_kind
                    );
                    entry.insert(source)
                }
            };
            if self.need_seek || self.current_clip != Some(index) {
                source.seek(clip.source_in + offset)?;
                self.need_seek = false;
                self.current_clip = Some(index);
            }
            match source.next_frame()? {
                Some(frame) if frame.pts_seconds < clip.source_out - 1e-6 => {
                    let time = clip_start + (frame.pts_seconds - clip.source_in).max(0.0);
                    self.position = time + source.frame_seconds() * 0.5;
                    return Ok(Some(TimelineFrame {
                        time,
                        frame,
                        colour: info.colour,
                        rotation: info.rotation,
                        clip: index,
                    }));
                }
                _ => {
                    // Past this clip's out point or the end of the file:
                    // move to the next clip.
                    let next = clip_start + clip.duration();
                    if index + 1 >= self.track.clips.len() {
                        self.position = next;
                        return Ok(None);
                    }
                    self.position = next;
                    self.need_seek = true;
                }
            }
        }
    }
}

struct AudioThread {
    media: Vec<MediaInfo>,
    track: Track,
    sources: HashMap<usize, Option<AudioSource>>,
    commands: Receiver<Command>,
    ring: Ring,
    rate: u32,
    errors: Sender<String>,
    position: f64,
    playing: bool,
    need_seek: bool,
    current_clip: Option<usize>,
    /// Sample frames of silence still to write for a clip without audio.
    silence_left: u64,
}

impl AudioThread {
    fn run(mut self) {
        let cap = (RING_SECONDS * f64::from(self.rate)) as usize * 2;
        loop {
            if !self.playing {
                match self.commands.recv() {
                    Ok(command) => {
                        if !self.handle(command) {
                            return;
                        }
                    }
                    Err(_) => return,
                }
                continue;
            }
            while let Ok(command) = self.commands.try_recv() {
                if !self.handle(command) {
                    return;
                }
            }
            if !self.playing {
                continue;
            }
            let queued = self
                .ring
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .len();
            if queued >= cap {
                match self.commands.recv_timeout(Duration::from_millis(5)) {
                    Ok(command) => {
                        if !self.handle(command) {
                            return;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
                continue;
            }
            match self.decode_some() {
                Ok(true) => {}
                Ok(false) => self.playing = false,
                Err(error) => {
                    let _ = self.errors.send(error.to_string());
                    self.playing = false;
                }
            }
        }
    }

    fn handle(&mut self, command: Command) -> bool {
        match command {
            Command::Seek(seconds) => {
                self.position = seconds;
                self.need_seek = true;
                self.silence_left = 0;
                self.ring
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clear();
            }
            Command::Play => self.playing = true,
            Command::Pause => {
                self.playing = false;
                self.ring
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clear();
                self.need_seek = true;
            }
            Command::SetTrack(track) => {
                self.track = track;
                self.need_seek = true;
            }
            Command::Stop => return false,
        }
        true
    }

    /// Pushes the next run of samples for `position`; `false` at the end
    /// of the track.
    fn decode_some(&mut self) -> Result<bool, VideoError> {
        let Some((index, offset)) = self.track.clip_at(self.position) else {
            return Ok(false);
        };
        if self.position >= self.track.duration() - 1e-6 {
            return Ok(false);
        }
        let clip = self.track.clips[index];
        let clip_start = self.track.start_of(index);
        let info = self.media[clip.media].clone();
        let rate = f64::from(self.rate);
        if !self.sources.contains_key(&clip.media) {
            let source = if info.has_audio {
                Some(AudioSource::open(&info.path, self.rate, 2)?)
            } else {
                None
            };
            self.sources.insert(clip.media, source);
        }
        let seek_here = self.need_seek || self.current_clip != Some(index);
        if seek_here {
            self.current_clip = Some(index);
            self.need_seek = false;
        }
        let Some(source) = self.sources.get_mut(&clip.media).expect("opened above") else {
            // No audio in this media: silence for the rest of the clip
            // keeps the clock moving.
            if seek_here {
                self.silence_left = ((clip.duration() - offset) * rate).round().max(0.0) as u64;
            }
            let run = self.silence_left.min(1024);
            self.push(vec![0.0; run as usize * 2]);
            self.silence_left -= run;
            self.position = clip_start + clip.duration() - self.silence_left as f64 / rate;
            if self.silence_left == 0 {
                self.position = clip_start + clip.duration();
                self.need_seek = true;
            }
            return Ok(true);
        };
        if seek_here {
            source.seek(clip.source_in + offset)?;
        }
        match source.next_samples()? {
            Some(chunk) => {
                let frames = chunk.samples.len() / 2;
                let end = chunk.pts_seconds + frames as f64 / rate;
                let keep_to = clip.source_out;
                let samples = if end > keep_to {
                    let keep = ((keep_to - chunk.pts_seconds) * rate).round().max(0.0) as usize;
                    chunk.samples[..(keep * 2).min(chunk.samples.len())].to_vec()
                } else {
                    chunk.samples
                };
                let kept = samples.len() / 2;
                self.push(samples);
                self.position =
                    clip_start + (chunk.pts_seconds - clip.source_in) + kept as f64 / rate;
                if end >= keep_to {
                    self.position = clip_start + clip.duration();
                    self.need_seek = true;
                }
            }
            None => {
                // The file ended before the clip's out point.
                let remaining = clip_start + clip.duration() - self.position;
                let frames = (remaining * rate).round().max(0.0) as usize;
                self.push(vec![0.0; frames.min(48_000) * 2]);
                self.position = clip_start + clip.duration();
                self.need_seek = true;
            }
        }
        Ok(true)
    }

    fn push(&self, samples: Vec<f32>) {
        self.ring
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .extend(samples);
    }
}
