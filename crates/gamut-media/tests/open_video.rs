//! Opens the M2 video fixtures and checks their size, rate, codec, frame
//! count, colour tags and audio. The public files come from the
//! fixtures-m2 release; the iPhone excerpt is local only and its tests
//! skip with a printed line when it is absent or when NVDEC is not there.

use ffmpeg_the_third::codec::Id;
use gamut_media::audio::AudioSource;
use gamut_media::hwaccel::nvdec_available;
use gamut_media::{Decoder, PlaneFormat, Transfer, VideoSource, YuvSpace, fixtures};

fn open(name: &str) -> VideoSource {
    let path = fixtures::require(name);
    let source = VideoSource::open(&path).unwrap_or_else(|error| panic!("open {name}: {error}"));
    println!("{name}: decoder: {:?}", source.decoder_kind);
    source
}

fn count_frames(source: &mut VideoSource, limit: usize) -> (usize, f64) {
    let mut count = 0;
    let mut last_pts = 0.0;
    while let Some(frame) = source.next_frame().expect("decode") {
        count += 1;
        last_pts = frame.pts_seconds;
        if count >= limit {
            break;
        }
    }
    (count, last_pts)
}

#[test]
fn sample_5s_is_h264_1080p_at_30_with_audio() {
    let mut source = open("sample-5s.mp4");
    assert_eq!((source.width, source.height), (1920, 1080));
    assert_eq!(source.codec(), Id::H264);
    assert!((f64::from(source.frame_rate) - 30.0).abs() < 0.01);
    assert!(source.has_audio);
    assert_eq!(source.colour.space, YuvSpace::Bt709);
    assert_eq!(source.colour.transfer, Transfer::Sdr);
    let first = source.next_frame().expect("decode").expect("a frame");
    assert_eq!(first.format, PlaneFormat::Nv12);
    assert_eq!(first.y.len(), 1920 * 1080);
    assert_eq!(first.uv.len(), 1920 * 540);
    let (count, last) = count_frames(&mut source, 1000);
    println!("sample-5s.mp4: {} frames, last pts {last:.3}", count + 1);
    assert!(count + 1 >= 140, "{count} frames");

    let mut audio =
        AudioSource::open(&fixtures::require("sample-5s.mp4"), 48_000, 2).expect("open the audio");
    assert_eq!(audio.out_rate, 48_000);
    let mut samples = 0usize;
    while let Some(chunk) = audio.next_samples().expect("decode audio") {
        samples += chunk.samples.len() / 2;
    }
    println!("sample-5s.mp4: {samples} audio frames at 48 kHz");
    assert!((230_000..=290_000).contains(&samples), "{samples} samples");
}

#[test]
fn big_buck_bunny_is_hevc_1080p_without_audio() {
    let mut source = open("Big_Buck_Bunny_1080_10s_5MB.mp4");
    assert_eq!((source.width, source.height), (1920, 1080));
    assert_eq!(source.codec(), Id::HEVC);
    assert!(!source.has_audio);
    assert!((source.duration_seconds - 10.0).abs() < 0.1);
    let (count, last) = count_frames(&mut source, 1000);
    println!("Big_Buck_Bunny: {count} frames, last pts {last:.3}");
    assert!((290..=310).contains(&count), "{count} frames");
    let audio = AudioSource::open(
        &fixtures::require("Big_Buck_Bunny_1080_10s_5MB.mp4"),
        48_000,
        2,
    );
    assert!(matches!(
        audio,
        Err(gamut_media::VideoError::NoAudioStream { .. })
    ));
}

#[test]
fn a_seek_lands_within_one_frame() {
    let mut source = open("Big_Buck_Bunny_1080_10s_5MB.mp4");
    source.seek(2.0).expect("seek");
    let frame = source.next_frame().expect("decode").expect("a frame");
    println!("seek to 2.0 gave pts {:.4}", frame.pts_seconds);
    assert!((frame.pts_seconds - 2.0).abs() <= source.frame_seconds() + 1e-6);
    source.seek(0.5).expect("seek back");
    let frame = source.next_frame().expect("decode").expect("a frame");
    assert!((frame.pts_seconds - 0.5).abs() <= source.frame_seconds() + 1e-6);
}

#[test]
fn the_iphone_excerpt_is_hevc_4k_hlg_p010() {
    let Some(path) = fixtures::local("iphone_hevc_10s.mov") else {
        println!("iphone_hevc_10s.mov is not on this machine, skipped");
        return;
    };
    let mut source = VideoSource::open(&path).expect("open the iPhone excerpt");
    println!("iphone_hevc_10s.mov: decoder: {:?}", source.decoder_kind);
    assert_eq!(source.codec(), Id::HEVC);
    assert_eq!((source.coded_width, source.coded_height), (3840, 2160));
    assert_eq!(source.rotation, 90);
    assert_eq!((source.width, source.height), (2160, 3840));
    assert_eq!(source.colour.transfer, Transfer::Hlg);
    assert_eq!(source.colour.space, YuvSpace::Bt2020);
    assert!(source.has_audio);
    let first = source.next_frame().expect("decode").expect("a frame");
    assert_eq!(first.format, PlaneFormat::P010);
    assert_eq!(first.y.len(), 3840 * 2160 * 2);
    let (count, last) = count_frames(&mut source, 1000);
    println!(
        "iphone_hevc_10s.mov: {} frames, last pts {last:.3}",
        count + 1
    );
    assert!((200..=300).contains(&(count + 1)), "{count} frames");
}

#[test]
fn nvdec_decodes_the_iphone_excerpt() {
    if !nvdec_available() {
        println!("no CUDA device, NVDEC test skipped");
        return;
    }
    let Some(path) = fixtures::local("iphone_hevc_10s.mov") else {
        println!("iphone_hevc_10s.mov is not on this machine, skipped");
        return;
    };
    let mut source = VideoSource::open(&path).expect("open the iPhone excerpt");
    println!("decoder: {:?}", source.decoder_kind);
    assert_eq!(source.decoder_kind, Decoder::Nvdec);
    let (count, _) = count_frames(&mut source, 60);
    assert_eq!(count, 60);
    println!(
        "last frame: decode {:.2} ms, copy {:.2} ms",
        source.last_timing.decode_ms, source.last_timing.copy_ms
    );
    let software = VideoSource::open_software(&path).expect("open in software");
    assert_eq!(software.decoder_kind, Decoder::Software);
}
