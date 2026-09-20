//! Exports the sample clip whole as a Reel and checks the file with
//! ffprobe: H.264 High 1080x1920 yuv420p at 30 fps, AAC 48 kHz stereo at
//! the 128 kbps setting, the moov atom before mdat, the duration within a
//! frame of the source, and a key frame every 30 frames. Skips with a
//! printed line when NVENC is not there.
//!
//! The AAC average the file measures depends on the sound: ffmpeg's own
//! encoder lands this clip at about 113 kbps for the 128 kbps setting, and
//! a 60 second sine at 128. So the check here is against what ffmpeg's
//! command line produces for the same audio at 128k, within 10 percent;
//! reel_timing.rs checks the 60 second clip against 128 kbps itself.

use std::path::Path;
use std::process::Command;

use gamut_app::reel::export_file;
use gamut_media::fixtures;
use gamut_media::video_export::nvenc_available;

fn ffprobe(args: &[&str]) -> String {
    let output = Command::new("ffprobe")
        .args(["-v", "error"])
        .args(args)
        .output()
        .expect("ffprobe runs (the ffmpeg bin folder is on PATH)");
    assert!(
        output.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The AAC bit rate ffmpeg's command line reaches for `source` at 128k,
/// 48 kHz stereo: the reference the writer is held to.
fn reference_audio_bit_rate(source: &Path, dir: &Path) -> f64 {
    let out = dir.join("reference_aac.mp4");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(source)
        .args([
            "-vn", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        ])
        .arg(&out)
        .status()
        .expect("ffmpeg runs (the ffmpeg bin folder is on PATH)");
    assert!(status.success(), "ffmpeg failed to write the reference");
    ffprobe(&[
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=bit_rate",
        "-of",
        "csv=p=0",
        &out.to_string_lossy(),
    ])
    .trim()
    .parse()
    .expect("a bit rate")
}

fn atom_offset(bytes: &[u8], tag: &[u8; 4]) -> Option<usize> {
    bytes.windows(4).position(|w| w == tag)
}

#[test]
fn the_sample_clip_exports_as_a_reel_that_ffprobe_accepts() {
    let _ = env_logger::try_init();
    if !nvenc_available() {
        println!("no NVENC, Reel export test skipped");
        return;
    }
    let source = fixtures::require("sample-5s.mp4");
    let dir = std::env::temp_dir().join("gamut-reel-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let out = dir.join("sample-5s_reel.mp4");
    let _ = std::fs::remove_file(&out);
    let seconds = export_file(&source, &out).expect("export the Reel");
    println!("exported {} in {seconds:.2} s", out.display());
    assert!(out.is_file());

    let path = out.to_string_lossy().into_owned();
    let streams: serde_json::Value = serde_json::from_str(&ffprobe(&[
        "-show_entries",
        "stream=codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate,sample_rate,channels,bit_rate:format=duration",
        "-of",
        "json",
        &path,
    ]))
    .expect("ffprobe json");
    let video = streams["streams"]
        .as_array()
        .expect("streams")
        .iter()
        .find(|s| s["codec_type"] == "video")
        .expect("a video stream")
        .clone();
    let audio = streams["streams"]
        .as_array()
        .expect("streams")
        .iter()
        .find(|s| s["codec_type"] == "audio")
        .expect("an audio stream")
        .clone();
    println!("video: {video}");
    println!("audio: {audio}");
    assert_eq!(video["codec_name"], "h264");
    assert_eq!(video["profile"], "High");
    assert_eq!(video["width"], 1080);
    assert_eq!(video["height"], 1920);
    assert_eq!(video["pix_fmt"], "yuv420p");
    assert_eq!(video["r_frame_rate"], "30/1");
    assert_eq!(audio["codec_name"], "aac");
    assert_eq!(audio["sample_rate"], "48000");
    assert_eq!(audio["channels"], 2);
    let audio_bit_rate: f64 = audio["bit_rate"]
        .as_str()
        .expect("audio bit rate")
        .parse()
        .expect("a number");
    let reference = reference_audio_bit_rate(&source, &dir);
    println!(
        "audio bit rate {audio_bit_rate} against ffmpeg's own {reference} at the 128k setting"
    );
    assert!(
        (audio_bit_rate - reference).abs() <= reference * 0.1,
        "audio bit rate {audio_bit_rate} is not within 10 percent of {reference}"
    );

    // The source is 5.7 seconds of video; the Reel is within one frame.
    let source_duration = ffprobe(&[
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "csv=p=0",
        &source.to_string_lossy(),
    ]);
    let source_duration: f64 = source_duration.trim().parse().expect("source duration");
    let duration: f64 = streams["format"]["duration"]
        .as_str()
        .expect("duration")
        .parse()
        .expect("a number");
    println!("duration {duration:.3} s against the source's {source_duration:.3} s");
    assert!(
        (duration - source_duration).abs() <= 1.0 / 30.0 + 1e-3,
        "duration {duration} against {source_duration}"
    );

    let bytes = std::fs::read(&out).expect("read the file");
    let moov = atom_offset(&bytes, b"moov").expect("a moov atom");
    let mdat = atom_offset(&bytes, b"mdat").expect("an mdat atom");
    println!("moov at {moov}, mdat at {mdat}");
    assert!(moov < mdat, "moov {moov} is not before mdat {mdat}");

    let key_frames = ffprobe(&[
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=key_frame",
        "-of",
        "csv=p=0",
        &path,
    ]);
    let flags: Vec<bool> = key_frames
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim() == "1")
        .collect();
    println!(
        "{} frames, key frames at {:?}",
        flags.len(),
        flags
            .iter()
            .enumerate()
            .filter(|(_, k)| **k)
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    );
    assert!(flags.len() >= 150, "{} frames", flags.len());
    for (i, key) in flags.iter().enumerate() {
        assert_eq!(*key, i % 30 == 0, "frame {i} key {key}");
    }
    assert!(Path::new(&path).is_file());
}
