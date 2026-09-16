//! Times the Reel export of the 60 second 4K30 timing clip placed whole
//! on the timeline. The wall time is printed and must be under 15 seconds
//! only when SLATE_TIMING_GATE=1 is set; the test skips when the clip is
//! not on the machine or NVENC is not there.

use slate_app::reel::export_file;
use slate_media::fixtures;
use slate_media::video_export::nvenc_available;

/// The wall time the gate demands, in seconds.
const GATE_SECONDS: f64 = 15.0;

const GATE: &str = "SLATE_TIMING_GATE";

#[test]
fn a_60_second_reel_exports_fast_enough() {
    let _ = env_logger::try_init();
    let Some(path) = fixtures::local("timing_4k30_60s.mp4") else {
        println!("timing_4k30_60s.mp4 is not on this machine, skipped");
        return;
    };
    if !nvenc_available() {
        println!("no NVENC, skipped");
        return;
    }
    let dir = std::env::temp_dir().join("slate-reel-timing");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let out = dir.join("timing_reel.mp4");
    let _ = std::fs::remove_file(&out);
    let seconds = export_file(&path, &out).expect("export the Reel");
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!("60 second Reel exported in {seconds:.2} s, {size} bytes");
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=bit_rate,duration",
            "-of",
            "default=nw=1",
        ])
        .arg(&out)
        .output()
        .expect("ffprobe runs (the ffmpeg bin folder is on PATH)");
    let probe = String::from_utf8_lossy(&probe.stdout);
    println!(
        "audio: {}",
        probe.trim().lines().collect::<Vec<_>>().join("; ")
    );
    let bit_rate: f64 = probe
        .lines()
        .find_map(|l| l.strip_prefix("bit_rate="))
        .expect("a bit rate")
        .trim()
        .parse()
        .expect("a number");
    assert!(
        (bit_rate - 128_000.0).abs() <= 12_800.0,
        "audio bit rate {bit_rate} is not within 10 percent of 128 kbps"
    );
    if std::env::var(GATE).as_deref() == Ok("1") {
        assert!(
            seconds < GATE_SECONDS,
            "{seconds:.2} s is not under {GATE_SECONDS} s"
        );
    } else {
        println!("{GATE} is not set, so the {GATE_SECONDS} s gate is printed and not asserted");
    }
}
