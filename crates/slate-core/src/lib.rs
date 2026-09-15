//! slate-core holds the project document. It carries the edit parameters for
//! a photo and the timeline model for a video. The timeline model covers
//! tracks, clips, in and out points, speed curves, keyframes, transitions and
//! overlays. Presets and undo history live here too. It is plain data with
//! serde, no GPU and no I/O. A photo edit is stored as a sidecar JSON next to
//! the original, a video project is a .slate JSON file with media paths
//! relative to it, and presets are JSON subsets of the photo parameters. All
//! three are text, so they diff in git.

#[cfg(test)]
mod tests {
    #[test]
    fn crate_links() {
        // The crate compiles and its test harness runs. Real tests arrive
        // with the milestone that fills the crate.
    }
}
