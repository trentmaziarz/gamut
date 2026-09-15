//! slate-gpu is the wgpu render graph and the WGSL shaders. Textures are
//! rgba16float in the working space and masks are r8unorm. The video
//! compositor lives here too, with transforms, transitions, overlays, text and
//! SVG rasterized into textures, and readback for export. The render graph
//! draws into an offscreen texture that egui then draws as an image. The
//! picture never leaves the GPU.

#[cfg(test)]
mod tests {
    #[test]
    fn crate_links() {
        // The crate compiles and its test harness runs. Real tests arrive
        // with the milestone that fills the crate.
    }
}
