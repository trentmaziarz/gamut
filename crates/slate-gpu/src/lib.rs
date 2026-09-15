//! slate-gpu is the wgpu render graph and the WGSL shaders. Textures are
//! rgba16float in the working space and masks are r8unorm. The video
//! compositor lives here too, with transforms, transitions, overlays, text and
//! SVG rasterized into textures, and readback for export. The render graph
//! draws into an offscreen texture that egui then draws as an image. The
//! picture never leaves the GPU.
//!
//! In M0 the graph holds one pass, the [`test_image`] and
//! the [`headless`] context that runs it without a window.

pub mod headless;
pub mod test_image;

pub use headless::Headless;
pub use test_image::TestImage;

/// A fullscreen triangle: three vertices, no vertex buffer. Every pass that
/// covers the whole target draws this.
pub(crate) const FULLSCREEN_VERTICES: u32 = 3;

/// The pipeline state shared by every fullscreen pass.
pub(crate) fn fullscreen_primitive() -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        unclipped_depth: false,
        polygon_mode: wgpu::PolygonMode::Fill,
        conservative: false,
    }
}
