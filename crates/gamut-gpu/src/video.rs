//! The GPU side of a video frame: two textures holding the luma and the
//! interleaved chroma planes of a decoded frame (R8Unorm and Rg8Unorm for
//! NV12, R16Unorm and Rg16Unorm for P010), written with `write_texture`
//! from the CPU planes, and the uniform the `yuv_to_working` pass reads.
//! The frame's rotation and colour tags travel with it; the pass applies
//! the rotation in its sampling.

use bytemuck::{Pod, Zeroable};
use gamut_color::video::{self, PlaneFormat, VideoColour};
use gamut_media::FramePlanes;

/// The device feature the 16 bit plane formats need. Asked for when the
/// adapter has it; without it a 10 bit clip cannot be drawn.
pub const P010_FEATURE: wgpu::Features = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;

/// The features slate asks a device for: [`P010_FEATURE`] when the
/// adapter offers it.
pub fn wanted_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    adapter.features() & P010_FEATURE
}

/// The luma and chroma formats of a plane format.
pub fn texture_formats(format: PlaneFormat) -> (wgpu::TextureFormat, wgpu::TextureFormat) {
    match format {
        PlaneFormat::Nv12 => (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm),
        PlaneFormat::P010 => (
            wgpu::TextureFormat::R16Unorm,
            wgpu::TextureFormat::Rg16Unorm,
        ),
    }
}

/// The uniform of the video pass, in the layout `yuv_to_working.wgsl`
/// declares.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VideoUniform {
    pub yuv: [[f32; 4]; 3],
    pub primaries: [[f32; 4]; 3],
    pub black: f32,
    pub luma_range: f32,
    pub chroma_range: f32,
    pub mid: f32,
    pub transfer: u32,
    pub rotation: u32,
    pub full_range: u32,
    pub _pad: u32,
    /// The part of the displayed frame the render covers, as x, y, width
    /// and height in 0 to 1.
    pub window: [f32; 4],
}

impl VideoUniform {
    pub fn new(format: PlaneFormat, colour: VideoColour, rotation: u32, window: [f32; 4]) -> Self {
        let range = video::range(format, colour.full_range);
        Self {
            yuv: video::yuv_matrix(colour.space).to_wgsl_columns(),
            primaries: video::primaries_matrix(colour.space).to_wgsl_columns(),
            black: range.black,
            luma_range: range.luma,
            chroma_range: range.chroma,
            mid: range.mid,
            transfer: colour.transfer.id(),
            rotation,
            full_range: u32::from(colour.full_range),
            _pad: 0,
            window,
        }
    }
}

/// The plane textures of one video source on the device.
pub struct VideoSource {
    luma: wgpu::Texture,
    chroma: wgpu::Texture,
    pub luma_view: wgpu::TextureView,
    pub chroma_view: wgpu::TextureView,
    /// The stored size, before the rotation.
    pub width: u32,
    pub height: u32,
    pub format: PlaneFormat,
    pub colour: VideoColour,
    /// Degrees clockwise the stored frame turns to be displayed.
    pub rotation: u32,
}

impl VideoSource {
    /// Creates the two plane textures for frames of the given stored size.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: PlaneFormat,
        colour: VideoColour,
        rotation: u32,
    ) -> Self {
        let (luma_format, chroma_format) = texture_formats(format);
        let make = |label: &str, format, width, height| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let luma = make("video luma", luma_format, width, height);
        let chroma = make(
            "video chroma",
            chroma_format,
            width.div_ceil(2),
            height.div_ceil(2),
        );
        Self {
            luma_view: luma.create_view(&wgpu::TextureViewDescriptor::default()),
            chroma_view: chroma.create_view(&wgpu::TextureViewDescriptor::default()),
            luma,
            chroma,
            width,
            height,
            format,
            colour,
            rotation,
        }
    }

    /// Whether `frame` fits these textures.
    pub fn accepts(&self, frame: &dyn FramePlanes) -> bool {
        frame.width() == self.width
            && frame.height() == self.height
            && frame.format() == self.format
    }

    /// Writes the frame's planes into the textures, padding and all. The
    /// frame must have the size and format the textures were made for.
    pub fn upload(&self, queue: &wgpu::Queue, frame: &dyn FramePlanes) {
        assert!(self.accepts(frame), "the frame matches the plane textures");
        queue.write_texture(
            self.luma.as_image_copy(),
            frame.y(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(frame.y_stride() as u32),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        let chroma_height = self.height.div_ceil(2);
        queue.write_texture(
            self.chroma.as_image_copy(),
            frame.uv(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(frame.uv_stride() as u32),
                rows_per_image: Some(chroma_height),
            },
            wgpu::Extent3d {
                width: self.width.div_ceil(2),
                height: chroma_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// The size after the rotation, which is what the viewer shows.
    pub fn display_size(&self) -> (u32, u32) {
        if self.rotation == 90 || self.rotation == 270 {
            (self.height, self.width)
        } else {
            (self.width, self.height)
        }
    }

    /// The uniform for this source, rendering `window` of the displayed
    /// frame.
    pub fn uniform(&self, window: [f32; 4]) -> VideoUniform {
        VideoUniform::new(self.format, self.colour, self.rotation, window)
    }
}
