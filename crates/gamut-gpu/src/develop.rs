//! The develop render graph: five passes from a photo to an 8-bit sRGB
//! picture, all on one device.
//!
//! 1. The source upload: the photo's RGBA bytes in a plain 8-bit format.
//!    The sRGB curve is decoded in the shader rather than by the sampler:
//!    the sampler's decode is implementation-defined and measured 1 to 2
//!    codes off on dark channels on this machine's driver, and every M1
//!    source shares the sRGB curve anyway; only the matrix differs.
//! 2. `input_transform.wgsl`: samples the source at the render size and
//!    writes linear Rec.2020 into the working texture.
//! 3. `blur.wgsl` twice: the luminance of the working texture under a
//!    gaussian, the base layer that highlights and shadows split on.
//! 4. `develop.wgsl`: the whole develop chain of gamut-color's
//!    `develop_pixel_with`, from a uniform that mirrors `PhotoEdit`, into
//!    the developed texture.
//! 5. `output.wgsl`: the crop out of the developed texture, Rec.2020 to
//!    sRGB, clipped, into an 8-bit sRGB texture that egui draws and
//!    `Readback` reads.
//!
//! Passes 2 and 3 depend only on the source and the render size, so they
//! run again only when one of those changes. A slider change runs 4 and 5.
//!
//! M3 adds three products the develop pass reads, each skipped while its
//! operator is at rest, so an edit that does not use them costs nothing:
//!
//! - The texture layer: `blur.wgsl` twice more at the finer sigma, into its
//!   own R16Float texture. A head-pass product: it runs once when texture
//!   leaves 0 and again only when pass 2 runs again.
//! - The transmission map of dehaze: `minimum.wgsl` twice (the dark channel
//!   over the atmospheric light under a square minimum filter), then
//!   `blur.wgsl` twice to smooth it, into its own R16Float texture. Also a
//!   head-pass product, run once when dehaze leaves 0. The atmospheric light
//!   is read from the photo on the CPU in `set_source`; a video frame takes
//!   white.
//! - The tone curve table: 1024 by 1 in Rgba32Float, baked by gamut-color
//!   and uploaded only when a curve changes. It is read with `textureLoad`,
//!   two loads and a mix, so it needs no filtering.
//!
//! A mask is an alpha the masked develop pass reads: `mask.wgsl` draws it from
//! the mask's components, a head-pass product. With Refine edges on,
//! `refine.wgsl` filters it against the working texture into a second alpha,
//! the refined one, and the develop pass and the overlay read that instead.
//! It is drawn again whenever the alpha is and when its own setting changes,
//! never by a slider of the develop chain, and an edit with no refined mask
//! runs none of its passes and holds none of its float targets.
//!
//! After Refine edges come the three edge controls of `edge.wgsl`: Shift
//! edge, Feather and Contrast, in that order. Each that is on keeps its
//! output in its own texture, and the last of them is the finished alpha the
//! develop pass and the overlay read. A mask with the three at rest runs
//! none of their passes and holds none of their textures.
//!
//! M2 lets a decoded video frame stand in for the photo: the source is
//! then the two plane textures of [`crate::video::VideoSource`] and pass 2
//! is `yuv_to_working.wgsl`, which writes the same linear Rec.2020 working
//! texture. Passes 3 to 5 do not know the difference. A new frame of the
//! same size rewrites the planes and reruns passes 2 and 3 without
//! rebuilding any texture.

use std::cmp::Ordering;

use bytemuck::{Pod, Zeroable};
use gamut_color::SourceSpace;
use gamut_color::basic;
use gamut_color::brush::Proxy as ProxyTwin;
use gamut_color::curve::{self, TABLE_SIZE};
use gamut_color::dehaze;
use gamut_color::edge::{self as edge_twin, Plan as EdgePlan};
use gamut_color::hsl::HslParams;
use gamut_color::local;
use gamut_color::mask::{self as mask_twin, Geometry};
use gamut_color::matrices;
use gamut_color::refine::{self as refine_twin, Plan};
use gamut_color::video::VideoColour;
use gamut_color::wheels::Cdl;
use gamut_core::brush::Brush;
use gamut_core::look::ToneCurves;
use gamut_core::mask::{
    Edge, MAX_COMPONENTS, MAX_MASKS, Mask, MaskOp, MaskShape, MaskSource, Refine,
};
use gamut_core::{Adjustments, CropRect, ExportPreset, PhotoEdit};
use gamut_media::{FramePlanes, Photo};

use crate::brush_layer::{AutoInputs, BrushPass, Drawn, Layers, has_auto};
use crate::edge::{EdgePass, EdgePasses, EdgeScratch, EdgeWork, Edged, Held as EdgeHeld};
use crate::proxy::{self, ProxyUniform, Tile};
use crate::refine::{RefinePass, Refined, Scratch};
use crate::video::{VideoSource, VideoUniform};
use crate::{FULLSCREEN_VERTICES, Readback, fullscreen_primitive};

/// The working-space format: linear Rec.2020 in half floats.
pub const WORKING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The base layer format: one half float of luminance.
pub const BASE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// The output format: 8-bit sRGB, encoded by the hardware on write.
pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The tone curve table format: full floats, read with `textureLoad`.
pub const TABLE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// The alpha of a mask: one byte per pixel.
pub const ALPHA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// How much of a frame built ahead one slice records, in texels read and
/// written: 700,000,000. Each pixel a head pass draws counts its reads plus
/// its one write, so a blur of radius r counts 2 + 2 ceil(r / 2) (the
/// centre tap and one bilinear read for each pair of taps on each side), a
/// minimum of radius r counts 2r + 2, and the input transform counts its
/// taps squared plus 1 (a video frame's 2 planes plus 1).
///
/// The reason is the frame time. At 5b8e1b1 replacing the padded window of
/// timing_24mp.jpg at 100 percent, every operator on, took 36.24 to 54.59
/// ms. That frame is 3104 by 3744 pixels; its nine head passes draw
/// 90,146,736 pixels and count 9,867,249,792 texels (blurs of radius 240,
/// 30 and 180 over 2628 columns or a 2628 by 3268 region, two minimums of
/// radius 30 and the input over all 11,621,376 pixels). Charging the whole
/// 54.59 ms to the head passes, the most any of them can cost, gives
/// 180,751,965 texels a millisecond, so 4 ms is 723,007,862. A slice of
/// 700,000,000 costs 3.87 ms at that rate, 2.57 ms at the 36.24 ms one,
/// and the frame takes 15 slices, one a UI frame. Beside a pan inside the
/// window (p95 1.38 ms) a UI frame stays well inside 16 ms, and the 15
/// frames are fewer than the 20 steps of 32 pixels a pan takes to cross
/// the 640 pixel pad. Counting texels rather than pixels keeps a slice's
/// cost even between an input pass that reads 2 texels a pixel and a base
/// blur that reads 242.
///
/// It is a schedule, not arithmetic: no pixel depends on it, and the timing
/// phase moves it on the evidence.
pub const AHEAD_SLICE_TEXELS: u64 = 700_000_000;

/// The rows of the tone curve table: the global edit, then one per mask.
pub const TABLE_ROWS: u32 = 1 + MAX_MASKS as u32;

/// The most samples per axis the input transform averages per output pixel.
pub const MAX_TAPS: u32 = 8;

/// The `flags` bits of the develop uniform.
const FLAG_CURVES: u32 = 1;
const FLAG_HSL: u32 = 2;
const FLAG_WHEELS: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InputUniform {
    matrix: [[f32; 4]; 3],
    render_size: [f32; 2],
    decode_srgb: u32,
    taps: u32,
    window: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurUniform {
    direction: [i32; 2],
    radius: i32,
    luma: u32,
    sigma: f32,
    _pad: [f32; 3],
}

/// The uniform of `minimum.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MinimumUniform {
    direction: [i32; 2],
    radius: i32,
    stage: u32,
    atmosphere: [f32; 4],
}

/// Mirrors the `Uniform` of `develop.wgsl` member for member; a test holds
/// the two layouts equal.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DevelopUniform {
    white_balance: [[f32; 4]; 3],
    white_balance_temperature: f32,
    white_balance_tint: f32,
    exposure: f32,
    contrast: f32,
    highlights: f32,
    shadows: f32,
    whites: f32,
    blacks: f32,
    vibrance: f32,
    saturation: f32,
    texture: f32,
    clarity: f32,
    dehaze: f32,
    flags: u32,
    table_row: u32,
    opacity: f32,
    atmosphere: [f32; 4],
    cdl_slope: [f32; 4],
    cdl_offset: [f32; 4],
    cdl_power: [f32; 4],
    hsl: [[f32; 4]; 8],
}

impl DevelopUniform {
    /// The uniform of the global pass. `atmosphere` is the atmospheric light
    /// of the source.
    fn new(edit: &Adjustments, atmosphere: [f32; 3]) -> Self {
        let wb =
            basic::white_balance_matrix(edit.white_balance_temperature, edit.white_balance_tint);
        let look = &edit.look;
        let mut flags = 0;
        if !look.curves.is_identity() {
            flags |= FLAG_CURVES;
        }
        let mut hsl = [[0.0; 4]; 8];
        if !look.hsl_is_identity() {
            flags |= FLAG_HSL;
            hsl = HslParams::new(&look.hsl).ranges;
        }
        let mut cdl = Cdl::IDENTITY;
        if !look.wheels.is_identity() {
            flags |= FLAG_WHEELS;
            cdl = Cdl::new(&look.wheels);
        }
        let wide = |v: [f32; 3]| [v[0], v[1], v[2], 1.0];
        Self {
            white_balance: wb.to_wgsl_columns(),
            white_balance_temperature: edit.white_balance_temperature,
            white_balance_tint: edit.white_balance_tint,
            exposure: edit.exposure,
            contrast: edit.contrast,
            highlights: edit.highlights,
            shadows: edit.shadows,
            whites: edit.whites,
            blacks: edit.blacks,
            vibrance: edit.vibrance,
            saturation: edit.saturation,
            texture: edit.texture,
            clarity: edit.clarity,
            dehaze: edit.dehaze,
            flags,
            table_row: 0,
            opacity: 1.0,
            atmosphere: wide(basic::exposed_atmosphere(&wb, atmosphere, edit.exposure)),
            cdl_slope: wide(cdl.slope),
            cdl_offset: wide(cdl.offset),
            cdl_power: wide(cdl.power),
            hsl,
        }
    }
}

impl DevelopUniform {
    /// The uniform of the pass of the mask at `index` of the list: its
    /// effective adjustments, its row of the curve table, its opacity. The
    /// curves run when the global edit or the mask has one.
    fn for_mask(global: &Adjustments, mask: &Mask, index: usize, atmosphere: [f32; 3]) -> Self {
        let effective = mask_twin::effective_adjustments(global, &mask.adjust);
        let mut uniform = Self::new(&effective, atmosphere);
        if !mask.adjust.look.curves.is_identity() {
            uniform.flags |= FLAG_CURVES;
        }
        uniform.table_row = index as u32 + 1;
        uniform.opacity = mask.opacity / 100.0;
        uniform
    }
}

/// One component of `mask.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskComponentUniform {
    header: [u32; 4],
    a: [f32; 4],
    b: [f32; 4],
}

/// Mirrors the `Uniform` of `mask.wgsl`; a test holds the two layouts equal.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskUniform {
    window: [f32; 4],
    render_size: [f32; 2],
    aspect: [f32; 2],
    count: u32,
    invert: u32,
    _pad: [u32; 2],
    components: [MaskComponentUniform; MAX_COMPONENTS],
}

impl MaskUniform {
    /// `mask` is sanitised. Everything the twin computes once per mask is
    /// computed here with the same arithmetic: the sine and the cosine of a
    /// rotation, the inner share of a feather, the floored falloffs.
    fn new(mask: &Mask, geometry: &Geometry) -> Self {
        let mut components = [MaskComponentUniform::zeroed(); MAX_COMPONENTS];
        let mut layer = 0;
        for (slot, component) in components.iter_mut().zip(&mask.components) {
            let mut spare = 0;
            let (kind, a, b) = match &component.source {
                MaskSource::Linear(g) => {
                    (0, [g.start[0], g.start[1], g.end[0], g.end[1]], [0.0; 4])
                }
                MaskSource::Radial(g) => {
                    let (sin, cos) = g.rotation.to_radians().sin_cos();
                    let inner = (1.0 - g.feather / 100.0).min(mask_twin::RADIAL_INNER_CEILING);
                    (
                        1,
                        [g.centre[0], g.centre[1], g.radius[0], g.radius[1]],
                        [sin, cos, inner, 0.0],
                    )
                }
                MaskSource::Luminance(r) => (
                    2,
                    [
                        r.low,
                        r.high,
                        r.falloff.max(mask_twin::LUMINANCE_FALLOFF_FLOOR),
                        0.0,
                    ],
                    [0.0; 4],
                ),
                MaskSource::Colour(r) => (
                    3,
                    [
                        r.hue,
                        r.hue_width / 2.0,
                        r.chroma_low,
                        r.falloff.max(mask_twin::HUE_FALLOFF_FLOOR),
                    ],
                    [0.0; 4],
                ),
                MaskSource::Brush(_) => {
                    // Its layer of the mask's array, in component order.
                    spare = layer;
                    layer += 1;
                    (4, [0.0; 4], [0.0; 4])
                }
            };
            let op = match component.op {
                MaskOp::Add => 0,
                MaskOp::Subtract => 1,
                MaskOp::Intersect => 2,
            };
            *slot = MaskComponentUniform {
                header: [kind, op, u32::from(component.invert), spare],
                a,
                b,
            };
        }
        let window = geometry.window;
        MaskUniform {
            window: [window.x, window.y, window.width, window.height],
            render_size: [geometry.size.0 as f32, geometry.size.1 as f32],
            aspect: geometry.aspect(),
            count: mask.components.len().min(MAX_COMPONENTS) as u32,
            invert: u32::from(mask.invert),
            _pad: [0; 2],
            components,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OutputUniform {
    matrix: [[f32; 4]; 3],
    crop: [f32; 4],
    overlay: [f32; 4],
}

/// What of the picture a render has to draw again because of the masks.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Damage {
    Nothing,
    /// Only where new dabs of a growing brush reach.
    Part(PixelRect),
    Whole,
}

impl Damage {
    fn with(self, rect: Option<PixelRect>) -> Damage {
        match (self, rect) {
            (Damage::Whole, _) => Damage::Whole,
            (damage, None) => damage,
            (Damage::Nothing, Some(rect)) => Damage::Part(rect),
            (Damage::Part(a), Some(b)) => {
                let (x0, y0) = (a.0.min(b.0), a.1.min(b.1));
                let (x1, y1) = ((a.0 + a.2).max(b.0 + b.2), (a.1 + a.3).max(b.1 + b.3));
                Damage::Part((x0, y0, x1 - x0, y1 - y0))
            }
        }
    }
}

/// Which alpha of a mask the develop pass and the overlay read: the edged
/// alpha while an edge control is on, else the refined alpha while Refine
/// edges is, else the alpha of the components.
fn product_of(slot: &FrameMask) -> Product {
    match slot.edged.as_ref().filter(|e| e.product().is_some()) {
        Some(edged) => Product::Edged(edged.product_id()),
        None if slot.refined.is_some() => Product::Refined,
        None => Product::Alpha,
    }
}

/// The view of [`product_of`].
fn product_view(slot: &FrameMask) -> &wgpu::TextureView {
    slot.edged
        .as_ref()
        .and_then(Edged::product)
        .or(slot.refined.as_ref().map(|refined| &refined.alpha))
        .unwrap_or(&slot.alpha.view)
}

/// `rect` with `by` pixels more on every side, inside a render of `size`.
/// Where one render drew each stage of a mask's shape again, in pixels of
/// its frame, for the cache test: the alpha, the refined alpha, the passes
/// of Shift edge, Feather's cells and the finished alpha. `None` where a
/// stage drew nothing.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShapeRedrawn {
    /// The size of the frame, and the part of it the develop passes draw.
    pub frame: (u32, u32),
    pub region: PixelRect,
    pub alpha: Option<PixelRect>,
    pub refine: Option<PixelRect>,
    pub shift: Option<PixelRect>,
    pub feather: Option<PixelRect>,
    pub finish: Option<PixelRect>,
}

fn grow(rect: PixelRect, by: u32, size: (u32, u32)) -> PixelRect {
    let (x0, y0) = (rect.0.saturating_sub(by), rect.1.saturating_sub(by));
    let x1 = (rect.0 + rect.2 + by).min(size.0).max(x0);
    let y1 = (rect.1 + rect.3 + by).min(size.1).max(y0);
    (x0, y0, x1 - x0, y1 - y0)
}

/// The part of `a` inside `b`, when there is any.
fn intersect(a: PixelRect, b: PixelRect) -> Option<PixelRect> {
    let (x0, y0) = (a.0.max(b.0), a.1.max(b.1));
    let (x1, y1) = ((a.0 + a.2).min(b.0 + b.2), (a.1 + a.3).min(b.1 + b.3));
    (x1 > x0 && y1 > y0).then(|| (x0, y0, x1 - x0, y1 - y0))
}

/// A pipeline and the layout of its one bind group.
struct Pass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

/// A render target and its view.
struct Target {
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// What the head of the graph reads.
enum SourceKind {
    /// A photo's RGBA texture, through the input transform.
    Photo {
        view: wgpu::TextureView,
        space: SourceSpace,
    },
    /// A video frame's planes, through the YUV pass. Boxed because the
    /// planes are far larger than the photo variant.
    Video(Box<VideoSource>),
}

struct Source {
    kind: SourceKind,
    /// The display size: the photo's, or the frame's after its rotation.
    width: u32,
    height: u32,
    /// Counts up when the textures are replaced.
    generation: u64,
    /// Counts up when the content changes but the textures stay.
    content: u64,
    /// The atmospheric light dehaze works against, in linear Rec.2020.
    atmosphere: [f32; 3],
}

/// Everything that depends on the source and the render size.
struct Frame {
    width: u32,
    height: u32,
    generation: u64,
    /// The source content the working and base textures hold.
    content: u64,
    /// The part of the source this render covers.
    window: CropRect,
    /// The size the blur sigma was taken from.
    sigma_size: (u32, u32),
    working: Target,
    ping: Target,
    base: Target,
    /// The texture layer; holds nothing until `texture_ready`.
    texture_base: Target,
    /// The transmission map; holds nothing until `transmission_ready`.
    transmission: Target,
    developed: Target,
    /// The other texture of the ordered blend. A mask pass reads what the
    /// passes before it left in one of the two and writes the other, and the
    /// last pass of a render writes `developed`.
    developed_other: Target,
    /// The texture layer matches the working texture and `products_region`.
    texture_ready: bool,
    /// The transmission map matches the working texture and
    /// `products_region`.
    transmission_ready: bool,
    /// The scissor the two products above were last drawn under.
    products_region: Option<(u32, u32, u32, u32)>,
    /// The edit and the scissor `developed` was last drawn for, or `None`
    /// when it has to be drawn again. While both hold, a render only runs
    /// the output pass.
    developed_for: Option<(PhotoEdit, (u32, u32, u32, u32))>,
    /// How many mask passes the two developed textures were last drawn by.
    /// Which of the two a pass writes follows from it, so a part is drawn
    /// over what they hold only while it is the same.
    developed_passes: usize,
    input_bind: wgpu::BindGroup,
    blur_h_bind: wgpu::BindGroup,
    blur_v_bind: wgpu::BindGroup,
    texture_h_bind: wgpu::BindGroup,
    texture_v_bind: wgpu::BindGroup,
    minimum_h_bind: wgpu::BindGroup,
    minimum_v_bind: wgpu::BindGroup,
    smooth_h_bind: wgpu::BindGroup,
    smooth_v_bind: wgpu::BindGroup,
    develop_bind: wgpu::BindGroup,
    /// What each mask of the list has on this frame, by its index in the
    /// list; nothing until the mask first runs.
    masks: [Option<FrameMask>; MAX_MASKS],
    /// The float targets Refine edges works in, shared by the masks; nothing
    /// while no mask of the edit is refined.
    refine_scratch: Option<Scratch>,
    /// The textures the edge controls work in, shared by the masks; nothing
    /// while no mask has Shift edge or Feather on.
    edge_scratch: EdgeScratch,
}

/// The scratch a refine drew with, as [`Develop::refine_last_hold`] reads
/// it.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefineHold {
    /// The cells a side the targets of the scratch hold.
    pub size: (u32, u32),
    /// The most cells a side a tile of the refine held.
    pub side: u32,
    /// Whether the scratch was made for this refine.
    pub made: bool,
    /// The branch of the rule: First, Fits, Cut or Floor.
    pub branch: &'static str,
}

/// A frame built ahead of a pan: the head passes of the window the pan will
/// reach, drawn in slices, each in a submit of its own, while the current
/// frame still serves every render.
struct NextFrame {
    /// The picture size and the padded window of the view it was built for.
    full: (u32, u32),
    window: PixelRect,
    frame: Frame,
    /// What the slices draw and how far they got.
    build: AheadBuild,
}

/// The progress of a frame built ahead: its head passes in the order they
/// are drawn, and the rows of them the slices submitted so far recorded.
struct AheadBuild {
    work: HeadWork,
    passes: Vec<HeadPass>,
    /// The pass the next slice begins in, and how many of its rows the
    /// slices before recorded. Every pass before it is submitted whole.
    pass: usize,
    row: u32,
    /// The source content the strips of the input transform read, while
    /// they all read the same one. A video frame that arrives between two
    /// of them leaves none, and the frame then holds no content until a
    /// render runs its head passes again.
    content: Option<u64>,
}

impl AheadBuild {
    /// Whether every pass is recorded and submitted.
    fn complete(&self) -> bool {
        self.pass == self.passes.len()
    }
}

/// A head pass of a frame, one render pass each in [`Develop::head_passes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeadPass {
    /// The input transform, or the YUV pass of a video frame, into working.
    Input,
    /// The base blur: working into ping, then ping into base.
    BlurH,
    BlurV,
    /// The texture layer's blur: working into ping, then into texture_base.
    TextureH,
    TextureV,
    /// The minimum of the transmission map: working into ping, then into
    /// transmission.
    MinimumH,
    MinimumV,
    /// The transmission map's smoothing: transmission into ping, then into
    /// transmission.
    TransmissionH,
    TransmissionV,
}

impl HeadPass {
    /// The passes that draw the source content when `input`, the texture
    /// layer when `texture` and the transmission map when `transmission`,
    /// in the order they run. Each horizontal blur writes ping and the
    /// vertical one after it reads it, before the next pass writes ping.
    fn plan(input: bool, texture: bool, transmission: bool) -> Vec<Self> {
        let mut passes = Vec::with_capacity(9);
        if input {
            passes.extend([Self::Input, Self::BlurH, Self::BlurV]);
        }
        if texture {
            passes.extend([Self::TextureH, Self::TextureV]);
        }
        if transmission {
            passes.extend([
                Self::MinimumH,
                Self::MinimumV,
                Self::TransmissionH,
                Self::TransmissionV,
            ]);
        }
        passes
    }

    /// The scissor of the pass: the columns under the products region for
    /// a horizontal blur, the region for a vertical one, none for the input
    /// transform and the minimum, which cover the whole render.
    fn scissor(self, work: &HeadWork) -> Option<(u32, u32, u32, u32)> {
        match self {
            Self::Input | Self::MinimumH | Self::MinimumV => None,
            Self::BlurH | Self::TextureH | Self::TransmissionH => {
                Some(scissor_for(work.products, work.size, true))
            }
            Self::BlurV | Self::TextureV | Self::TransmissionV => {
                Some(scissor_for(work.products, work.size, false))
            }
        }
    }

    /// The pixels the pass draws: its scissor, or the whole render.
    fn rect(self, work: &HeadWork) -> (u32, u32, u32, u32) {
        self.scissor(work)
            .unwrap_or((0, 0, work.size.0, work.size.1))
    }
}

/// What [`Develop::render_view`] renders for a view: the arguments of
/// `render_window`.
struct ViewGeometry {
    crop: CropRect,
    render_size: (u32, u32),
    output_size: (u32, u32),
    window: CropRect,
    sigma_size: (u32, u32),
    products: CropRect,
}

/// What the head passes of a frame draw: the render, the part of the source
/// it covers, the size the sigmas come from, the part the products cover,
/// and whether the texture layer and the transmission map are wanted.
struct HeadWork {
    size: (u32, u32),
    window: CropRect,
    sigma_size: (u32, u32),
    products: CropRect,
    texture: bool,
    transmission: bool,
}

/// Which alpha of a mask the develop pass and the overlay read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Product {
    /// The alpha of the components.
    Alpha,
    /// The refined alpha of Refine edges.
    Refined,
    /// The alpha of the edge controls: shifted, or finished by Feather and
    /// Contrast. It holds the product id, so a bind group made before the
    /// shifted or the finished alpha was made or dropped is made again.
    Edged(u64),
}

/// The head-pass product of one mask: its alpha, cached like the texture
/// layer and the transmission map.
struct FrameMask {
    alpha: Target,
    /// The shape `alpha` holds, with no Refine edges in it, or `None` when it
    /// has to be drawn again: the head passes reran or the scissor moved. A
    /// slider of the mask's adjustments is no part of the shape, so it never
    /// redraws the alpha.
    shape: Option<MaskShape>,
    /// The pixels `alpha` was last drawn over whole. A refined mask reads
    /// its alpha a filter's reach beyond the products.
    alpha_region: PixelRect,
    /// The refined alpha, while the mask has Refine edges on, and the
    /// setting it holds, or `None` when it has to be drawn again.
    refined: Option<Refined>,
    refine: Option<Refine>,
    /// The pixels `refined` was last drawn over whole.
    refined_region: PixelRect,
    /// The products of the edge controls, while one of them is on.
    edged: Option<Edged>,
    /// Which alpha `develop_binds` read.
    binds_product: Product,
    /// The layers of the mask's brush components, which `raster_bind` holds.
    layers: Layers,
    raster_bind: wgpu::BindGroup,
    /// The mask pass reading `developed`, then the one reading
    /// `developed_other`; each is drawn into the texture it does not read.
    develop_binds: [wgpu::BindGroup; 2],
}

/// One tile of the proxy build: the head pass draws the source pixels of
/// `plan` into the tile texture at their own size, and `proxy.wgsl` averages
/// them into the proxy pixels of `plan`.
struct ProxyTile {
    plan: Tile,
    head_uniform: wgpu::Buffer,
    head_bind: wgpu::BindGroup,
    reduce_bind: wgpu::BindGroup,
}

/// The reference proxy of the source, which the dabs of an auto stroke read
/// their reference colour from: a head-pass product of the source and of no
/// window. It exists only once an edit holds an auto stroke.
struct ProxyProduct {
    /// The source textures the tiles are bound to.
    generation: u64,
    /// The source content the proxy holds, or `None` before it is built.
    content: Option<u64>,
    target: Target,
    /// The tile texture: source pixels in the working format.
    tile: Target,
    tiles: Vec<ProxyTile>,
    /// Told apart from every proxy before it, for the bind groups that hold
    /// its view.
    id: u64,
}

struct Output {
    target: Target,
    bind: wgpu::BindGroup,
    frame_generation: u64,
    /// The mask whose alpha the bind group holds for the overlay, and which
    /// of its alphas that is.
    overlay: Option<(usize, Product)>,
}

/// The develop graph on one device. Build it once, set a source, render as
/// often as the edit or the view changes.
pub struct Develop {
    device: wgpu::Device,
    queue: wgpu::Queue,
    input: Pass,
    video: Pass,
    blur: Pass,
    minimum: Pass,
    develop: Pass,
    /// `mask.wgsl`: draws the alpha of one mask.
    mask: Pass,
    /// `brush.wgsl`: stamps the dabs of a painted source into its layer.
    brush: BrushPass,
    /// `proxy.wgsl`: reduces a tile of the source into the proxy.
    reduce: Pass,
    /// `refine.wgsl`: filters the alpha of a mask into its refined alpha.
    refine: RefinePass,
    /// `edge.wgsl`: Shift edge, Feather and Contrast after Refine edges.
    edge: EdgePass,
    /// How many times a refined alpha was drawn whole, and how many times
    /// over the reach of new dabs only, for the cache test.
    refine_builds: u64,
    refine_patches: u64,
    /// How many times Shift edge, Feather's cells and the finished alpha
    /// were drawn over their whole frame, and how many times over the reach
    /// of new dabs only, for the cache test.
    edge_builds: [u64; 3],
    edge_patches: [u64; 3],
    /// Where the last render drew each stage of each mask's shape again.
    shape_redrawn: [ShapeRedrawn; MAX_MASKS],
    proxy: Option<ProxyProduct>,
    /// How many proxies have been set up, and how many times one was built,
    /// for the cache test.
    proxy_ids: u64,
    proxy_builds: u64,
    /// `develop.wgsl` through `fs_masked`: develops one mask over what the
    /// passes before it left in the other developed texture.
    masked: Pass,
    output: Pass,
    input_uniform: wgpu::Buffer,
    video_uniform: wgpu::Buffer,
    blur_h_uniform: wgpu::Buffer,
    blur_v_uniform: wgpu::Buffer,
    texture_h_uniform: wgpu::Buffer,
    texture_v_uniform: wgpu::Buffer,
    minimum_h_uniform: wgpu::Buffer,
    minimum_v_uniform: wgpu::Buffer,
    smooth_h_uniform: wgpu::Buffer,
    smooth_v_uniform: wgpu::Buffer,
    develop_uniform: wgpu::Buffer,
    /// Per mask of the list: the uniform of its alpha and of its develop.
    mask_uniforms: Vec<wgpu::Buffer>,
    masked_uniforms: Vec<wgpu::Buffer>,
    /// The tone curve table and the curves it was baked from: row 0 from
    /// the global curves, row 1 + i from the global curves and those of
    /// mask i.
    curve_table: wgpu::Texture,
    curve_table_view: wgpu::TextureView,
    curves_uploaded: Option<ToneCurves>,
    mask_curves_uploaded: [Option<(ToneCurves, ToneCurves)>; MAX_MASKS],
    /// How many mask alphas have been drawn, for the cache test.
    alpha_builds: u64,
    /// How many brush layers have been drawn whole, and how many times new
    /// dabs were stamped onto a layer, for the cache test.
    layer_builds: u64,
    layer_appends: u64,
    /// How many times an alpha, and how many times the developed picture,
    /// was drawn again over the reach of new dabs only.
    alpha_patches: u64,
    develop_patches: u64,
    output_uniform: wgpu::Buffer,
    /// The mask of the list shown as a red overlay, when one is.
    overlay: Option<usize>,
    /// One zero texel: the overlay alpha of an output that shows none.
    no_overlay: Target,
    sampler: wgpu::Sampler,
    readback: Readback,
    source: Option<Source>,
    frame: Option<Frame>,
    /// A frame built ahead of a pan for the window the pan will reach, kept
    /// beside `frame` and swapped in when a zoomed view asks for its window.
    next: Option<NextFrame>,
    /// How many times a render built its frame and ran the head passes in
    /// its own submit, and how many times it took a frame built ahead.
    window_replaces: u64,
    window_swaps: u64,
    /// The texels one slice of a frame built ahead records at most, and how
    /// many slices have been submitted.
    ahead_slice_texels: u64,
    ahead_slices: u64,
    out: Option<Output>,
    generation: u64,
    frame_generation: u64,
    out_generation: u64,
}

impl Develop {
    /// Builds the passes on the shared device.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let input = make_pass(
            device,
            "input transform",
            include_str!("shaders/input_transform.wgsl"),
            WORKING_FORMAT,
            &[uniform_entry(0), texture_entry(1), sampler_entry(2)],
        );
        let video = make_pass(
            device,
            "video",
            include_str!("shaders/yuv_to_working.wgsl"),
            WORKING_FORMAT,
            &[
                uniform_entry(0),
                texture_entry(1),
                texture_entry(2),
                sampler_entry(3),
            ],
        );
        let blur = make_pass(
            device,
            "blur",
            include_str!("shaders/blur.wgsl"),
            BASE_FORMAT,
            &[uniform_entry(0), texture_entry(1), sampler_entry(2)],
        );
        let minimum = make_pass(
            device,
            "minimum",
            include_str!("shaders/minimum.wgsl"),
            BASE_FORMAT,
            &[uniform_entry(0), texture_entry(1)],
        );
        let develop = make_pass(
            device,
            "develop",
            include_str!("shaders/develop.wgsl"),
            WORKING_FORMAT,
            &[
                uniform_entry(0),
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                table_entry(5),
            ],
        );
        let mask = make_pass(
            device,
            "mask",
            include_str!("shaders/mask.wgsl"),
            ALPHA_FORMAT,
            &[uniform_entry(0), texture_entry(1), layers_entry(2)],
        );
        let masked = make_pass_with(
            device,
            "masked develop",
            include_str!("shaders/develop.wgsl"),
            WORKING_FORMAT,
            &[
                uniform_entry(0),
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                table_entry(5),
                texture_entry(6),
                texture_entry(7),
            ],
            "fs_masked",
            wgpu::BlendState::REPLACE,
        );
        let curve_table = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tone curve table"),
            size: wgpu::Extent3d {
                width: TABLE_SIZE as u32,
                height: TABLE_ROWS,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TABLE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let curve_table_view = curve_table.create_view(&wgpu::TextureViewDescriptor::default());
        let output = make_pass(
            device,
            "output",
            include_str!("shaders/output.wgsl"),
            OUTPUT_FORMAT,
            &[
                uniform_entry(0),
                texture_entry(1),
                sampler_entry(2),
                texture_entry(3),
            ],
        );
        let uniform = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("develop sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            device: device.clone(),
            queue: queue.clone(),
            input,
            video,
            blur,
            minimum,
            develop,
            mask,
            brush: BrushPass::new(device),
            reduce: make_pass(
                device,
                "proxy",
                include_str!("shaders/proxy.wgsl"),
                WORKING_FORMAT,
                &[uniform_entry(0), texture_entry(1)],
            ),
            refine: RefinePass::new(device, ALPHA_FORMAT),
            edge: EdgePass::new(device),
            refine_builds: 0,
            refine_patches: 0,
            edge_builds: [0; 3],
            edge_patches: [0; 3],
            shape_redrawn: [ShapeRedrawn::default(); MAX_MASKS],
            proxy: None,
            proxy_ids: 0,
            proxy_builds: 0,
            masked,
            output,
            input_uniform: uniform("input uniform", size_of::<InputUniform>() as u64),
            video_uniform: uniform("video uniform", size_of::<VideoUniform>() as u64),
            blur_h_uniform: uniform("blur h uniform", size_of::<BlurUniform>() as u64),
            blur_v_uniform: uniform("blur v uniform", size_of::<BlurUniform>() as u64),
            texture_h_uniform: uniform("texture h uniform", size_of::<BlurUniform>() as u64),
            texture_v_uniform: uniform("texture v uniform", size_of::<BlurUniform>() as u64),
            minimum_h_uniform: uniform("minimum h uniform", size_of::<MinimumUniform>() as u64),
            minimum_v_uniform: uniform("minimum v uniform", size_of::<MinimumUniform>() as u64),
            smooth_h_uniform: uniform("smooth h uniform", size_of::<BlurUniform>() as u64),
            smooth_v_uniform: uniform("smooth v uniform", size_of::<BlurUniform>() as u64),
            develop_uniform: uniform("develop uniform", size_of::<DevelopUniform>() as u64),
            mask_uniforms: (0..MAX_MASKS)
                .map(|_| uniform("mask uniform", size_of::<MaskUniform>() as u64))
                .collect(),
            masked_uniforms: (0..MAX_MASKS)
                .map(|_| uniform("masked develop uniform", size_of::<DevelopUniform>() as u64))
                .collect(),
            curve_table,
            curve_table_view,
            curves_uploaded: None,
            mask_curves_uploaded: Default::default(),
            alpha_builds: 0,
            layer_builds: 0,
            layer_appends: 0,
            alpha_patches: 0,
            develop_patches: 0,
            output_uniform: uniform("output uniform", size_of::<OutputUniform>() as u64),
            overlay: None,
            no_overlay: zero_texel(device, queue),
            sampler,
            readback: Readback::new(device),
            source: None,
            frame: None,
            next: None,
            window_replaces: 0,
            window_swaps: 0,
            ahead_slice_texels: AHEAD_SLICE_TEXELS,
            ahead_slices: 0,
            out: None,
            generation: 0,
            frame_generation: 0,
            out_generation: 0,
        }
    }

    /// Uploads a photo as the source. A photo wider or taller than the
    /// device allows is refused with a log line and the old source stays.
    pub fn set_source(&mut self, photo: &Photo) {
        let limit = self.device.limits().max_texture_dimension_2d;
        if photo.width > limit || photo.height > limit {
            log::error!(
                "photo of {}x{} exceeds the {limit} pixel texture limit",
                photo.width,
                photo.height
            );
            return;
        }
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let size = wgpu::Extent3d {
            width: photo.width,
            height: photo.height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("develop source"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            &photo.rgba8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(photo.width * 4),
                rows_per_image: Some(photo.height),
            },
            size,
        );
        self.generation += 1;
        self.source = Some(Source {
            kind: SourceKind::Photo {
                view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
                space: photo.source,
            },
            width: photo.width,
            height: photo.height,
            generation: self.generation,
            content: self.generation,
            atmosphere: dehaze::atmosphere_rgba8(
                &photo.rgba8,
                photo.width,
                photo.height,
                photo.source,
            ),
        });
        self.frame = None;
        self.next = None;
        log::info!(
            "develop source set: {}x{} {:?} as {format:?}",
            photo.width,
            photo.height,
            photo.source
        );
    }

    /// Uploads a decoded video frame as the source. The plane textures are
    /// kept between frames of one size and format; a frame of another
    /// size, format, colour or rotation replaces them.
    pub fn set_video_frame(&mut self, frame: &dyn FramePlanes, colour: VideoColour, rotation: u32) {
        let reuse = matches!(
            &self.source,
            Some(Source {
                kind: SourceKind::Video(planes),
                ..
            }) if planes.accepts(frame) && planes.colour == colour && planes.rotation == rotation
        );
        if !reuse {
            let planes = VideoSource::new(
                &self.device,
                frame.width(),
                frame.height(),
                frame.format(),
                colour,
                rotation,
            );
            let (width, height) = planes.display_size();
            self.generation += 1;
            self.source = Some(Source {
                kind: SourceKind::Video(Box::new(planes)),
                width,
                height,
                generation: self.generation,
                content: 0,
                // A frame's own atmospheric light waits for per-clip
                // analysis; until then haze is measured against white.
                atmosphere: dehaze::WHITE_ATMOSPHERE,
            });
            self.frame = None;
            self.next = None;
            log::info!(
                "develop video source set: {}x{} {:?} rotation {rotation} {colour:?}",
                frame.width(),
                frame.height(),
                frame.format()
            );
        }
        let source = self.source.as_mut().expect("set above");
        let SourceKind::Video(planes) = &source.kind else {
            unreachable!("the source is the video planes");
        };
        planes.upload(&self.queue, frame);
        source.content += 1;
    }

    /// Whether the source is a video frame.
    pub fn has_video(&self) -> bool {
        matches!(
            &self.source,
            Some(Source {
                kind: SourceKind::Video(_),
                ..
            })
        )
    }

    /// Shows the mask at this index of the edit's list as a red overlay on
    /// every render from now on, or none. The overlay is for the window and
    /// for a screenshot that asks for it: [`render_export`](Self::render_export)
    /// never draws it.
    pub fn set_overlay(&mut self, mask: Option<usize>) {
        self.overlay = mask;
    }

    /// The mask shown as an overlay, when one is.
    pub fn overlay(&self) -> Option<usize> {
        self.overlay
    }

    /// How many mask alphas this graph has drawn since it was built. A test
    /// reads it to hold that a slider of a mask never redraws its alpha.
    #[doc(hidden)]
    pub fn mask_alpha_builds(&self) -> u64 {
        self.alpha_builds
    }

    /// The scratch the last refine of this graph drew with: the cells a side
    /// it holds, the side its tiles were cut at, whether it was made for
    /// that refine, and the branch of the rule that gave it. `None` before
    /// any refine. A test reads it to hold two masks at different steps
    /// inside the budget.
    #[doc(hidden)]
    pub fn refine_last_hold(&self) -> Option<RefineHold> {
        self.refine.last_hold.map(|hold| RefineHold {
            size: hold.size,
            side: hold.side,
            made: hold.made,
            branch: hold.branch.name(),
        })
    }

    /// The refined alpha of mask `index` on the frame of the last render,
    /// and the size of that frame. `None` while the mask has Refine edges
    /// off. A test reads its bytes to hold a refine beside another mask to
    /// the same refine alone.
    #[doc(hidden)]
    pub fn refined_alpha(&self, index: usize) -> Option<(&wgpu::TextureView, (u32, u32))> {
        let frame = self.frame.as_ref()?;
        let slot = frame.masks.get(index)?.as_ref()?;
        let refined = slot.refined.as_ref()?;
        Some((&refined.alpha, (frame.width, frame.height)))
    }

    /// How many times a render replaced its frame and ran the head passes in
    /// the same submit: the first render of a window, a jump, a zoom change,
    /// or a pan that left the window with no frame built ahead for it. A
    /// test reads it to hold that a pan onto a window built ahead pays none.
    #[doc(hidden)]
    pub fn window_replaces(&self) -> u64 {
        self.window_replaces
    }

    /// How many times a render took the frame built ahead instead of
    /// replacing its own.
    #[doc(hidden)]
    pub fn window_swaps(&self) -> u64 {
        self.window_swaps
    }

    /// The picture size and the padded window of the frame built ahead, when
    /// one is held, complete or still building. A zoomed viewer asks for
    /// this window once what is seen leaves the one it renders.
    pub fn window_ahead(&self) -> Option<((u32, u32), PixelRect)> {
        self.next.as_ref().map(|next| (next.full, next.window))
    }

    /// Whether a frame built ahead is held with slices left to record. The
    /// viewer calls [`build_ahead`](Self::build_ahead) once a UI frame while
    /// it is.
    pub fn ahead_building(&self) -> bool {
        self.next
            .as_ref()
            .is_some_and(|next| !next.build.complete())
    }

    /// How many slices of frames built ahead have been submitted.
    #[doc(hidden)]
    pub fn ahead_slices(&self) -> u64 {
        self.ahead_slices
    }

    /// Sets how many texels one slice of a frame built ahead records at most,
    /// [`AHEAD_SLICE_TEXELS`] until set. A slice records one row of a pass at
    /// least, so 1 records one row a slice and `u64::MAX` the whole frame in
    /// one. A test sets it; the pixels drawn do not depend on it.
    #[doc(hidden)]
    pub fn set_ahead_slice_texels(&mut self, texels: u64) {
        self.ahead_slice_texels = texels.max(1);
    }

    /// The seven textures of the current frame, or of the frame built ahead,
    /// by name, with their size, for a test to read back and compare.
    #[doc(hidden)]
    pub fn frame_textures(
        &self,
        ahead: bool,
    ) -> Vec<(&'static str, &wgpu::TextureView, (u32, u32))> {
        let frame = if ahead {
            self.next.as_ref().map(|next| &next.frame)
        } else {
            self.frame.as_ref()
        };
        let Some(frame) = frame else {
            return Vec::new();
        };
        let size = (frame.width, frame.height);
        [
            ("working", &frame.working),
            ("ping", &frame.ping),
            ("base", &frame.base),
            ("texture base", &frame.texture_base),
            ("transmission", &frame.transmission),
            ("developed", &frame.developed),
            ("developed other", &frame.developed_other),
        ]
        .into_iter()
        .map(|(name, target)| (name, &target.view, size))
        .collect()
    }

    /// How many brush layers have been stamped whole since the graph was
    /// built. A layer is kept until its strokes or the window change.
    pub fn brush_layer_builds(&self) -> u64 {
        self.layer_builds
    }

    /// How many times new dabs were stamped onto a layer that was kept.
    pub fn brush_layer_appends(&self) -> u64 {
        self.layer_appends
    }

    /// How many of the mask alphas drawn were drawn over the reach of new
    /// dabs only, and how many times the developed picture was.
    pub fn brush_patches(&self) -> (u64, u64) {
        (self.alpha_patches, self.develop_patches)
    }

    /// How many times a refined alpha was drawn whole, and how many times
    /// over the reach of new dabs only.
    pub fn refine_builds(&self) -> (u64, u64) {
        (self.refine_builds, self.refine_patches)
    }

    /// How many times Refine edges took the moments of the source over the
    /// whole work of a tile, in tiles. They are kept while the source, the
    /// side of a cell, the grid and the tile stay the same: a Refine slider,
    /// a new stroke, another mask and a Radius step that keeps the side of a
    /// cell take none.
    pub fn refine_source_builds(&self) -> u64 {
        self.refine.source_builds
    }

    /// How many strips Refine edges took the moments of the source over: the
    /// cells a refine works over that the moments held do not cover, at most
    /// four rectangles a tile.
    pub fn refine_source_strips(&self) -> u64 {
        self.refine.source_strips
    }

    /// How many passes of Refine edges this graph has drawn. A develop slider
    /// draws none.
    pub fn refine_passes(&self) -> u32 {
        self.refine.refine_passes
    }

    /// Whether Refine edges draws the moments of the source in one pass of
    /// three targets and its solve in one pass of four, on a device that
    /// draws into 64 bytes a sample, rather than each in two passes: the
    /// source and each gather then draw one pass fewer.
    pub fn refine_fused(&self) -> bool {
        self.refine.fused()
    }

    /// How many tiles of Refine edges this graph has drawn. A grid of cells
    /// the scratch budget holds is one tile.
    pub fn refine_tiles(&self) -> u32 {
        self.refine.refine_tiles
    }

    /// The same graph with a Refine edges scratch of at most `bytes`, so a
    /// test can draw a refine in more tiles than the default budget gives.
    #[doc(hidden)]
    pub fn with_refine_scratch_budget(mut self, bytes: u64) -> Self {
        self.refine = self.refine.with_scratch_budget(bytes);
        self
    }

    /// The same graph with every box of Refine edges summed in the direct
    /// loop when `on`, and no block pass, so a test can hold the block sums
    /// to it.
    #[doc(hidden)]
    pub fn with_refine_direct_box(mut self, on: bool) -> Self {
        self.refine = self.refine.with_direct_box(on);
        self
    }

    /// The same graph with the work textures of Shift edge held to at most
    /// `side` pixels a side, below the device's limit, so a test can cut
    /// their pad short of the runs' half the way a photo near the limit does
    /// and hold the combine pass's loop over the samples to the twin.
    #[doc(hidden)]
    pub fn with_edge_pad_limit(mut self, side: u32) -> Self {
        self.edge = self.edge.with_pad_limit(side);
        self
    }

    /// How many pixels the work textures of Shift edge hold past the frame
    /// on each side after the last render, or `None` while none are held.
    #[doc(hidden)]
    pub fn edge_pad(&self) -> Option<u32> {
        self.frame
            .as_ref()
            .and_then(|frame| frame.edge_scratch.pad())
    }

    /// How many passes of Shift edge, of Feather's cells and of the finished
    /// alpha this graph has drawn. A develop slider draws none of them.
    pub fn edge_passes(&self) -> EdgePasses {
        self.edge.passes
    }

    /// How many times Shift edge, Feather's cells and the finished alpha
    /// were drawn over their whole frame, and how many times over the reach
    /// of new dabs only. A test reads it to hold that a stroke appended into
    /// an edged mask draws each stage over its own reach.
    #[doc(hidden)]
    pub fn edge_builds(&self) -> ([u64; 3], [u64; 3]) {
        (self.edge_builds, self.edge_patches)
    }

    /// Where the last render drew each stage of the shape of mask `index`
    /// again. Nothing for a mask that render did not draw.
    #[doc(hidden)]
    pub fn shape_redrawn(&self, index: usize) -> ShapeRedrawn {
        self.shape_redrawn.get(index).copied().unwrap_or_default()
    }

    /// How many textures the edge controls hold on the frame: those of the
    /// masks and those the masks share. None while every mask has the three
    /// at rest.
    pub fn edge_textures(&self) -> usize {
        self.frame.as_ref().map_or(0, |frame| {
            frame.edge_scratch.held()
                + frame
                    .masks
                    .iter()
                    .flatten()
                    .filter_map(|slot| slot.edged.as_ref())
                    .map(Edged::held_textures)
                    .sum::<usize>()
        })
    }

    /// The alpha the edge controls of mask `index` hand to the blend on the
    /// frame of the last render, and the size of that frame: the shifted
    /// alpha while Shift edge alone is on. `None` while the three are at
    /// rest. A test reads its bytes to hold a schedule to the one before it.
    #[doc(hidden)]
    pub fn edge_product(&self, index: usize) -> Option<(&wgpu::TextureView, (u32, u32))> {
        let frame = self.frame.as_ref()?;
        let slot = frame.masks.get(index)?.as_ref()?;
        let view = slot.edged.as_ref()?.product()?;
        Some((view, (frame.width, frame.height)))
    }

    /// How far Refine edges and the edge controls read around a pixel at a
    /// render of `full`: the widest reach among the masks this edit draws,
    /// each Refine edges' and its edge controls' together, and the margin of
    /// the products with it. Above the reach of the head passes it goes up in
    /// steps, so a Radius or an edge slider replaces the window of a zoomed
    /// viewer a few times over its travel and not at every step.
    fn refine_reach(&self, edit: &PhotoEdit, full: (u32, u32)) -> u32 {
        const STEP: u32 = 64;
        let shown = self.overlay.and_then(|index| edit.masks.get(index));
        let reach = mask_twin::active_masks(edit)
            .iter()
            .map(|(_, mask)| mask)
            .chain(shown)
            .map(|mask| {
                refine_twin::reach(&mask.refine.shape(), full) + edge_twin::reach(&mask.edge, full)
            })
            .max()
            .unwrap_or(0);
        if reach == 0 {
            0
        } else {
            (reach + PRODUCTS_MARGIN).div_ceil(STEP) * STEP
        }
    }

    /// How many times the reference proxy of an auto brush was built. It is
    /// built once a source content and by no window and no slider.
    pub fn proxy_builds(&self) -> u64 {
        self.proxy_builds
    }

    /// The size of the source photo, when one is set.
    pub fn source_size(&self) -> Option<(u32, u32)> {
        self.source.as_ref().map(|s| (s.width, s.height))
    }

    /// Renders the whole photo at `render_size`, then writes the part of it
    /// under `crop` into an output of `output_size` pixels. Returns the
    /// output view, or `None` when no source is set. The work is submitted
    /// but not waited for.
    pub fn render(
        &mut self,
        edit: &PhotoEdit,
        crop: CropRect,
        render_size: (u32, u32),
        output_size: (u32, u32),
    ) -> Option<&wgpu::TextureView> {
        // Only a zoomed view pans onto a frame built ahead.
        self.next = None;
        self.render_window(
            edit,
            crop,
            render_size,
            output_size,
            CropRect::FULL,
            render_size,
            crop,
        )
    }

    /// Renders only what `crop` needs at the scale where it lands on
    /// `output_size` one to one: the crop plus the reach of the base blur
    /// on every side, so the picture inside the crop is the one
    /// [`render`](Self::render) would give at the full size. The rest of
    /// the frame is never drawn, which is what makes a Reel frame cheap.
    pub fn render_crop(
        &mut self,
        edit: &PhotoEdit,
        crop: CropRect,
        output_size: (u32, u32),
    ) -> Option<&wgpu::TextureView> {
        // Only a zoomed view pans onto a frame built ahead.
        self.next = None;
        let full = render_size_for_crop(crop, output_size);
        let (full_w, full_h) = (full.0 as f32, full.1 as f32);
        let reach = (basic::blur_radius(basic::base_sigma(full.0, full.1)).max(0) as u32)
            .max(self.refine_reach(edit, full)) as f32;
        let x0 = (crop.x * full_w - reach).floor().max(0.0);
        let y0 = (crop.y * full_h - reach).floor().max(0.0);
        let x1 = ((crop.x + crop.width) * full_w + reach).ceil().min(full_w);
        let y1 = ((crop.y + crop.height) * full_h + reach).ceil().min(full_h);
        let window = CropRect {
            x: x0 / full_w,
            y: y0 / full_h,
            width: (x1 - x0) / full_w,
            height: (y1 - y0) / full_h,
        };
        let inner = CropRect {
            x: (crop.x - window.x) / window.width,
            y: (crop.y - window.y) / window.height,
            width: crop.width / window.width,
            height: crop.height / window.height,
        };
        let window_size = ((x1 - x0).round() as u32, (y1 - y0).round() as u32);
        self.render_window(edit, inner, window_size, output_size, window, full, inner)
    }

    /// Renders what a zoomed viewer shows. The head-pass products (the base
    /// blur, the texture layer, the transmission map, the mask alphas) cover
    /// the whole of `view.window`, a padded rectangle around what is seen;
    /// the develop passes cover `view.visible` and the output takes it out
    /// one to one. A pan that stays inside the window therefore runs no
    /// head pass, and a slider step develops only what is seen.
    /// The frame holds the window plus the reach of the widest head pass on
    /// every side, or of Refine edges when a mask's reaches further, so the
    /// picture is the one [`render`](Self::render) gives at `view.full`.
    ///
    /// `view.full` must not pass the size of the source: above one source
    /// pixel per output pixel the blur radius meets its cap and the look
    /// would change with the zoom. A viewer magnifies the 100 percent render
    /// instead.
    pub fn render_view(
        &mut self,
        edit: &PhotoEdit,
        view: &ViewWindow,
    ) -> Option<&wgpu::TextureView> {
        let full = (view.full.0.max(1), view.full.1.max(1));
        // A frame built ahead at another zoom serves no view of this one.
        if self.next.as_ref().is_some_and(|next| next.full != full) {
            self.next = None;
        }
        // The view's window is the frame built ahead when the pan reached
        // it, and render_window swaps that frame in.
        let geometry = self.view_geometry(edit, view);
        self.render_window(
            edit,
            geometry.crop,
            geometry.render_size,
            geometry.output_size,
            geometry.window,
            geometry.sigma_size,
            geometry.products,
        )
    }

    /// Builds the frame of `window`, a padded window of a zoomed view of the
    /// picture at `full`, ahead of the pan that will reach it, one slice a
    /// call. The first call makes the frame; each call records at most one
    /// slice of the head passes left, [`AHEAD_SLICE_TEXELS`] of work, into
    /// an encoder of its own and submits it before it returns. The current
    /// frame keeps serving every render meanwhile, and once every slice is
    /// submitted [`render_view`](Self::render_view) takes this frame when it
    /// asks for that window, instead of replacing its own. A frame held for
    /// the same window is kept and built on; one held for another is
    /// dropped.
    ///
    /// Whether this call completed the frame: true from the call that
    /// submits its last slice, false while slices are left and from a call
    /// that finds it complete already or `window` rendered by the current
    /// frame. [`ahead_building`](Self::ahead_building) tells whether slices
    /// are left.
    ///
    /// The frame holds the head passes of the source content it was built
    /// from. A video frame that arrives before the swap is drawn into it
    /// at the swap, as into the current frame at every new video frame.
    pub fn build_ahead(&mut self, edit: &PhotoEdit, full: (u32, u32), window: PixelRect) -> bool {
        let full = (full.0.max(1), full.1.max(1));
        let view = ViewWindow {
            full,
            window,
            visible: window,
        };
        let geometry = self.view_geometry(edit, &view);
        let Some(source) = self.source.as_ref() else {
            return false;
        };
        let (width, height) = (geometry.render_size.0.max(1), geometry.render_size.1.max(1));
        let same = |f: &Frame| {
            (f.width, f.height, f.generation) == (width, height, source.generation)
                && f.window == geometry.window
                && f.sigma_size == geometry.sigma_size
        };
        if self.frame.as_ref().is_some_and(same) {
            return false;
        }
        if !self.next.as_ref().is_some_and(|next| same(&next.frame)) {
            // The frame held for another window goes first, so no more than
            // two frames are held at once.
            self.next = None;
            let frame =
                self.build_frame(source, width, height, geometry.window, geometry.sigma_size);
            let (texture, transmission) =
                head_products_wanted(edit, &mask_twin::active_masks(edit));
            let passes = HeadPass::plan(frame.content != source.content, texture, transmission);
            self.next = Some(NextFrame {
                full,
                window,
                frame,
                build: AheadBuild {
                    work: HeadWork {
                        size: (width, height),
                        window: geometry.window,
                        sigma_size: geometry.sigma_size,
                        products: geometry.products,
                        texture,
                        transmission,
                    },
                    passes,
                    pass: 0,
                    row: 0,
                    content: None,
                },
            });
        }
        self.ahead_slice()
    }

    /// Records the next slice of the frame built ahead and submits it:
    /// strips of whole rows of its passes, in their order, until the next
    /// strip would pass the slice's texels. The first strip of a pass clears
    /// its target, as the pass does when it runs whole, and the strips after
    /// it load it, so every pixel of the target ends as the whole pass leaves
    /// it. A pass without a scissor is cut into strips of the whole width.
    /// Whether this slice completed the frame.
    ///
    /// The uniforms of the head passes are Develop's, shared with the
    /// current frame. A slice writes every one its strips read and submits
    /// before it returns, so a render's writes, which land at the render's
    /// own submit, never reach its commands.
    fn ahead_slice(&mut self) -> bool {
        let Some(mut next) = self.next.take() else {
            return false;
        };
        if next.build.complete() {
            self.next = Some(next);
            return false;
        }
        let source = self.source.as_ref().expect("a frame ahead has a source");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("develop ahead encoder"),
            });
        let build = &mut next.build;
        let mut budget = self.ahead_slice_texels;
        let mut strips = 0u32;
        let mut finished = Vec::new();
        while let Some(&pass) = build.passes.get(build.pass) {
            let (x, y, w, h) = pass.rect(&build.work);
            let per_row = u64::from(w) * self.head_pass_texels(pass, &build.work);
            let left = h - build.row;
            let fits = (budget / per_row).min(u64::from(left)) as u32;
            // A slice records one row at least, so a budget under one row
            // still moves the build.
            let rows = if strips == 0 { fits.max(1) } else { fits };
            if rows == 0 {
                break;
            }
            if pass == HeadPass::Input {
                build.content = match build.content {
                    _ if build.row == 0 => Some(source.content),
                    Some(content) if content == source.content => Some(content),
                    _ => None,
                };
            }
            let load = if build.row == 0 {
                wgpu::LoadOp::Clear(wgpu::Color::BLACK)
            } else {
                wgpu::LoadOp::Load
            };
            self.write_head_uniform(pass, &build.work);
            self.draw_head_pass(
                &mut encoder,
                &next.frame,
                pass,
                Some((x, y + build.row, w, rows)),
                load,
            );
            strips += 1;
            budget = budget.saturating_sub(u64::from(rows) * per_row);
            build.row += rows;
            if build.row == h {
                finished.push(pass);
                build.pass += 1;
                build.row = 0;
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.ahead_slices += 1;
        // What the frame holds is known only now that its commands are
        // submitted.
        let frame = &mut next.frame;
        for pass in finished {
            match pass {
                HeadPass::Input => {
                    if let Some(content) = build.content {
                        frame.content = content;
                    }
                }
                HeadPass::BlurV => {
                    frame.products_region =
                        Some(scissor_for(build.work.products, build.work.size, false));
                }
                HeadPass::TextureV => frame.texture_ready = true,
                HeadPass::TransmissionV => frame.transmission_ready = true,
                _ => {}
            }
        }
        let complete = build.complete();
        self.next = Some(next);
        complete
    }

    /// What `render_window` renders for a zoomed view: the frame holds the
    /// window plus the reach of the widest head pass on every side, or of
    /// Refine edges when a mask's reaches further.
    fn view_geometry(&self, edit: &PhotoEdit, view: &ViewWindow) -> ViewGeometry {
        let full = (view.full.0.max(1), view.full.1.max(1));
        let (wx, wy, ww, wh) = clamp_rect(view.window, (0, 0, full.0, full.1));
        let (vx, vy, vw, vh) = clamp_rect(view.visible, (wx, wy, ww, wh));
        let reach = head_pass_reach(full).max(self.refine_reach(edit, full));
        let (x0, y0) = (wx.saturating_sub(reach), wy.saturating_sub(reach));
        let (x1, y1) = ((wx + ww + reach).min(full.0), (wy + wh + reach).min(full.1));
        let frame = ((x1 - x0) as f32, (y1 - y0) as f32);
        let window = CropRect {
            x: x0 as f32 / full.0 as f32,
            y: y0 as f32 / full.1 as f32,
            width: frame.0 / full.0 as f32,
            height: frame.1 / full.1 as f32,
        };
        let within = |(x, y, w, h): (u32, u32, u32, u32)| CropRect {
            x: (x - x0) as f32 / frame.0,
            y: (y - y0) as f32 / frame.1,
            width: w as f32 / frame.0,
            height: h as f32 / frame.1,
        };
        ViewGeometry {
            crop: within((vx, vy, vw, vh)),
            render_size: (x1 - x0, y1 - y0),
            output_size: (vw, vh),
            window,
            sigma_size: full,
            products: within((wx, wy, ww, wh)),
        }
    }

    /// The shared body: renders `window` of the source at `render_size`
    /// with the blur sigma of `sigma_size`, draws the head-pass products
    /// over `products` of that render and the develop passes over `crop`,
    /// then crops `crop` of it into the output. `crop` lies inside
    /// `products`.
    #[allow(clippy::too_many_arguments)]
    fn render_window(
        &mut self,
        edit: &PhotoEdit,
        crop: CropRect,
        render_size: (u32, u32),
        output_size: (u32, u32),
        window: CropRect,
        sigma_size: (u32, u32),
        products: CropRect,
    ) -> Option<&wgpu::TextureView> {
        let source = self.source.as_ref()?;
        let (width, height) = (render_size.0.max(1), render_size.1.max(1));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("develop encoder"),
            });

        let stale = self.frame.as_ref().is_none_or(|f| {
            (f.width, f.height, f.generation) != (width, height, source.generation)
                || f.window != window
                || f.sigma_size != sigma_size
        });
        if stale {
            // A frame built ahead for this window is swapped in once every
            // slice of it is submitted: its head passes ran in submits of
            // their own. Without one the frame is replaced and its head
            // passes run in this submit; one still building for this window
            // is dropped, as the frame replaced here serves the window.
            let ahead = self.next.take_if(|next| {
                let f = &next.frame;
                (f.width, f.height, f.generation) == (width, height, source.generation)
                    && f.window == window
                    && f.sigma_size == sigma_size
            });
            match ahead {
                Some(next) if next.build.complete() => {
                    self.frame = Some(next.frame);
                    self.window_swaps += 1;
                }
                building => {
                    drop(building);
                    self.frame = Some(self.build_frame(source, width, height, window, sigma_size));
                    self.window_replaces += 1;
                }
            }
            self.frame_generation += 1;
        }
        // The masks that change the picture, and the head-pass products they
        // and the global edit want.
        let masks = mask_twin::active_masks(edit);
        // The overlay shows a mask whether or not it adjusts anything yet,
        // so its alpha is drawn even when the mask itself is not.
        let shown = self.overlay.filter(|index| *index < MAX_MASKS);
        let overlaid: Option<(usize, Mask)> = shown.and_then(|index| {
            let mask = edit.masks.get(index)?.sanitised();
            Some((index, mask))
        });
        let (texture, transmission) = head_products_wanted(edit, &masks);
        let work = HeadWork {
            size: (width, height),
            window,
            sigma_size,
            products,
            texture,
            transmission,
        };
        let mut frame = self.frame.take().expect("frame built above");
        self.head_passes(&mut encoder, &mut frame, &work);
        self.frame = Some(frame);
        let frame = self.frame.as_mut().expect("frame built above");
        let region = scissor_for(products, (width, height), false);
        // The develop passes cover only what the output shows. For a zoomed
        // viewer that is far less than the products cover, so a slider step
        // costs what it costs on a fitted picture, and a pan inside the
        // window develops what came into view and runs no head pass.
        let seen = scissor_for(crop, (width, height), false);
        let curves = &edit.look.curves;
        if !curves.is_identity() && self.curves_uploaded.as_ref() != Some(curves) {
            write_table_row(&self.queue, &self.curve_table, 0, &curve::bake(curves));
            self.curves_uploaded = Some(curves.clone());
        }
        // The alpha of each mask, drawn once and kept until its shape, the
        // head passes or the scissor change; its row of the curve table,
        // uploaded only when the composed curves change; its uniform.
        let geometry = Geometry {
            window,
            size: (width, height),
            photo: (source.width, source.height),
        };
        // The proxy an auto stroke reads its references from: set up when an
        // edit first holds such a stroke, built once a source content.
        let wants_proxy = masks
            .iter()
            .map(|(_, mask)| mask)
            .chain(overlaid.iter().map(|(_, mask)| mask))
            .flat_map(|mask| &mask.components)
            .any(|component| matches!(&component.source, MaskSource::Brush(b) if has_auto(b)));
        if wants_proxy {
            let built = build_proxy(
                &self.device,
                &self.queue,
                &mut encoder,
                ProxyPasses {
                    input: &self.input,
                    video: &self.video,
                    reduce: &self.reduce,
                    sampler: &self.sampler,
                },
                source,
                &mut self.proxy,
                &mut self.proxy_ids,
            );
            self.proxy_builds += u64::from(built);
        } else if self
            .proxy
            .as_ref()
            .is_some_and(|proxy| proxy.generation != source.generation)
        {
            self.proxy = None;
        }
        // What the masks changed of the picture in this render. A brush that
        // only grew changes what its new dabs reach and nothing else.
        let mut damage = Damage::Nothing;
        self.shape_redrawn = [ShapeRedrawn::default(); MAX_MASKS];
        let only_shown = overlaid
            .iter()
            .filter(|(index, _)| masks.iter().all(|(active, _)| active != index));
        for (index, mask, develops) in masks
            .iter()
            .map(|(index, mask)| (*index, mask, true))
            .chain(only_shown.map(|(index, mask)| (*index, mask, false)))
        {
            let brushes: Vec<&Brush> = mask
                .components
                .iter()
                .filter_map(|component| match &component.source {
                    MaskSource::Brush(brush) => Some(brush),
                    _ => None,
                })
                .collect();
            let raster_bind = |layers: &Layers| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mask bind group"),
                    layout: &self.mask.layout,
                    entries: &[
                        buffer_binding(0, &self.mask_uniforms[index]),
                        texture_binding(1, &frame.working.view),
                        texture_binding(2, &layers.array),
                    ],
                })
            };
            // The mask pass reads this alpha: the mask's own, or its refined
            // one.
            let develop_binds = |alpha: &wgpu::TextureView| {
                [&frame.developed, &frame.developed_other].map(|before| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("masked develop bind group"),
                        layout: &self.masked.layout,
                        entries: &[
                            buffer_binding(0, &self.masked_uniforms[index]),
                            texture_binding(1, &frame.working.view),
                            texture_binding(2, &frame.base.view),
                            texture_binding(3, &frame.texture_base.view),
                            texture_binding(4, &frame.transmission.view),
                            texture_binding(5, &self.curve_table_view),
                            texture_binding(6, alpha),
                            texture_binding(7, &before.view),
                        ],
                    })
                })
            };
            let slot = frame.masks[index].get_or_insert_with(|| {
                let alpha = create_target(&self.device, "mask alpha", ALPHA_FORMAT, width, height);
                let layers = Layers::new(&self.device, brushes.len(), width, height);
                let raster_bind = raster_bind(&layers);
                let develop_binds = develop_binds(&alpha.view);
                FrameMask {
                    alpha,
                    shape: None,
                    alpha_region: (0, 0, 0, 0),
                    refined: None,
                    refine: None,
                    refined_region: (0, 0, 0, 0),
                    edged: None,
                    binds_product: Product::Alpha,
                    layers,
                    raster_bind,
                    develop_binds,
                }
            });
            // A layer for each brush of the mask. The layers are kept while
            // the frame is; each is stamped again only when its strokes are
            // not the ones it holds.
            if slot.layers.len() != brushes.len() {
                slot.layers = Layers::new(&self.device, brushes.len(), width, height);
                slot.raster_bind = raster_bind(&slot.layers);
                slot.shape = None;
            }
            let mut stamped = Damage::Nothing;
            for (layer, brush) in brushes.iter().enumerate() {
                let auto = self
                    .proxy
                    .as_ref()
                    .filter(|_| has_auto(brush))
                    .map(|proxy| AutoInputs {
                        proxy: &proxy.target.view,
                        working: &frame.working.view,
                        generation: proxy.id,
                    });
                let drawn = slot.layers.update(
                    layer,
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &mut self.brush,
                    brush,
                    &geometry,
                    auto,
                );
                match drawn {
                    Drawn::Nothing => {}
                    Drawn::Appended(reach) => {
                        self.layer_appends += 1;
                        stamped = stamped.with(reach);
                    }
                    Drawn::Whole => {
                        self.layer_builds += 1;
                        stamped = Damage::Whole;
                    }
                }
            }
            // Refine edges, then the edge controls, are the last steps of the
            // shape. The alpha of the components is kept under the shape
            // without them, so a Refine or an edge slider filters the alpha
            // again and never redraws it.
            let refine = mask.refine.shape();
            let origin = (
                (window.x * sigma_size.0 as f32).round().max(0.0) as u32,
                (window.y * sigma_size.1 as f32).round().max(0.0) as u32,
            );
            let plan =
                (!refine.is_off()).then(|| Plan::new(&refine, sigma_size, origin, (width, height)));
            let edge_plan = EdgePlan::new(&mask.edge, sigma_size, origin, (width, height));
            let edged_on = !edge_plan.is_off();
            let frame_rect = (0, 0, width, height);
            // The filter reads the alpha its reach beyond what it writes. A
            // mask with an edge control on keeps its alpha and its refined
            // alpha over the whole frame, which is padded by both reaches, so
            // an edge slider, whose reach moves, never draws them again.
            let alpha_region = if edged_on {
                frame_rect
            } else {
                plan.map_or(region, |plan| grow(region, plan.reach(), (width, height)))
            };
            let refined_region = if edged_on { frame_rect } else { region };
            let shape = MaskShape {
                refine: Refine::default(),
                edge: Edge::default(),
                ..mask.shape()
            };
            // What of the alpha this render drew again.
            let mut redrawn = Damage::Nothing;
            if slot.shape.as_ref() != Some(&shape) || !holds(slot.alpha_region, alpha_region) {
                self.queue.write_buffer(
                    &self.mask_uniforms[index],
                    0,
                    bytemuck::bytes_of(&MaskUniform::new(mask, &geometry)),
                );
                // A shape that differs from the one the alpha holds only by
                // strokes stamped onto its layers is drawn again where those
                // dabs reach; anything else is drawn again whole.
                let grew = stamped != Damage::Whole
                    && holds(slot.alpha_region, alpha_region)
                    && slot
                        .shape
                        .as_ref()
                        .is_some_and(|held| held.same_but_strokes(&shape));
                if grew {
                    let reach = match stamped {
                        Damage::Part(reach) => intersect(reach, alpha_region),
                        _ => None,
                    };
                    if let Some(reach) = reach {
                        draw_over(
                            &mut encoder,
                            "mask alpha",
                            &self.mask.pipeline,
                            &slot.raster_bind,
                            &slot.alpha.view,
                            Some(reach),
                        );
                        self.alpha_patches += 1;
                    }
                    self.shape_redrawn[index].alpha = reach;
                    redrawn = redrawn.with(reach);
                } else {
                    draw(
                        &mut encoder,
                        "mask alpha",
                        &self.mask.pipeline,
                        &slot.raster_bind,
                        &slot.alpha.view,
                        Some(alpha_region),
                    );
                    slot.alpha_region = alpha_region;
                    self.shape_redrawn[index].alpha = Some(alpha_region);
                    redrawn = Damage::Whole;
                }
                slot.shape = Some(shape);
                self.alpha_builds += 1;
            }
            // What of the alpha Refine edges hands on changed.
            let mut handed = Damage::Nothing;
            match plan {
                None => {
                    slot.refined = None;
                    slot.refine = None;
                    handed = redrawn;
                }
                Some(plan) => {
                    let refined = slot.refined.get_or_insert_with(|| {
                        self.refine
                            .refined(&self.device, ALPHA_FORMAT, width, height)
                    });
                    // New dabs change the refined alpha the filter's reach
                    // further out than they change the alpha, and no
                    // further.
                    let over = match redrawn {
                        _ if slot.refine != Some(refine) => Some(refined_region),
                        _ if !holds(slot.refined_region, refined_region) => Some(refined_region),
                        Damage::Whole => Some(refined_region),
                        Damage::Part(reach) => {
                            intersect(grow(reach, plan.reach(), (width, height)), refined_region)
                        }
                        Damage::Nothing => None,
                    };
                    self.shape_redrawn[index].refine = over;
                    if let Some(over) = over {
                        self.refine.run(
                            &self.device,
                            &self.queue,
                            &mut encoder,
                            &mut frame.refine_scratch,
                            refined,
                            &frame.working.view,
                            &slot.alpha.view,
                            &plan,
                            over,
                        );
                        if over == refined_region {
                            self.refine_builds += 1;
                            slot.refined_region = refined_region;
                            handed = Damage::Whole;
                        } else {
                            self.refine_patches += 1;
                            handed = Damage::Part(over);
                        }
                    }
                    slot.refine = Some(refine);
                }
            }
            // The edge controls, each over the reach of what changed before
            // it, or whole when its own setting changed.
            let changed = if edged_on {
                let refined_input = slot.refined.is_some();
                let edged = slot
                    .edged
                    .get_or_insert_with(|| self.edge.edged(&self.device));
                let made =
                    self.edge
                        .prepare(&self.device, &mut frame.edge_scratch, edged, &edge_plan);
                let held = edged.held.filter(|held| {
                    held.refined == refined_input
                        && (held.plan.full, held.plan.origin, held.plan.size)
                            == (edge_plan.full, edge_plan.origin, edge_plan.size)
                });
                let p = &edge_plan;
                // A product made again is drawn whole; one kept is drawn
                // again only where its setting or the stage before changed.
                let shift_changed = made.shifted
                    || held.is_none_or(|h| {
                        (h.plan.grow, h.plan.axis, h.plan.diagonal) != (p.grow, p.axis, p.diagonal)
                    });
                let feather_changed = made.cells
                    || held.is_none_or(|h| {
                        (h.plan.sigma, h.plan.step, h.plan.radius_cells)
                            != (p.sigma, p.step, p.radius_cells)
                    });
                let contrast_changed =
                    made.finished || held.is_none_or(|h| h.plan.contrast != p.contrast);
                let size = (width, height);
                let grown = |damage: Damage, by: u32| match damage {
                    Damage::Part(rect) => Damage::Part(grow(rect, by, size)),
                    other => other,
                };
                let shifted = if shift_changed {
                    Damage::Whole
                } else {
                    grown(handed, p.shift_reach())
                };
                let feathered = if feather_changed {
                    Damage::Whole
                } else {
                    grown(shifted, p.feather_reach())
                };
                let finished = if contrast_changed {
                    Damage::Whole
                } else {
                    feathered
                };
                let pixels = |damage: Damage, whole: PixelRect| match damage {
                    Damage::Whole => Some(whole),
                    Damage::Part(rect) => intersect(rect, whole),
                    Damage::Nothing => None,
                };
                let work = EdgeWork {
                    shift: pixels(shifted, frame_rect).filter(|_| p.shifts()),
                    feather: pixels(feathered, region).filter(|_| p.feathers()),
                    finish: pixels(finished, region).filter(|_| p.feathers() || p.contrasts()),
                };
                let stages = [
                    (work.shift, frame_rect),
                    (work.feather, region),
                    (work.finish, region),
                ];
                for (stage, (drawn, whole)) in stages.into_iter().enumerate() {
                    match drawn {
                        Some(drawn) if drawn == whole => self.edge_builds[stage] += 1,
                        Some(_) => self.edge_patches[stage] += 1,
                        None => {}
                    }
                }
                let traced = &mut self.shape_redrawn[index];
                (traced.shift, traced.feather, traced.finish) =
                    (work.shift, work.feather, work.finish);
                let input = slot
                    .refined
                    .as_ref()
                    .map_or(&slot.alpha.view, |refined| &refined.alpha);
                self.edge.run(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    &frame.edge_scratch,
                    edged,
                    input,
                    p,
                    &work,
                );
                edged.held = Some(EdgeHeld {
                    plan: edge_plan,
                    refined: refined_input,
                });
                if p.feathers() || p.contrasts() {
                    finished
                } else {
                    shifted
                }
            } else {
                slot.edged = None;
                handed
            };
            self.shape_redrawn[index].frame = (width, height);
            self.shape_redrawn[index].region = region;
            damage = match changed {
                Damage::Part(reach) => damage.with(intersect(reach, region)),
                Damage::Whole => Damage::Whole,
                Damage::Nothing => damage,
            };
            let product = product_of(slot);
            if slot.binds_product != product {
                slot.develop_binds = develop_binds(product_view(slot));
                slot.binds_product = product;
            }
            if !develops {
                continue;
            }
            let (ours, theirs) = (&edit.look.curves, &mask.adjust.look.curves);
            let composed = (!ours.is_identity() || !theirs.is_identity())
                .then(|| (ours.clone(), theirs.clone()));
            if composed.is_some() && self.mask_curves_uploaded[index] != composed {
                let tables = curve::bake_composed(ours, theirs);
                write_table_row(&self.queue, &self.curve_table, index as u32 + 1, &tables);
                self.mask_curves_uploaded[index] = composed;
            }
            self.queue.write_buffer(
                &self.masked_uniforms[index],
                0,
                bytemuck::bytes_of(&DevelopUniform::for_mask(
                    &edit.adjust,
                    mask,
                    index,
                    source.atmosphere,
                )),
            );
        }
        let frame = self.frame.as_mut().expect("frame built above");
        if frame
            .masks
            .iter()
            .flatten()
            .all(|slot| slot.refined.is_none())
        {
            frame.refine_scratch = None;
        }
        let (runs, cells) = frame
            .masks
            .iter()
            .flatten()
            .filter_map(|slot| slot.edged.as_ref())
            .fold((false, false), |(runs, cells), edged| {
                (runs || edged.shifts(), cells || edged.feathers())
            });
        frame.edge_scratch.keep(runs, cells);

        // The developed texture is kept while the edit and the scissor are
        // the ones it was drawn for: a pan changes neither.
        let developed = frame
            .developed_for
            .as_ref()
            .is_some_and(|(drawn, scissor)| *scissor == seen && drawn == edit);
        if !developed {
            // While a stroke grows the edit differs from the one developed
            // only by strokes, and the picture only where the new dabs
            // reach: the passes run over that part of what is seen alone.
            let grew = damage != Damage::Whole
                && frame.developed_passes == masks.len()
                && frame
                    .developed_for
                    .as_ref()
                    .is_some_and(|(drawn, scissor)| {
                        *scissor == seen && drawn.same_but_strokes(edit)
                    });
            let over = match damage {
                _ if !grew => Some(seen),
                Damage::Part(reach) => intersect(reach, seen),
                _ => None,
            };
            if let Some(over) = over {
                self.queue.write_buffer(
                    &self.develop_uniform,
                    0,
                    bytemuck::bytes_of(&DevelopUniform::new(&edit.adjust, source.atmosphere)),
                );
                // A part is drawn over what the texture holds around it.
                let global = if grew { draw_over } else { draw };
                // The passes take turns at the two developed textures, and
                // the global pass starts where the last one ends in
                // `developed`.
                let targets = [&frame.developed, &frame.developed_other];
                let mut written = masks.len() % 2;
                global(
                    &mut encoder,
                    "develop",
                    &self.develop.pipeline,
                    &frame.develop_bind,
                    &targets[written].view,
                    Some(over),
                );
                // The ordered blend: each mask in list order over what the
                // passes before it left. A mask pass writes every pixel of
                // its scissor, so it clears nothing.
                for (index, _) in &masks {
                    let slot = frame.masks[*index].as_ref().expect("built above");
                    draw_over(
                        &mut encoder,
                        "masked develop",
                        &self.masked.pipeline,
                        &slot.develop_binds[written],
                        &targets[1 - written].view,
                        Some(over),
                    );
                    written = 1 - written;
                }
                if grew {
                    self.develop_patches += 1;
                }
            }
            frame.developed_for = Some((edit.clone(), seen));
            frame.developed_passes = masks.len();
        }
        let frame = self.frame.as_ref().expect("frame built above");

        let (out_width, out_height) = (output_size.0.max(1), output_size.1.max(1));
        let overlay = overlaid.as_ref().map(|(index, _)| {
            let product = frame.masks[*index]
                .as_ref()
                .map_or(Product::Alpha, product_of);
            (*index, product)
        });
        let out_stale = self.out.as_ref().is_none_or(|o| {
            (o.target.width, o.target.height) != (out_width, out_height)
                || o.frame_generation != self.frame_generation
                || o.overlay != overlay
        });
        if out_stale {
            let target = create_target(
                &self.device,
                "develop output",
                OUTPUT_FORMAT,
                out_width,
                out_height,
            );
            let overlay_alpha = overlay
                .and_then(|(index, _)| frame.masks[index].as_ref())
                .map_or(&self.no_overlay.view, product_view);
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("output bind group"),
                layout: &self.output.layout,
                entries: &[
                    buffer_binding(0, &self.output_uniform),
                    texture_binding(1, &frame.developed.view),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    texture_binding(3, overlay_alpha),
                ],
            });
            self.out = Some(Output {
                target,
                bind,
                frame_generation: self.frame_generation,
                overlay,
            });
            self.out_generation += 1;
        }
        let out = self.out.as_ref().expect("output built above");
        self.queue.write_buffer(
            &self.output_uniform,
            0,
            bytemuck::bytes_of(&OutputUniform {
                matrix: matrices::rec2020_to_srgb().to_wgsl_columns(),
                crop: [crop.x, crop.y, crop.width, crop.height],
                overlay: [
                    overlay.map_or(0.0, |_| mask_twin::OVERLAY_STRENGTH),
                    0.0,
                    0.0,
                    0.0,
                ],
            }),
        );
        draw(
            &mut encoder,
            "output",
            &self.output.pipeline,
            &out.bind,
            &out.target.view,
            None,
        );
        self.queue.submit(Some(encoder.finish()));
        Some(&out.target.view)
    }

    /// The last output view, when a render has happened.
    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.out.as_ref().map(|o| &o.target.view)
    }

    /// The size of the last output.
    pub fn output_size(&self) -> Option<(u32, u32)> {
        self.out.as_ref().map(|o| (o.target.width, o.target.height))
    }

    /// Counts up every time the output texture is replaced, so a caller
    /// that registered the view elsewhere knows to register it again.
    pub fn output_generation(&self) -> u64 {
        self.out_generation
    }

    /// Renders `crop` at the full source resolution, resamples it to the
    /// preset's size with linear filtering (halving first while the ratio
    /// is above 2, so no step skips pixels) and reads it back as sRGB RGBA
    /// bytes. Returns `None` when no source is set. The viewer's frame is
    /// replaced by the full-size one, so the next viewer render rebuilds.
    pub fn render_export(
        &mut self,
        edit: &PhotoEdit,
        crop: CropRect,
        preset: ExportPreset,
    ) -> Option<Vec<u8>> {
        let (source_width, source_height) = self.source_size()?;
        let crop_size = crop.pixel_size(source_width, source_height);
        // An export is the picture, never the view of a mask.
        let overlay = self.overlay.take();
        let rendered = self
            .render(edit, crop, (source_width, source_height), crop_size)
            .is_some();
        self.overlay = overlay;
        if !rendered {
            return None;
        }
        let target = preset.size();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("export encoder"),
            });
        let mut view = self.out.as_ref()?.target.view.clone();
        let mut size = crop_size;
        let mut steps = Vec::new();
        while size.0 > target.0 * 2 && size.1 > target.1 * 2 {
            size = (size.0.div_ceil(2), size.1.div_ceil(2));
            let step = self.resample(&mut encoder, &view, size);
            view = step.0.view.clone();
            steps.push(step);
        }
        let last = self.resample(&mut encoder, &view, target);
        steps.push(last);
        self.queue.submit(Some(encoder.finish()));
        let (final_target, _, _) = steps.last().expect("at least one step");
        log::info!(
            "exported {crop_size:?} of {source_width}x{source_height} to {target:?} in {} steps",
            steps.len()
        );
        Some(self.readback.read(
            &self.device,
            &self.queue,
            &final_target.view,
            target.0,
            target.1,
        ))
    }

    /// One linear resample of `source` into a new sRGB target of `size`,
    /// through the output pass with the identity matrix.
    fn resample(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        size: (u32, u32),
    ) -> (Target, wgpu::Buffer, wgpu::BindGroup) {
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resample uniform"),
            size: size_of::<OutputUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &uniform,
            0,
            bytemuck::bytes_of(&OutputUniform {
                matrix: matrices::Mat3::IDENTITY.to_wgsl_columns(),
                crop: [0.0, 0.0, 1.0, 1.0],
                overlay: [0.0; 4],
            }),
        );
        let target = create_target(
            &self.device,
            "resample target",
            OUTPUT_FORMAT,
            size.0,
            size.1,
        );
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resample bind group"),
            layout: &self.output.layout,
            entries: &[
                buffer_binding(0, &uniform),
                texture_binding(1, source),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                texture_binding(3, &self.no_overlay.view),
            ],
        });
        draw(
            encoder,
            "resample",
            &self.output.pipeline,
            &bind,
            &target.view,
            None,
        );
        (target, uniform, bind)
    }

    /// The head passes of `frame`: the input transform and the base blur
    /// when it does not hold the source content yet, then the texture layer
    /// and the transmission map when `work` wants them and they were not
    /// drawn since. They read only the frame and the uniforms they write, so
    /// a frame built ahead runs them in submits of its own
    /// ([`ahead_slice`](Self::ahead_slice)); here each runs whole.
    fn head_passes(&self, encoder: &mut wgpu::CommandEncoder, frame: &mut Frame, work: &HeadWork) {
        let source = self.source.as_ref().expect("a frame has a source");
        let rerun = frame.content != source.content;
        if rerun {
            frame.content = source.content;
        }
        // The head-pass products of M3. Each is drawn once when its slider
        // leaves 0 and again only after the head passes ran or the scissor
        // moved, never on a plain slider change.
        let region = scissor_for(work.products, work.size, false);
        if rerun || frame.products_region != Some(region) {
            frame.texture_ready = false;
            frame.transmission_ready = false;
            frame.products_region = Some(region);
            frame.developed_for = None;
            // The moments Refine edges holds of the source are of the
            // working texture as it was.
            if let Some(scratch) = frame.refine_scratch.as_mut() {
                scratch.forget_source();
            }
            for mask in frame.masks.iter_mut().flatten() {
                mask.shape = None;
                mask.refine = None;
                if let Some(edged) = mask.edged.as_mut() {
                    edged.held = None;
                }
                // A layer with an auto stroke reads the working texture: it
                // is stamped again when the head passes ran. One without
                // reads no source pixel and is kept.
                if rerun {
                    mask.layers.forget_auto();
                }
            }
        }
        let texture = work.texture && !frame.texture_ready;
        let transmission = work.transmission && !frame.transmission_ready;
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        for pass in HeadPass::plan(rerun, texture, transmission) {
            self.write_head_uniform(pass, work);
            self.draw_head_pass(encoder, frame, pass, pass.scissor(work), clear);
        }
        if texture {
            frame.texture_ready = true;
        }
        if transmission {
            frame.transmission_ready = true;
        }
    }

    /// Writes the uniform `pass` reads for `work`. Each head pass has a
    /// buffer of its own, so a submit holds every pass's value at once.
    fn write_head_uniform(&self, pass: HeadPass, work: &HeadWork) {
        let source = self.source.as_ref().expect("a frame has a source");
        let (width, height) = work.size;
        let (window, sigma_size) = (work.window, work.sigma_size);
        match pass {
            HeadPass::Input => {
                let window_uniform = [window.x, window.y, window.width, window.height];
                match &source.kind {
                    SourceKind::Photo { space, .. } => {
                        let taps =
                            input_taps((source.width, source.height), window, (width, height));
                        self.queue.write_buffer(
                            &self.input_uniform,
                            0,
                            bytemuck::bytes_of(&InputUniform {
                                matrix: matrices::input_matrix(*space).to_wgsl_columns(),
                                render_size: [width as f32, height as f32],
                                decode_srgb: 1,
                                taps,
                                window: window_uniform,
                            }),
                        );
                    }
                    SourceKind::Video(planes) => {
                        self.queue.write_buffer(
                            &self.video_uniform,
                            0,
                            bytemuck::bytes_of(&planes.uniform(window_uniform)),
                        );
                    }
                }
            }
            HeadPass::BlurH | HeadPass::BlurV => {
                let sigma = basic::base_sigma(sigma_size.0, sigma_size.1);
                if pass == HeadPass::BlurH {
                    write_blur(&self.queue, &self.blur_h_uniform, [1, 0], 1, sigma);
                } else {
                    write_blur(&self.queue, &self.blur_v_uniform, [0, 1], 0, sigma);
                }
            }
            HeadPass::TextureH | HeadPass::TextureV => {
                let sigma = local::texture_sigma(sigma_size.0, sigma_size.1);
                if pass == HeadPass::TextureH {
                    write_blur(&self.queue, &self.texture_h_uniform, [1, 0], 1, sigma);
                } else {
                    write_blur(&self.queue, &self.texture_v_uniform, [0, 1], 0, sigma);
                }
            }
            HeadPass::MinimumH | HeadPass::MinimumV => {
                let radius = dehaze::patch_radius(sigma_size.0, sigma_size.1);
                let a = source.atmosphere;
                let (buffer, direction, stage) = if pass == HeadPass::MinimumH {
                    (&self.minimum_h_uniform, [1, 0], 0)
                } else {
                    (&self.minimum_v_uniform, [0, 1], 1)
                };
                self.queue.write_buffer(
                    buffer,
                    0,
                    bytemuck::bytes_of(&MinimumUniform {
                        direction,
                        radius,
                        stage,
                        atmosphere: [a[0], a[1], a[2], 1.0],
                    }),
                );
            }
            HeadPass::TransmissionH | HeadPass::TransmissionV => {
                let sigma = dehaze::smoothing_sigma(sigma_size.0, sigma_size.1);
                if pass == HeadPass::TransmissionH {
                    write_blur(&self.queue, &self.smooth_h_uniform, [1, 0], 0, sigma);
                } else {
                    write_blur(&self.queue, &self.smooth_v_uniform, [0, 1], 0, sigma);
                }
            }
        }
    }

    /// Records `pass` of `frame` under `scissor`, beginning with `load`: a
    /// clear for the whole pass or its first strip, a load for the strips
    /// after it.
    fn draw_head_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        pass: HeadPass,
        scissor: Option<(u32, u32, u32, u32)>,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        let source = self.source.as_ref().expect("a frame has a source");
        let blur = &self.blur.pipeline;
        let minimum = &self.minimum.pipeline;
        let (label, pipeline, bind, target) = match pass {
            HeadPass::Input => {
                let (label, pipeline) = match &source.kind {
                    SourceKind::Photo { .. } => ("input transform", &self.input.pipeline),
                    SourceKind::Video(_) => ("video", &self.video.pipeline),
                };
                (label, pipeline, &frame.input_bind, &frame.working)
            }
            HeadPass::BlurH => ("blur h", blur, &frame.blur_h_bind, &frame.ping),
            HeadPass::BlurV => ("blur v", blur, &frame.blur_v_bind, &frame.base),
            HeadPass::TextureH => ("texture blur h", blur, &frame.texture_h_bind, &frame.ping),
            HeadPass::TextureV => (
                "texture blur v",
                blur,
                &frame.texture_v_bind,
                &frame.texture_base,
            ),
            HeadPass::MinimumH => ("minimum h", minimum, &frame.minimum_h_bind, &frame.ping),
            HeadPass::MinimumV => (
                "minimum v",
                minimum,
                &frame.minimum_v_bind,
                &frame.transmission,
            ),
            HeadPass::TransmissionH => (
                "transmission blur h",
                blur,
                &frame.smooth_h_bind,
                &frame.ping,
            ),
            HeadPass::TransmissionV => (
                "transmission blur v",
                blur,
                &frame.smooth_v_bind,
                &frame.transmission,
            ),
        };
        draw_loading(encoder, label, pipeline, bind, &target.view, scissor, load);
    }

    /// What one pixel of `pass` costs, in texels read and written (see
    /// [`AHEAD_SLICE_TEXELS`]): the reads its shader makes for the pixel
    /// plus its one write.
    fn head_pass_texels(&self, pass: HeadPass, work: &HeadWork) -> u64 {
        let source = self.source.as_ref().expect("a frame has a source");
        let sigma_size = work.sigma_size;
        // A blur reads its centre and one bilinear read for each pair of
        // taps on each side (blur.wgsl).
        let blur = |sigma: f32| {
            let radius = basic::blur_radius(sigma).max(0) as u64;
            1 + 2 * radius.div_ceil(2)
        };
        let reads = match pass {
            HeadPass::Input => match &source.kind {
                SourceKind::Photo { .. } => {
                    let taps = u64::from(input_taps(
                        (source.width, source.height),
                        work.window,
                        work.size,
                    ));
                    taps * taps
                }
                // The luma plane and the chroma plane.
                SourceKind::Video(_) => 2,
            },
            HeadPass::BlurH | HeadPass::BlurV => {
                blur(basic::base_sigma(sigma_size.0, sigma_size.1))
            }
            HeadPass::TextureH | HeadPass::TextureV => {
                blur(local::texture_sigma(sigma_size.0, sigma_size.1))
            }
            HeadPass::MinimumH | HeadPass::MinimumV => {
                let radius = dehaze::patch_radius(sigma_size.0, sigma_size.1).max(0) as u64;
                2 * radius + 1
            }
            HeadPass::TransmissionH | HeadPass::TransmissionV => {
                blur(dehaze::smoothing_sigma(sigma_size.0, sigma_size.1))
            }
        };
        reads + 1
    }

    fn build_frame(
        &self,
        source: &Source,
        width: u32,
        height: u32,
        window: CropRect,
        sigma_size: (u32, u32),
    ) -> Frame {
        let device = &self.device;
        let working = create_target(device, "working", WORKING_FORMAT, width, height);
        let ping = create_target(device, "blur ping", BASE_FORMAT, width, height);
        let base = create_target(device, "base", BASE_FORMAT, width, height);
        let texture_base = create_target(device, "texture base", BASE_FORMAT, width, height);
        let transmission = create_target(device, "transmission", BASE_FORMAT, width, height);
        let developed = create_target(device, "developed", WORKING_FORMAT, width, height);
        let developed_other =
            create_target(device, "developed other", WORKING_FORMAT, width, height);
        let input_bind = match &source.kind {
            SourceKind::Photo { view, .. } => {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("input bind group"),
                    layout: &self.input.layout,
                    entries: &[
                        buffer_binding(0, &self.input_uniform),
                        texture_binding(1, view),
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            }
            SourceKind::Video(planes) => device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("video bind group"),
                layout: &self.video.layout,
                entries: &[
                    buffer_binding(0, &self.video_uniform),
                    texture_binding(1, &planes.luma_view),
                    texture_binding(2, &planes.chroma_view),
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            }),
        };
        let blur_h_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur h bind group"),
            layout: &self.blur.layout,
            entries: &[
                buffer_binding(0, &self.blur_h_uniform),
                texture_binding(1, &working.view),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let blur_v_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur v bind group"),
            layout: &self.blur.layout,
            entries: &[
                buffer_binding(0, &self.blur_v_uniform),
                texture_binding(1, &ping.view),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let blur_bind = |label: &str, uniform: &wgpu::Buffer, view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.blur.layout,
                entries: &[
                    buffer_binding(0, uniform),
                    texture_binding(1, view),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let texture_h_bind = blur_bind(
            "texture h bind group",
            &self.texture_h_uniform,
            &working.view,
        );
        let texture_v_bind = blur_bind("texture v bind group", &self.texture_v_uniform, &ping.view);
        let smooth_h_bind = blur_bind(
            "smooth h bind group",
            &self.smooth_h_uniform,
            &transmission.view,
        );
        let smooth_v_bind = blur_bind("smooth v bind group", &self.smooth_v_uniform, &ping.view);
        let minimum_bind = |label: &str, uniform: &wgpu::Buffer, view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.minimum.layout,
                entries: &[buffer_binding(0, uniform), texture_binding(1, view)],
            })
        };
        let minimum_h_bind = minimum_bind(
            "minimum h bind group",
            &self.minimum_h_uniform,
            &working.view,
        );
        let minimum_v_bind =
            minimum_bind("minimum v bind group", &self.minimum_v_uniform, &ping.view);
        let develop_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("develop bind group"),
            layout: &self.develop.layout,
            entries: &[
                buffer_binding(0, &self.develop_uniform),
                texture_binding(1, &working.view),
                texture_binding(2, &base.view),
                texture_binding(3, &texture_base.view),
                texture_binding(4, &transmission.view),
                texture_binding(5, &self.curve_table_view),
            ],
        });
        Frame {
            width,
            height,
            generation: source.generation,
            // The head passes have not run into these textures yet.
            content: source.content.wrapping_sub(1),
            window,
            sigma_size,
            working,
            ping,
            base,
            texture_base,
            transmission,
            developed,
            developed_other,
            texture_ready: false,
            transmission_ready: false,
            products_region: None,
            developed_for: None,
            developed_passes: 0,
            input_bind,
            blur_h_bind,
            blur_v_bind,
            texture_h_bind,
            texture_v_bind,
            minimum_h_bind,
            minimum_v_bind,
            smooth_h_bind,
            smooth_v_bind,
            develop_bind,
            masks: Default::default(),
            refine_scratch: None,
            edge_scratch: EdgeScratch::default(),
        }
    }
}

/// A 1 by 1 alpha texture holding 0.
/// The passes and the sampler a proxy build draws with.
struct ProxyPasses<'a> {
    input: &'a Pass,
    video: &'a Pass,
    reduce: &'a Pass,
    sampler: &'a wgpu::Sampler,
}

/// Brings the proxy in `slot` to the content of `source`: sets it up for the
/// source textures when it is not, and builds it when it holds another
/// content. Whether it was built.
///
/// Each tile is drawn by the head pass of the source at the source's own
/// size, one sample a pixel, so it holds the working pixels every render at
/// that scale holds, and is then averaged into its block of the proxy.
fn build_proxy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    passes: ProxyPasses,
    source: &Source,
    slot: &mut Option<ProxyProduct>,
    ids: &mut u64,
) -> bool {
    let size = (source.width, source.height);
    let proxy_size = ProxyTwin::size_for(size);
    if slot
        .as_ref()
        .is_none_or(|product| product.generation != source.generation)
    {
        let (w, h) = proxy_size;
        let target = create_target(device, "proxy", WORKING_FORMAT, w, h);
        let (w, h) = (size.0.min(proxy::TILE), size.1.min(proxy::TILE));
        let tile = create_target(device, "proxy tile", WORKING_FORMAT, w, h);
        let buffer = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let tiles = proxy::tiles(size, proxy_size)
            .into_iter()
            .map(|plan| {
                let (head_uniform, head_bind) = match &source.kind {
                    SourceKind::Photo { view, .. } => {
                        let uniform = buffer("proxy input uniform", size_of::<InputUniform>());
                        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("proxy input bind group"),
                            layout: &passes.input.layout,
                            entries: &[
                                buffer_binding(0, &uniform),
                                texture_binding(1, view),
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(passes.sampler),
                                },
                            ],
                        });
                        (uniform, bind)
                    }
                    SourceKind::Video(planes) => {
                        let uniform = buffer("proxy video uniform", size_of::<VideoUniform>());
                        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("proxy video bind group"),
                            layout: &passes.video.layout,
                            entries: &[
                                buffer_binding(0, &uniform),
                                texture_binding(1, &planes.luma_view),
                                texture_binding(2, &planes.chroma_view),
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::Sampler(passes.sampler),
                                },
                            ],
                        });
                        (uniform, bind)
                    }
                };
                let reduce_uniform = buffer("proxy uniform", size_of::<ProxyUniform>());
                queue.write_buffer(
                    &reduce_uniform,
                    0,
                    bytemuck::bytes_of(&plan.uniform(size, proxy_size)),
                );
                let reduce_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("proxy bind group"),
                    layout: &passes.reduce.layout,
                    entries: &[
                        buffer_binding(0, &reduce_uniform),
                        texture_binding(1, &tile.view),
                    ],
                });
                ProxyTile {
                    plan,
                    head_uniform,
                    head_bind,
                    reduce_bind,
                }
            })
            .collect();
        *ids += 1;
        *slot = Some(ProxyProduct {
            generation: source.generation,
            content: None,
            target,
            tile,
            tiles,
            id: *ids,
        });
    }
    let product = slot.as_mut().expect("set up above");
    if product.content == Some(source.content) {
        return false;
    }
    for tile in &product.tiles {
        let (_, _, w, h) = tile.plan.source;
        let window = tile.plan.window(size);
        let head = match &source.kind {
            SourceKind::Photo { space, .. } => {
                queue.write_buffer(
                    &tile.head_uniform,
                    0,
                    bytemuck::bytes_of(&InputUniform {
                        matrix: matrices::input_matrix(*space).to_wgsl_columns(),
                        render_size: [w as f32, h as f32],
                        decode_srgb: 1,
                        taps: 1,
                        window,
                    }),
                );
                &passes.input.pipeline
            }
            SourceKind::Video(planes) => {
                queue.write_buffer(
                    &tile.head_uniform,
                    0,
                    bytemuck::bytes_of(&planes.uniform(window)),
                );
                &passes.video.pipeline
            }
        };
        draw_in_viewport(
            encoder,
            "proxy tile",
            head,
            &tile.head_bind,
            &product.tile.view,
            (w, h),
        );
        draw_over(
            encoder,
            "proxy",
            &passes.reduce.pipeline,
            &tile.reduce_bind,
            &product.target.view,
            Some(tile.plan.proxy),
        );
    }
    product.content = Some(source.content);
    true
}

fn zero_texel(device: &wgpu::Device, queue: &wgpu::Queue) -> Target {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("no overlay"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ALPHA_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &[0],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(1),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    Target {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        width: 1,
        height: 1,
    }
}

/// Writes one row of the tone curve table.
fn write_table_row(queue: &wgpu::Queue, table: &wgpu::Texture, row: u32, tables: &curve::Tables) {
    let texels: Vec<[f32; 4]> = (0..TABLE_SIZE)
        .map(|i| [tables.red[i], tables.green[i], tables.blue[i], 1.0])
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: table,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: row, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TABLE_SIZE as u32 * 16),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: TABLE_SIZE as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

/// Whether the edit wants the texture layer and the transmission map. A mask
/// that alone turns on texture or dehaze asks for the product the same way
/// the global slider does.
fn head_products_wanted(edit: &PhotoEdit, masks: &[(usize, Mask)]) -> (bool, bool) {
    let effective: Vec<Adjustments> = masks
        .iter()
        .map(|(_, mask)| mask_twin::effective_adjustments(&edit.adjust, &mask.adjust))
        .collect();
    let wants_texture = edit.texture != 0.0 || effective.iter().any(|e| e.texture != 0.0);
    let wants_transmission = edit.dehaze != 0.0 || effective.iter().any(|e| e.dehaze != 0.0);
    (wants_texture, wants_transmission)
}

/// What a zoomed viewer shows, in pixels of the whole picture rendered at
/// `full`. Rectangles are x, y, width, height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewWindow {
    /// The size the whole picture would have at this zoom, never above the
    /// size of the source.
    pub full: (u32, u32),
    /// The padded rectangle the products and the develop pass cover.
    pub window: PixelRect,
    /// The part of `window` the output shows, one to one.
    pub visible: PixelRect,
}

/// A rectangle of pixels: x, y, width, height.
pub type PixelRect = (u32, u32, u32, u32);

/// `visible` with `pad` on every side, its edges moved outward onto `grid`,
/// kept inside `full`.
pub fn padded_window(
    full: (u32, u32),
    visible: PixelRect,
    pad: (u32, u32),
    grid: u32,
) -> PixelRect {
    let grid = grid.max(1);
    let axis = |start: u32, size: u32, pad: u32, full: u32| {
        let low = start.saturating_sub(pad) / grid * grid;
        let high = (start + size + pad).div_ceil(grid) * grid;
        let high = high.min(full).max(low + 1);
        (low, high - low)
    };
    let (x, width) = axis(visible.0, visible.2, pad.0, full.0);
    let (y, height) = axis(visible.1, visible.3, pad.1, full.1);
    (x, y, width, height)
}

/// Where `visible` first lies outside `window` when a pan keeps moving it by
/// the step from `previous` to `visible`, kept inside `full`. `None` while the
/// pan does not move, or when it moves only toward edges of `window` that are
/// edges of `full`, so that it never leaves.
pub fn pan_exit(
    full: (u32, u32),
    window: PixelRect,
    previous: PixelRect,
    visible: PixelRect,
) -> Option<PixelRect> {
    let step = (
        i64::from(visible.0) - i64::from(previous.0),
        i64::from(visible.1) - i64::from(previous.1),
    );
    // The steps until the pan passes the edge of the window on one axis.
    let steps = |step: i64, start: u32, size: u32, low: u32, extent: u32, full: u32| {
        let (start, size, low, high) = (
            i64::from(start),
            i64::from(size),
            i64::from(low),
            i64::from(low) + i64::from(extent),
        );
        match step {
            0 => None,
            s if s > 0 && high < i64::from(full) => Some((high - start - size).max(0) / s + 1),
            s if s < 0 && low > 0 => Some((start - low).max(0) / -s + 1),
            _ => None,
        }
    };
    let across = steps(step.0, visible.0, visible.2, window.0, window.2, full.0);
    let down = steps(step.1, visible.1, visible.3, window.1, window.3, full.1);
    let taken = match (across, down) {
        (Some(a), Some(d)) => a.min(d),
        (Some(n), None) | (None, Some(n)) => n,
        (None, None) => return None,
    };
    let moved = |start: u32, size: u32, step: i64, full: u32| {
        let last = i64::from(full.saturating_sub(size));
        (i64::from(start) + taken * step).clamp(0, last) as u32
    };
    Some((
        moved(visible.0, visible.2, step.0, full.0),
        moved(visible.1, visible.3, step.1, full.1),
        visible.2,
        visible.3,
    ))
}

/// The window to build ahead of a pan: the padded window of what is seen
/// where the pan leaves `window` (see [`pan_exit`]), moved half the pad
/// further in the direction of the pan on each axis it moves along. It
/// holds that rectangle with half the pad behind it, so a pan whose steps
/// vary by less than that still leaves `window` inside it, and reaches one
/// and a half pads ahead of it. `None` when the pan never leaves `window`.
pub fn window_ahead(
    full: (u32, u32),
    window: PixelRect,
    previous: PixelRect,
    visible: PixelRect,
    pad: (u32, u32),
    grid: u32,
) -> Option<PixelRect> {
    let exit = pan_exit(full, window, previous, visible)?;
    let ahead = |start: u32, size: u32, pan: Ordering, pad: u32, full: u32| match pan {
        Ordering::Greater => start.saturating_add(pad).min(full.saturating_sub(size)),
        Ordering::Less => start.saturating_sub(pad),
        Ordering::Equal => start,
    };
    let (x, y) = (
        ahead(
            exit.0,
            exit.2,
            visible.0.cmp(&previous.0),
            pad.0 / 2,
            full.0,
        ),
        ahead(
            exit.1,
            exit.3,
            visible.1.cmp(&previous.1),
            pad.1 / 2,
            full.1,
        ),
    );
    let moved = padded_window(full, (x, y, exit.2, exit.3), pad, grid);
    Some(if holds(moved, exit) {
        moved
    } else {
        padded_window(full, exit, pad, grid)
    })
}

/// Whether `inner` lies inside `window`.
pub fn holds(window: PixelRect, inner: PixelRect) -> bool {
    inner.0 >= window.0
        && inner.1 >= window.1
        && inner.0 + inner.2 <= window.0 + window.2
        && inner.1 + inner.3 <= window.1 + window.3
}

/// How far the widest head pass reads around a pixel at a render of `full`:
/// the base blur, the texture blur, or the dark channel patch plus its
/// smoothing.
pub fn head_pass_reach(full: (u32, u32)) -> u32 {
    let base = basic::blur_radius(basic::base_sigma(full.0, full.1));
    let texture = basic::blur_radius(local::texture_sigma(full.0, full.1));
    let haze = dehaze::patch_radius(full.0, full.1)
        + basic::blur_radius(dehaze::smoothing_sigma(full.0, full.1));
    base.max(texture).max(haze).max(0) as u32
}

/// `rect` moved and shrunk until it lies inside `bounds`, never empty.
fn clamp_rect(rect: (u32, u32, u32, u32), bounds: (u32, u32, u32, u32)) -> (u32, u32, u32, u32) {
    let (bx, by, bw, bh) = (bounds.0, bounds.1, bounds.2.max(1), bounds.3.max(1));
    let (w, h) = (rect.2.clamp(1, bw), rect.3.clamp(1, bh));
    let x = rect.0.clamp(bx, bx + bw - w);
    let y = rect.1.clamp(by, by + bh - h);
    (x, y, w, h)
}

/// The margin of [`scissor_for`] in pixels.
const PRODUCTS_MARGIN: u32 = 2;

/// The pixels the blur must reach: the crop with a margin of two pixels
/// for the output pass's sampling, over the whole height when
/// `full_height` (the horizontal pass feeds every row the vertical pass
/// reads). As x, y, width, height of a render of `size`.
fn scissor_for(crop: CropRect, size: (u32, u32), full_height: bool) -> (u32, u32, u32, u32) {
    const MARGIN: f32 = PRODUCTS_MARGIN as f32;
    let (w, h) = (size.0 as f32, size.1 as f32);
    let x0 = (crop.x * w - MARGIN).floor().clamp(0.0, w - 1.0);
    let x1 = ((crop.x + crop.width) * w + MARGIN)
        .ceil()
        .clamp(x0 + 1.0, w);
    let (y0, y1) = if full_height {
        (0.0, h)
    } else {
        let y0 = (crop.y * h - MARGIN).floor().clamp(0.0, h - 1.0);
        (
            y0,
            ((crop.y + crop.height) * h + MARGIN)
                .ceil()
                .clamp(y0 + 1.0, h),
        )
    };
    (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
}

/// How many source pixels a side the input transform averages into one pixel
/// of a render of `window`: the source pixels a render pixel covers, rounded
/// up. The fractions of a window are f32, so a window at one source pixel a
/// pixel can read a hair over 1 (586 of 1100 pixels reads 1.0000001), and a
/// second tap there would soften that window and no other. A tap is added
/// only past that noise.
fn input_taps(source: (u32, u32), window: CropRect, render: (u32, u32)) -> u32 {
    const NOISE: f32 = 1e-4;
    let covered = (source.0 as f32 * window.width / render.0 as f32)
        .max(source.1 as f32 * window.height / render.1 as f32);
    (covered - NOISE).ceil().clamp(1.0, MAX_TAPS as f32) as u32
}

/// The size to render the whole photo at so that `crop` lands on an output
/// of `output` pixels one to one.
pub fn render_size_for_crop(crop: CropRect, output: (u32, u32)) -> (u32, u32) {
    (
        (output.0 as f32 / crop.width.max(1e-6)).round().max(1.0) as u32,
        (output.1 as f32 / crop.height.max(1e-6)).round().max(1.0) as u32,
    )
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// The brush layers of one mask: an array read with `textureLoad`.
fn layers_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

/// The tone curve table: full floats, which are not filterable without a
/// device feature, read with `textureLoad` only.
fn table_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Writes the uniform of one blur pass.
fn write_blur(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    direction: [i32; 2],
    luma: u32,
    sigma: f32,
) {
    queue.write_buffer(
        buffer,
        0,
        bytemuck::bytes_of(&BlurUniform {
            direction,
            radius: basic::blur_radius(sigma),
            luma,
            sigma,
            _pad: [0.0; 3],
        }),
    );
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn buffer_binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn texture_binding(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn make_pass(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    format: wgpu::TextureFormat,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Pass {
    make_pass_with(
        device,
        label,
        source,
        format,
        entries,
        "fs_main",
        wgpu::BlendState::REPLACE,
    )
}

/// [`make_pass`] with the fragment entry point and the blend named.
fn make_pass_with(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    format: wgpu::TextureFormat,
    entries: &[wgpu::BindGroupLayoutEntry],
    fragment: &str,
    blend: wgpu::BlendState,
) -> Pass {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: fullscreen_primitive(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    Pass { pipeline, layout }
}

fn create_target(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Target {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    Target {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        width,
        height,
    }
}

fn draw(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    scissor: Option<(u32, u32, u32, u32)>,
) {
    let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
    draw_loading(encoder, label, pipeline, bind_group, target, scissor, clear);
}

/// [`draw`] into the top left `size` pixels of a larger target: the viewport
/// is that part, so the pass sees a render of that size.
fn draw_in_viewport(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    size: (u32, u32),
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_viewport(0.0, 0.0, size.0 as f32, size.1 as f32, 0.0, 1.0);
    pass.set_scissor_rect(0, 0, size.0, size.1);
    pass.draw(0..FULLSCREEN_VERTICES, 0..1);
}

/// [`draw`] over what the target holds, with no clear.
fn draw_over(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    scissor: Option<(u32, u32, u32, u32)>,
) {
    let load = wgpu::LoadOp::Load;
    draw_loading(encoder, label, pipeline, bind_group, target, scissor, load);
}

fn draw_loading(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    scissor: Option<(u32, u32, u32, u32)>,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    if let Some((x, y, w, h)) = scissor {
        pass.set_scissor_rect(x, y, w, h);
    }
    pass.draw(0..FULLSCREEN_VERTICES, 0..1);
}

#[cfg(test)]
mod tests {
    use super::{
        DevelopUniform, FLAG_CURVES, Geometry, MAX_TAPS, MaskComponentUniform, MaskUniform,
        MinimumUniform, OutputUniform, PixelRect, holds, input_taps, padded_window, pan_exit,
        scissor_for, window_ahead,
    };
    use gamut_core::mask::{Component, LinearGradient, MaskOp, MaskSource, RadialGradient};
    use gamut_core::{Adjustments, CropRect, Mask};
    use std::mem::offset_of;

    /// The window a pan will reach, for each of the four directions of a
    /// steady pan of 32 pixels a step at 100 percent of a 6000 by 4000 photo
    /// in a 1280 by 1600 tab: the rectangle seen where the pan first leaves
    /// the window is the one the steps reach, and the window ahead holds it,
    /// keeps the cross axis of the window it follows and reaches past that
    /// window in the pan's direction.
    #[test]
    fn a_window_ahead_lies_where_a_pan_leaves_in_each_of_four_directions() {
        const FULL: (u32, u32) = (6000, 4000);
        const PAD: (u32, u32) = (640, 800);
        const GRID: u32 = 64;
        const STEP: i64 = 32;
        let visible: PixelRect = (2359, 1199, 1281, 1601);
        let window = padded_window(FULL, visible, PAD, GRID);
        assert_eq!(window, (1664, 384, 2624, 3264));
        let at = |rect: PixelRect, (dx, dy): (i64, i64)| -> PixelRect {
            (
                (i64::from(rect.0) + dx) as u32,
                (i64::from(rect.1) + dy) as u32,
                rect.2,
                rect.3,
            )
        };
        // Right, down, left, up: the step, the exit and the window ahead.
        let cases = [
            ((STEP, 0), (3031, 1199), (2688, 384, 2624, 3264)),
            ((0, STEP), (2359, 2063), (1664, 1536, 2624, 2464)),
            ((-STEP, 0), (1655, 1199), (640, 384, 2624, 3264)),
            ((0, -STEP), (2359, 367), (1664, 0, 2624, 2432)),
        ];
        for (step, exit, expected) in cases {
            let previous = at(visible, (-step.0, -step.1));
            let left = pan_exit(FULL, window, previous, visible).expect("the pan leaves");
            // The steps themselves: the first rectangle outside the window.
            let mut walked = visible;
            while holds(window, walked) {
                walked = at(walked, step);
            }
            assert_eq!(left, walked, "exit of the pan {step:?}");
            assert_eq!((left.0, left.1), exit, "exit of the pan {step:?}");
            let ahead =
                window_ahead(FULL, window, previous, visible, PAD, GRID).expect("a window ahead");
            assert_eq!(ahead, expected, "window ahead of the pan {step:?}");
            assert!(holds(ahead, left), "{ahead:?} holds the exit {left:?}");
            assert!(
                !holds(window, left),
                "the exit {left:?} is out of {window:?}"
            );
            let reach = |rect: PixelRect| match step {
                (dx, _) if dx > 0 => i64::from(rect.0 + rect.2),
                (dx, _) if dx < 0 => -i64::from(rect.0),
                (_, dy) if dy > 0 => i64::from(rect.1 + rect.3),
                _ => -i64::from(rect.1),
            };
            assert!(
                reach(ahead) > reach(window),
                "{ahead:?} reaches past {window:?} toward {step:?}"
            );
            // The cross axis is the window's own.
            if step.0 == 0 {
                assert_eq!((ahead.0, ahead.2), (window.0, window.2));
            } else {
                assert_eq!((ahead.1, ahead.3), (window.1, window.3));
            }
        }
        // No step, no window ahead; a pan toward an edge of the picture the
        // window already touches never leaves it.
        assert_eq!(
            window_ahead(FULL, window, visible, visible, PAD, GRID),
            None
        );
        let corner = padded_window(FULL, (0, 0, 1281, 1601), PAD, GRID);
        let still = (0, 0, 1281, 1601);
        assert_eq!(pan_exit(FULL, corner, (32, 0, 1281, 1601), still), None);
        assert_eq!(pan_exit(FULL, corner, (0, 32, 1281, 1601), still), None);
    }

    #[test]
    fn a_window_at_one_source_pixel_a_pixel_takes_one_tap() {
        // Every window of whole pixels of a photo 1100 wide, rendered one to
        // one: 586 of them read 1.0000001 source pixels a pixel in f32.
        for pixels in 1..=1100u32 {
            let window = CropRect {
                x: 0.0,
                y: 0.0,
                width: pixels as f32 / 1100.0,
                height: 1.0,
            };
            assert_eq!(
                input_taps((1100, 600), window, (pixels, 600)),
                1,
                "{pixels} pixels"
            );
        }
        // A render at half the size takes two, at a third three, and one a
        // little under the source's size two, as before.
        assert_eq!(input_taps((1100, 600), CropRect::FULL, (550, 300)), 2);
        assert_eq!(input_taps((1200, 600), CropRect::FULL, (400, 200)), 3);
        assert_eq!(input_taps((1100, 600), CropRect::FULL, (1000, 545)), 2);
        assert_eq!(input_taps((6000, 4000), CropRect::FULL, (60, 40)), MAX_TAPS);
    }
    use wgpu::naga;

    /// The size of the struct named `Uniform` in a shader and the offset of
    /// every member, as naga lays them out.
    fn wgsl_uniform(source: &str) -> (u32, Vec<(String, u32)>) {
        wgsl_struct(source, "Uniform")
    }

    /// The same for the struct of any name.
    fn wgsl_struct(source: &str, name: &str) -> (u32, Vec<(String, u32)>) {
        let module = naga::front::wgsl::parse_str(source).expect("the shader parses");
        for (_, ty) in module.types.iter() {
            if ty.name.as_deref() != Some(name) {
                continue;
            }
            if let naga::TypeInner::Struct { members, span } = &ty.inner {
                let offsets = members
                    .iter()
                    .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
                    .collect();
                return (*span, offsets);
            }
        }
        panic!("the shader has no struct named {name}");
    }

    #[test]
    fn the_develop_uniform_matches_the_wgsl_struct() {
        let (size, offsets) = wgsl_uniform(include_str!("shaders/develop.wgsl"));
        assert_eq!(size as usize, size_of::<DevelopUniform>());
        let offset = |name: &str| {
            offsets
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("no member {name}"))
                .1 as usize
        };
        assert_eq!(
            offset("white_balance"),
            offset_of!(DevelopUniform, white_balance)
        );
        assert_eq!(
            offset("white_balance_temperature"),
            offset_of!(DevelopUniform, white_balance_temperature)
        );
        assert_eq!(offset("saturation"), offset_of!(DevelopUniform, saturation));
        assert_eq!(
            offset("texture_amount"),
            offset_of!(DevelopUniform, texture)
        );
        assert_eq!(offset("clarity"), offset_of!(DevelopUniform, clarity));
        assert_eq!(offset("dehaze"), offset_of!(DevelopUniform, dehaze));
        assert_eq!(offset("flags"), offset_of!(DevelopUniform, flags));
        assert_eq!(offset("table_row"), offset_of!(DevelopUniform, table_row));
        assert_eq!(offset("opacity"), offset_of!(DevelopUniform, opacity));
        assert_eq!(offset("atmosphere"), offset_of!(DevelopUniform, atmosphere));
        assert_eq!(offset("cdl_slope"), offset_of!(DevelopUniform, cdl_slope));
        assert_eq!(offset("cdl_offset"), offset_of!(DevelopUniform, cdl_offset));
        assert_eq!(offset("cdl_power"), offset_of!(DevelopUniform, cdl_power));
        assert_eq!(offset("hsl"), offset_of!(DevelopUniform, hsl));
        assert_eq!(offsets.len(), 22, "a member was added without a check here");
        assert_eq!(size, 304, "the two new members took the old padding");
    }

    #[test]
    fn the_output_uniform_matches_the_wgsl_struct() {
        let (size, offsets) = wgsl_uniform(include_str!("shaders/output.wgsl"));
        assert_eq!(size as usize, size_of::<OutputUniform>());
        let expected = [
            ("matrix", offset_of!(OutputUniform, matrix)),
            ("crop", offset_of!(OutputUniform, crop)),
            ("overlay", offset_of!(OutputUniform, overlay)),
        ];
        assert_eq!(offsets.len(), expected.len());
        for ((name, offset), (wanted_name, wanted)) in offsets.iter().zip(expected) {
            assert_eq!(name, wanted_name);
            assert_eq!(*offset as usize, wanted, "{name}");
        }
    }

    #[test]
    fn the_mask_uniform_matches_the_wgsl_struct() {
        let source = include_str!("shaders/mask.wgsl");
        let (size, offsets) = wgsl_uniform(source);
        assert_eq!(size as usize, size_of::<MaskUniform>());
        let expected = [
            ("window", offset_of!(MaskUniform, window)),
            ("render_size", offset_of!(MaskUniform, render_size)),
            ("aspect", offset_of!(MaskUniform, aspect)),
            ("count", offset_of!(MaskUniform, count)),
            ("invert", offset_of!(MaskUniform, invert)),
            ("components", offset_of!(MaskUniform, components)),
        ];
        assert_eq!(offsets.len(), expected.len());
        for ((name, offset), (wanted_name, wanted)) in offsets.iter().zip(expected) {
            assert_eq!(name, wanted_name);
            assert_eq!(*offset as usize, wanted, "{name}");
        }
        let (size, offsets) = wgsl_struct(source, "Component");
        assert_eq!(size as usize, size_of::<MaskComponentUniform>());
        let expected = [
            ("header", offset_of!(MaskComponentUniform, header)),
            ("a", offset_of!(MaskComponentUniform, a)),
            ("b", offset_of!(MaskComponentUniform, b)),
        ];
        assert_eq!(offsets.len(), expected.len());
        for ((name, offset), (wanted_name, wanted)) in offsets.iter().zip(expected) {
            assert_eq!(name, wanted_name);
            assert_eq!(*offset as usize, wanted, "{name}");
        }
    }

    #[test]
    fn a_mask_uniform_carries_what_the_twin_computes_once() {
        let mut mask = Mask::new(
            "Both",
            MaskSource::Radial(RadialGradient {
                centre: [0.4, 0.6],
                radius: [0.3, 0.2],
                rotation: 90.0,
                feather: 25.0,
            }),
        );
        mask.invert = true;
        mask.components.push(Component {
            op: MaskOp::Intersect,
            source: MaskSource::Linear(LinearGradient {
                start: [0.1, 0.2],
                end: [0.3, 0.4],
            }),
            invert: true,
        });
        let geometry = Geometry {
            window: CropRect {
                x: 0.25,
                y: 0.0,
                width: 0.5,
                height: 1.0,
            },
            size: (200, 100),
            photo: (4000, 2000),
        };
        let uniform = MaskUniform::new(&mask, &geometry);
        assert_eq!(uniform.window, [0.25, 0.0, 0.5, 1.0]);
        assert_eq!(uniform.render_size, [200.0, 100.0]);
        assert_eq!(uniform.aspect, [1.0, 0.5]);
        assert_eq!((uniform.count, uniform.invert), (2, 1));
        let radial = uniform.components[0];
        assert_eq!(radial.header, [1, 0, 0, 0]);
        assert_eq!(radial.a, [0.4, 0.6, 0.3, 0.2]);
        assert!((radial.b[0] - 1.0).abs() < 1e-6 && radial.b[1].abs() < 1e-6);
        assert_eq!(radial.b[2], 0.75);
        let linear = uniform.components[1];
        assert_eq!(linear.header, [0, 2, 1, 0]);
        assert_eq!(linear.a, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(uniform.components[2].header, [0; 4]);
    }

    #[test]
    fn a_mask_pass_reads_its_own_row_and_runs_the_curves_of_either_side() {
        let global = Adjustments {
            exposure: 0.5,
            ..Adjustments::default()
        };
        let mut mask = Mask::new("Sky", MaskSource::default());
        mask.adjust.exposure = -1.25;
        mask.opacity = 40.0;
        let uniform = DevelopUniform::for_mask(&global, &mask, 3, [1.0; 3]);
        assert_eq!(uniform.table_row, 4);
        assert_eq!(uniform.opacity, 0.4);
        assert_eq!(uniform.exposure, -0.75, "the mask adds to the global edit");
        assert_eq!(uniform.flags & FLAG_CURVES, 0);
        mask.adjust.look.curves.red.points.insert(1, [0.5, 0.6]);
        let uniform = DevelopUniform::for_mask(&global, &mask, 3, [1.0; 3]);
        assert_eq!(uniform.flags & FLAG_CURVES, FLAG_CURVES);
        let plain = DevelopUniform::new(&global, [1.0; 3]);
        assert_eq!((plain.table_row, plain.opacity), (0, 1.0));
    }

    #[test]
    fn the_minimum_uniform_matches_the_wgsl_struct() {
        let (size, offsets) = wgsl_uniform(include_str!("shaders/minimum.wgsl"));
        assert_eq!(size as usize, size_of::<MinimumUniform>());
        assert_eq!(offsets[3], ("atmosphere".to_string(), 16));
    }

    /// The list after `name` in a shader, as numbers.
    fn wgsl_list(source: &str, name: &str) -> Vec<f32> {
        let start = source.find(name).expect("the constant is in the shader");
        let rest = &source[start..];
        let open = rest.find(">(").expect("a constructor follows") + 2;
        let close = rest[open..].find(')').expect("the constructor closes") + open;
        rest[open..close]
            .split(',')
            .map(|n| n.trim().parse().expect("a number"))
            .collect()
    }

    #[test]
    fn the_shader_hue_centres_equal_the_gamut_color_list() {
        let shader = wgsl_list(
            include_str!("shaders/develop.wgsl"),
            "const HUE_CENTRES: array<f32, 8> =",
        );
        let twin = gamut_color::hue::range_centres();
        assert_eq!(shader.len(), twin.len());
        for (k, (s, t)) in shader.iter().zip(twin).enumerate() {
            assert!(
                (s - t).abs() < 1e-6,
                "centre {k}: shader {s}, gamut-color {t}"
            );
        }
    }

    #[test]
    fn the_shader_constants_equal_the_gamut_color_constants() {
        let source = include_str!("shaders/develop.wgsl");
        let value = |name: &str| -> f32 {
            let start = source
                .find(&format!("const {name}: f32 = "))
                .unwrap_or_else(|| panic!("no constant {name}"));
            let rest = &source[start..];
            let from = rest.find("= ").expect("an equals sign") + 2;
            let to = rest.find(';').expect("a semicolon");
            rest[from..to].trim().parse().expect("a number")
        };
        use gamut_color::{acescct, dehaze, hsl, local, mask, wheels};
        assert_eq!(value("ACES_LINEAR_CUT"), acescct::LINEAR_CUT);
        assert_eq!(value("ACES_ENCODED_CUT"), acescct::ENCODED_CUT);
        assert_eq!(value("ACES_SLOPE"), acescct::SLOPE);
        assert_eq!(value("ACES_OFFSET"), acescct::OFFSET);
        assert_eq!(value("ACES_LOG_SHIFT"), acescct::LOG_SHIFT);
        assert_eq!(value("ACES_LOG_SCALE"), acescct::LOG_SCALE);
        assert_eq!(value("LOCAL_STRENGTH"), local::STRENGTH);
        assert_eq!(value("CLARITY_HALF_WIDTH"), local::CLARITY_HALF_WIDTH);
        assert_eq!(value("TRANSMISSION_FLOOR"), dehaze::TRANSMISSION_FLOOR);
        assert_eq!(value("RECOVERY_FLOOR"), dehaze::RECOVERY_FLOOR);
        assert_eq!(value("CHROMA_FLOOR"), hsl::CHROMA_FLOOR);
        assert_eq!(value("CHROMA_FULL"), hsl::CHROMA_FULL);
        assert_eq!(value("SHADOWS_END"), wheels::SHADOWS_END);
        assert_eq!(value("HIGHLIGHTS_START"), wheels::HIGHLIGHTS_START);

        // mask.wgsl repeats what it needs of acescct.rs and adds mask.rs.
        let source = include_str!("shaders/mask.wgsl");
        let value = |name: &str| -> f32 {
            let start = source
                .find(&format!("const {name}: f32 = "))
                .unwrap_or_else(|| panic!("no constant {name}"));
            let rest = &source[start..];
            let from = rest.find("= ").expect("an equals sign") + 2;
            let to = rest.find(';').expect("a semicolon");
            rest[from..to].trim().parse().expect("a number")
        };
        assert_eq!(value("ACES_LINEAR_CUT"), acescct::LINEAR_CUT);
        assert_eq!(value("ACES_SLOPE"), acescct::SLOPE);
        assert_eq!(value("ACES_OFFSET"), acescct::OFFSET);
        assert_eq!(value("ACES_LOG_SHIFT"), acescct::LOG_SHIFT);
        assert_eq!(value("ACES_LOG_SCALE"), acescct::LOG_SCALE);
        assert_eq!(value("CHROMA_RAMP"), mask::CHROMA_RAMP);
        assert_eq!(value("LINEAR_LENGTH_FLOOR"), mask::LINEAR_LENGTH_FLOOR);
        assert_eq!(value("DEGREES"), 1f32.to_degrees());

        // brush.wgsl repeats the same for the gate of an auto dab and adds
        // brush.rs.
        let source = include_str!("shaders/brush.wgsl");
        let value = |name: &str| -> f32 {
            let start = source
                .find(&format!("const {name}: f32 = "))
                .unwrap_or_else(|| panic!("no constant {name}"));
            let rest = &source[start..];
            let from = rest.find("= ").expect("an equals sign") + 2;
            let to = rest.find(';').expect("a semicolon");
            rest[from..to].trim().parse().expect("a number")
        };
        use gamut_color::brush;
        assert_eq!(value("ACES_LINEAR_CUT"), acescct::LINEAR_CUT);
        assert_eq!(value("ACES_SLOPE"), acescct::SLOPE);
        assert_eq!(value("ACES_OFFSET"), acescct::OFFSET);
        assert_eq!(value("ACES_LOG_SHIFT"), acescct::LOG_SHIFT);
        assert_eq!(value("ACES_LOG_SCALE"), acescct::LOG_SCALE);
        assert_eq!(value("GATE_CHROMA_WEIGHT"), brush::GATE_CHROMA_WEIGHT);
        assert_eq!(value("GATE_FALL"), brush::GATE_FALL);
        let mask = include_str!("shaders/mask.wgsl");
        for list in ["const LUMA", "const E1", "const E2"] {
            assert_eq!(wgsl_list(source, list), wgsl_list(mask, list), "{list}");
        }
    }

    #[test]
    fn the_proxy_uniform_matches_the_wgsl_struct() {
        use crate::proxy::ProxyUniform;
        let (size, offsets) = wgsl_uniform(include_str!("shaders/proxy.wgsl"));
        assert_eq!(size as usize, size_of::<ProxyUniform>());
        assert_eq!(offsets[0], ("source_size".to_string(), 0));
        assert_eq!(offsets[1], ("proxy_size".to_string(), 8));
        assert_eq!(offsets[2], ("tile_origin".to_string(), 16));
        assert_eq!(offset_of!(ProxyUniform, proxy_size), 8);
        assert_eq!(offset_of!(ProxyUniform, tile_origin), 16);
    }

    #[test]
    fn the_scissor_covers_the_crop_and_a_margin_inside_the_render() {
        let crop = CropRect {
            x: 0.25,
            y: 0.0,
            width: 0.5,
            height: 1.0,
        };
        assert_eq!(scissor_for(crop, (400, 200), true), (98, 0, 204, 200));
        assert_eq!(scissor_for(crop, (400, 200), false), (98, 0, 204, 200));
        assert_eq!(scissor_for(CropRect::FULL, (64, 64), false), (0, 0, 64, 64));
        let tiny = CropRect {
            x: 0.5,
            y: 0.5,
            width: 0.01,
            height: 0.01,
        };
        let (x, y, w, h) = scissor_for(tiny, (100, 100), false);
        assert!(x <= 50 && y <= 50 && w >= 1 && h >= 1 && x + w <= 100 && y + h <= 100);
    }
}
