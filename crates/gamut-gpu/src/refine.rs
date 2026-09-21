//! Refine edges on the GPU: the pass family of `refine.wgsl`, the float
//! targets it works in, and the refined alpha of one mask.
//!
//! The arithmetic and the plan of a render (the side of a cell, the radius of
//! the box, eps) are gamut-color's [`Plan`]; this module only lays the work
//! out. The moments of the filter are differences of near-equal numbers, so
//! they live in Rgba32Float targets read with `textureLoad`. Those targets
//! belong to the frame and not to a mask, exist only once an edit holds a
//! refined mask, and hold one tile of the cell grid at a time, so their size
//! is bounded whatever the render: a render larger than a tile is refined a
//! tile after another. A default device draws into at most 32 bytes a sample,
//! which is two of them a pass.
//!
//! A tile is drawn in ten passes: the moments in two, their box means across
//! and down in two each, the solve, the box means of `a` and `b` across and
//! down, and the apply, which alone runs at full resolution and writes the
//! r8unorm refined alpha.

use bytemuck::{Pod, Zeroable};
use gamut_color::refine::Plan;

use crate::{FULLSCREEN_VERTICES, fullscreen_primitive};

/// The format of the float targets.
pub(crate) const MOMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// The most cells a side the float targets hold, unless one box needs more.
pub(crate) const TILE_CELLS: u32 = 768;

/// A rectangle of pixels: x, y, width, height.
pub(crate) type Rect = (u32, u32, u32, u32);

/// Mirrors the `Uniform` of `refine.wgsl`; a test holds the two layouts equal.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct RefineUniform {
    pub(crate) origin: [u32; 2],
    pub(crate) size: [u32; 2],
    pub(crate) grid_first: [u32; 2],
    pub(crate) grid_count: [u32; 2],
    pub(crate) tile_first: [u32; 2],
    pub(crate) tile_count: [u32; 2],
    pub(crate) step: u32,
    pub(crate) cells: u32,
    pub(crate) eps: f32,
    pub(crate) amount: f32,
}

/// One tile of a refine: the pixels of the render it writes and the cells
/// the float targets hold while it is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tile {
    /// The pixels of the render the apply pass writes.
    pub(crate) out: Rect,
    /// The first cell held and how many, on each axis.
    pub(crate) first: (u32, u32),
    pub(crate) count: (u32, u32),
}

impl Tile {
    pub(crate) fn uniform(&self, plan: &Plan) -> RefineUniform {
        let (grid_first, grid_count) = plan.grid();
        RefineUniform {
            origin: [plan.origin.0, plan.origin.1],
            size: [plan.size.0, plan.size.1],
            grid_first: [grid_first.0, grid_first.1],
            grid_count: [grid_count.0, grid_count.1],
            tile_first: [self.first.0, self.first.1],
            tile_count: [self.count.0, self.count.1],
            step: plan.step,
            cells: plan.cells,
            eps: plan.eps,
            amount: plan.amount,
        }
    }
}

/// The most cells a side a tile of this plan may hold: [`TILE_CELLS`], or
/// what one box and a few pixels around it need when that is more.
pub(crate) fn tile_side(plan: &Plan) -> u32 {
    TILE_CELLS.max(margin(plan) * 2 + 64)
}

/// The cells a tile holds on each side of the ones its pixels lie among: the
/// box of the moments and the box of `a` and `b`.
fn margin(plan: &Plan) -> u32 {
    2 * plan.cells
}

/// The cells one axis of a span of pixels of the render needs: the cells its
/// pixels lie among (the one before the first centre, the one after the
/// last) and the margin around them, inside the grid. First cell and count.
fn cells_for(plan: &Plan, origin: u32, start: u32, length: u32, grid: (u32, u32)) -> (u32, u32) {
    let step = plan.step;
    // The cell whose centre lies at or before the centre of a pixel: the
    // twin's floor((X + 0.5) / step - 0.5), which is -1 before the first
    // centre and held at the grid there.
    let before = |pixel: u32| (2 * (origin + pixel) + 1).saturating_sub(step) / (2 * step);
    let low = before(start).saturating_sub(margin(plan)).max(grid.0);
    let high = (before(start + length - 1) + 1 + margin(plan)).min(grid.0 + grid.1 - 1);
    (low, high.max(low) - low + 1)
}

/// The tiles that refine `over` of a render, each inside `side` cells a side.
pub(crate) fn tiles(plan: &Plan, over: Rect, side: u32) -> Vec<Tile> {
    let (grid_first, grid_count) = plan.grid();
    // The pixels a tile may write along one axis so that its cells fit: the
    // margin on both sides and the two cells around the span come off.
    let span = (side.saturating_sub(2 * margin(plan) + 3)).max(1) * plan.step;
    let cuts = |start: u32, length: u32| -> Vec<(u32, u32)> {
        (0..length.div_ceil(span))
            .map(|k| (start + k * span, span.min(length - k * span)))
            .collect()
    };
    let mut out = Vec::new();
    for (y, height) in cuts(over.1, over.3) {
        for (x, width) in cuts(over.0, over.2) {
            let (first_x, columns) =
                cells_for(plan, plan.origin.0, x, width, (grid_first.0, grid_count.0));
            let (first_y, rows) =
                cells_for(plan, plan.origin.1, y, height, (grid_first.1, grid_count.1));
            out.push(Tile {
                out: (x, y, width, height),
                first: (first_x, first_y),
                count: (columns, rows),
            });
        }
    }
    out
}

struct FloatTarget {
    view: wgpu::TextureView,
}

/// The float targets of a frame: the moments in `m`, and `n` for the other
/// side of every box mean, for `a` and `b` and for theirs.
pub(crate) struct Scratch {
    m: [FloatTarget; 4],
    n: [FloatTarget; 4],
    size: (u32, u32),
    /// Told apart from every scratch before it, for the bind groups that
    /// hold its views.
    id: u64,
}

/// The refined alpha of one mask and what draws it.
pub(crate) struct Refined {
    pub(crate) alpha: wgpu::TextureView,
    uniform: wgpu::Buffer,
    /// How many tiles the uniform buffer holds.
    slots: u32,
    /// The bind groups, and the scratch they were made for.
    binds: Option<(u64, Binds)>,
}

/// One bind group for each set of textures a pass reads. A texture is never
/// in the group of a pass that writes it.
struct Binds {
    /// n0 to n3: the moments passes and the box down of the first pair.
    n: wgpu::BindGroup,
    /// n2, n3, n0, n1: the box down of the second pair.
    n_second: wgpu::BindGroup,
    /// m0 to m3: the box across of the first pair, and the solve.
    m: wgpu::BindGroup,
    /// m2, m3, m0, m1: the box across of the second pair.
    m_second: wgpu::BindGroup,
    /// n0 first: the box across of `a` and `b`, and the apply.
    ab: wgpu::BindGroup,
    /// n1 first: the box down of `a` and `b`.
    ab_across: wgpu::BindGroup,
}

/// The pipelines of `refine.wgsl`.
pub(crate) struct RefinePass {
    layout: wgpu::BindGroupLayout,
    moments_a: wgpu::RenderPipeline,
    moments_b: wgpu::RenderPipeline,
    box_h2: wgpu::RenderPipeline,
    box_v2: wgpu::RenderPipeline,
    solve: wgpu::RenderPipeline,
    box_h1: wgpu::RenderPipeline,
    box_v1: wgpu::RenderPipeline,
    apply: wgpu::RenderPipeline,
    /// The distance between two tiles in the uniform buffer.
    stride: u32,
    scratches: u64,
}

const SHADER: &str = include_str!("shaders/refine.wgsl");

impl RefinePass {
    pub(crate) fn new(device: &wgpu::Device, alpha_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("refine"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let size = std::mem::size_of::<RefineUniform>() as u64;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("refine"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(size),
                    },
                    count: None,
                },
                texture(1),
                texture(2),
                texture(3),
                texture(4),
                texture(5),
                texture(6),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("refine"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |fragment: &str, formats: &[wgpu::TextureFormat]| {
            let targets: Vec<Option<wgpu::ColorTargetState>> = formats
                .iter()
                .map(|format| {
                    Some(wgpu::ColorTargetState {
                        format: *format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                })
                .collect();
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
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
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pair = [MOMENT_FORMAT; 2];
        let one = [MOMENT_FORMAT];
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        RefinePass {
            moments_a: pipeline("fs_moments_a", &pair),
            moments_b: pipeline("fs_moments_b", &pair),
            box_h2: pipeline("fs_box_h2", &pair),
            box_v2: pipeline("fs_box_v2", &pair),
            solve: pipeline("fs_solve", &one),
            box_h1: pipeline("fs_box_h1", &one),
            box_v1: pipeline("fs_box_v1", &one),
            apply: pipeline("fs_apply", &[alpha_format]),
            layout,
            stride: (size as u32).div_ceil(alignment) * alignment,
            scratches: 0,
        }
    }

    /// The refined alpha of a mask on a render of this size, not drawn yet.
    pub(crate) fn refined(
        &self,
        device: &wgpu::Device,
        alpha_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Refined {
        Refined {
            alpha: target(device, "refined alpha", alpha_format, width, height).view,
            uniform: self.uniform_buffer(device, 1),
            slots: 1,
            binds: None,
        }
    }

    fn uniform_buffer(&self, device: &wgpu::Device, slots: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("refine uniforms"),
            size: u64::from(self.stride) * u64::from(slots),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Draws the refined alpha of one mask over `over` of the render, from
    /// its finished alpha and the working texture. `scratch` is the frame's:
    /// it is made here when there is none or when the plan needs a larger
    /// one. How many tiles were drawn.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scratch: &mut Option<Scratch>,
        refined: &mut Refined,
        working: &wgpu::TextureView,
        alpha: &wgpu::TextureView,
        plan: &Plan,
        over: Rect,
    ) -> u32 {
        let (_, grid) = plan.grid();
        let side = tile_side(plan);
        let wanted = (grid.0.min(side), grid.1.min(side));
        if scratch
            .as_ref()
            .is_none_or(|held| held.size.0 < wanted.0 || held.size.1 < wanted.1)
        {
            let size = scratch.as_ref().map_or(wanted, |held| {
                (held.size.0.max(wanted.0), held.size.1.max(wanted.1))
            });
            self.scratches += 1;
            let float = |label: &str| FloatTarget {
                view: target(device, label, MOMENT_FORMAT, size.0, size.1).view,
            };
            *scratch = Some(Scratch {
                m: [0; 4].map(|_| float("refine moments")),
                n: [0; 4].map(|_| float("refine means")),
                size,
                id: self.scratches,
            });
        }
        let scratch = scratch.as_ref().expect("made above");
        let tiles = tiles(plan, over, side);
        if refined.slots < tiles.len() as u32 {
            refined.slots = (tiles.len() as u32).next_power_of_two();
            refined.uniform = self.uniform_buffer(device, refined.slots);
            refined.binds = None;
        }
        if refined
            .binds
            .as_ref()
            .is_none_or(|(id, _)| *id != scratch.id)
        {
            let group = |label: &str, textures: [&FloatTarget; 4]| {
                fn view(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
                    wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::TextureView(view),
                    }
                }
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &refined.uniform,
                                offset: 0,
                                size: wgpu::BufferSize::new(
                                    std::mem::size_of::<RefineUniform>() as u64
                                ),
                            }),
                        },
                        view(1, working),
                        view(2, alpha),
                        view(3, &textures[0].view),
                        view(4, &textures[1].view),
                        view(5, &textures[2].view),
                        view(6, &textures[3].view),
                    ],
                })
            };
            let (m, n) = (&scratch.m, &scratch.n);
            refined.binds = Some((
                scratch.id,
                Binds {
                    n: group("refine n", [&n[0], &n[1], &n[2], &n[3]]),
                    n_second: group("refine n second", [&n[2], &n[3], &n[0], &n[1]]),
                    m: group("refine m", [&m[0], &m[1], &m[2], &m[3]]),
                    m_second: group("refine m second", [&m[2], &m[3], &m[0], &m[1]]),
                    ab: group("refine ab", [&n[0], &m[1], &m[2], &m[3]]),
                    ab_across: group("refine ab across", [&n[1], &m[1], &m[2], &m[3]]),
                },
            ));
        }
        let (_, binds) = refined.binds.as_ref().expect("made above");
        let (m, n) = (&scratch.m, &scratch.n);
        for (slot, tile) in tiles.iter().enumerate() {
            let offset = slot as u32 * self.stride;
            queue.write_buffer(
                &refined.uniform,
                u64::from(offset),
                bytemuck::bytes_of(&tile.uniform(plan)),
            );
            let cells = (0, 0, tile.count.0, tile.count.1);
            let mut pass = |label: &str,
                            pipeline: &wgpu::RenderPipeline,
                            bind: &wgpu::BindGroup,
                            targets: &[&wgpu::TextureView],
                            area: Rect| {
                draw(encoder, label, pipeline, bind, offset, targets, area);
            };
            fn pair<'a>(a: &'a FloatTarget, b: &'a FloatTarget) -> [&'a wgpu::TextureView; 2] {
                [&a.view, &b.view]
            }
            pass(
                "refine moments a",
                &self.moments_a,
                &binds.n,
                &pair(&m[0], &m[1]),
                cells,
            );
            pass(
                "refine moments b",
                &self.moments_b,
                &binds.n,
                &pair(&m[2], &m[3]),
                cells,
            );
            pass(
                "refine box h a",
                &self.box_h2,
                &binds.m,
                &pair(&n[0], &n[1]),
                cells,
            );
            pass(
                "refine box h b",
                &self.box_h2,
                &binds.m_second,
                &pair(&n[2], &n[3]),
                cells,
            );
            pass(
                "refine box v a",
                &self.box_v2,
                &binds.n,
                &pair(&m[0], &m[1]),
                cells,
            );
            pass(
                "refine box v b",
                &self.box_v2,
                &binds.n_second,
                &pair(&m[2], &m[3]),
                cells,
            );
            pass("refine solve", &self.solve, &binds.m, &[&n[0].view], cells);
            pass("refine ab h", &self.box_h1, &binds.ab, &[&n[1].view], cells);
            pass(
                "refine ab v",
                &self.box_v1,
                &binds.ab_across,
                &[&n[0].view],
                cells,
            );
            pass(
                "refine apply",
                &self.apply,
                &binds.ab,
                &[&refined.alpha],
                tile.out,
            );
        }
        tiles.len() as u32
    }
}

struct MadeTarget {
    view: wgpu::TextureView,
}

fn target(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> MadeTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    MadeTarget {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
    }
}

/// One pass over `area` of its targets, which keep what they hold outside
/// it. Every pass of the filter writes every pixel of its area.
fn draw(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind: &wgpu::BindGroup,
    offset: u32,
    targets: &[&wgpu::TextureView],
    area: Rect,
) {
    let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = targets
        .iter()
        .copied()
        .map(|view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })
        })
        .collect();
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[offset]);
    pass.set_scissor_rect(area.0, area.1, area.2, area.3);
    pass.draw(0..FULLSCREEN_VERTICES, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamut_color::acescct;
    use gamut_color::refine::SOLVED;
    use gamut_core::mask::Refine;
    use std::mem::offset_of;
    use wgpu::naga;

    fn plan(radius: f32, full: (u32, u32), origin: (u32, u32), size: (u32, u32)) -> Plan {
        let refine = Refine {
            amount: 100.0,
            radius,
            sensitivity: 50.0,
        };
        Plan::new(&refine, full, origin, size)
    }

    #[test]
    fn the_refine_uniform_matches_the_wgsl_struct() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("the shader parses");
        let (span, members) = module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { members, span }
                    if ty.name.as_deref() == Some("Uniform") =>
                {
                    Some((*span, members.clone()))
                }
                _ => None,
            })
            .expect("the shader has a Uniform");
        assert_eq!(span as usize, size_of::<RefineUniform>());
        let offset = |name: &str| {
            members
                .iter()
                .find(|m| m.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("no member {name}"))
                .offset as usize
        };
        assert_eq!(offset("origin"), offset_of!(RefineUniform, origin));
        assert_eq!(offset("size"), offset_of!(RefineUniform, size));
        assert_eq!(offset("grid_first"), offset_of!(RefineUniform, grid_first));
        assert_eq!(offset("grid_count"), offset_of!(RefineUniform, grid_count));
        assert_eq!(offset("tile_first"), offset_of!(RefineUniform, tile_first));
        assert_eq!(offset("tile_count"), offset_of!(RefineUniform, tile_count));
        assert_eq!(offset("step"), offset_of!(RefineUniform, step));
        assert_eq!(offset("cells"), offset_of!(RefineUniform, cells));
        assert_eq!(offset("eps"), offset_of!(RefineUniform, eps));
        assert_eq!(offset("amount"), offset_of!(RefineUniform, amount));
        assert_eq!(members.len(), 10);
    }

    /// The value of `const NAME: f32 = value;` in the shader.
    fn wgsl_constant(name: &str) -> f32 {
        let start = SHADER
            .find(&format!("const {name}: f32 = "))
            .unwrap_or_else(|| panic!("no constant {name}"));
        let text = &SHADER[start..];
        let value = &text[text.find("= ").expect("an equals sign") + 2..];
        value[..value.find(';').expect("a semicolon")]
            .parse()
            .expect("a number")
    }

    #[test]
    fn the_shader_constants_equal_the_gamut_color_constants() {
        assert_eq!(wgsl_constant("ACES_LINEAR_CUT"), acescct::LINEAR_CUT);
        assert_eq!(wgsl_constant("ACES_SLOPE"), acescct::SLOPE);
        assert_eq!(wgsl_constant("ACES_OFFSET"), acescct::OFFSET);
        assert_eq!(wgsl_constant("ACES_LOG_SHIFT"), acescct::LOG_SHIFT);
        assert_eq!(wgsl_constant("ACES_LOG_SCALE"), acescct::LOG_SCALE);
        // What the twin solves a cell fits four float targets, which two
        // passes of two targets fill: 32 bytes a sample, the most a default
        // device draws into.
        assert_eq!(SOLVED.div_ceil(4), 4, "four targets of four channels");
        assert_eq!(MOMENT_FORMAT.target_pixel_byte_cost(), Some(16));
    }

    #[test]
    fn a_render_that_fits_is_one_tile_of_its_whole_grid() {
        let plan = plan(0.01, (1280, 1600), (0, 0), (1280, 1600));
        assert_eq!((plan.step, plan.cells), (4, 4));
        let tiles = tiles(&plan, (0, 0, 1280, 1600), tile_side(&plan));
        assert_eq!(
            tiles,
            vec![Tile {
                out: (0, 0, 1280, 1600),
                first: (0, 0),
                count: (320, 400),
            }]
        );
        let uniform = tiles[0].uniform(&plan);
        assert_eq!(uniform.grid_count, [320, 400]);
        assert_eq!(uniform.tile_count, [320, 400]);
        assert_eq!((uniform.step, uniform.cells), (4, 4));
    }

    /// Every pixel of what is refined is written by exactly one tile, and a
    /// tile holds every cell the twin reads for its pixels: the two cells a
    /// pixel lies among, the box of `a` and `b` around them, and the box of
    /// the moments around that, inside the grid of the render.
    #[test]
    fn the_tiles_write_every_pixel_once_and_hold_every_cell_they_read() {
        let window = ((6000, 4000), (1021, 513), (3104, 2005));
        let cases = [
            // Cells of one pixel on a large window: many tiles.
            (
                plan(0.0012, window.0, window.1, window.2),
                (3, 2, 3098, 2001),
            ),
            // The default radius at 100 percent.
            (plan(0.01, window.0, window.1, window.2), (3, 2, 3098, 2001)),
            // The widest box: 75 cells either side.
            (
                plan(0.05, (6000, 4000), (0, 0), (6000, 4000)),
                (0, 0, 6000, 4000),
            ),
            // A patch under new dabs.
            (
                plan(0.02, window.0, window.1, window.2),
                (1500, 700, 301, 277),
            ),
        ];
        for (plan, over) in cases {
            let side = tile_side(&plan);
            let tiles = tiles(&plan, over, side);
            let (grid_first, grid_count) = plan.grid();
            let mut written = vec![0u8; (over.2 * over.3) as usize];
            for tile in &tiles {
                assert!(tile.count.0 <= side && tile.count.1 <= side, "{tile:?}");
                for y in tile.out.1..tile.out.1 + tile.out.3 {
                    for x in tile.out.0..tile.out.0 + tile.out.2 {
                        written[((y - over.1) * over.2 + x - over.0) as usize] += 1;
                    }
                }
                // Start and length of the pixels, the origin of the render,
                // the grid, and what the tile holds, on each axis.
                let axes = [
                    (
                        (tile.out.0, tile.out.2, plan.origin.0),
                        (grid_first.0, grid_count.0),
                        (tile.first.0, tile.count.0),
                    ),
                    (
                        (tile.out.1, tile.out.3, plan.origin.1),
                        (grid_first.1, grid_count.1),
                        (tile.first.1, tile.count.1),
                    ),
                ];
                for ((start, length, origin), (first, count), (held_first, held_count)) in axes {
                    for pixel in [start, start + length - 1] {
                        // The twin's place of a pixel among the cell centres.
                        let place = ((origin + pixel) as f32 + 0.5) / plan.step as f32 - 0.5;
                        let low = place.floor() as i64;
                        let reach = 2 * i64::from(plan.cells);
                        let clamp =
                            |cell: i64| cell.clamp(i64::from(first), i64::from(first + count - 1));
                        let (need_low, need_high) = (clamp(low - reach), clamp(low + 1 + reach));
                        assert!(
                            i64::from(held_first) <= need_low
                                && need_high <= i64::from(held_first + held_count - 1),
                            "{tile:?}: pixel {pixel} reads cells {need_low} to {need_high}"
                        );
                    }
                }
            }
            assert!(written.iter().all(|count| *count == 1), "{plan:?}");
            println!("{} tiles of at most {side} cells a side", tiles.len());
        }
    }

    #[test]
    fn a_box_wider_than_a_tile_gets_a_tile_that_holds_it() {
        // A picture of 40,000 pixels: a radius of 0.05 is 2,000 pixels, 500
        // cells either side for each of the two boxes.
        let plan = plan(0.05, (40_000, 30_000), (0, 0), (40_000, 30_000));
        assert_eq!((plan.step, plan.cells), (4, 500));
        assert!(tile_side(&plan) > TILE_CELLS);
        let tiles = tiles(&plan, (0, 0, 4000, 3000), tile_side(&plan));
        assert!(tiles.iter().all(|tile| tile.out.2 >= 4 && tile.out.3 >= 4));
    }
}
