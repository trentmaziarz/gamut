// Output transform: reads the crop out of the developed texture, takes
// linear Rec.2020 to linear sRGB and clips to 0 and 1. The sRGB target
// format applies the OETF on write. This is the only place values clip.
//
// The selected mask can be shown over the picture: red mixed in by the
// alpha of the mask (mask.rs in slate-color holds the twin). The overlay is
// a view of the mask for the person at the window; an export never sets it.

struct Uniform {
    matrix: mat3x3<f32>,
    crop: vec4<f32>,
    // x: how much red a full alpha mixes in, 0 when no overlay shows.
    overlay: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var developed: texture_2d<f32>;
@group(0) @binding(2) var developed_sampler: sampler;
// The alpha of the mask the overlay shows; one zero texel when none does.
@group(0) @binding(3) var overlay_alpha: texture_2d<f32>;

const OVERLAY_COLOUR: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index & 2u) * 2 - 1);
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = u.crop.xy + in.uv * u.crop.zw;
    let linear = textureSampleLevel(developed, developed_sampler, uv, 0.0).rgb;
    let srgb = clamp(u.matrix * linear, vec3<f32>(0.0), vec3<f32>(1.0));
    let alpha = textureSampleLevel(overlay_alpha, developed_sampler, uv, 0.0).r;
    return vec4<f32>(mix(srgb, OVERLAY_COLOUR, alpha * u.overlay.x), 1.0);
}
