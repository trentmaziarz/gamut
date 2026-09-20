// The video pass: samples a decoded frame's luma and chroma planes at the
// render size, turns the frame by its display rotation, expands the range,
// applies the YCbCr matrix, decodes the transfer and writes linear
// Rec.2020. Every step mirrors gamut-color's video.rs line for line.
// The HLG branch is provisional: the inverse OETF and a clip at 1.0, no
// tone map, until the BT.2390 pass lands.

struct Uniform {
    yuv: mat3x3<f32>,
    primaries: mat3x3<f32>,
    black: f32,
    luma_range: f32,
    chroma_range: f32,
    mid: f32,
    transfer: u32,
    rotation: u32,
    full_range: u32,
    _pad: u32,
    // The part of the displayed frame this render covers, as x, y, width,
    // height in 0 to 1; the whole frame is 0, 0, 1, 1.
    window: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var luma: texture_2d<f32>;
@group(0) @binding(2) var chroma: texture_2d<f32>;
@group(0) @binding(3) var plane_sampler: sampler;

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

fn srgb_eotf(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(low, high, c > vec3<f32>(0.04045));
}

const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.55991073;

fn hlg_inverse_oetf(c: vec3<f32>) -> vec3<f32> {
    let low = c * c / 3.0;
    let high = (exp((c - vec3<f32>(HLG_C)) / HLG_A) + vec3<f32>(HLG_B)) / 12.0;
    return min(select(low, high, c > vec3<f32>(0.5)), vec3<f32>(1.0));
}

// The stored frame turns clockwise by the rotation to be displayed, so a
// display position maps back onto the stored planes.
fn source_uv(uv: vec2<f32>) -> vec2<f32> {
    switch u.rotation {
        case 90u: {
            return vec2<f32>(uv.y, 1.0 - uv.x);
        }
        case 180u: {
            return vec2<f32>(1.0 - uv.x, 1.0 - uv.y);
        }
        case 270u: {
            return vec2<f32>(1.0 - uv.y, uv.x);
        }
        default: {
            return uv;
        }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = source_uv(u.window.xy + in.uv * u.window.zw);
    let y = textureSampleLevel(luma, plane_sampler, at, 0.0).r;
    let c = textureSampleLevel(chroma, plane_sampler, at, 0.0).rg;
    let yp = (y - u.black) / u.luma_range;
    let cb = (c.r - u.mid) / u.chroma_range;
    let cr = (c.g - u.mid) / u.chroma_range;
    let rgb = clamp(u.yuv * vec3<f32>(yp, cb, cr), vec3<f32>(0.0), vec3<f32>(1.0));
    var linear: vec3<f32>;
    if (u.transfer == 1u) {
        linear = hlg_inverse_oetf(rgb);
    } else {
        linear = srgb_eotf(rgb);
    }
    return vec4<f32>(u.primaries * linear, 1.0);
}
