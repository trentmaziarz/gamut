// The Basic panel on the GPU: the six operators in the ruled order on a
// linear Rec.2020 pixel. Every function mirrors its twin in slate-color's
// basic.rs line for line; the golden tests hold the two together. Nothing
// here clips.

struct Uniform {
    white_balance: mat3x3<f32>,
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
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var working: texture_2d<f32>;
@group(0) @binding(2) var base: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index & 2u) * 2 - 1);
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

const LUMA: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);
const GREY: f32 = 0.18;
const HIGHLIGHTS_SHADOWS_STOPS: f32 = 1.0;
const HIGHLIGHTS_RANGE: f32 = 3.0;
const SHADOWS_RANGE: f32 = 4.0;
const CONTRAST_STOPS: f32 = 1.0;
const CONTRAST_WIDTH: f32 = 2.0;
const WHITES_RANGE: f32 = 0.25;
const BLACKS_RANGE: f32 = 0.05;
const EPSILON: f32 = 1e-6;

fn luma(px: vec3<f32>) -> f32 {
    return dot(px, LUMA);
}

fn exposure(px: vec3<f32>, ev: f32) -> vec3<f32> {
    return px * exp2(ev);
}

fn highlights_shadows(px: vec3<f32>, base_luma: f32, highlights: f32, shadows: f32) -> vec3<f32> {
    let l = log2(max(base_luma, EPSILON) / GREY);
    let highlight_weight = smoothstep(0.0, HIGHLIGHTS_RANGE, l);
    let shadow_weight = smoothstep(0.0, SHADOWS_RANGE, -l);
    let stops = HIGHLIGHTS_SHADOWS_STOPS
        * (highlights / 100.0 * highlight_weight + shadows / 100.0 * shadow_weight);
    return px * exp2(stops);
}

fn whites_blacks(px: vec3<f32>, whites: f32, blacks: f32) -> vec3<f32> {
    let white = 1.0 - WHITES_RANGE * whites / 100.0;
    let black = -BLACKS_RANGE * blacks / 100.0;
    return (px - vec3<f32>(black)) / (white - black);
}

fn contrast(px: vec3<f32>, amount: f32) -> vec3<f32> {
    let l = log2(max(luma(px), EPSILON) / GREY);
    let push = amount / 100.0 * CONTRAST_STOPS * tanh(l / CONTRAST_WIDTH);
    return px * exp2(push);
}

fn vibrance_saturation(px: vec3<f32>, vibrance: f32, saturation: f32) -> vec3<f32> {
    let l = luma(px);
    let chroma = px - vec3<f32>(l);
    let normalised = clamp(length(chroma) / max(l, EPSILON), 0.0, 1.0);
    let gain = (1.0 + vibrance / 100.0 * (1.0 - normalised)) * (1.0 + saturation / 100.0);
    return vec3<f32>(l) + chroma * gain;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = vec2<i32>(in.position.xy);
    var px = textureLoad(working, at, 0).rgb;
    let base_luma = textureLoad(base, at, 0).r;
    px = u.white_balance * px;
    px = exposure(px, u.exposure);
    let base_after_exposure = base_luma * exp2(u.exposure);
    px = highlights_shadows(px, base_after_exposure, u.highlights, u.shadows);
    px = whites_blacks(px, u.whites, u.blacks);
    px = contrast(px, u.contrast);
    px = vibrance_saturation(px, u.vibrance, u.saturation);
    return vec4<f32>(px, 1.0);
}
