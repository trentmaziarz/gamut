// The develop chain on the GPU in the ruled order on a linear Rec.2020
// pixel: white balance, exposure, highlights and shadows, whites and blacks,
// contrast, texture and clarity, dehaze; then in ACEScct the tone curves,
// the HSL mixer and the colour wheels; then back in linear light vibrance
// and saturation. Every function mirrors its twin in slate-color line for
// line (basic.rs, local.rs, dehaze.rs, acescct.rs, curve.rs, hue.rs, hsl.rs,
// wheels.rs); the golden tests hold the two together. Every operator is
// skipped at its neutral value. Nothing here clips.

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
    texture_amount: f32,
    clarity: f32,
    dehaze: f32,
    // Bit 0: the tone curves run. Bit 1: the mixer runs. Bit 2: the wheels run.
    flags: u32,
    // The atmospheric light after the white balance and the exposure.
    atmosphere: vec4<f32>,
    cdl_slope: vec4<f32>,
    cdl_offset: vec4<f32>,
    cdl_power: vec4<f32>,
    // Per range: the hue turn in radians, the chroma scale, the luminance
    // offset in ACEScct units, a spare.
    hsl: array<vec4<f32>, 8>,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var working: texture_2d<f32>;
@group(0) @binding(2) var base: texture_2d<f32>;
// The texture layer: luminance under the finer gaussian.
@group(0) @binding(3) var texture_base: texture_2d<f32>;
// The transmission map of dehaze.
@group(0) @binding(4) var transmission_map: texture_2d<f32>;
// The baked tone curves: 1024 by 1, red, green and blue tables.
@group(0) @binding(5) var curve_table: texture_2d<f32>;

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

const FLAG_CURVES: u32 = 1u;
const FLAG_HSL: u32 = 2u;
const FLAG_WHEELS: u32 = 4u;

// local.rs
const LOCAL_STRENGTH: f32 = 0.5;
const CLARITY_HALF_WIDTH: f32 = 3.0;

// dehaze.rs
const TRANSMISSION_FLOOR: f32 = 0.01;
const RECOVERY_FLOOR: f32 = 0.1;

// acescct.rs
const ACES_LINEAR_CUT: f32 = 0.0078125;
const ACES_ENCODED_CUT: f32 = 0.15525114;
const ACES_SLOPE: f32 = 10.540237;
const ACES_OFFSET: f32 = 0.07290553;
const ACES_LOG_SHIFT: f32 = 9.72;
const ACES_LOG_SCALE: f32 = 17.52;
const ACES_BLACK: f32 = ACES_OFFSET;
const ACES_WHITE: f32 = ACES_LOG_SHIFT / ACES_LOG_SCALE;

// curve.rs
const TABLE_SIZE: i32 = 1024;

// hue.rs
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;
const E1: vec3<f32> = vec3<f32>(0.8164966, -0.4082483, -0.4082483);
const E2: vec3<f32> = vec3<f32>(0.0, 0.70710677, -0.70710677);
// The hue angles of sRGB red, orange, yellow, green, aqua, blue, purple and
// magenta through the working transform. slate-color computes the same list
// in hue::range_centres and a test in slate-gpu holds the two equal.
const HUE_CENTRES: array<f32, 8> = array<f32, 8>(0.40264693, 0.726648, 1.0599453, 1.4989543, 3.192117, 4.492467, 4.761667, 5.119506);

// hsl.rs
const CHROMA_FLOOR: f32 = 0.01;
const CHROMA_FULL: f32 = 0.012;

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

fn midtone_bell(exposed_base: f32) -> f32 {
    let l = log2(max(exposed_base, EPSILON) / GREY);
    return 1.0 - smoothstep(0.0, CLARITY_HALF_WIDTH, abs(l));
}

fn local_gain(
    input_luma: f32,
    base_luma: f32,
    texture_luma: f32,
    exposed_base: f32,
    texture_amount: f32,
    clarity: f32,
) -> f32 {
    let l = max(input_luma, EPSILON);
    let clarity_power = clarity / 100.0 * LOCAL_STRENGTH * midtone_bell(exposed_base);
    let texture_power = texture_amount / 100.0 * LOCAL_STRENGTH;
    return pow(l / max(base_luma, EPSILON), clarity_power)
        * pow(l / max(texture_luma, EPSILON), texture_power);
}

fn recover(px: vec3<f32>, transmission: f32, atmosphere: vec3<f32>, dehaze: f32) -> vec3<f32> {
    let divisor = max(pow(max(transmission, TRANSMISSION_FLOOR), dehaze / 100.0), RECOVERY_FLOOR);
    return (px - atmosphere) / divisor + atmosphere;
}

fn acescct_encode(lin: vec3<f32>) -> vec3<f32> {
    let linear_part = ACES_SLOPE * lin + vec3<f32>(ACES_OFFSET);
    let log_part = (log2(max(lin, vec3<f32>(ACES_LINEAR_CUT))) + vec3<f32>(ACES_LOG_SHIFT))
        / ACES_LOG_SCALE;
    return select(log_part, linear_part, lin <= vec3<f32>(ACES_LINEAR_CUT));
}

fn acescct_decode(v: vec3<f32>) -> vec3<f32> {
    let linear_part = (v - vec3<f32>(ACES_OFFSET)) / ACES_SLOPE;
    let log_part = exp2(v * ACES_LOG_SCALE - vec3<f32>(ACES_LOG_SHIFT));
    return select(log_part, linear_part, v <= vec3<f32>(ACES_ENCODED_CUT));
}

fn curve_lookup(x: f32, channel: i32) -> f32 {
    if (x < 0.0) {
        return textureLoad(curve_table, vec2<i32>(0, 0), 0)[channel] + x;
    }
    if (x > 1.0) {
        return textureLoad(curve_table, vec2<i32>(TABLE_SIZE - 1, 0), 0)[channel] + (x - 1.0);
    }
    let position = x * f32(TABLE_SIZE - 1);
    let i = min(i32(floor(position)), TABLE_SIZE - 2);
    let f = position - f32(i);
    let a = textureLoad(curve_table, vec2<i32>(i, 0), 0)[channel];
    let b = textureLoad(curve_table, vec2<i32>(i + 1, 0), 0)[channel];
    return a * (1.0 - f) + b * f;
}

fn curves(v: vec3<f32>) -> vec3<f32> {
    let span = ACES_WHITE - ACES_BLACK;
    let n = (v - vec3<f32>(ACES_BLACK)) / span;
    let out = vec3<f32>(curve_lookup(n.r, 0), curve_lookup(n.g, 1), curve_lookup(n.b, 2));
    return out * span + vec3<f32>(ACES_BLACK);
}

fn wrap_angle(angle: f32) -> f32 {
    let wrapped = angle - TAU * floor(angle / TAU);
    return select(wrapped, 0.0, wrapped >= TAU);
}

fn from_plane(p: vec2<f32>) -> vec3<f32> {
    return p.x * E1 + p.y * E2;
}

fn hsl(v: vec3<f32>) -> vec3<f32> {
    let plane = vec2<f32>(dot(v, E1), dot(v, E2));
    let chroma = length(plane);
    let strength = smoothstep(CHROMA_FLOOR, CHROMA_FULL, chroma);
    if (strength <= 0.0) {
        return v;
    }
    let hue = wrap_angle(atan2(plane.y, plane.x));
    var centres = HUE_CENTRES;
    var lower = 0;
    var nearest = TAU;
    for (var k = 0; k < 8; k = k + 1) {
        let d = wrap_angle(hue - centres[k]);
        if (d < nearest) {
            nearest = d;
            lower = k;
        }
    }
    let upper = (lower + 1) % 8;
    let span = wrap_angle(centres[upper] - centres[lower]);
    let t = min(nearest / span, 1.0);
    let weight = 0.5 - 0.5 * cos(PI * t);
    let mixed = u.hsl[lower] * (1.0 - weight) + u.hsl[upper] * weight;
    let turn = mixed.x * strength;
    let scale = 1.0 + (mixed.y - 1.0) * strength;
    let offset = mixed.z * strength;

    let s = sin(turn);
    let c = cos(turn);
    let turned = vec2<f32>(plane.x * c - plane.y * s, plane.x * s + plane.y * c) * scale;
    let coloured = from_plane(turned);
    let grey = luma(v) - luma(coloured) + offset;
    return coloured + vec3<f32>(grey);
}

fn wheels(v: vec3<f32>) -> vec3<f32> {
    return pow(max(v * u.cdl_slope.rgb + u.cdl_offset.rgb, vec3<f32>(0.0)), u.cdl_power.rgb);
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
    let input_luma = luma(px);
    px = u.white_balance * px;
    px = exposure(px, u.exposure);
    let base_after_exposure = base_luma * exp2(u.exposure);
    px = highlights_shadows(px, base_after_exposure, u.highlights, u.shadows);
    px = whites_blacks(px, u.whites, u.blacks);
    px = contrast(px, u.contrast);
    if (u.texture_amount != 0.0 || u.clarity != 0.0) {
        var texture_luma = input_luma;
        if (u.texture_amount != 0.0) {
            texture_luma = textureLoad(texture_base, at, 0).r;
        }
        px = px * local_gain(input_luma, base_luma, texture_luma, base_after_exposure, u.texture_amount, u.clarity);
    }
    if (u.dehaze != 0.0) {
        let transmission = textureLoad(transmission_map, at, 0).r;
        px = recover(px, transmission, u.atmosphere.rgb, u.dehaze);
    }
    if (u.flags != 0u) {
        var v = acescct_encode(px);
        if ((u.flags & FLAG_CURVES) != 0u) {
            v = curves(v);
        }
        if ((u.flags & FLAG_HSL) != 0u) {
            v = hsl(v);
        }
        if ((u.flags & FLAG_WHEELS) != 0u) {
            v = wheels(v);
        }
        px = acescct_decode(v);
    }
    px = vibrance_saturation(px, u.vibrance, u.saturation);
    return vec4<f32>(px, 1.0);
}
