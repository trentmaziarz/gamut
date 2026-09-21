// Stamps the dabs of a painted mask source into one layer at render size.
// One instance is one dab; gamut-color's brush.rs places them and its
// Dab::strength and Dab::apply are the fragment stage and the two blend
// states this shader is drawn through: paint takes the layer a to
// s + a (1 - s), erase to a (1 - s). Dabs are drawn in the order they were
// painted, and the golden tests hold the layer to the twin.

struct Uniform {
    // The part of the photo the render covers: x, y, width, height.
    window: vec4<f32>,
    render_size: vec2<f32>,
    // Each side of the photo over its longer side.
    aspect: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniform;

struct Dab {
    // The centre, normalised to the photo.
    @location(0) centre: vec2<f32>,
    // The radius as a share of the longer side, the inner share of the
    // radius where the feather starts, and the flow as a share.
    @location(1) brush: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) centre: vec2<f32>,
    @location(1) @interpolate(flat) brush: vec3<f32>,
}

// A square around the dab, one render pixel wider than its radius on every
// side so that no pixel centre inside the radius is left out.
@vertex
fn vs_main(@builtin(vertex_index) index: u32, dab: Dab) -> VertexOutput {
    let corner = vec2<f32>(f32(index & 1u), f32((index >> 1u) & 1u)) * 2.0 - 1.0;
    let pixel = u.window.zw / u.render_size;
    let at = dab.centre + corner * (dab.brush.x / u.aspect + pixel);
    let across = (at - u.window.xy) / u.window.zw;
    var out: VertexOutput;
    out.position = vec4<f32>(across.x * 2.0 - 1.0, 1.0 - across.y * 2.0, 0.0, 1.0);
    out.centre = dab.centre;
    out.brush = dab.brush;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = u.window.xy + in.position.xy / u.render_size * u.window.zw;
    let d = (at - in.centre) * u.aspect;
    let radius = in.brush.x;
    var s = 0.0;
    if (abs(d.x) < radius && abs(d.y) < radius) {
        s = (1.0 - smoothstep(in.brush.y, 1.0, length(d) / radius)) * in.brush.z;
    }
    return vec4<f32>(s, 0.0, 0.0, 1.0);
}

// The auto stroke. A dab of one paints only the colour under its centre:
// gamut-color's Gate. The vertex stage reads the reference of the dab, one
// bilinear sample of the proxy of the whole source at the centre
// (Proxy::sample), and the fragment stage holds the source pixel of the
// working texture against it. Both are the source before any operator, so no
// slider moves the mask; the proxy belongs to the source and not to the
// window, so every render reads the same reference.

@group(0) @binding(1) var proxy: texture_2d<f32>;
@group(0) @binding(2) var working: texture_2d<f32>;

const LUMA: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);

// acescct.rs
const ACES_LINEAR_CUT: f32 = 0.0078125;
const ACES_SLOPE: f32 = 10.540237;
const ACES_OFFSET: f32 = 0.07290553;
const ACES_LOG_SHIFT: f32 = 9.72;
const ACES_LOG_SCALE: f32 = 17.52;
const ACES_BLACK: f32 = ACES_OFFSET;
const ACES_WHITE: f32 = ACES_LOG_SHIFT / ACES_LOG_SCALE;

// hue.rs
const E1: vec3<f32> = vec3<f32>(0.8164966, -0.4082483, -0.4082483);
const E2: vec3<f32> = vec3<f32>(0.0, 0.70710677, -0.70710677);

// brush.rs
const GATE_CHROMA_WEIGHT: f32 = 2.0;
const GATE_FALL: f32 = 0.5;

struct AutoDab {
    @location(0) centre: vec2<f32>,
    @location(1) brush: vec3<f32>,
    // The colour distance that passes whole.
    @location(2) pass_distance: f32,
}

struct AutoOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) centre: vec2<f32>,
    @location(1) @interpolate(flat) brush: vec3<f32>,
    // gate_place of the reference colour.
    @location(2) @interpolate(flat) place: vec3<f32>,
    @location(3) @interpolate(flat) pass_distance: f32,
}

fn acescct_encode(lin: vec3<f32>) -> vec3<f32> {
    let linear_part = ACES_SLOPE * lin + vec3<f32>(ACES_OFFSET);
    let log_part = (log2(max(lin, vec3<f32>(ACES_LINEAR_CUT))) + vec3<f32>(ACES_LOG_SHIFT))
        / ACES_LOG_SCALE;
    return select(log_part, linear_part, lin <= vec3<f32>(ACES_LINEAR_CUT));
}

// The tone, then the chroma plane weighted against it.
fn gate_place(px: vec3<f32>) -> vec3<f32> {
    let v = acescct_encode(px);
    let n = clamp((dot(v, LUMA) - ACES_BLACK) / (ACES_WHITE - ACES_BLACK), 0.0, 1.0);
    return vec3<f32>(n, dot(v, E1) * GATE_CHROMA_WEIGHT, dot(v, E2) * GATE_CHROMA_WEIGHT);
}

fn proxy_texel(p: vec2<f32>, hi: vec2<f32>) -> vec3<f32> {
    return textureLoad(proxy, vec2<i32>(clamp(p, vec2<f32>(0.0), hi)), 0).rgb;
}

fn proxy_sample(at: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(proxy));
    let p = clamp(at, vec2<f32>(0.0), vec2<f32>(1.0)) * size - 0.5;
    let p0 = floor(p);
    let f = p - p0;
    let hi = size - 1.0;
    let a = proxy_texel(p0, hi);
    let b = proxy_texel(p0 + vec2<f32>(1.0, 0.0), hi);
    let c = proxy_texel(p0 + vec2<f32>(0.0, 1.0), hi);
    let d = proxy_texel(p0 + vec2<f32>(1.0, 1.0), hi);
    let top = a + (b - a) * f.x;
    let bottom = c + (d - c) * f.x;
    return top + (bottom - top) * f.y;
}

@vertex
fn vs_auto(@builtin(vertex_index) index: u32, dab: AutoDab) -> AutoOutput {
    let corner = vec2<f32>(f32(index & 1u), f32((index >> 1u) & 1u)) * 2.0 - 1.0;
    let pixel = u.window.zw / u.render_size;
    let at = dab.centre + corner * (dab.brush.x / u.aspect + pixel);
    let across = (at - u.window.xy) / u.window.zw;
    var out: AutoOutput;
    out.position = vec4<f32>(across.x * 2.0 - 1.0, 1.0 - across.y * 2.0, 0.0, 1.0);
    out.centre = dab.centre;
    out.brush = dab.brush;
    out.place = gate_place(proxy_sample(dab.centre));
    out.pass_distance = dab.pass_distance;
    return out;
}

@fragment
fn fs_auto(in: AutoOutput) -> @location(0) vec4<f32> {
    let at = u.window.xy + in.position.xy / u.render_size * u.window.zw;
    let d = (at - in.centre) * u.aspect;
    let radius = in.brush.x;
    var s = 0.0;
    if (abs(d.x) < radius && abs(d.y) < radius) {
        s = (1.0 - smoothstep(in.brush.y, 1.0, length(d) / radius)) * in.brush.z;
        let px = textureLoad(working, vec2<i32>(in.position.xy), 0).rgb;
        let away = distance(gate_place(px), in.place);
        s = s * (1.0 - smoothstep(in.pass_distance, in.pass_distance * (1.0 + GATE_FALL), away));
    }
    return vec4<f32>(s, 0.0, 0.0, 1.0);
}
