// Rasterises the alpha of one mask into an r8unorm texture at render size.
// Every function mirrors its twin in gamut-color's mask.rs line for line;
// the golden tests hold the two together. A source is measured on where the
// pixel is on the photo and on the source pixel of the working texture,
// before any operator, so the alpha never depends on what the mask adjusts.
// A painted source is read from its layer, which brush.wgsl stamped over the
// same window.

struct Component {
    // The source (0 linear, 1 radial, 2 luminance, 3 colour, 4 brush), the
    // operator (0 add, 1 subtract, 2 intersect), whether it is inverted, and
    // for a brush its layer.
    header: vec4<u32>,
    // Linear: the start and the end. Radial: the centre and the two radii.
    // Luminance: low, high, the floored falloff. Colour: the hue and half
    // the width in degrees, the low chroma, the floored falloff in degrees.
    a: vec4<f32>,
    // Radial: the sine and the cosine of the rotation, and the inner share
    // of the radius where the feather starts.
    b: vec4<f32>,
}

struct Uniform {
    // The part of the photo the render covers: x, y, width, height.
    window: vec4<f32>,
    render_size: vec2<f32>,
    // Each side of the photo over its longer side.
    aspect: vec2<f32>,
    count: u32,
    invert: u32,
    components: array<Component, 8>,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var working: texture_2d<f32>;
@group(0) @binding(2) var layers: texture_2d_array<f32>;

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
const TAU: f32 = 6.2831853;
const DEGREES: f32 = 57.29578;

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

// mask.rs
const CHROMA_RAMP: f32 = 0.01;
const LINEAR_LENGTH_FLOOR: f32 = 1e-12;

fn acescct_encode(lin: vec3<f32>) -> vec3<f32> {
    let linear_part = ACES_SLOPE * lin + vec3<f32>(ACES_OFFSET);
    let log_part = (log2(max(lin, vec3<f32>(ACES_LINEAR_CUT))) + vec3<f32>(ACES_LOG_SHIFT))
        / ACES_LOG_SCALE;
    return select(log_part, linear_part, lin <= vec3<f32>(ACES_LINEAR_CUT));
}

fn wrap_angle(angle: f32) -> f32 {
    let wrapped = angle - TAU * floor(angle / TAU);
    return select(wrapped, 0.0, wrapped >= TAU);
}

fn linear_gradient(c: Component, at: vec2<f32>) -> f32 {
    let start = c.a.xy * u.aspect;
    let end = c.a.zw * u.aspect;
    let p = at * u.aspect - start;
    let along = end - start;
    let length = max(dot(along, along), LINEAR_LENGTH_FLOOR);
    return smoothstep(0.0, 1.0, dot(p, along) / length);
}

fn radial_gradient(c: Component, at: vec2<f32>) -> f32 {
    let p = (at - c.a.xy) * u.aspect;
    let s = c.b.x;
    let k = c.b.y;
    let q = vec2<f32>(p.x * k + p.y * s, p.y * k - p.x * s) / c.a.zw;
    return 1.0 - smoothstep(c.b.z, 1.0, length(q));
}

fn luminance_range(c: Component, px: vec3<f32>) -> f32 {
    let v = acescct_encode(px);
    let n = clamp((dot(v, LUMA) - ACES_BLACK) / (ACES_WHITE - ACES_BLACK), 0.0, 1.0);
    let falloff = c.a.z;
    return smoothstep(c.a.x - falloff, c.a.x, n)
        * (1.0 - smoothstep(c.a.y, c.a.y + falloff, n));
}

fn colour_range(c: Component, px: vec3<f32>) -> f32 {
    let v = acescct_encode(px);
    let plane = vec2<f32>(dot(v, E1), dot(v, E2));
    let degrees = wrap_angle(atan2(plane.y, plane.x)) * DEGREES;
    let turned = degrees - c.a.x + 180.0;
    let away = abs(turned - 360.0 * floor(turned / 360.0) - 180.0);
    let by_hue = 1.0 - smoothstep(c.a.y, c.a.y + c.a.w, away);
    let by_chroma = smoothstep(c.a.z, c.a.z + CHROMA_RAMP, length(plane));
    return by_hue * by_chroma;
}

fn combine(a: f32, b: f32, op: u32) -> f32 {
    if (op == 1u) {
        return a * (1.0 - b);
    }
    if (op == 2u) {
        return a * b;
    }
    return a + b - a * b;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = u.window.xy + in.position.xy / u.render_size * u.window.zw;
    let px = textureLoad(working, vec2<i32>(in.position.xy), 0).rgb;
    var alpha = 0.0;
    for (var i = 0u; i < u.count; i = i + 1u) {
        let c = u.components[i];
        var b = 0.0;
        switch c.header.x {
            case 0u: {
                b = linear_gradient(c, at);
            }
            case 1u: {
                b = radial_gradient(c, at);
            }
            case 2u: {
                b = luminance_range(c, px);
            }
            case 4u: {
                b = textureLoad(layers, vec2<i32>(in.position.xy), i32(c.header.w), 0).r;
            }
            default: {
                b = colour_range(c, px);
            }
        }
        if (c.header.z != 0u) {
            b = 1.0 - b;
        }
        alpha = combine(alpha, b, c.header.y);
    }
    if (u.invert != 0u) {
        alpha = 1.0 - alpha;
    }
    return vec4<f32>(alpha, 0.0, 0.0, 1.0);
}
