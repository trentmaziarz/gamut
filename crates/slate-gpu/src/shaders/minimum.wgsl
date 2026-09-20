// One direction of the square minimum filter of dehaze, clamped at the
// edges; the twin of dehaze::minimum and dehaze::raw_transmission in
// slate-color. The first pass reads the working texture and takes the dark
// channel, the smallest channel of the pixel over the atmospheric light. The
// second reads the first's red channel and writes the transmission,
// 1 - 0.95 times the minimum, held between the floor and 1.

struct Uniform {
    direction: vec2<i32>,
    radius: i32,
    // 0: read the dark channel of the working texture, write the minimum.
    // 1: read the red channel, write the transmission of the minimum.
    stage: u32,
    atmosphere: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var source: texture_2d<f32>;

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

const OMEGA: f32 = 0.95;
const TRANSMISSION_FLOOR: f32 = 0.01;

fn value_at(at: vec2<i32>) -> f32 {
    let texel = textureLoad(source, at, 0);
    if (u.stage == 0u) {
        let over = texel.rgb / u.atmosphere.rgb;
        return min(over.r, min(over.g, over.b));
    }
    return texel.r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let limit = vec2<i32>(textureDimensions(source)) - vec2<i32>(1, 1);
    let centre = vec2<i32>(in.position.xy);
    var least = 3.4e38;
    for (var i = -u.radius; i <= u.radius; i = i + 1) {
        let at = clamp(centre + i * u.direction, vec2<i32>(0, 0), limit);
        least = min(least, value_at(at));
    }
    if (u.stage == 1u) {
        least = clamp(1.0 - OMEGA * least, TRANSMISSION_FLOOR, 1.0);
    }
    return vec4<f32>(least, 0.0, 0.0, 1.0);
}
