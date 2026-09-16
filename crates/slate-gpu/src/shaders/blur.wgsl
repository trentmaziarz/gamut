// One direction of a separable gaussian blur on luminance, clamped at the
// edges. The first pass reads the working texture and takes its Rec.2020
// luminance; the second reads the first's red channel. The weights are
// computed here from sigma, in the same order as the CPU twin. Taps are
// read in pairs: one bilinear sample placed between two texels at the
// ratio of their weights returns their weighted sum, so a kernel of
// 2r + 1 taps costs r + 1 fetches. Luminance is linear in the texel
// values, so the sample of the working texture gives the same luminance
// as the luminance of its two texels.

struct Uniform {
    direction: vec2<i32>,
    radius: i32,
    luma: u32,
    sigma: f32,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;

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

fn weight(i: i32) -> f32 {
    return exp(-f32(i * i) / (2.0 * u.sigma * u.sigma));
}

fn value_at(uv: vec2<f32>) -> f32 {
    let texel = textureSampleLevel(source, source_sampler, uv, 0.0);
    if (u.luma == 1u) {
        return dot(texel.rgb, LUMA);
    }
    return texel.r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(source));
    let centre = in.position.xy;
    let dir = vec2<f32>(u.direction);
    // The centre tap alone, then the rest in pairs on each side.
    var sum = weight(0) * value_at(centre / size);
    var weight_sum = weight(0);
    for (var i = 1; i <= u.radius; i = i + 2) {
        let w0 = weight(i);
        let w1 = select(0.0, weight(i + 1), i + 1 <= u.radius);
        let pair = w0 + w1;
        let offset = f32(i) + w1 / pair;
        sum = sum + pair * value_at((centre + offset * dir) / size);
        sum = sum + pair * value_at((centre - offset * dir) / size);
        weight_sum = weight_sum + 2.0 * pair;
    }
    return vec4<f32>(sum / weight_sum, 0.0, 0.0, 1.0);
}
