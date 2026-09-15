// One direction of a separable gaussian blur on luminance, clamped at the
// edges. The first pass reads the working texture and takes its Rec.2020
// luminance; the second reads the first's red channel. The weights are
// computed here from sigma, in the same order as the CPU twin.

struct Uniform {
    direction: vec2<i32>,
    radius: i32,
    luma: u32,
    sigma: f32,
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

const LUMA: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let limit = vec2<i32>(textureDimensions(source)) - vec2<i32>(1, 1);
    let centre = vec2<i32>(in.position.xy);
    var sum = 0.0;
    var weight_sum = 0.0;
    for (var i = -u.radius; i <= u.radius; i = i + 1) {
        let weight = exp(-f32(i * i) / (2.0 * u.sigma * u.sigma));
        let at = clamp(centre + i * u.direction, vec2<i32>(0, 0), limit);
        let texel = textureLoad(source, at, 0);
        var value = texel.r;
        if (u.luma == 1u) {
            value = dot(texel.rgb, LUMA);
        }
        sum = sum + weight * value;
        weight_sum = weight_sum + weight;
    }
    return vec4<f32>(sum / weight_sum, 0.0, 0.0, 1.0);
}
