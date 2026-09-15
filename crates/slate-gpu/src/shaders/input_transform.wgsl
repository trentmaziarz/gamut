// Input transform: samples the source photo at the render size and writes
// linear Rec.2020. A box of taps by taps bilinear samples covers the
// footprint of one output pixel, so a large photo shrinks without aliasing.
// Every source is decoded here with the sRGB curve when decode_srgb is
// set; the sources differ only in their matrix.

struct Uniform {
    matrix: mat3x3<f32>,
    render_size: vec2<f32>,
    decode_srgb: u32,
    taps: u32,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let footprint = 1.0 / u.render_size;
    let n = f32(u.taps);
    var sum = vec3<f32>(0.0);
    for (var j = 0u; j < u.taps; j = j + 1u) {
        for (var i = 0u; i < u.taps; i = i + 1u) {
            let offset = (vec2<f32>(f32(i), f32(j)) + 0.5) / n - 0.5;
            var c = textureSampleLevel(source, source_sampler, in.uv + offset * footprint, 0.0).rgb;
            if (u.decode_srgb == 1u) {
                c = srgb_eotf(c);
            }
            sum = sum + c;
        }
    }
    let linear = sum / (n * n);
    return vec4<f32>(u.matrix * linear, 1.0);
}
