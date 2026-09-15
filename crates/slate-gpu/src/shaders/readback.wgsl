// Copies a working-space texture into an 8-bit sRGB target, one texel per
// pixel. The target format does the sRGB encoding on write.

@group(0) @binding(0) var source: texture_2d<f32>;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let limit = vec2<i32>(textureDimensions(source)) - vec2<i32>(1, 1);
    let texel = clamp(vec2<i32>(in.position.xy), vec2<i32>(0, 0), limit);
    return textureLoad(source, texel, 0);
}
