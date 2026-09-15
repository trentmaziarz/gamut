// The test image: a horizontal red-to-blue gradient under a 32 pixel checker.
// One fullscreen triangle, no vertex buffer, no bindings.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    // Vertices 0, 1 and 2 land at (-1, -1), (3, -1) and (-1, 3), a triangle
    // that covers the whole of clip space once it is clipped to the target.
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index & 2u) * 2 - 1);
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

const CHECKER_CELL: f32 = 32.0;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // The outer one percent on each side is held at pure red and pure blue,
    // so the edge columns read back as clean primaries.
    let t = clamp(in.uv.x * 1.02 - 0.01, 0.0, 1.0);
    let gradient = vec3<f32>(1.0 - t, 0.0, t);
    let cell = vec2<u32>(floor(in.position.xy / CHECKER_CELL));
    let dim = (cell.x + cell.y) % 2u == 1u;
    let tint = select(1.0, 0.7, dim);
    return vec4<f32>(gradient * tint, 1.0);
}
