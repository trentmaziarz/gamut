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
