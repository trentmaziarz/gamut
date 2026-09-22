// The edge controls of one mask: Shift edge, Feather and Contrast, applied in
// that order to the alpha Refine edges hands to the blend. Every function
// mirrors its twin in gamut-color's edge.rs; the golden tests hold the two
// together.
//
// The passes, in order, each skipped while its control is at rest:
//   Shift edge, for each of the four runs of the octagon (x, y, (1, 1),
//   (1, -1)) that takes a step
//     fs_run          a doubling pass of the forward run, then of the
//                     backward run: the run so far here and `offset` steps on
//     fs_run_last     the last backward pass, which also takes the forward
//                     run in
//   Feather
//     fs_cells        the mean of the pixels of each cell of the whole
//                     picture's grid
//     fs_blur         the Gaussian over the cells, across, then down
//   Feather and Contrast
//     fs_finish       the four cells around each pixel mixed bilinearly (or
//                     the alpha as it is while Feather is at rest), then
//                     Contrast: the finished alpha
//
// A maximum or a minimum makes no new value, so the runs keep the 8-bit codes
// they read. The cells are 32 bit floats. Nothing here reads the source or
// the developed picture.

struct Uniform {
    // A run: its direction and how many steps on the far sample lies.
    direction: vec2<i32>,
    offset: i32,
    // 1 while Shift edge grows the mask (a maximum), 0 while it shrinks it.
    grow: u32,
    // Where the render begins on the whole picture, in pixels, and its size.
    origin: vec2<u32>,
    size: vec2<u32>,
    // The first cell the render touches and how many it touches. Texel
    // (0, 0) of a cell target is the cell `grid_first`.
    grid_first: vec2<u32>,
    grid_count: vec2<u32>,
    // The side of a cell in pixels, the radius of the kernel in cells, and
    // sigma in cells.
    step: u32,
    radius: i32,
    sigma: f32,
    // 1 when the finished alpha reads the cells, 0 when it reads `source`.
    feathered: u32,
    // Contrast, 0 to 100, and its gain below 100.
    contrast: f32,
    gain: f32,
}

@group(0) @binding(0) var<uniform> u: Uniform;
// The alpha a pass reads: the input of the chain, or the run so far.
@group(0) @binding(1) var source: texture_2d<f32>;
// The forward run, which the last backward pass takes in.
@group(0) @binding(2) var other: texture_2d<f32>;
// The cells a blur or the finished alpha reads.
@group(0) @binding(3) var cells: texture_2d<f32>;

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

const CONTRAST_MIDDLE: f32 = 0.5;
const MAX_CONTRAST: f32 = 100.0;

// An alpha of the render, its place held inside the render.
fn alpha_at(t: texture_2d<f32>, at: vec2<i32>) -> f32 {
    let limit = vec2<i32>(u.size) - vec2<i32>(1, 1);
    return textureLoad(t, clamp(at, vec2<i32>(0, 0), limit), 0).r;
}

// The maximum while Shift edge grows the mask, the minimum while it shrinks
// it.
fn pick(a: f32, b: f32) -> f32 {
    if (u.grow == 1u) {
        return max(a, b);
    }
    return min(a, b);
}

@fragment
fn fs_run(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = vec2<i32>(in.position.xy);
    let here = alpha_at(source, at);
    let far = alpha_at(source, at + u.offset * u.direction);
    return vec4<f32>(pick(here, far), 0.0, 0.0, 1.0);
}

@fragment
fn fs_run_last(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = vec2<i32>(in.position.xy);
    let here = alpha_at(source, at);
    let far = alpha_at(source, at + u.offset * u.direction);
    let forward = alpha_at(other, at);
    return vec4<f32>(pick(forward, pick(here, far)), 0.0, 0.0, 1.0);
}

// The mean of the pixels of one cell that the render holds.
@fragment
fn fs_cells(in: VertexOutput) -> @location(0) vec4<f32> {
    let cell = vec2<u32>(in.position.xy) + u.grid_first;
    let low = max(cell * u.step, u.origin) - u.origin;
    let high = min((cell + vec2<u32>(1u, 1u)) * u.step, u.origin + u.size) - u.origin;
    var sum = 0.0;
    for (var y = low.y; y < high.y; y = y + 1u) {
        for (var x = low.x; x < high.x; x = x + 1u) {
            sum = sum + textureLoad(source, vec2<i32>(i32(x), i32(y)), 0).r;
        }
    }
    let count = f32(max((high.x - low.x) * (high.y - low.y), 1u));
    return vec4<f32>(sum / count, 0.0, 0.0, 1.0);
}

fn weight(i: i32) -> f32 {
    return exp(-f32(i * i) / (2.0 * u.sigma * u.sigma));
}

// One direction of the Gaussian over the cells, held at the ends of the grid.
@fragment
fn fs_blur(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = vec2<i32>(in.position.xy);
    let last = vec2<i32>(u.grid_count) - vec2<i32>(1, 1);
    var sum = 0.0;
    var weight_sum = 0.0;
    for (var i = -u.radius; i <= u.radius; i = i + 1) {
        let w = weight(i);
        let cell = clamp(at + i * u.direction, vec2<i32>(0, 0), last);
        sum = sum + w * textureLoad(cells, cell, 0).r;
        weight_sum = weight_sum + w;
    }
    return vec4<f32>(sum / weight_sum, 0.0, 0.0, 1.0);
}

// Where a pixel lies among the centres of the cells on one axis: the cell
// before it, the one after, and the share of the second.
fn among(pixel: u32, origin: u32, first: u32, count: u32) -> vec3<f32> {
    let place = (f32(origin + pixel) + 0.5) / f32(u.step) - 0.5 - f32(first);
    let low = floor(place);
    let last = f32(count - 1u);
    return vec3<f32>(clamp(low, 0.0, last), clamp(low + 1.0, 0.0, last), place - low);
}

fn contrast(p: f32) -> f32 {
    if (u.contrast >= MAX_CONTRAST) {
        return select(0.0, 1.0, p >= CONTRAST_MIDDLE);
    }
    if (u.contrast > 0.0) {
        return clamp(CONTRAST_MIDDLE + (p - CONTRAST_MIDDLE) * u.gain, 0.0, 1.0);
    }
    return p;
}

@fragment
fn fs_finish(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = vec2<u32>(in.position.xy);
    var p: f32;
    if (u.feathered == 1u) {
        let x = among(at.x, u.origin.x, u.grid_first.x, u.grid_count.x);
        let y = among(at.y, u.origin.y, u.grid_first.y, u.grid_count.y);
        let tl = textureLoad(cells, vec2<i32>(i32(x.x), i32(y.x)), 0).r;
        let tr = textureLoad(cells, vec2<i32>(i32(x.y), i32(y.x)), 0).r;
        let bl = textureLoad(cells, vec2<i32>(i32(x.x), i32(y.y)), 0).r;
        let br = textureLoad(cells, vec2<i32>(i32(x.y), i32(y.y)), 0).r;
        let top = tl + (tr - tl) * x.z;
        let bottom = bl + (br - bl) * x.z;
        p = top + (bottom - top) * y.z;
    } else {
        p = textureLoad(source, vec2<i32>(at), 0).r;
    }
    return vec4<f32>(contrast(p), 0.0, 0.0, 1.0);
}
