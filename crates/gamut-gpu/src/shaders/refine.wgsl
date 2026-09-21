// Refine edges: the finished alpha of one mask moved onto the edges of the
// picture under it by the colour guided filter of He, Sun and Tang. Every
// function mirrors its twin in gamut-color's refine.rs, sum for sum; the
// golden tests hold the two together.
//
// The input p is the alpha of the mask as mask.wgsl stored it. The guide I is
// the SOURCE pixel of the working texture in ACEScct, never the developed
// one, so no slider of the develop chain moves a refined mask.
//
// The moments are taken on a grid of cells that lies on the pixel grid of the
// whole picture at the scale of the render. They live in 32 bit float
// targets: they are differences of near-equal numbers, and a half float would
// lose them. The targets hold one tile of the grid at a time, so their size
// does not grow with the render; texel (0, 0) is the cell `tile_first`.
//
// The passes of one tile, in order:
//   fs_moments_a, fs_moments_b   the means of p, I p, I and I I over each cell
//   fs_box_h2, fs_box_v2         their means over the box, twice (two pairs)
//   fs_solve                     a and b of each cell
//   fs_box_h1, fs_box_v1         the means of a and b over the box
//   fs_apply                     q at full resolution, mixed into p by amount

struct Uniform {
    // Where the render begins on the whole picture, in pixels, and its size.
    origin: vec2<u32>,
    size: vec2<u32>,
    // The first cell the render touches and how many it touches.
    grid_first: vec2<u32>,
    grid_count: vec2<u32>,
    // The cell texel (0, 0) of the float targets holds, and how many cells of
    // the tile they hold.
    tile_first: vec2<u32>,
    tile_count: vec2<u32>,
    // The side of a cell in pixels and the radius of the box in cells.
    step: u32,
    cells: u32,
    eps: f32,
    // How much of the refined alpha is taken, 0 to 1.
    amount: f32,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var working: texture_2d<f32>;
@group(0) @binding(2) var alpha: texture_2d<f32>;
@group(0) @binding(3) var tex_a: texture_2d<f32>;
@group(0) @binding(4) var tex_b: texture_2d<f32>;
@group(0) @binding(5) var tex_c: texture_2d<f32>;
@group(0) @binding(6) var tex_d: texture_2d<f32>;

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

// acescct.rs
const ACES_LINEAR_CUT: f32 = 0.0078125;
const ACES_SLOPE: f32 = 10.540237;
const ACES_OFFSET: f32 = 0.07290553;
const ACES_LOG_SHIFT: f32 = 9.72;
const ACES_LOG_SCALE: f32 = 17.52;

fn acescct_encode(lin: vec3<f32>) -> vec3<f32> {
    let linear_part = ACES_SLOPE * lin + vec3<f32>(ACES_OFFSET);
    let log_part = (log2(max(lin, vec3<f32>(ACES_LINEAR_CUT))) + vec3<f32>(ACES_LOG_SHIFT))
        / ACES_LOG_SCALE;
    return select(log_part, linear_part, lin <= vec3<f32>(ACES_LINEAR_CUT));
}

// The 13 sums of one cell, in the order of refine.rs: p, I p, I, then I I as
// rr, rg, rb, gg, gb, bb.
struct Moments {
    p: f32,
    ip: vec3<f32>,
    i: vec3<f32>,
    ii_r: vec3<f32>,
    ii_g: vec3<f32>,
}

fn cell_moments(texel: vec2<u32>) -> Moments {
    let cell = u.tile_first + texel;
    let low = max(cell * u.step, u.origin) - u.origin;
    let high = min((cell + vec2<u32>(1u)) * u.step, u.origin + u.size) - u.origin;
    var p = 0.0;
    var ip = vec3<f32>(0.0);
    var i = vec3<f32>(0.0);
    // rr, rg, rb, then gg, gb, bb.
    var ii_r = vec3<f32>(0.0);
    var ii_g = vec3<f32>(0.0);
    for (var y = low.y; y < high.y; y = y + 1u) {
        for (var x = low.x; x < high.x; x = x + 1u) {
            let at = vec2<i32>(i32(x), i32(y));
            let a = textureLoad(alpha, at, 0).r;
            let g = acescct_encode(textureLoad(working, at, 0).rgb);
            p = p + a;
            ip = ip + g * a;
            i = i + g;
            ii_r = ii_r + g * g.r;
            ii_g = ii_g + vec3<f32>(g.g * g.g, g.g * g.b, g.b * g.b);
        }
    }
    let span = high - low;
    let count = f32(max(span.x * span.y, 1u));
    var out: Moments;
    out.p = p / count;
    out.ip = ip / count;
    out.i = i / count;
    out.ii_r = ii_r / count;
    out.ii_g = ii_g / count;
    return out;
}

struct Pair {
    @location(0) one: vec4<f32>,
    @location(1) two: vec4<f32>,
}

// (p, I p) and (I, rr).
@fragment
fn fs_moments_a(in: VertexOutput) -> Pair {
    let m = cell_moments(vec2<u32>(in.position.xy));
    var out: Pair;
    out.one = vec4<f32>(m.p, m.ip);
    out.two = vec4<f32>(m.i, m.ii_r.x);
    return out;
}

// (rg, rb, gg, gb) and (bb).
@fragment
fn fs_moments_b(in: VertexOutput) -> Pair {
    let m = cell_moments(vec2<u32>(in.position.xy));
    var out: Pair;
    out.one = vec4<f32>(m.ii_r.y, m.ii_r.z, m.ii_g.x, m.ii_g.y);
    out.two = vec4<f32>(m.ii_g.z, 0.0, 0.0, 0.0);
    return out;
}

// The texel of the cell `offset` cells along `direction` from the one at
// `texel`, held inside the grid of the render as the twin clamps it, and
// inside the tile.
fn beside(texel: vec2<u32>, direction: vec2<i32>, offset: i32) -> vec2<i32> {
    let cell = vec2<i32>(u.tile_first + texel) + direction * offset;
    let low = vec2<i32>(u.grid_first);
    let high = vec2<i32>(u.grid_first + u.grid_count) - vec2<i32>(1);
    let held = clamp(cell, low, high) - vec2<i32>(u.tile_first);
    return clamp(held, vec2<i32>(0), vec2<i32>(u.tile_count) - vec2<i32>(1));
}

fn box_pair(texel: vec2<u32>, direction: vec2<i32>) -> Pair {
    let radius = i32(u.cells);
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    for (var i = -radius; i <= radius; i = i + 1) {
        let at = beside(texel, direction, i);
        a = a + textureLoad(tex_a, at, 0);
        b = b + textureLoad(tex_b, at, 0);
    }
    let count = f32(2 * radius + 1);
    var out: Pair;
    out.one = a / count;
    out.two = b / count;
    return out;
}

fn box_one(texel: vec2<u32>, direction: vec2<i32>) -> vec4<f32> {
    let radius = i32(u.cells);
    var a = vec4<f32>(0.0);
    for (var i = -radius; i <= radius; i = i + 1) {
        a = a + textureLoad(tex_a, beside(texel, direction, i), 0);
    }
    return a / f32(2 * radius + 1);
}

@fragment
fn fs_box_h2(in: VertexOutput) -> Pair {
    return box_pair(vec2<u32>(in.position.xy), vec2<i32>(1, 0));
}

@fragment
fn fs_box_v2(in: VertexOutput) -> Pair {
    return box_pair(vec2<u32>(in.position.xy), vec2<i32>(0, 1));
}

@fragment
fn fs_box_h1(in: VertexOutput) -> @location(0) vec4<f32> {
    return box_one(vec2<u32>(in.position.xy), vec2<i32>(1, 0));
}

@fragment
fn fs_box_v1(in: VertexOutput) -> @location(0) vec4<f32> {
    return box_one(vec2<u32>(in.position.xy), vec2<i32>(0, 1));
}

// a and b of one cell from the means of its moments: `tex_a` to `tex_d` hold
// (p, I p), (I, rr), (rg, rb, gg, gb) and (bb).
@fragment
fn fs_solve(in: VertexOutput) -> @location(0) vec4<f32> {
    let at = vec2<i32>(in.position.xy);
    let m0 = textureLoad(tex_a, at, 0);
    let m1 = textureLoad(tex_b, at, 0);
    let m2 = textureLoad(tex_c, at, 0);
    let m3 = textureLoad(tex_d, at, 0);
    let p = m0.x;
    let mu = m1.xyz;
    let cov = m0.yzw - mu * p;
    let rr = m1.w - mu.r * mu.r + u.eps;
    let rg = m2.x - mu.r * mu.g;
    let rb = m2.y - mu.r * mu.b;
    let gg = m2.z - mu.g * mu.g + u.eps;
    let gb = m2.w - mu.g * mu.b;
    let bb = m3.x - mu.b * mu.b + u.eps;
    let c_rr = gg * bb - gb * gb;
    let c_rg = rb * gb - rg * bb;
    let c_rb = rg * gb - rb * gg;
    let c_gg = rr * bb - rb * rb;
    let c_gb = rg * rb - rr * gb;
    let c_bb = rr * gg - rg * rg;
    let det = rr * c_rr + rg * c_rg + rb * c_rb;
    let scale = 1.0 / max(det, u.eps * u.eps * u.eps);
    let a = vec3<f32>(
        (c_rr * cov.x + c_rg * cov.y + c_rb * cov.z) * scale,
        (c_rg * cov.x + c_gg * cov.y + c_gb * cov.z) * scale,
        (c_rb * cov.x + c_gb * cov.y + c_bb * cov.z) * scale,
    );
    let b = p - (a.x * mu.r + a.y * mu.g + a.z * mu.b);
    return vec4<f32>(a, b);
}

// Where a pixel lies among the centres of the cells on one axis: the texel of
// the cell before it, of the one after, and the share of the second.
fn among(pixel: u32, origin: u32, grid_first: u32, grid_count: u32, tile_first: u32) -> vec3<f32> {
    let place = (f32(origin + pixel) + 0.5) / f32(u.step) - 0.5 - f32(grid_first);
    let low = floor(place);
    let last = f32(grid_count - 1u);
    let shift = f32(grid_first) - f32(tile_first);
    return vec3<f32>(
        clamp(low, 0.0, last) + shift,
        clamp(low + 1.0, 0.0, last) + shift,
        place - low,
    );
}

// `tex_a` holds the means of a and b over the box.
@fragment
fn fs_apply(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(in.position.xy);
    let at = vec2<i32>(pixel);
    let p = textureLoad(alpha, at, 0).r;
    let g = acescct_encode(textureLoad(working, at, 0).rgb);
    let x = among(pixel.x, u.origin.x, u.grid_first.x, u.grid_count.x, u.tile_first.x);
    let y = among(pixel.y, u.origin.y, u.grid_first.y, u.grid_count.y, u.tile_first.y);
    let tl = textureLoad(tex_a, vec2<i32>(i32(x.x), i32(y.x)), 0);
    let tr = textureLoad(tex_a, vec2<i32>(i32(x.y), i32(y.x)), 0);
    let bl = textureLoad(tex_a, vec2<i32>(i32(x.x), i32(y.y)), 0);
    let br = textureLoad(tex_a, vec2<i32>(i32(x.y), i32(y.y)), 0);
    let top = tl + (tr - tl) * x.z;
    let bottom = bl + (br - bl) * x.z;
    let ab = top + (bottom - top) * y.z;
    let q = clamp(ab.x * g.r + ab.y * g.g + ab.z * g.b + ab.w, 0.0, 1.0);
    return vec4<f32>(p + (q - p) * u.amount, 0.0, 0.0, 1.0);
}
