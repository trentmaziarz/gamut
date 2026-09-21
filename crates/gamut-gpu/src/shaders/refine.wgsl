// Refine edges: the finished alpha of one mask moved onto the edges of the
// picture under it. Every function mirrors its twin in gamut-color's
// refine.rs, sum for sum; the golden tests hold the two together.
//
// The input p is the alpha of the mask as mask.wgsl stored it. The guide I is
// the SOURCE pixel of the working texture in ACEScct, never the developed
// one, so no slider of the develop chain moves a refined mask.
//
// The filter gathers the colours of the guide inside the mask and outside it
// over a box, solves the one direction of colour that best tells the two
// classes apart, and moves every pixel of the mask AS DRAWN by where its
// colour lies along that direction. It gathers three times: the first gather
// weighs the classes by p and each later one by the q the gather before
// wrote, which is only ever a weight.
//
// The moments are taken on a grid of cells that lies on the pixel grid of the
// whole picture at the scale of the render. They live in 32 bit float
// targets: they are differences of near-equal numbers, and a half float would
// lose them. The targets hold one tile of the grid at a time, so their size
// does not grow with the render; texel (0, 0) is the cell `tile_first`. The q
// between two gathers is a 32 bit float too, so no store rounds it; its texel
// (0, 0) is the pixel `q_first` of the render.
//
// The passes of one tile, in order:
//   once a source and radius (kept while neither changes)
//     fs_source_a, fs_source_b    the means of I and I I over each cell
//     fs_box_h2, fs_box_h1        their means over the box, across
//     fs_box_v2, fs_box_v1        and down
//   each of the three gathers
//     fs_gather_first / fs_gather the means of q and q I over each cell (and
//                                 of p and p p, with the first)
//     fs_box_h2 / fs_box_h1       their means over the box, across
//     fs_box_v2 / fs_box_v1       and down
//     fs_solve_a, fs_solve_b      the 15 numbers of each cell
//     fs_move / fs_apply          q at full resolution; the last gather mixes
//                                 it into p by amount and stores the alpha

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
    // The pixel of the render texel (0, 0) of the q target holds.
    q_first: vec2<u32>,
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
@group(0) @binding(3) var moved_before: texture_2d<f32>;
@group(0) @binding(4) var tex_a: texture_2d<f32>;
@group(0) @binding(5) var tex_b: texture_2d<f32>;
@group(0) @binding(6) var tex_c: texture_2d<f32>;
@group(0) @binding(7) var tex_d: texture_2d<f32>;
@group(0) @binding(8) var tex_e: texture_2d<f32>;

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

// refine.rs
const COVER_LOW: f32 = 0.0;
const COVER_HIGH: f32 = 0.1;
const HARD_LOW: f32 = 0.3;
const HARD_HIGH: f32 = 0.6;
const SEPARATE_LOW: f32 = 0.15;
const SEPARATE_HIGH: f32 = 0.8;
const LINE_LOW: f32 = 0.6;
const LINE_HIGH: f32 = 1.2;
const OUT_LOW: f32 = 0.15;
const OUT_HIGH: f32 = 0.4;
const IN_LOW: f32 = 0.6;
const IN_HIGH: f32 = 0.85;
const SHARE_FLOOR: f32 = 0.000001;
const SEPARATION_FLOOR: f32 = 0.000000001;

fn acescct_encode(lin: vec3<f32>) -> vec3<f32> {
    let linear_part = ACES_SLOPE * lin + vec3<f32>(ACES_OFFSET);
    let log_part = (log2(max(lin, vec3<f32>(ACES_LINEAR_CUT))) + vec3<f32>(ACES_LOG_SHIFT))
        / ACES_LOG_SCALE;
    return select(log_part, linear_part, lin <= vec3<f32>(ACES_LINEAR_CUT));
}

// The smoothstep written out, as the twin writes it: never the builtin, whose
// rounding a driver is free to choose.
fn smooth_between(x: f32, low: f32, high: f32) -> f32 {
    let t = clamp((x - low) / (high - low), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// The pixels of the render inside the cell at `texel` of the float targets:
// the first, and one past the last.
struct Span {
    low: vec2<u32>,
    high: vec2<u32>,
}

fn cell_span(texel: vec2<u32>) -> Span {
    let cell = u.tile_first + texel;
    var span: Span;
    span.low = max(cell * u.step, u.origin) - u.origin;
    span.high = min((cell + vec2<u32>(1u)) * u.step, u.origin + u.size) - u.origin;
    return span;
}

fn span_count(span: Span) -> f32 {
    let size = span.high - span.low;
    return f32(max(size.x * size.y, 1u));
}

struct Pair {
    @location(0) one: vec4<f32>,
    @location(1) two: vec4<f32>,
}

// The 9 sums of the source in one cell, in the order of refine.rs: I, then
// I I as rr, rg, rb, gg, gb, bb.
struct SourceMoments {
    i: vec3<f32>,
    ii_r: vec3<f32>,
    ii_g: vec3<f32>,
}

fn source_moments(texel: vec2<u32>) -> SourceMoments {
    let span = cell_span(texel);
    var i = vec3<f32>(0.0);
    // rr, rg, rb, then gg, gb, bb.
    var ii_r = vec3<f32>(0.0);
    var ii_g = vec3<f32>(0.0);
    for (var y = span.low.y; y < span.high.y; y = y + 1u) {
        for (var x = span.low.x; x < span.high.x; x = x + 1u) {
            let g = acescct_encode(textureLoad(working, vec2<i32>(i32(x), i32(y)), 0).rgb);
            i = i + g;
            ii_r = ii_r + g * g.r;
            ii_g = ii_g + vec3<f32>(g.g * g.g, g.g * g.b, g.b * g.b);
        }
    }
    let count = span_count(span);
    var out: SourceMoments;
    out.i = i / count;
    out.ii_r = ii_r / count;
    out.ii_g = ii_g / count;
    return out;
}

// (I, rr) and (rg, rb, gg, gb).
@fragment
fn fs_source_a(in: VertexOutput) -> Pair {
    let m = source_moments(vec2<u32>(in.position.xy));
    var out: Pair;
    out.one = vec4<f32>(m.i, m.ii_r.x);
    out.two = vec4<f32>(m.ii_r.y, m.ii_r.z, m.ii_g.x, m.ii_g.y);
    return out;
}

// (bb).
@fragment
fn fs_source_b(in: VertexOutput) -> @location(0) vec4<f32> {
    let m = source_moments(vec2<u32>(in.position.xy));
    return vec4<f32>(m.ii_g.z, 0.0, 0.0, 0.0);
}

// The first gather weighs the classes by the mask as drawn: (q, q I) with q
// the alpha, and the moments of the mask itself, (p, p p).
@fragment
fn fs_gather_first(in: VertexOutput) -> Pair {
    let span = cell_span(vec2<u32>(in.position.xy));
    var q = 0.0;
    var qi = vec3<f32>(0.0);
    var pp = 0.0;
    for (var y = span.low.y; y < span.high.y; y = y + 1u) {
        for (var x = span.low.x; x < span.high.x; x = x + 1u) {
            let at = vec2<i32>(i32(x), i32(y));
            let a = textureLoad(alpha, at, 0).r;
            let g = acescct_encode(textureLoad(working, at, 0).rgb);
            q = q + a;
            qi = qi + g * a;
            pp = pp + a * a;
        }
    }
    let count = span_count(span);
    var out: Pair;
    out.one = vec4<f32>(q / count, qi / count);
    out.two = vec4<f32>(q / count, pp / count, 0.0, 0.0);
    return out;
}

// A later gather weighs them by the q the gather before wrote: (q, q I).
@fragment
fn fs_gather(in: VertexOutput) -> @location(0) vec4<f32> {
    let span = cell_span(vec2<u32>(in.position.xy));
    var q = 0.0;
    var qi = vec3<f32>(0.0);
    for (var y = span.low.y; y < span.high.y; y = y + 1u) {
        for (var x = span.low.x; x < span.high.x; x = x + 1u) {
            let at = vec2<i32>(i32(x), i32(y));
            let a = textureLoad(moved_before, at - vec2<i32>(u.q_first), 0).r;
            let g = acescct_encode(textureLoad(working, at, 0).rgb);
            q = q + a;
            qi = qi + g * a;
        }
    }
    let count = span_count(span);
    return vec4<f32>(q / count, qi / count);
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

// What one gather solves in one cell, in the order of refine.rs: a and s,
// then d2, c and P as rr, rg, rb, gg, gb, bb, then mid.
struct Solved {
    a_s: vec4<f32>,
    d2_c_p: vec4<f32>,
    p: vec4<f32>,
    mid: vec4<f32>,
}

// How far the box holds both classes and a mask that means an edge.
fn both(p: f32, pp: f32) -> f32 {
    let hardness = max(pp - p * p, 0.0) / max(p * (1.0 - p), SHARE_FLOOR);
    return smooth_between(min(p, 1.0 - p), COVER_LOW, COVER_HIGH)
        * smooth_between(hardness, HARD_LOW, HARD_HIGH);
}

// `tex_a` to `tex_c` hold the means of the source's moments, (I, rr),
// (rg, rb, gg, gb) and (bb); `tex_d` those of the gather, (q, q I); `tex_e`
// those of the mask, (p, p p).
fn solve(at: vec2<i32>) -> Solved {
    let s0 = textureLoad(tex_a, at, 0);
    let s1 = textureLoad(tex_b, at, 0);
    let s2 = textureLoad(tex_c, at, 0);
    let gather = textureLoad(tex_d, at, 0);
    let mask = textureLoad(tex_e, at, 0);
    let n1 = gather.x;
    let n0 = 1.0 - n1;
    let inside = max(n1, SHARE_FLOOR);
    let outside = max(n0, SHARE_FLOOR);
    let mu1 = gather.yzw / inside;
    let mu0 = (s0.xyz - gather.yzw) / outside;
    // The scatter inside the classes, symmetric: rr, rg, rb, gg, gb, bb.
    let rr = max(s0.w - n1 * mu1.r * mu1.r - n0 * mu0.r * mu0.r, 0.0) + u.eps;
    let rg = s1.x - n1 * mu1.r * mu1.g - n0 * mu0.r * mu0.g;
    let rb = s1.y - n1 * mu1.r * mu1.b - n0 * mu0.r * mu0.b;
    let gg = max(s1.z - n1 * mu1.g * mu1.g - n0 * mu0.g * mu0.g, 0.0) + u.eps;
    let gb = s1.w - n1 * mu1.g * mu1.b - n0 * mu0.g * mu0.b;
    let bb = max(s2.x - n1 * mu1.b * mu1.b - n0 * mu0.b * mu0.b, 0.0) + u.eps;
    // Its inverse by cofactors.
    let c_rr = gg * bb - gb * gb;
    let c_rg = rb * gb - rg * bb;
    let c_rb = rg * gb - rb * gg;
    let c_gg = rr * bb - rb * rb;
    let c_gb = rg * rb - rr * gb;
    let c_bb = rr * gg - rg * rg;
    let det = rr * c_rr + rg * c_rg + rb * c_rb;
    let scale = 1.0 / max(det, u.eps * u.eps * u.eps);
    let i_rr = c_rr * scale;
    let i_rg = c_rg * scale;
    let i_rb = c_rb * scale;
    let i_gg = c_gg * scale;
    let i_gb = c_gb * scale;
    let i_bb = c_bb * scale;
    let delta = mu1 - mu0;
    let a = vec3<f32>(
        i_rr * delta.x + i_rg * delta.y + i_rb * delta.z,
        i_rg * delta.x + i_gg * delta.y + i_gb * delta.z,
        i_rb * delta.x + i_gb * delta.y + i_bb * delta.z,
    );
    let d2 = max(delta.x * a.x + delta.y * a.y + delta.z * a.z, 0.0);
    let c = smooth_between(d2, SEPARATE_LOW, SEPARATE_HIGH) * both(mask.x, mask.y);
    let over = max(d2, SEPARATION_FLOOR);
    let mid = (mu1 + mu0) / 2.0;
    let s = a.x * mid.x + a.y * mid.y + a.z * mid.z;
    var out: Solved;
    out.a_s = vec4<f32>(a, s);
    out.d2_c_p = vec4<f32>(d2, c, i_rr - a.x * a.x / over, i_rg - a.x * a.y / over);
    out.p = vec4<f32>(
        i_rb - a.x * a.z / over,
        i_gg - a.y * a.y / over,
        i_gb - a.y * a.z / over,
        i_bb - a.z * a.z / over,
    );
    out.mid = vec4<f32>(mid, 0.0);
    return out;
}

@fragment
fn fs_solve_a(in: VertexOutput) -> Pair {
    let solved = solve(vec2<i32>(in.position.xy));
    var out: Pair;
    out.one = solved.a_s;
    out.two = solved.d2_c_p;
    return out;
}

@fragment
fn fs_solve_b(in: VertexOutput) -> Pair {
    let solved = solve(vec2<i32>(in.position.xy));
    var out: Pair;
    out.one = solved.p;
    out.two = solved.mid;
    return out;
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

// One of the four targets of what was solved, mixed from the four cells
// around a pixel.
fn mixed(solved: texture_2d<f32>, x: vec3<f32>, y: vec3<f32>) -> vec4<f32> {
    let tl = textureLoad(solved, vec2<i32>(i32(x.x), i32(y.x)), 0);
    let tr = textureLoad(solved, vec2<i32>(i32(x.y), i32(y.x)), 0);
    let bl = textureLoad(solved, vec2<i32>(i32(x.x), i32(y.y)), 0);
    let br = textureLoad(solved, vec2<i32>(i32(x.y), i32(y.y)), 0);
    let top = tl + (tr - tl) * x.z;
    let bottom = bl + (br - bl) * x.z;
    return top + (bottom - top) * y.z;
}

// The mask as drawn moved at the pixel `pixel` of the render. `tex_a` to
// `tex_d` hold what the gather solved.
fn moved(pixel: vec2<u32>) -> vec2<f32> {
    let at = vec2<i32>(pixel);
    let p = textureLoad(alpha, at, 0).r;
    let g = acescct_encode(textureLoad(working, at, 0).rgb);
    let x = among(pixel.x, u.origin.x, u.grid_first.x, u.grid_count.x, u.tile_first.x);
    let y = among(pixel.y, u.origin.y, u.grid_first.y, u.grid_count.y, u.tile_first.y);
    let a_s = mixed(tex_a, x, y);
    let d2_c_p = mixed(tex_b, x, y);
    let rest = mixed(tex_c, x, y);
    let mid = mixed(tex_d, x, y);
    let along = (a_s.x * g.r + a_s.y * g.g + a_s.z * g.b) - a_s.w;
    let est = clamp(0.5 + along / max(d2_c_p.x, SEPARATION_FLOOR), 0.0, 1.0);
    let e = g - mid.xyz;
    let m = max(
        d2_c_p.z * e.x * e.x + rest.y * e.y * e.y + rest.w * e.z * e.z
            + 2.0 * (d2_c_p.w * e.x * e.y + rest.x * e.x * e.z + rest.z * e.y * e.z),
        0.0,
    );
    let on_line = 1.0 - smooth_between(m, LINE_LOW, LINE_HIGH);
    let out = on_line * (1.0 - smooth_between(est, OUT_LOW, OUT_HIGH));
    let into = on_line * smooth_between(est, IN_LOW, IN_HIGH);
    return vec2<f32>(p, p + d2_c_p.y * (into * (1.0 - p) - out * p));
}

// The q of a gather before the last, into the q target.
@fragment
fn fs_move(in: VertexOutput) -> @location(0) vec4<f32> {
    let q = moved(vec2<u32>(in.position.xy) + u.q_first).y;
    return vec4<f32>(q, 0.0, 0.0, 1.0);
}

// The q of the last gather, mixed into p by amount: the refined alpha.
@fragment
fn fs_apply(in: VertexOutput) -> @location(0) vec4<f32> {
    let pq = moved(vec2<u32>(in.position.xy));
    return vec4<f32>(clamp(pq.x + (pq.y - pq.x) * u.amount, 0.0, 1.0), 0.0, 0.0, 1.0);
}
