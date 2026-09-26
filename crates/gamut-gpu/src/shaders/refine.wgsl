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
// weighs the classes by p and each later one by q, the mask moved by what the
// gather before solved, which is only ever a weight.
//
// A reached field rf on the cell grid says how far each cell is joined to the
// outside of the mask through like colour: seeded by the outside as drawn,
// 1 - Ec[p], and spread by passes of doubling steps. A pixel weighs the inside
// class by wq = q keep + max(q - p, 0) (1 - keep), keep 1 less the smoothstep
// of rf from 0.7 to 1.0, the move comes in only where some of the inside in
// the box is unreached, and a pixel leaves the mask only where rf reaches it.
//
// The moments are taken on a grid of cells that lies on the pixel grid of the
// whole picture at the scale of the render. They live in 32 bit float
// targets: they are differences of near-equal numbers, and a half float would
// lose them. The targets hold one tile of the grid at a time, so their size
// does not grow with the render; texel (0, 0) is the cell `tile_first`. The q
// of a later gather is never stored: the gather moves each pixel it sums from
// what the gather before solved, so no store rounds it.
//
// The passes of one tile, in order:
//   once a source and radius (kept while neither changes)
//     fs_source                   the means of I and I I over each cell, in
//                                 three targets; fs_source_a, fs_source_b
//                                 in a pass of two and a pass of one on a
//                                 device that draws into 32 bytes a sample
//     fs_box_h2, fs_box_h1        their means over the box, across
//     fs_box_v2, fs_box_v1        and down
//   once a mask
//     fs_flood_seed               the seed of the reached field, 1 - Ec[p],
//                                 into one of the two flood targets
//     fs_flood                    one pass a doubling step s = 1, 2, 4, ...
//                                 cells while the steps sum to at most the
//                                 flood of the plan, from one flood target
//                                 into the other, the last into flood 0
//   each of the three gathers
//     fs_gather_first             the means of wq and wq I over each cell,
//       / fs_gather_moved         and of p, p p and p keep with the first; a
//                                 later gather moves each pixel into q as it
//                                 sums
//     fs_box_h2 / fs_box_h1       their means over the box, across
//     fs_box_v2 / fs_box_v1       and down
//     fs_solve                    the 15 numbers of each cell, in four
//                                 targets; fs_solve_a, fs_solve_b in two
//                                 passes of two on a device that draws
//                                 into 32 bytes a sample
//   after the last gather
//     fs_apply                    q at full resolution, mixed into p by
//                                 amount: the refined alpha
//
// A box of BLOCK_TAPS cells or more (2 cells + 1, from a radius of 12
// cells) is drawn in two passes: fs_block_h2, fs_block_v2, fs_block_h1 or
// fs_block_v1 sums BLOCK cells from every cell on along the axis, and the box
// then adds those sums BLOCK cells apart and the cells left over. It sums the
// same cells, each held as the direct loop holds it; only the order of the
// sum changes. A box under BLOCK_TAPS cells sums its cells in the direct
// loop.

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
    // How many blocks of BLOCK cells a box adds, and 0 when it sums every
    // cell in the direct loop.
    blocks: u32,
    // The step of a pass of fs_flood in cells, and 0 in every other pass.
    flood_step: u32,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var working: texture_2d<f32>;
@group(0) @binding(2) var alpha: texture_2d<f32>;
// The reached field on the cell grid, flood 0 after the last pass of
// fs_flood: read by fs_gather_first, and by fs_solve, which carries each
// cell's into the fourth channel of mid for fs_gather_moved and fs_apply. A
// group whose pass reads no reached field binds the alpha here.
@group(0) @binding(3) var reached_field: texture_2d<f32>;
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
const LIKENESS: f32 = 2.0;
const KEEP_LOW: f32 = 0.7;
const KEEP_HIGH: f32 = 1.0;
const UNREACHED_LOW: f32 = 0.0;
const UNREACHED_HIGH: f32 = 0.02;
const LEAVE_LOW: f32 = 0.2;
const LEAVE_HIGH: f32 = 0.5;
const KAPPA: f32 = 0.12857144;
const BETA_LOW: f32 = 0.95;
const BETA_HIGH: f32 = 1.1;
const GAIN_LOW: f32 = 0.45;
const GAIN_HIGH: f32 = 0.6;
const GAIN_ROOT: f32 = 0.0223607;

// The cells a block pass sums, and the fewest cells a box adds from block
// sums (refine.rs in gamut-gpu).
const BLOCK: u32 = 8u;
const BLOCK_TAPS: u32 = 24u;

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

// The three targets of the fused source, on a device that draws into 64
// bytes a sample.
struct Three {
    @location(0) one: vec4<f32>,
    @location(1) two: vec4<f32>,
    @location(2) three: vec4<f32>,
}

// The four targets of the fused solve, on a device that draws into 64 bytes
// a sample.
struct Four {
    @location(0) one: vec4<f32>,
    @location(1) two: vec4<f32>,
    @location(2) three: vec4<f32>,
    @location(3) four: vec4<f32>,
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

// (I, rr), (rg, rb, gg, gb) and (bb) in one pass of three targets, on a
// device that draws into 64 bytes a sample: the targets of fs_source_a, then
// that of fs_source_b, from one call of source_moments.
@fragment
fn fs_source(in: VertexOutput) -> Three {
    let m = source_moments(vec2<u32>(in.position.xy));
    var out: Three;
    out.one = vec4<f32>(m.i, m.ii_r.x);
    out.two = vec4<f32>(m.ii_r.y, m.ii_r.z, m.ii_g.x, m.ii_g.y);
    out.three = vec4<f32>(m.ii_g.z, 0.0, 0.0, 0.0);
    return out;
}

// The same in a pass of two targets and a pass of one, on a device that
// draws into 32 bytes a sample: (I, rr) and (rg, rb, gg, gb).
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

// The seed of the reached field in one cell: the outside as drawn, 1 - Ec[p],
// with Ec[p] the mean of the alpha over the cell's pixels summed as
// fs_gather_first sums it.
@fragment
fn fs_flood_seed(in: VertexOutput) -> @location(0) vec4<f32> {
    let span = cell_span(vec2<u32>(in.position.xy));
    var p = 0.0;
    for (var y = span.low.y; y < span.high.y; y = y + 1u) {
        for (var x = span.low.x; x < span.high.x; x = x + 1u) {
            p = p + textureLoad(alpha, vec2<i32>(i32(x), i32(y)), 0).r;
        }
    }
    return vec4<f32>(1.0 - p / span_count(span), 0.0, 0.0, 0.0);
}

// The reached field of the cell `direction` times the step from the one at
// `texel`, weighed by the likeness of the two cells' mean colours: `tex_a`
// holds the field of the pass before and `tex_b` the source's moments, whose
// first three are Ec[I]. Reads are held to the grid of the render as the
// twin clamps them.
fn likened(texel: vec2<u32>, here: vec3<f32>, direction: vec2<i32>, scale: f32) -> f32 {
    let at = beside(texel, direction, i32(u.flood_step));
    let there = textureLoad(tex_b, at, 0).xyz;
    let d = here - there;
    let like = exp(-(d.x * d.x + d.y * d.y + d.z * d.z) / scale);
    return textureLoad(tex_a, at, 0).r * like;
}

// One pass of the flood at the step `flood_step`: the most of the cell's own
// field and of the 8 cells a step away across, down and on the diagonals,
// each weighed by its likeness exp(-|Ec[I](x) - Ec[I](n)|^2 / (2 eps)), in
// the order of the twin.
@fragment
fn fs_flood(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = vec2<u32>(in.position.xy);
    let scale = LIKENESS * u.eps;
    let here = textureLoad(tex_b, vec2<i32>(texel), 0).xyz;
    var most = textureLoad(tex_a, vec2<i32>(texel), 0).r;
    most = max(most, likened(texel, here, vec2<i32>(1, 0), scale));
    most = max(most, likened(texel, here, vec2<i32>(-1, 0), scale));
    most = max(most, likened(texel, here, vec2<i32>(0, 1), scale));
    most = max(most, likened(texel, here, vec2<i32>(0, -1), scale));
    most = max(most, likened(texel, here, vec2<i32>(1, 1), scale));
    most = max(most, likened(texel, here, vec2<i32>(1, -1), scale));
    most = max(most, likened(texel, here, vec2<i32>(-1, 1), scale));
    most = max(most, likened(texel, here, vec2<i32>(-1, -1), scale));
    return vec4<f32>(most, 0.0, 0.0, 0.0);
}

// How much of a pixel the inside class keeps where the reached field is `rf`.
fn keep_of(rf: f32) -> f32 {
    return 1.0 - smooth_between(rf, KEEP_LOW, KEEP_HIGH);
}

// The weight of the inside class at a pixel: the mask so far `q` where the
// outside does not reach it, and alpha a move added above the mask as drawn
// `p` whether reached or not.
fn weight(q: f32, p: f32, keep: f32) -> f32 {
    return q * keep + max(q - p, 0.0) * (1.0 - keep);
}

// The first gather weighs the classes by the mask as drawn where the outside
// does not reach it: (wq, wq I) with wq = p keep, and the moments of the mask
// itself with the mean of that weight, (p, p p, p keep), which every solve
// takes the gate of the move from.
@fragment
fn fs_gather_first(in: VertexOutput) -> Pair {
    let span = cell_span(vec2<u32>(in.position.xy));
    var q = 0.0;
    var qi = vec3<f32>(0.0);
    var p = 0.0;
    var pp = 0.0;
    // The four cells of the reached field last read: the pixels that lie
    // among the same four read them once.
    var held = vec4<f32>(-1.0);
    var rf = vec4<f32>(0.0);
    for (var y = span.low.y; y < span.high.y; y = y + 1u) {
        let cy = among(y, u.origin.y, u.grid_first.y, u.grid_count.y, u.tile_first.y);
        for (var x = span.low.x; x < span.high.x; x = x + 1u) {
            let at = vec2<i32>(i32(x), i32(y));
            let cx = among(x, u.origin.x, u.grid_first.x, u.grid_count.x, u.tile_first.x);
            let cells = vec4<f32>(cx.x, cx.y, cy.x, cy.y);
            if any(cells != held) {
                held = cells;
                rf = corners_r(reached_field, cx, cy);
            }
            let a = textureLoad(alpha, at, 0).r;
            let g = acescct_encode(textureLoad(working, at, 0).rgb);
            let wq = weight(a, a, keep_of(mix_r(rf, cx.z, cy.z)));
            q = q + wq;
            qi = qi + g * wq;
            p = p + a;
            pp = pp + a * a;
        }
    }
    let count = span_count(span);
    var out: Pair;
    out.one = vec4<f32>(q / count, qi / count);
    out.two = vec4<f32>(p / count, pp / count, q / count, 0.0);
    return out;
}

// A later gather weighs them by wq of q, the mask moved at each pixel by what
// the gather before solved, `tex_a` to `tex_d`: (wq, wq I). q is taken here
// and never stored. It is moved() at each pixel, the same sums in the same
// order, with the four cells' texels read once for the pixels that lie among
// the same four.
@fragment
fn fs_gather_moved(in: VertexOutput) -> @location(0) vec4<f32> {
    let span = cell_span(vec2<u32>(in.position.xy));
    var q = 0.0;
    var qi = vec3<f32>(0.0);
    // The four cells last read in the four solved targets, the reached
    // field riding in mid's fourth channel: the pixels that lie among the
    // same four read them once.
    var held = vec4<f32>(-1.0);
    var a_s: Corners;
    var d2_c_p: Corners;
    var rest: Corners;
    var mid: Corners;
    for (var y = span.low.y; y < span.high.y; y = y + 1u) {
        let cy = among(y, u.origin.y, u.grid_first.y, u.grid_count.y, u.tile_first.y);
        for (var x = span.low.x; x < span.high.x; x = x + 1u) {
            let at = vec2<i32>(i32(x), i32(y));
            let cx = among(x, u.origin.x, u.grid_first.x, u.grid_count.x, u.tile_first.x);
            let cells = vec4<f32>(cx.x, cx.y, cy.x, cy.y);
            if any(cells != held) {
                held = cells;
                a_s = corners(tex_a, cx, cy);
                d2_c_p = corners(tex_b, cx, cy);
                rest = corners(tex_c, cx, cy);
                mid = corners(tex_d, cx, cy);
            }
            let p = textureLoad(alpha, at, 0).r;
            let g = acescct_encode(textureLoad(working, at, 0).rgb);
            let mid_here = mix_corners(mid, cx.z, cy.z);
            let pqr = moved_from(
                p,
                g,
                mid_here.w,
                mix_corners(a_s, cx.z, cy.z),
                mix_corners(d2_c_p, cx.z, cy.z),
                mix_corners(rest, cx.z, cy.z),
                mid_here,
            );
            let wq = weight(pqr.y, pqr.x, keep_of(pqr.z));
            q = q + wq;
            qi = qi + g * wq;
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

// The sum of the BLOCK cells from the one at `texel` on along `direction`,
// each held by beside() as the direct loop holds it. Not divided: the box
// divides.
fn block_pair(texel: vec2<u32>, direction: vec2<i32>) -> Pair {
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    for (var k = 0; k < i32(BLOCK); k = k + 1) {
        let at = beside(texel, direction, k);
        a = a + textureLoad(tex_a, at, 0);
        b = b + textureLoad(tex_b, at, 0);
    }
    var out: Pair;
    out.one = a;
    out.two = b;
    return out;
}

fn block_one(texel: vec2<u32>, direction: vec2<i32>) -> vec4<f32> {
    var a = vec4<f32>(0.0);
    for (var k = 0; k < i32(BLOCK); k = k + 1) {
        a = a + textureLoad(tex_a, beside(texel, direction, k), 0);
    }
    return a;
}

@fragment
fn fs_block_h2(in: VertexOutput) -> Pair {
    return block_pair(vec2<u32>(in.position.xy), vec2<i32>(1, 0));
}

@fragment
fn fs_block_v2(in: VertexOutput) -> Pair {
    return block_pair(vec2<u32>(in.position.xy), vec2<i32>(0, 1));
}

@fragment
fn fs_block_h1(in: VertexOutput) -> @location(0) vec4<f32> {
    return block_one(vec2<u32>(in.position.xy), vec2<i32>(1, 0));
}

@fragment
fn fs_block_v1(in: VertexOutput) -> @location(0) vec4<f32> {
    return block_one(vec2<u32>(in.position.xy), vec2<i32>(0, 1));
}

// Whether the block `offset` cells along `direction` from `texel` is the one
// the block pass wrote at `at`, its first cell held by beside(). It is while
// its first cell lies at or past the near end of what beside() holds, and
// past the far end too, where the block pass summed the last cell BLOCK
// times. Before the near end its cells are summed one at a time, so each is
// still held as the direct loop holds it.
fn block_written(texel: vec2<u32>, direction: vec2<i32>, offset: i32, at: vec2<i32>) -> bool {
    let unheld = vec2<i32>(texel) + direction * offset;
    return dot(at - unheld, direction) <= 0;
}

// The box of box_pair from the block sums of the block pass (tex_c, tex_d)
// BLOCK cells apart from -cells, and the cells left over at the far end from
// tex_a and tex_b.
fn box_pair_blocks(texel: vec2<u32>, direction: vec2<i32>) -> Pair {
    let radius = i32(u.cells);
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    for (var j = 0; j < i32(u.blocks); j = j + 1) {
        let offset = -radius + j * i32(BLOCK);
        let at = beside(texel, direction, offset);
        if block_written(texel, direction, offset, at) {
            a = a + textureLoad(tex_c, at, 0);
            b = b + textureLoad(tex_d, at, 0);
        } else {
            for (var k = 0; k < i32(BLOCK); k = k + 1) {
                let cell = beside(texel, direction, offset + k);
                a = a + textureLoad(tex_a, cell, 0);
                b = b + textureLoad(tex_b, cell, 0);
            }
        }
    }
    for (var i = -radius + i32(u.blocks * BLOCK); i <= radius; i = i + 1) {
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

fn box_one_blocks(texel: vec2<u32>, direction: vec2<i32>) -> vec4<f32> {
    let radius = i32(u.cells);
    var a = vec4<f32>(0.0);
    for (var j = 0; j < i32(u.blocks); j = j + 1) {
        let offset = -radius + j * i32(BLOCK);
        let at = beside(texel, direction, offset);
        if block_written(texel, direction, offset, at) {
            a = a + textureLoad(tex_c, at, 0);
        } else {
            for (var k = 0; k < i32(BLOCK); k = k + 1) {
                a = a + textureLoad(tex_a, beside(texel, direction, offset + k), 0);
            }
        }
    }
    for (var i = -radius + i32(u.blocks * BLOCK); i <= radius; i = i + 1) {
        a = a + textureLoad(tex_a, beside(texel, direction, i), 0);
    }
    return a / f32(2 * radius + 1);
}

// Whether a box adds block sums: the uniform carries its blocks, and a box
// under BLOCK_TAPS cells sums every cell in the direct loop whatever it says.
fn from_blocks() -> bool {
    return u.blocks > 0u && 2u * u.cells + 1u >= BLOCK_TAPS;
}

@fragment
fn fs_box_h2(in: VertexOutput) -> Pair {
    let texel = vec2<u32>(in.position.xy);
    if from_blocks() {
        return box_pair_blocks(texel, vec2<i32>(1, 0));
    }
    return box_pair(texel, vec2<i32>(1, 0));
}

@fragment
fn fs_box_v2(in: VertexOutput) -> Pair {
    let texel = vec2<u32>(in.position.xy);
    if from_blocks() {
        return box_pair_blocks(texel, vec2<i32>(0, 1));
    }
    return box_pair(texel, vec2<i32>(0, 1));
}

@fragment
fn fs_box_h1(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = vec2<u32>(in.position.xy);
    if from_blocks() {
        return box_one_blocks(texel, vec2<i32>(1, 0));
    }
    return box_one(texel, vec2<i32>(1, 0));
}

@fragment
fn fs_box_v1(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = vec2<u32>(in.position.xy);
    if from_blocks() {
        return box_one_blocks(texel, vec2<i32>(0, 1));
    }
    return box_one(texel, vec2<i32>(0, 1));
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

// bothA, how far the move comes in over a box: both() of the mask's moments,
// times how much of the inside the outside does not reach, E[p keep], which
// is gather 1's own mean weight. The same number in every gather: E[p keep]
// is carried in the mask's means, which no later gather writes.
fn gate(p: f32, pp: f32, unreached: f32) -> f32 {
    return both(p, pp) * smooth_between(unreached, UNREACHED_LOW, UNREACHED_HIGH);
}

// The box's half width against the mask's own ramp, from the means of the
// mask's moments: E[p] - E[p p] is the box mean of p (1 - p), the ramp is
// spp (2 b + 1) step / KAPPA pixels wide, and beta is b step over it, with b
// the box's radius in cells. The twin takes it once a mask; every solve here
// forms it again from the same means, as it does the gate.
fn beta_of(p: f32, pp: f32) -> f32 {
    let spp = max(p - pp, 0.0);
    let width = spp * f32(2u * u.cells + 1u) * f32(u.step) / KAPPA;
    return f32(u.cells) * f32(u.step) / max(width, SHARE_FLOOR);
}

// The gain of the classes: GAIN_ROOT over the length of the class step
// mu1 - mu0, summed as the twin sums it.
fn gain_of(delta: vec3<f32>) -> f32 {
    let apart = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    return GAIN_ROOT / max(apart, SEPARATION_FLOOR);
}

// How far a gather's move comes in over a soft rim, 0 to 1: 0 where the box
// is no wider than the mask's ramp and the classes lie close, 1 where the box
// is wider or the edge strong. A soft rim over a weak edge holds still.
fn hold(beta: f32, gain: f32) -> f32 {
    return 1.0
        - (1.0 - smooth_between(beta, BETA_LOW, BETA_HIGH))
            * smooth_between(gain, GAIN_LOW, GAIN_HIGH);
}

// `tex_a` to `tex_c` hold the means of the source's moments, (I, rr),
// (rg, rb, gg, gb) and (bb); `tex_d` those of the gather, (wq, wq I);
// `tex_e` those of the mask, (p, p p, p keep). The cell's reached field
// rides in mid's fourth channel, which holds no solved number, so moved()
// mixes it from the four cells it reads for mid. The move c is scaled by
// hold() of the mask's beta and the gather's own gain, as the twin's solve.
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
    let gain = gain_of(delta);
    let beta = beta_of(mask.x, mask.y);
    let c = smooth_between(d2, SEPARATE_LOW, SEPARATE_HIGH) * gate(mask.x, mask.y, mask.z)
        * hold(beta, gain);
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
    out.mid = vec4<f32>(mid, textureLoad(reached_field, at, 0).r);
    return out;
}

// The 15 numbers of each cell in one pass of four targets, on a device that
// draws into 64 bytes a sample: the targets of fs_solve_a, then those of
// fs_solve_b, from one call of solve.
@fragment
fn fs_solve(in: VertexOutput) -> Four {
    let solved = solve(vec2<i32>(in.position.xy));
    var out: Four;
    out.one = solved.a_s;
    out.two = solved.d2_c_p;
    out.three = solved.p;
    out.four = solved.mid;
    return out;
}

// The same 15 numbers in two passes of two targets, on a device that draws
// into 32 bytes a sample.
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

// The texels of the four cells around a pixel in one target: top left, top
// right, bottom left and bottom right.
struct Corners {
    tl: vec4<f32>,
    tr: vec4<f32>,
    bl: vec4<f32>,
    br: vec4<f32>,
}

fn corners(solved: texture_2d<f32>, x: vec3<f32>, y: vec3<f32>) -> Corners {
    var c: Corners;
    c.tl = textureLoad(solved, vec2<i32>(i32(x.x), i32(y.x)), 0);
    c.tr = textureLoad(solved, vec2<i32>(i32(x.y), i32(y.x)), 0);
    c.bl = textureLoad(solved, vec2<i32>(i32(x.x), i32(y.y)), 0);
    c.br = textureLoad(solved, vec2<i32>(i32(x.y), i32(y.y)), 0);
    return c;
}

// The four cells' texels mixed at the shares `sx` across and `sy` down.
fn mix_corners(c: Corners, sx: f32, sy: f32) -> vec4<f32> {
    let top = c.tl + (c.tr - c.tl) * sx;
    let bottom = c.bl + (c.br - c.bl) * sx;
    return top + (bottom - top) * sy;
}

// One of the four targets of what was solved, mixed from the four cells
// around a pixel.
fn mixed(solved: texture_2d<f32>, x: vec3<f32>, y: vec3<f32>) -> vec4<f32> {
    return mix_corners(corners(solved, x, y), x.z, y.z);
}

// The first channel of the four cells around a pixel, in the order of
// Corners, and the same mix of them.
fn corners_r(field: texture_2d<f32>, x: vec3<f32>, y: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(
        textureLoad(field, vec2<i32>(i32(x.x), i32(y.x)), 0).r,
        textureLoad(field, vec2<i32>(i32(x.y), i32(y.x)), 0).r,
        textureLoad(field, vec2<i32>(i32(x.x), i32(y.y)), 0).r,
        textureLoad(field, vec2<i32>(i32(x.y), i32(y.y)), 0).r,
    );
}

fn mix_r(c: vec4<f32>, sx: f32, sy: f32) -> f32 {
    let top = c.x + (c.y - c.x) * sx;
    let bottom = c.z + (c.w - c.z) * sx;
    return top + (bottom - top) * sy;
}

// The mask as drawn moved at the pixel `pixel` of the render: p, q and the
// reached field there. `tex_a` to `tex_d` hold what the gather solved, and
// `tex_d` the cells' reached field in its fourth channel: mixed from the four
// cells around the pixel as fs_gather_first mixes it, the same number. A pixel
// leaves the mask only where the reached field reaches it.
fn moved(pixel: vec2<u32>) -> vec3<f32> {
    let at = vec2<i32>(pixel);
    let p = textureLoad(alpha, at, 0).r;
    let g = acescct_encode(textureLoad(working, at, 0).rgb);
    let x = among(pixel.x, u.origin.x, u.grid_first.x, u.grid_count.x, u.tile_first.x);
    let y = among(pixel.y, u.origin.y, u.grid_first.y, u.grid_count.y, u.tile_first.y);
    let a_s = mixed(tex_a, x, y);
    let d2_c_p = mixed(tex_b, x, y);
    let rest = mixed(tex_c, x, y);
    let mid = mixed(tex_d, x, y);
    return moved_from(p, g, mid.w, a_s, d2_c_p, rest, mid);
}

// moved() from the numbers it reads and mixes: p, the guide g, the reached
// field and the four solved targets mixed at the pixel.
fn moved_from(
    p: f32,
    g: vec3<f32>,
    rf: f32,
    a_s: vec4<f32>,
    d2_c_p: vec4<f32>,
    rest: vec4<f32>,
    mid: vec4<f32>,
) -> vec3<f32> {
    let along = (a_s.x * g.r + a_s.y * g.g + a_s.z * g.b) - a_s.w;
    let est = clamp(0.5 + along / max(d2_c_p.x, SEPARATION_FLOOR), 0.0, 1.0);
    let e = g - mid.xyz;
    let m = max(
        d2_c_p.z * e.x * e.x + rest.y * e.y * e.y + rest.w * e.z * e.z
            + 2.0 * (d2_c_p.w * e.x * e.y + rest.x * e.x * e.z + rest.z * e.y * e.z),
        0.0,
    );
    let on_line = 1.0 - smooth_between(m, LINE_LOW, LINE_HIGH);
    let out = on_line * (1.0 - smooth_between(est, OUT_LOW, OUT_HIGH))
        * smooth_between(rf, LEAVE_LOW, LEAVE_HIGH);
    let into = on_line * smooth_between(est, IN_LOW, IN_HIGH);
    return vec3<f32>(p, p + d2_c_p.y * (into * (1.0 - p) - out * p), rf);
}

// The q of the last gather, mixed into p by amount: the refined alpha.
@fragment
fn fs_apply(in: VertexOutput) -> @location(0) vec4<f32> {
    let pq = moved(vec2<u32>(in.position.xy));
    return vec4<f32>(clamp(pq.x + (pq.y - pq.x) * u.amount, 0.0, 1.0), 0.0, 0.0, 1.0);
}
