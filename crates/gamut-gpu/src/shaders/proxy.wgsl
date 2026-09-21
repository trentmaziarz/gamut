// Reduces the source to the reference proxy an auto brush reads: a box
// filter in linear Rec.2020, gamut-color's Proxy::from_source line for line.
//
// The source is first drawn a tile at a time at its own size by the pass
// every render reads it through (the input transform or the video pass, one
// sample a pixel on the pixel's centre), so a tile holds the working pixels
// of the source. This pass then averages, for each proxy pixel of the tile,
// the source pixels it covers, each weighted by the part of it that is
// covered. It is drawn under a scissor of the proxy pixels the tile serves.

struct Uniform {
    source_size: vec2<f32>,
    proxy_size: vec2<f32>,
    // The source pixel texel (0, 0) of the tile holds.
    tile_origin: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniform;
@group(0) @binding(1) var tile: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let span = u.source_size / u.proxy_size;
    let low = floor(position.xy) * span;
    let high = low + span;
    let first = vec2<i32>(floor(low));
    let last = min(vec2<i32>(ceil(high)), vec2<i32>(u.source_size));
    let origin = vec2<i32>(u.tile_origin);
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var sy = first.y; sy < last.y; sy = sy + 1) {
        let wy = min(high.y, f32(sy) + 1.0) - max(low.y, f32(sy));
        if (wy <= 0.0) {
            continue;
        }
        for (var sx = first.x; sx < last.x; sx = sx + 1) {
            let wx = min(high.x, f32(sx) + 1.0) - max(low.x, f32(sx));
            if (wx <= 0.0) {
                continue;
            }
            let weight = wx * wy;
            sum = sum + textureLoad(tile, vec2<i32>(sx, sy) - origin, 0).rgb * weight;
            total = total + weight;
        }
    }
    return vec4<f32>(sum / total, 1.0);
}
