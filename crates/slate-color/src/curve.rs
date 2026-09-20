//! The point tone curve: monotone cubic Hermite interpolation with
//! Fritsch-Carlson tangents, baked into one table per channel. The CPU twin
//! and the shader both read the table with linear interpolation between
//! neighbours, so they agree by construction.
//!
//! The axis is normalised ACEScct (see [`crate::acescct::normalise`]): 0 is
//! black and 1 is diffuse white. Past either end the curve continues with
//! slope 1 from the end of the table.

use slate_core::look::{Curve, ToneCurves};

use crate::acescct;

/// The number of entries in a baked table.
pub const TABLE_SIZE: usize = 1024;

/// One baked channel.
pub type Table = [f32; TABLE_SIZE];

/// The three baked channels. Each already composes the channel curve, then
/// the master curve.
#[derive(Clone, Debug, PartialEq)]
pub struct Tables {
    pub red: Box<Table>,
    pub green: Box<Table>,
    pub blue: Box<Table>,
}

/// The Fritsch-Carlson tangents of a sanitised point list. A monotone list
/// gets tangents that keep the interpolant monotone, so it never overshoots.
pub fn tangents(points: &[[f32; 2]]) -> Vec<f32> {
    let n = points.len();
    let secants: Vec<f32> = points
        .windows(2)
        .map(|w| (w[1][1] - w[0][1]) / (w[1][0] - w[0][0]))
        .collect();
    let mut m = vec![0.0; n];
    m[0] = secants[0];
    m[n - 1] = secants[n - 2];
    for i in 1..n - 1 {
        m[i] = if secants[i - 1] * secants[i] <= 0.0 {
            0.0
        } else {
            (secants[i - 1] + secants[i]) / 2.0
        };
    }
    for (i, &d) in secants.iter().enumerate() {
        if d == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let a = m[i] / d;
        let b = m[i + 1] / d;
        let length = a * a + b * b;
        if length > 9.0 {
            let tau = 3.0 / length.sqrt();
            m[i] = tau * a * d;
            m[i + 1] = tau * b * d;
        }
    }
    m
}

/// The curve at `x` in 0 to 1. Left of the first point and right of the last
/// the curve is flat at that point's y.
pub fn evaluate(curve: &Curve, x: f32) -> f32 {
    let curve = curve.sanitised();
    evaluate_sanitised(&curve.points, &tangents(&curve.points), x)
}

fn evaluate_sanitised(points: &[[f32; 2]], m: &[f32], x: f32) -> f32 {
    let last = points.len() - 1;
    if x <= points[0][0] {
        return points[0][1];
    }
    if x >= points[last][0] {
        return points[last][1];
    }
    let i = points.partition_point(|p| p[0] <= x) - 1;
    let [x0, y0] = points[i];
    let [x1, y1] = points[i + 1];
    let h = x1 - x0;
    let t = (x - x0) / h;
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * h * m[i]
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * h * m[i + 1]
}

/// One channel's table: the channel curve, then the master curve, sampled at
/// i over 1023.
pub fn bake_channel(channel: &Curve, master: &Curve) -> Box<Table> {
    let channel = channel.sanitised();
    let master = master.sanitised();
    let channel_m = tangents(&channel.points);
    let master_m = tangents(&master.points);
    let mut table = Box::new([0.0; TABLE_SIZE]);
    for (i, entry) in table.iter_mut().enumerate() {
        let x = i as f32 / (TABLE_SIZE - 1) as f32;
        let y = evaluate_sanitised(&channel.points, &channel_m, x);
        *entry = evaluate_sanitised(&master.points, &master_m, y);
    }
    table
}

/// The three tables of a set of curves.
pub fn bake(curves: &ToneCurves) -> Tables {
    Tables {
        red: bake_channel(&curves.red, &curves.master),
        green: bake_channel(&curves.green, &curves.master),
        blue: bake_channel(&curves.blue, &curves.master),
    }
}

/// The table at `x`: linear interpolation between the two neighbours inside
/// 0 to 1, slope 1 from the end entry outside. The shader reads its table
/// texture with the same arithmetic.
pub fn lookup(table: &Table, x: f32) -> f32 {
    if x < 0.0 {
        return table[0] + x;
    }
    if x > 1.0 {
        return table[TABLE_SIZE - 1] + (x - 1.0);
    }
    let position = x * (TABLE_SIZE - 1) as f32;
    let i = (position.floor() as usize).min(TABLE_SIZE - 2);
    let f = position - i as f32;
    table[i] * (1.0 - f) + table[i + 1] * f
}

/// The tone curves on one ACEScct pixel.
pub fn apply(v: [f32; 3], tables: &Tables) -> [f32; 3] {
    let channel =
        |value: f32, table: &Table| acescct::denormalise(lookup(table, acescct::normalise(value)));
    [
        channel(v[0], &tables.red),
        channel(v[1], &tables.green),
        channel(v[2], &tables.blue),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(points: &[[f32; 2]]) -> Curve {
        Curve {
            points: points.to_vec(),
        }
    }

    #[test]
    fn the_default_curve_bakes_to_the_diagonal() {
        let table = bake_channel(&Curve::default(), &Curve::default());
        for (i, y) in table.iter().enumerate() {
            let x = i as f32 / (TABLE_SIZE - 1) as f32;
            assert!((y - x).abs() < 1e-6, "{x} maps to {y}");
        }
        for x in [-0.3, 0.0, 0.2501, 0.9, 1.0, 1.7] {
            assert!((lookup(&table, x) - x).abs() < 1e-6, "{x}");
        }
    }

    #[test]
    fn a_monotone_point_list_gives_a_monotone_table_through_every_point() {
        let lists: [&[[f32; 2]]; 3] = [
            &[[0.0, 0.0], [0.25, 0.15], [0.75, 0.85], [1.0, 1.0]],
            &[[0.0, 0.0], [0.1, 0.6], [0.2, 0.62], [0.9, 0.65], [1.0, 1.0]],
            &[[0.0, 0.1], [0.5, 0.5], [0.52, 0.9], [1.0, 0.95]],
        ];
        for points in lists {
            let c = curve(points);
            let table = bake_channel(&c, &Curve::default());
            for pair in table.windows(2) {
                assert!(pair[1] >= pair[0] - 1e-7, "{points:?} is not monotone");
            }
            for [x, y] in points {
                assert!((evaluate(&c, *x) - y).abs() < 1e-4, "{points:?} at {x}");
            }
            let (low, high) = (points[0][1], points[points.len() - 1][1]);
            assert!(table.iter().all(|y| *y >= low - 1e-6 && *y <= high + 1e-6));
        }
    }

    #[test]
    fn the_channel_curve_runs_before_the_master() {
        let channel = curve(&[[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]]);
        let master = curve(&[[0.0, 0.0], [0.25, 0.75], [1.0, 1.0]]);
        let table = bake_channel(&channel, &master);
        assert!((lookup(&table, 0.5) - 0.75).abs() < 1e-3);
    }

    #[test]
    fn past_either_end_the_table_continues_with_slope_one() {
        let table = bake_channel(&curve(&[[0.0, 0.1], [1.0, 0.8]]), &Curve::default());
        assert!((lookup(&table, -0.5) - (0.1 - 0.5)).abs() < 1e-6);
        assert!((lookup(&table, 1.5) - (0.8 + 0.5)).abs() < 1e-6);
    }

    #[test]
    fn the_curves_move_only_their_own_channel() {
        let curves = ToneCurves {
            red: curve(&[[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]]),
            ..ToneCurves::default()
        };
        let tables = bake(&curves);
        let v = [0.3, 0.3, 0.3];
        let out = apply(v, &tables);
        assert!(out[0] > v[0] + 0.01);
        assert!((out[1] - v[1]).abs() < 1e-6 && (out[2] - v[2]).abs() < 1e-6);
    }
}
