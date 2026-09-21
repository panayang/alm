//! Probe a representation that has already been produced.
//!
//! The patient readout built its features from 2.5M ticks and then scored them
//! with 300 steps of full-batch gradient descent, the same 300 for every feature
//! set. Full-batch descent converges at a rate set by the conditioning of the
//! features, and the sets differ enormously: the count-and-gap control has a few
//! dozen columns, the model state has 2560. If 300 steps settles the control and
//! does not settle the state, the comparison is between two amounts of
//! optimisation rather than between two representations -- and the direction of
//! that bias is always toward the narrower set.
//!
//! There is already a symptom in the record. Adding bigrams to the counts
//! *lowered* the score, 0.7936 to 0.7544, which cannot happen for an information
//! reason: the wider set contains the narrower one. It was read as overfitting
//! and answered by sweeping the regularisation, and the regularisation sweep
//! chose an interior value, so that was not the answer either.
//!
//! So sweep the iteration count as well, and print whether the best is at the
//! edge of the grid. A number taken at the edge is a lower bound on what that
//! representation holds, not a measurement of it.

use crate::patient::{auroc_pub, logistic_pub};

pub struct Dump {
    pub ids: Vec<u32>,
    pub y: Vec<u8>,
    pub sets: Vec<(String, usize, Vec<Vec<f32>>)>,
}

fn rd_u64(b: &[u8], p: &mut usize) -> u64 {
    let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}

pub fn load(path: &str) -> Dump {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
    let mut p = 0usize;
    let n = rd_u64(&b, &mut p) as usize;
    let nsets = rd_u64(&b, &mut p) as usize;
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        ids.push(u32::from_le_bytes(b[p..p + 4].try_into().unwrap()));
        p += 4;
    }
    let y = b[p..p + n].to_vec();
    p += n;
    let mut sets = Vec::with_capacity(nsets);
    for _ in 0..nsets {
        let ln = rd_u64(&b, &mut p) as usize;
        let name = String::from_utf8_lossy(&b[p..p + ln]).to_string();
        p += ln;
        let d = rd_u64(&b, &mut p) as usize;
        let mut rows = Vec::with_capacity(n);
        for _ in 0..n {
            let mut r = Vec::with_capacity(d);
            for _ in 0..d {
                r.push(f32::from_le_bytes(b[p..p + 4].try_into().unwrap()));
                p += 4;
            }
            rows.push(r);
        }
        sets.push((name, d, rows));
    }
    Dump { ids, y, sets }
}

fn cv(x: &[Vec<f32>], y: &[u8], ids: &[u32], l2: f64, iters: usize) -> f64 {
    let folds = 5usize;
    let mut score = vec![0.0f64; x.len()];
    for f in 0..folds {
        let train: Vec<usize> = (0..x.len()).filter(|&i| ids[i] as usize % folds != f).collect();
        let test: Vec<usize> = (0..x.len()).filter(|&i| ids[i] as usize % folds == f).collect();
        if train.is_empty() || test.is_empty() {
            continue;
        }
        let s = logistic_pub(x, y, &train, &test, l2, iters);
        for (k, &i) in test.iter().enumerate() {
            score[i] = s[k];
        }
    }
    auroc_pub(&score, y)
}

pub fn run(path: &str) {
    let dm = load(path);
    let l2s = [1e-4f64, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0];
    let iters = [300usize, 1000, 3000, 10000];
    println!("probing {}: {} patients, {} sets", path, dm.ids.len(), dm.sets.len());
    println!("  both grids swept per set. an optimum at the edge of either is a lower bound.\n");
    println!(
        "{:>10} {:>7} {:>9} {:>8} {:>9} {:>12} {:>10}",
        "set", "width", "AUROC", "L2", "iters", "at 300 iters", "edge?"
    );
    for (name, d, x) in dm.sets.iter() {
        let mut best = (0.0f64, 0.0f64, 0usize);
        let mut at300 = 0.0f64;
        for &it in iters.iter() {
            for &l2 in l2s.iter() {
                let a = cv(x, &dm.y, &dm.ids, l2, it);
                if it == 300 && a > at300 {
                    at300 = a;
                }
                if a > best.0 {
                    best = (a, l2, it);
                }
            }
        }
        let edge = if best.2 == *iters.last().unwrap() && best.1 == *l2s.last().unwrap() {
            "both"
        } else if best.2 == *iters.last().unwrap() {
            "iters"
        } else if best.1 == *l2s.last().unwrap() {
            "L2"
        } else {
            "-"
        };
        println!(
            "{:>10} {:>7} {:>9.4} {:>8} {:>9} {:>12.4} {:>10}",
            name, d, best.0, best.1, best.2, at300, edge
        );
    }
    println!("\n  the `at 300 iters` column is what the previous readout reported. where it");
    println!("  is far below the swept number, that set was being scored on how far 300");
    println!("  steps got rather than on what it holds -- and the wider the set, the less");
    println!("  far 300 steps gets, so the bias always favours the narrower one.");
}
