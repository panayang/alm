//! Asked at the bedside, at any moment.
//!
//! The patient readout took a snapshot -- the features at the last tick, or
//! pooled over the stay -- and fitted a probe to it. The foundations draft says
//! what that can and cannot see: an evaluation that can be redone at any time
//! sees only terminal structure, and seeing present ability needs individuation
//! by time, which cannot be redone. And the response machinery -- a question
//! arriving, a silence in which the response unfolds, a commitment -- took no
//! part in it at all.
//!
//! So ask it the way this architecture answers. The outcome is an event the
//! world speaks at the end of a stay, charged and written like any other token;
//! nothing about it reaches the model by another route. During the stay the
//! question is put at a quarter, a half and three quarters of the way through,
//! as a probe: writes frozen, the volatile state saved and restored, so asking
//! does not disturb the patient being asked about. At the end the question is
//! put for real, and the world answers.
//!
//! Memory is continual across every patient and never reset. The first half of
//! the patients are where it learns what outcomes follow what; the second half
//! is read. That is a prequential split, not the five-fold cross-validation the
//! snapshot probe used, so the numbers here are not comparable to that one and
//! should not be read against it.
//!
//! What comes out is a curve over the stay, not a number.

use crate::clinical::flat_streams_valued;
use crate::config::Config;
use crate::model::Model;
use crate::patient::{auroc_pub, logistic_pub};

const SILENCE: usize = 6;
const FRACS: [f64; 4] = [0.25, 0.5, 0.75, 1.0];

#[inline]
fn silence_ticks(g: u32, cap: usize) -> usize {
    ((32 - g.max(1).leading_zeros()).saturating_sub(1) as usize).min(cap)
}

fn load_death(outcomes: &str) -> std::collections::HashMap<u32, u8> {
    let text = std::fs::read_to_string(outcomes).unwrap_or_else(|e| panic!("{}: {}", outcomes, e));
    let mut m = std::collections::HashMap::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if f.len() >= 6 {
            if let (Ok(id), Ok(d)) = (f[0].parse::<u32>(), f[5].parse::<u8>()) {
                m.insert(id, d);
            }
        }
    }
    m
}

struct Asked {
    /// P(died | {died, survived}) at each fraction of the stay.
    q: [f64; 4],
    changes: [u32; 4],
    bits: f64,
}

fn ask(m: &mut Model, q_tok: usize, died: usize, survived: usize, changes: &mut u32) -> f64 {
    let read = |m: &Model| {
        let sc = m.spread_now();
        let a = sc.prob_of(&m.store, died as u32) as f64;
        let b = sc.prob_of(&m.store, survived as u32) as f64;
        a / (a + b).max(1e-30)
    };
    m.tick(Some(q_tok), false);
    let mut prev = read(m) >= 0.5;
    for _ in 0..SILENCE {
        m.tick(None, false);
        let now = read(m) >= 0.5;
        if now != prev {
            *changes += 1;
            prev = now;
        }
    }
    read(m)
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    streams: &[Vec<(u32, u32)>],
    labels: &[u8],
    vp: usize,
    d: usize,
    banks: usize,
    cap: usize,
    seed: u64,
    wiped: bool,
) -> Vec<Asked> {
    let (q_tok, died, survived) = (vp, vp + 1, vp + 2);
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = vp + 3;
    cfg.d = d;
    cfg.cleanup_floor_mult = 1.1;
    cfg.mem_banks = banks;
    cfg.derive();
    let mut m = Model::new(cfg);
    let blank = m.volatile();
    let n = streams.len();
    let t0 = std::time::Instant::now();
    let mut out = Vec::with_capacity(n);
    for (pi, (s, &y)) in streams.iter().zip(labels).enumerate() {
        if pi > 0 && pi % (n / 20).max(1) == 0 {
            let el = t0.elapsed().as_secs_f64();
            eprintln!(
                "  {} arm: {}/{} patients, {:.0}s elapsed, ~{:.0}s left",
                if wiped { "wiped" } else { "full" },
                pi,
                n,
                el,
                el * (n - pi) as f64 / pi as f64
            );
        }
        m.restore(blank.clone());
        let marks: Vec<usize> =
            FRACS[..3].iter().map(|f| ((s.len() as f64 * f) as usize).max(1)).collect();
        let mut a = Asked { q: [0.5; 4], changes: [0; 4], bits: 0.0 };
        for (k, &(gap, tok)) in s.iter().enumerate() {
            for _ in 0..silence_ticks(gap, cap) {
                m.tick(None, false);
            }
            m.tick(Some(tok as usize), false);
            for (fi, &mk) in marks.iter().enumerate() {
                if k + 1 == mk {
                    // A probe: nothing written, and the patient's situation put
                    // back exactly as it was once the question has been asked.
                    let saved = m.volatile();
                    m.frozen = true;
                    if wiped {
                        m.restore(blank.clone());
                    }
                    a.q[fi] = ask(&mut m, q_tok, died, survived, &mut a.changes[fi]);
                    m.restore(saved);
                    m.frozen = false;
                }
            }
        }
        // The end of the stay: asked for real, and the world answers.
        if wiped {
            m.restore(blank.clone());
        }
        a.q[3] = ask(&mut m, q_tok, died, survived, &mut a.changes[3]);
        let o = m.tick(Some(if y == 1 { died } else { survived }), false);
        a.bits = o.bits;
        out.push(a);
    }
    out
}

/// The count-and-gap summary over the first `frac` of a stay: what an
/// order-free counter could know at that moment.
fn summary(s: &[(u32, u32)], vp: usize, frac: f64) -> Vec<f32> {
    let upto = ((s.len() as f64 * frac) as usize).max(1).min(s.len());
    let mut cnt = vec![0.0f32; vp];
    let mut gapsum = vec![0.0f32; vp];
    for &(gap, tok) in s[..upto].iter() {
        cnt[tok as usize] += 1.0;
        gapsum[tok as usize] += gap as f32;
    }
    let mut f = cnt.clone();
    for (k, g) in gapsum.iter().enumerate() {
        f.push(g / cnt[k].max(1.0));
    }
    f.push(upto as f32);
    f
}

/// The counter on the same split: L2 chosen by five-fold cross-validation inside
/// the first half only, then fitted on the whole first half and scored on the
/// second. The same protocol the model is held to, with its own sweep.
fn counter_auroc(x: &[Vec<f32>], y: &[u8], half: usize) -> (f64, f64) {
    let train: Vec<usize> = (0..half).collect();
    let mut best = (0.0f64, 1.0f64);
    for &l2 in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0].iter() {
        let mut score = vec![0.0f64; half];
        for f in 0..5usize {
            let tr: Vec<usize> = train.iter().copied().filter(|i| i % 5 != f).collect();
            let te: Vec<usize> = train.iter().copied().filter(|i| i % 5 == f).collect();
            let s = logistic_pub(x, y, &tr, &te, l2, 300);
            for (k, &i) in te.iter().enumerate() {
                score[i] = s[k];
            }
        }
        let a = auroc_pub(&score, &y[..half]);
        if a > best.0 {
            best = (a, l2);
        }
    }
    let test: Vec<usize> = (half..x.len()).collect();
    let s = logistic_pub(x, y, &train, &test, best.1, 300);
    (auroc_pub(&s, &y[half..]), best.1)
}

/// The same counter held to the model's own regime: online, one pass, one step
/// per patient, predicting each before learning from it.
///
/// The batch counter above fits 300 full-batch steps over every labelled
/// patient in the first half -- each outcome seen hundreds of times. The model
/// sees each outcome once, the moment the world speaks it. Comparing the two
/// compares acquisition regimes as much as representations, so this is the
/// comparator on the model's terms: same features, same order, same single
/// pass. Features are log-scaled and standardised by running statistics of the
/// patients seen so far, never by anything ahead. The step size is chosen by
/// its prequential AUROC on the first half and read on the second.
fn online_counter_auroc(x: &[Vec<f32>], y: &[u8], half: usize) -> (f64, f64) {
    let d = x[0].len();
    let mut best = (0.0f64, 0.0f64, 0.0f64);
    for &lr in [0.003f64, 0.01, 0.03, 0.1, 0.3].iter() {
        let mut w = vec![0.0f64; d];
        let mut b = 0.0f64;
        let mut mean = vec![0.0f64; d];
        let mut m2 = vec![0.0f64; d];
        let mut pred = Vec::with_capacity(x.len());
        for (i, row) in x.iter().enumerate() {
            let n = i as f64;
            let z: Vec<f64> = row
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let v = (1.0 + v.max(0.0) as f64).ln();
                    let sd = if n > 1.0 { (m2[j] / (n - 1.0)).sqrt().max(1e-6) } else { 1.0 };
                    (v - mean[j]) / sd
                })
                .collect();
            let s: f64 = b + w.iter().zip(&z).map(|(a, c)| a * c).sum::<f64>();
            let p = 1.0 / (1.0 + (-s).exp());
            pred.push(s);
            let e = y[i] as f64 - p;
            for j in 0..d {
                w[j] += lr * e * z[j];
            }
            b += lr * e;
            // Running statistics, updated after the prediction so nothing ahead
            // leaks into the standardisation.
            for (j, &v) in row.iter().enumerate() {
                let v = (1.0 + v.max(0.0) as f64).ln();
                let dlt = v - mean[j];
                mean[j] += dlt / (n + 1.0);
                m2[j] += dlt * (v - mean[j]);
            }
        }
        let first = auroc_pub(&pred[..half], &y[..half]);
        if first > best.0 {
            best = (first, lr, auroc_pub(&pred[half..], &y[half..]));
        }
    }
    (best.2, best.1)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    dir: &str,
    outcomes: &str,
    d: usize,
    patients: usize,
    seed: u64,
    cap: usize,
    bins: usize,
    banks: usize,
) {
    let death = load_death(outcomes);
    let (streams_all, ids, vp) = flat_streams_valued(dir, bins);
    let mut streams = Vec::new();
    let mut labels = Vec::new();
    for (s, id) in streams_all.into_iter().zip(ids) {
        if let Some(&y) = death.get(&id) {
            streams.push(s);
            labels.push(y);
        }
    }
    if patients > 0 && streams.len() > patients {
        streams.truncate(patients);
        labels.truncate(patients);
    }
    let n = streams.len();
    let half = n / 2;
    let rate = labels[..half].iter().filter(|&&y| y == 1).count() as f64 / half.max(1) as f64;
    println!("asked at the bedside, at any moment; the outcome is spoken by the world at the end of the stay");
    println!(
        "  {} patients ({} learned from, {} read), in-hospital death {:.3} in the first half, V = {} + 3",
        n,
        half,
        n - half,
        rate,
        vp
    );
    println!("  prequential split: not comparable to the five-fold snapshot probe.\n");

    let (full, wiped) = std::thread::scope(|sc| {
        let (s, l) = (&streams, &labels);
        let a = sc.spawn(move || run_arm(s, l, vp, d, banks, cap, seed, false));
        let b = sc.spawn(move || run_arm(s, l, vp, d, banks, cap, seed, true));
        (a.join().unwrap(), b.join().unwrap())
    });

    let y_test = &labels[half..];
    println!(
        "{:>12} {:>10} {:>10} {:>14} {:>16} {:>14}",
        "asked at", "ours", "wiped", "counter online", "counter batch", "changes/ask"
    );
    for (fi, f) in FRACS.iter().enumerate() {
        let q: Vec<f64> = full[half..].iter().map(|a| a.q[fi]).collect();
        let qw: Vec<f64> = wiped[half..].iter().map(|a| a.q[fi]).collect();
        let x: Vec<Vec<f32>> = streams.iter().map(|s| summary(s, vp, *f)).collect();
        let (ca, cl2) = counter_auroc(&x, &labels, half);
        let (oa, olr) = online_counter_auroc(&x, &labels, half);
        let ch: f64 =
            full[half..].iter().map(|a| a.changes[fi] as f64).sum::<f64>() / (n - half) as f64;
        println!(
            "{:>11.0}% {:>10.4} {:>10.4} {:>14} {:>16} {:>14.3}",
            f * 100.0,
            auroc_pub(&q, y_test),
            auroc_pub(&qw, y_test),
            format!("{:.4} lr {}", oa, olr),
            format!("{:.4} L2 {}", ca, cl2),
            ch
        );
    }
    let bits: f64 = full[half..].iter().map(|a| a.bits).sum::<f64>() / (n - half) as f64;
    let h = -(rate * rate.log2() + (1.0 - rate) * (1.0 - rate).log2());
    println!(
        "\n  charge on the spoken outcome, second half: {:.4} bits (full vocabulary)   entropy of the base rate {:.4}",
        bits, h
    );
    println!("  the online counter is the comparator on the model's terms: one pass, one step per");
    println!("  patient, each predicted before it is learned from. the batch counter sees every");
    println!("  first-half outcome hundreds of times and is a different acquisition regime.");
    println!("  the wiped arm is the same memory with this stay erased before each question:");
    println!("  what it knows is only what outcomes are like in general, so it sits at 0.5.");
}
