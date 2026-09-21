//! Does consolidation cost plasticity?
//!
//! The readout loses to a counter in one narrow band: associations the counter
//! has seen between once and a dozen times. A counter holds an association after
//! one occurrence; the delta rule needs several corrections to move the same
//! mass. The obvious repair is to make the first corrections larger and later
//! ones smaller -- to consolidate.
//!
//! What drives that shrinkage decides whether it is consolidation at all.
//!
//!   * `eta / (1 + n)` is frequency wearing a different hat. 1/n is exactly the
//!     weight that turns a running update into an empirical mean, so it moves
//!     the counted prior out of the emission and into the step size. And n only
//!     grows, so a row that settles on a world which then changes can never move
//!     again.
//!   * `eta * (running mean of |err|)` measures whether the row is still wrong
//!     rather than how often it has been touched. It is reversible by
//!     construction: when the world changes the error returns, the mean rises,
//!     and the step grows back.
//!
//! The difference is not rhetorical and it has a known answer. Run a stream
//! whose conditional entropy is exactly zero, change the rule halfway, and
//! watch. Both before and after the change the right charge is zero bits. A
//! rule that converges and then cannot recover is not consolidating, it is
//! ossifying.
//!
//! # What it measured
//!
//! Charge by segment, 200k events, the map replaced between segments 10 and 11:
//!
//! ```text
//!   rule              seg1   seg10   seg11   seg20
//!   fixed             1.06    0.11    1.49    0.12
//!   inverse count     5.42    5.30    5.94    5.79
//!   error driven      4.72    0.90    5.95    0.96
//! ```
//!
//! `eta/(1+n)` does not lose plasticity so much as never acquire anything: a
//! row needs many corrections to build its vector and 1/n throttles it before
//! it arrives. The error-driven rule does keep plasticity -- it recovers as
//! well as it first learned -- so settledness measured by error rather than by
//! count is a real distinction. But it converges an order of magnitude worse,
//! because the error falls as the answer is approached, so a row throttles
//! itself while still most of a bit from right. Both fail the same way: neither
//! can tell "small error because this row has learned" from "small error
//! because it is nearly there". Fixed stays.
//!
//! # And there is no floor to trade against
//!
//! The residual on a stationary stretch was taken for LMS misadjustment -- the
//! hover a fixed step keeps around the solution -- and a trade was argued from
//! it: that the hover and the ability to track are one quantity, so the first
//! cannot be removed without the second. The measurement refuses the premise.
//! On a stationary stream of 600k events, truth zero bits:
//!
//! ```text
//!   eta        seg1     seg6    seg10    seg11    seg12
//!   0.5      0.2667   0.0445   0.0318   0.0317   0.0307
//!   0.2      0.4019   0.0641   0.0469   0.0444   0.0426
//!   0.05     0.8608   0.1346   0.1038   0.0985   0.0908
//! ```
//!
//! Every step size is still descending at 600k, and the smaller the step the
//! *higher* the charge -- the opposite of a hover, which grows with the step.
//! This is an unfinished descent, not a floor. At 100k the same quantity read
//! 0.11 for eta 0.2; at 600k it reads 0.043 and is still falling.
//!
//! So there is nothing here to defend as an acceptable noise floor, because at
//! the scales this project runs at there is no floor. What looks like one is
//! convergence in progress, and the same is true of the band where a counter
//! beats us: we are slower per observation and still improving where a counter
//! has levelled off. That is a rate, not a price.

use crate::config::{Config, StepRule};
use crate::model::Model;
use crate::num::cbrng;

const SEGMENTS: usize = 20;

/// Per-event charge on the successor, for the floor-and-recovery reading.
fn charge_trace(rule: StepRule, eta: f32, v: usize, n: usize) -> Vec<f32> {
    let mut cfg = Config::local();
    cfg.seed = 0x71A5;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.eta = eta;
    cfg.step_rule = rule;
    cfg.derive();
    let mut m = Model::new(cfg);
    let mut out = Vec::with_capacity(n / 2);
    let mut cue = 0usize;
    for i in 0..n {
        let tok = if i % 2 == 0 {
            cue = (cbrng(0xC0E, i as u64) % v as u64) as usize;
            cue
        } else if i * 2 < n {
            (cue * 7 + 3) % v
        } else {
            (cue * 11 + 29) % v
        };
        for _ in 0..2 {
            m.tick(None, false);
        }
        let o = m.tick(Some(tok), false);
        if o.charged && i % 2 == 1 {
            out.push(o.bits as f32);
        }
    }
    out
}

/// The floor is what the charge settles to on a stationary stretch; the
/// recovery is how long it takes to get back there after the world moves.
///
/// Both are read off the same trace. The floor is the mean over the last tenth
/// before the change; the recovery is the number of scored events after the
/// change until a sliding window first falls back under twice that floor.
fn floor_and_recovery(trace: &[f32]) -> (f64, Option<usize>, f64) {
    let half = trace.len() / 2;
    let tail = half / 10;
    let floor: f64 =
        trace[half - tail..half].iter().map(|x| *x as f64).sum::<f64>() / tail.max(1) as f64;
    let spike: f64 = trace[half..half + 200.min(trace.len() - half)]
        .iter()
        .map(|x| *x as f64)
        .sum::<f64>()
        / 200.0f64.min((trace.len() - half) as f64);
    let win = 400usize;
    let target = 2.0 * floor;
    let mut rec = None;
    let mut acc: f64 = 0.0;
    for k in half..trace.len() {
        acc += trace[k] as f64;
        if k >= half + win {
            acc -= trace[k - win] as f64;
            if acc / win as f64 <= target {
                rec = Some(k - half);
                break;
            }
        }
    }
    (floor, rec, spike)
}

/// Is there a floor at all?
///
/// The first sweep assumed the residual on a stationary stretch was LMS
/// misadjustment -- the hover a fixed step keeps around the solution, which
/// theory says grows with the step. The measurement said the opposite: at
/// 100k events the residual was 0.085 at eta 0.5 and 0.238 at eta 0.05, larger
/// for the smaller step. That is not a hover, it is a descent that has not
/// finished, and reading it as a floor was reading the wrong quantity.
///
/// So ask the prior question. On a stationary stream, run long, and watch
/// whether the charge flattens. If it keeps falling there is no floor and the
/// trade this file was written to examine does not exist. If it flattens, the
/// level it flattens at should rise with eta, and only then is it a hover.
fn is_there_a_floor(v: usize, n: usize) {
    println!("\n-- is there a floor at all? stationary stream, truth 0 bits, {} events --", n);
    println!("  a hover grows with the step; a descent that has not finished shrinks with it.");
    let etas = [0.5f32, 0.2, 0.05];
    let segs = 12usize;
    let rows: Vec<(f32, Vec<f64>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = etas
            .iter()
            .map(|e| {
                let e = *e;
                sc.spawn(move || {
                    let mut cfg = Config::local();
                    cfg.seed = 0x71A5;
                    cfg.vocab = v;
                    cfg.d = 128;
                    cfg.mem_banks = 4096;
                    cfg.cleanup_floor_mult = 1.1;
                    cfg.eta = e;
                    cfg.derive();
                    let mut m = Model::new(cfg);
                    let mut acc = vec![(0.0f64, 0u64); segs];
                    let mut cue = 0usize;
                    for i in 0..n {
                        let tok = if i % 2 == 0 {
                            cue = (cbrng(0xC0E, i as u64) % v as u64) as usize;
                            cue
                        } else {
                            (cue * 7 + 3) % v
                        };
                        for _ in 0..2 {
                            m.tick(None, false);
                        }
                        let o = m.tick(Some(tok), false);
                        if o.charged && i % 2 == 1 {
                            let sg = (i * segs / n).min(segs - 1);
                            acc[sg].0 += o.bits;
                            acc[sg].1 += 1;
                        }
                    }
                    (
                        e,
                        acc.iter()
                            .map(|(b, c)| if *c == 0 { 0.0 } else { b / *c as f64 })
                            .collect::<Vec<f64>>(),
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!(
        "{:>8} | {}",
        "eta",
        (1..=segs).map(|s| format!("{:>7}", s)).collect::<Vec<_>>().join("")
    );
    for (e, c) in rows.iter() {
        println!(
            "{:>8} | {}",
            e,
            c.iter().map(|x| format!("{:>7.4}", x)).collect::<Vec<_>>().join("")
        );
    }
    for (e, c) in rows.iter() {
        let last = c[segs - 1];
        let prev = c[segs - 2];
        println!(
            "  eta {:<5} last two segments {:.4} -> {:.4}   {}",
            e,
            prev,
            last,
            if last < prev * 0.97 { "still falling" } else { "flat" }
        );
    }
}

/// Is the noise floor a defect, or one face of a conserved trade?
///
/// A fixed step makes the row an exponentially weighted average of its past
/// gradients with a time constant of about 1/eta. That is a *finite memory
/// horizon*, and it is the same property twice: the residual hover on a
/// stationary stretch, and the ability to follow the world when it moves. LMS
/// theory says the excess error goes as eta and the tracking time as 1/eta, so
/// their product should be roughly flat -- and a flat product is the statement
/// that the floor cannot be engineered away, only traded.
///
/// If the product is not flat, that argument is wrong and this prints the
/// evidence against it.
fn floor_recovery_tradeoff(v: usize, n: usize) {
    println!("
-- is the floor a defect, or the price of being able to follow? --");
    println!("  floor: mean charge over the last tenth before the change (truth is 0 bits)");
    println!("  recovery: scored events after the change until a 400-wide window is back under 2x floor");
    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>14}",
        "eta", "floor", "spike", "recovery", "floor x recovery"
    );
    let etas = [0.5f32, 0.3, 0.2, 0.1, 0.05];
    let rows: Vec<(f32, (f64, Option<usize>, f64))> = std::thread::scope(|sc| {
        let hs: Vec<_> = etas
            .iter()
            .map(|e| {
                let e = *e;
                sc.spawn(move || (e, floor_and_recovery(&charge_trace(StepRule::Fixed, e, v, n))))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (e, (f, r, sp)) in rows.iter() {
        match r {
            Some(k) => println!(
                "{:>8} {:>10.4} {:>12.3} {:>12} {:>14.2}",
                e,
                f,
                sp,
                k,
                f * *k as f64
            ),
            None => println!("{:>8} {:>10.4} {:>12.3} {:>12} {:>14}", e, f, sp, "never", "-"),
        }
    }
    println!("  a flat last column means the floor is not a defect that better optimisation");
    println!("  removes -- it is one face of a conserved exchange, and where to sit on it");
    println!("  is a question about the world rather than about the algorithm.");
}


/// Charge on the successor, in segments of stream position. The map from cue to
/// successor is replaced at the halfway point.
fn run_rule(rule: StepRule, eta: f32, v: usize, n: usize) -> Vec<f64> {
    let mut cfg = Config::local();
    cfg.seed = 0x71A5;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.eta = eta;
    cfg.step_rule = rule;
    cfg.derive();
    let mut m = Model::new(cfg);

    let mut acc = vec![(0.0f64, 0u64); SEGMENTS];
    let mut cue = 0usize;
    for i in 0..n {
        let tok = if i % 2 == 0 {
            cue = (cbrng(0xC0E, i as u64) % v as u64) as usize;
            cue
        } else if i * 2 < n {
            (cue * 7 + 3) % v
        } else {
            // A different permutation of the same alphabet: every association
            // the model has settled on is now wrong, and every one it needs is
            // one it has actively been pushed away from.
            (cue * 11 + 29) % v
        };
        for _ in 0..2 {
            m.tick(None, false);
        }
        let out = m.tick(Some(tok), false);
        if out.charged && i % 2 == 1 {
            let s = (i * SEGMENTS / n).min(SEGMENTS - 1);
            acc[s].0 += out.bits;
            acc[s].1 += 1;
        }
    }
    acc.iter().map(|(b, c)| if *c == 0 { 0.0 } else { b / *c as f64 }).collect()
}

pub fn run(n: usize) {
    let v = 64usize;
    println!(
        "change point: cue -> successor is deterministic throughout, and the map is replaced at the halfway mark."
    );
    println!("the right charge is 0 bits on both sides. {} events, V = {}.\n", n, v);
    println!(
        "{:>16} {:>7} | {}",
        "rule",
        "eta",
        (1..=SEGMENTS).map(|s| format!("{:>6}", s)).collect::<Vec<_>>().join("")
    );
    let rules = [
        ("fixed", StepRule::Fixed),
        ("inverse count", StepRule::InverseCount),
        ("error driven", StepRule::ErrorDriven),
    ];
    let rows: Vec<(&str, f32, Vec<f64>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = rules
            .iter()
            .map(|(name, r)| {
                let (name, r) = (*name, *r);
                sc.spawn(move || (name, 0.2f32, run_rule(r, 0.2, v, n)))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (name, eta, c) in rows.iter() {
        println!(
            "{:>16} {:>7} | {}",
            name,
            eta,
            c.iter().map(|x| format!("{:>6.2}", x)).collect::<Vec<_>>().join("")
        );
    }
    is_there_a_floor(v, n * 3);
    floor_recovery_tradeoff(v, n);

    println!("\n  the halfway mark is between segments 10 and 11. read two things: how low");
    println!("  each rule gets before it, and whether it comes back after. a rule that");
    println!("  converges and then cannot recover has not consolidated, it has ossified.");
}
