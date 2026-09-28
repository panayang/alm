//! A stateful judge.
//!
//! **Open-loop instrument.** The world's next event here is fixed in advance
//! and ignores what the model says, so this measures memory and prediction,
//! not response. It is kept as a record; it is not the test this design is
//! for (see the top of `lib.rs`).
//!
//! A decision model in the shape the field has just started asking for: the
//! application declares the valid answers, the model returns one of them with a
//! probability, and it may decline to answer when it is unsure. The one on the
//! market is stateless and trained by reinforcement for calibration. This
//! architecture is neither, so the question is not whether it can match that
//! model on that model's ground -- it is whether being stateful buys anything a
//! stateless judge cannot have.
//!
//! The foundations draft says exactly where to look. A behaviour has a record
//! model iff it is public and undisturbed, and otherwise the least hidden state
//! that can reproduce it is its number of residual behaviours. Read as a
//! judging task: if the right answer does not depend on history, a stateless
//! judge is optimal and state buys nothing. So the tasks here are ones whose
//! answer depends on history by construction, and a judge that sees only the
//! question is at chance.
//!
//! # The stream
//!
//! An episode is a run of item tokens, a question token, a silence, and then
//! the answer -- YES or NO -- spoken by the world as an ordinary event. The
//! label is part of the world, not a separate channel: the model is charged on
//! it and writes on it like anything else, and nothing about the answer is
//! trained by any other route.
//!
//!   * presence: did T occur anywhere in this episode?
//!   * recency: of T and U, was T the later one? (order-dependent)
//!
//! # What is read
//!
//! Probabilities on the declared answer set {YES, NO}, renormalised, as a
//! decision model reports them. Commitment is the architecture's own: the
//! commit rule fires when evidence stops rising, so "committed during the
//! silence" is accept and "did not" is escalate. And the change number -- how
//! often the standing answer flips before the world speaks -- which is the
//! quantity the stabilization theorem is about, and which a stateless judge
//! answering once does not have.
//!
//! None of these is a target. They are read, together, as what the judge does.

use std::collections::HashMap;

use crate::config::Config;
use crate::model::Model;
use crate::num::cbrng;

const K: usize = 16;
const T_TOK: usize = 0;
const U_TOK: usize = 1;
const Q_PRES: usize = K;
const Q_REC: usize = K + 1;
const YES: usize = K + 2;
const NO: usize = K + 3;
const V: usize = 32;
const SILENCE: usize = 6;

struct Ppm {
    order: usize,
    vocab: usize,
    ctx: HashMap<u64, HashMap<u32, u32>>,
    hist: Vec<u32>,
}

impl Ppm {
    fn new(order: usize, vocab: usize) -> Self {
        Ppm { order, vocab, ctx: HashMap::new(), hist: Vec::new() }
    }
    fn key(&self, o: usize) -> u64 {
        let mut k = 0xcbf2_9ce4_8422_2325u64 ^ o as u64;
        for &h in self.hist[self.hist.len() - o..].iter() {
            k = (k ^ h as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        k
    }
    fn prob(&self, sym: u32) -> f64 {
        let maxo = self.order.min(self.hist.len());
        let (mut p, mut esc) = (0.0f64, 1.0f64);
        for o in (0..=maxo).rev() {
            if let Some(c) = self.ctx.get(&self.key(o)) {
                let tot: u32 = c.values().sum();
                if tot > 0 {
                    let e = c.len() as f64 / (c.len() as f64 + tot as f64);
                    p += esc * (1.0 - e) * (*c.get(&sym).unwrap_or(&0) as f64 / tot as f64);
                    esc *= e;
                }
            }
        }
        p + esc / self.vocab as f64
    }
    fn observe(&mut self, sym: u32) {
        let maxo = self.order.min(self.hist.len());
        for o in 0..=maxo {
            *self.ctx.entry(self.key(o)).or_default().entry(sym).or_insert(0) += 1;
        }
        self.hist.push(sym);
    }
    fn reset_hist(&mut self) {
        self.hist.clear();
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Full,
    /// The same model, with its volatile state reset the instant before the
    /// question is asked. Memory is untouched; only what this episode put in
    /// the state is gone. By construction this judge sees the question and
    /// nothing else, so it is the stateless judge built out of our own parts.
    Wiped,
}

struct Rec {
    /// Full-vocabulary mass on the first and second choices after the silence:
    /// what the commit rule sees, as against what the judge reports.
    top1: f64,
    top2: f64,
    task: u8,
    label: bool,
    /// P(YES | {YES, NO}) the moment the question arrives, and again after the
    /// silence, when the world is about to answer.
    q0: f64,
    q: f64,
    changes: u32,
    commit: Option<bool>,
    bits: f64,
    ppm: [f64; 4],
}

fn yes_given_declared(m: &Model) -> f64 {
    let sc = m.spread_now();
    let py = sc.prob_of(&m.store, YES as u32) as f64;
    let pn = sc.prob_of(&m.store, NO as u32) as f64;
    py / (py + pn).max(1e-30)
}

fn run_arm(arm: Arm, episodes: usize, seed: u64) -> Vec<Rec> {
    run_arm_with(arm, episodes, seed, |_| {})
}

fn run_arm_with(arm: Arm, episodes: usize, seed: u64, set: fn(&mut Config)) -> Vec<Rec> {
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = V;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    set(&mut cfg);
    cfg.derive();
    let mut m = Model::new(cfg);
    let blank = m.volatile();
    let mut ppms: Vec<Ppm> = (1..=4).map(|o| Ppm::new(o, V)).collect();
    let mut out = Vec::with_capacity(episodes);

    for e in 0..episodes {
        if episodes >= 10_000 && e > 0 && e % (episodes / 10).max(1) == 0 {
            eprintln!("  arm {}: {}/{} episodes", arm as u8, e, episodes);
        }
        m.restore(blank.clone());
        for p in ppms.iter_mut() {
            p.reset_hist();
        }
        let len = 4 + (cbrng(seed ^ 0x1E1, e as u64) % 13) as usize;
        let items: Vec<usize> = (0..len)
            .map(|i| (cbrng(seed ^ 0x17E, (e * 64 + i) as u64) % K as u64) as usize)
            .collect();
        let last_t = items.iter().rposition(|&x| x == T_TOK);
        let last_u = items.iter().rposition(|&x| x == U_TOK);
        let rec_ok = last_t.is_some() || last_u.is_some();
        let ask_rec = rec_ok && cbrng(seed ^ 0xA5C, e as u64) % 2 == 0;
        let (q_tok, label) = if ask_rec {
            let later_t = match (last_t, last_u) {
                (Some(a), Some(b)) => a > b,
                (Some(_), None) => true,
                _ => false,
            };
            (Q_REC, later_t)
        } else {
            (Q_PRES, last_t.is_some())
        };

        for &it in items.iter() {
            m.tick(None, false);
            m.tick(Some(it), false);
            for p in ppms.iter_mut() {
                p.observe(it as u32);
            }
        }

        if arm == Arm::Wiped {
            m.restore(blank.clone());
        }
        let o = m.tick(Some(q_tok), false);
        for p in ppms.iter_mut() {
            p.observe(q_tok as u32);
        }
        let q0 = yes_given_declared(&m);

        let mut commit: Option<bool> = None;
        let note = |t: Option<usize>, c: &mut Option<bool>| {
            if c.is_none() {
                if let Some(t) = t {
                    if t == YES || t == NO {
                        *c = Some(t == YES);
                    }
                }
            }
        };
        note(o.overt, &mut commit);
        let mut prev = q0 >= 0.5;
        let mut changes = 0u32;
        for _ in 0..SILENCE {
            let o = m.tick(None, false);
            note(o.overt, &mut commit);
            let now = yes_given_declared(&m) >= 0.5;
            if now != prev {
                changes += 1;
                prev = now;
            }
        }
        let q = yes_given_declared(&m);
        let (top1, top2) = {
            let sc = m.spread_now();
            let mut v: Vec<f64> = sc.rows.iter().map(|&(_, e)| (e / sc.z) as f64).collect();
            v.sort_by(|a, b| b.partial_cmp(a).unwrap());
            (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0))
        };

        let mut pp = [0.0f64; 4];
        for (i, p) in ppms.iter().enumerate() {
            let py = p.prob(YES as u32);
            let pn = p.prob(NO as u32);
            pp[i] = py / (py + pn).max(1e-30);
        }

        let ans = if label { YES } else { NO };
        let o = m.tick(Some(ans), false);
        for p in ppms.iter_mut() {
            p.observe(ans as u32);
        }

        out.push(Rec {
            top1,
            top2,
            task: if ask_rec { 1 } else { 0 },
            label,
            q0,
            q,
            changes,
            commit,
            bits: o.bits,
            ppm: pp,
        });
    }
    out
}

fn auroc(score: &[f64], label: &[bool]) -> f64 {
    let y: Vec<u8> = label.iter().map(|&b| b as u8).collect();
    crate::patient::auroc_pub(score, &y)
}

fn brier(q: &[f64], y: &[bool]) -> f64 {
    q.iter().zip(y).map(|(p, &l)| (p - if l { 1.0 } else { 0.0 }).powi(2)).sum::<f64>()
        / q.len().max(1) as f64
}

fn acc(q: &[f64], y: &[bool]) -> f64 {
    q.iter().zip(y).filter(|(p, &l)| (**p >= 0.5) == l).count() as f64 / q.len().max(1) as f64
}

/// Ten-bin expected calibration error on the confidence of the chosen answer.
fn ece(q: &[f64], y: &[bool]) -> f64 {
    let mut bins = vec![(0u64, 0.0f64, 0u64); 10];
    for (p, &l) in q.iter().zip(y) {
        let c = p.max(1.0 - p);
        let b = (((c - 0.5) * 2.0 * 10.0) as usize).min(9);
        bins[b].0 += 1;
        bins[b].1 += c;
        if (*p >= 0.5) == l {
            bins[b].2 += 1;
        }
    }
    let n = q.len().max(1) as f64;
    bins.iter()
        .filter(|b| b.0 > 0)
        .map(|b| (b.0 as f64 / n) * (b.1 / b.0 as f64 - b.2 as f64 / b.0 as f64).abs())
        .sum()
}

/// Accuracy on the most confident fraction, at several coverages: the
/// risk-coverage curve, which is a curve and is reported as one.
fn risk_coverage(q: &[f64], y: &[bool]) -> Vec<(f64, f64)> {
    let mut idx: Vec<usize> = (0..q.len()).collect();
    idx.sort_by(|&a, &b| {
        let ca = q[a].max(1.0 - q[a]);
        let cb = q[b].max(1.0 - q[b]);
        cb.partial_cmp(&ca).unwrap()
    });
    [0.2, 0.4, 0.6, 0.8, 1.0]
        .iter()
        .map(|&c| {
            let n = ((q.len() as f64 * c) as usize).max(1);
            let hit = idx[..n].iter().filter(|&&i| (q[i] >= 0.5) == y[i]).count();
            (c, hit as f64 / n as f64)
        })
        .collect()
}

fn report(name: &str, recs: &[Rec]) {
    let half = recs.len() / 2;
    let test = &recs[half..];
    for (task, tname) in [(0u8, "presence"), (1, "recency (order-dependent)")] {
        let r: Vec<&Rec> = test.iter().filter(|x| x.task == task).collect();
        if r.is_empty() {
            continue;
        }
        let y: Vec<bool> = r.iter().map(|x| x.label).collect();
        let q: Vec<f64> = r.iter().map(|x| x.q).collect();
        let q0: Vec<f64> = r.iter().map(|x| x.q0).collect();
        let base = recs[..half].iter().filter(|x| x.task == task && x.label).count() as f64
            / recs[..half].iter().filter(|x| x.task == task).count().max(1) as f64;
        // The counter at its best order on this task, by Brier, chosen once.
        let mut best = 0usize;
        let mut best_b = f64::INFINITY;
        for o in 0..4 {
            let qp: Vec<f64> = r.iter().map(|x| x.ppm[o]).collect();
            let b = brier(&qp, &y);
            if b < best_b {
                best_b = b;
                best = o;
            }
        }
        let qp: Vec<f64> = r.iter().map(|x| x.ppm[best]).collect();
        let committed: Vec<&&Rec> = r.iter().filter(|x| x.commit.is_some()).collect();
        let c_acc = committed.iter().filter(|x| x.commit == Some(x.label)).count() as f64
            / committed.len().max(1) as f64;
        let chg: f64 = r.iter().map(|x| x.changes as f64).sum::<f64>() / r.len() as f64;
        let bits: f64 = r.iter().map(|x| x.bits).sum::<f64>() / r.len() as f64;

        println!("\n  [{}] {}   {} questions, base rate YES {:.3}", name, tname, r.len(), base);
        println!(
            "    {:<34} {:>8} {:>8} {:>8} {:>8}",
            "", "accuracy", "AUROC", "Brier", "ECE"
        );
        println!(
            "    {:<34} {:>8.3} {:>8.3} {:>8.4} {:>8.4}",
            "judge, after the silence",
            acc(&q, &y),
            auroc(&q, &y),
            brier(&q, &y),
            ece(&q, &y)
        );
        println!(
            "    {:<34} {:>8.3} {:>8.3} {:>8.4} {:>8.4}",
            "judge, the moment it is asked",
            acc(&q0, &y),
            auroc(&q0, &y),
            brier(&q0, &y),
            ece(&q0, &y)
        );
        println!(
            "    {:<34} {:>8.3} {:>8.3} {:>8.4} {:>8}",
            format!("PPM-C, best order {}", best + 1),
            acc(&qp, &y),
            auroc(&qp, &y),
            best_b,
            "-"
        );
        println!(
            "    {:<34} {:>8.3} {:>8.3} {:>8.4} {:>8}",
            "stateless: the base rate",
            base.max(1.0 - base),
            0.5,
            base * (1.0 - base),
            "-"
        );
        let rc = risk_coverage(&q, &y);
        println!(
            "    risk-coverage (accuracy on the most confident fraction): {}",
            rc.iter().map(|(c, a)| format!("{:.0}%={:.3}", c * 100.0, a)).collect::<Vec<_>>().join("  ")
        );
        println!(
            "    commits during the silence: {:.1}% of questions, accuracy when it does {:.3}",
            100.0 * committed.len() as f64 / r.len() as f64,
            c_acc
        );
        let t1: f64 = r.iter().map(|x| x.top1).sum::<f64>() / r.len() as f64;
        let t2: f64 = r.iter().map(|x| x.top2).sum::<f64>() / r.len() as f64;
        println!(
            "    what the commit rule sees (full vocabulary): first choice {:.3}, second {:.3}, margin {:.3}",
            t1,
            t2,
            t1 - t2
        );
        println!(
            "    standing answer changes during the silence: {:.3} per question   charge on the answer {:.4} bits",
            chg, bits
        );
        // Within one judge: does the change number sort right from wrong? That
        // is the question that makes it usable -- a label-free sign of whether
        // an answer is grounded, read off the dynamics rather than off a
        // trained confidence head. Across two judges it already separates
        // grounded from guessing; this asks whether it does so inside one.
        let mut by = std::collections::BTreeMap::<u32, (u64, u64)>::new();
        for x in r.iter() {
            let e = by.entry(x.changes.min(3)).or_insert((0, 0));
            e.0 += 1;
            if (x.q >= 0.5) == x.label {
                e.1 += 1;
            }
        }
        println!(
            "    accuracy by changes during the silence: {}",
            by.iter()
                .map(|(k, (n, h))| format!(
                    "{}{}: {:.3} (n={})",
                    k,
                    if *k == 3 { "+" } else { "" },
                    *h as f64 / *n as f64,
                    n
                ))
                .collect::<Vec<_>>()
                .join("   ")
        );
    }
}

pub fn run(episodes: usize, seed: u64) {
    println!("a stateful judge: the answer depends on this episode's history, and is spoken by the world");
    println!(
        "  {} episodes, second half read. items from {} types, 4-16 per episode, silence {} ticks.",
        episodes, K, SILENCE
    );
    println!("  answers are read on the declared set {{YES, NO}}, as a decision model reports them.");
    let rows: Vec<(&str, Vec<Rec>)> = std::thread::scope(|sc| {
        let a = sc.spawn(move || ("full", run_arm(Arm::Full, episodes, seed)));
        let b = sc.spawn(move || ("history wiped before the question", run_arm(Arm::Wiped, episodes, seed)));
        vec![a.join().unwrap(), b.join().unwrap()]
    });
    for (name, recs) in rows.iter() {
        report(name, recs);
    }
    println!("\n  the wiped arm is the same model with this episode erased from its state the");
    println!("  instant before the question: memory intact, history gone. it is at chance by");
    println!("  construction, and the distance to it is what the state is carrying.");
}

/// AUROC on (presence, recency) over the second half, for a configuration.
/// Used by the facts matrix; the full report is `run`.
pub(crate) fn auroc_pair(episodes: usize, seed: u64, set: fn(&mut Config)) -> (f64, f64) {
    let recs = run_arm_with(Arm::Full, episodes, seed, set);
    let test = &recs[recs.len() / 2..];
    let mut out = [0.5f64; 2];
    for task in 0..2u8 {
        let r: Vec<&Rec> = test.iter().filter(|x| x.task == task).collect();
        let y: Vec<bool> = r.iter().map(|x| x.label).collect();
        let q: Vec<f64> = r.iter().map(|x| x.q).collect();
        if !r.is_empty() {
            out[task as usize] = auroc(&q, &y);
        }
    }
    (out[0], out[1])
}

/// Does the judge's dependence on order match the task's?
///
/// The foundations draft: the future of a behaviour is independent of the
/// order questions are asked iff they pairwise commute, and order spectra are
/// non-scalar invariants. On this stream there is a known answer for each task.
/// Whether T occurred does not depend on the order the items came in, so the
/// judgment should not move when the same items are permuted. Which of T and U
/// came last does depend on it, so the judgment should move exactly when the
/// label does.
///
/// So the question is not whether the judge is right but whether it is
/// sensitive to order where the world is and insensitive where it is not. A
/// judge that ignored order would pass the first and fail the second; one that
/// latched onto order indiscriminately would do the reverse.
pub fn order_spectrum(train_episodes: usize, seed: u64) {
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = V;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.derive();
    let mut m = Model::new(cfg);
    let blank = m.volatile();

    // Learn, exactly as the judge does.
    for e in 0..train_episodes {
        if e > 0 && e % (train_episodes / 10).max(1) == 0 {
            eprintln!("  training: {}/{} episodes", e, train_episodes);
        }
        m.restore(blank.clone());
        let len = 4 + (cbrng(seed ^ 0x1E1, e as u64) % 13) as usize;
        let items: Vec<usize> = (0..len)
            .map(|i| (cbrng(seed ^ 0x17E, (e * 64 + i) as u64) % K as u64) as usize)
            .collect();
        let last_t = items.iter().rposition(|&x| x == T_TOK);
        let last_u = items.iter().rposition(|&x| x == U_TOK);
        let ask_rec = (last_t.is_some() || last_u.is_some()) && cbrng(seed ^ 0xA5C, e as u64) % 2 == 0;
        let (q_tok, label) = if ask_rec {
            (Q_REC, matches!((last_t, last_u), (Some(a), Some(b)) if a > b) || (last_t.is_some() && last_u.is_none()))
        } else {
            (Q_PRES, last_t.is_some())
        };
        for &it in items.iter() {
            m.tick(None, false);
            m.tick(Some(it), false);
        }
        m.tick(Some(q_tok), false);
        for _ in 0..SILENCE {
            m.tick(None, false);
        }
        m.tick(Some(if label { YES } else { NO }), false);
    }

    // Probe, frozen: permutations of one multiset of items, each asked afresh.
    m.frozen = true;
    let sets = 400usize;
    let perms = 8usize;
    let mut pres_spread = 0.0f64;
    let mut pres_hit = 0usize;
    let mut pres_n = 0usize;
    let mut rec_hit = 0usize;
    let mut rec_n = 0usize;
    let mut rec_moves = 0usize;
    let mut rec_label_moves = 0usize;
    let mut rec_agree = 0usize;
    for si in 0..sets {
        // Eight items that always contain both T and U, so recency is always
        // defined and can flip under permutation.
        let mut base_items: Vec<usize> = vec![T_TOK, U_TOK];
        for i in 0..6 {
            base_items.push(2 + (cbrng(seed ^ 0x5E7, (si * 16 + i) as u64) % (K as u64 - 2)) as usize);
        }
        let mut pres_q: Vec<f64> = Vec::new();
        let mut rec: Vec<(f64, bool)> = Vec::new();
        for pi in 0..perms {
            let mut it = base_items.clone();
            for i in (1..it.len()).rev() {
                let j = (cbrng(seed ^ 0x9E1, (si * 1024 + pi * 32 + i) as u64) % (i as u64 + 1)) as usize;
                it.swap(i, j);
            }
            let later_t = it.iter().rposition(|&x| x == T_TOK) > it.iter().rposition(|&x| x == U_TOK);
            for (q_tok, slot) in [(Q_PRES, 0u8), (Q_REC, 1u8)] {
                m.restore(blank.clone());
                for &x in it.iter() {
                    m.tick(None, false);
                    m.tick(Some(x), false);
                }
                m.tick(Some(q_tok), false);
                for _ in 0..SILENCE {
                    m.tick(None, false);
                }
                let q = yes_given_declared(&m);
                if slot == 0 {
                    pres_q.push(q);
                } else {
                    rec.push((q, later_t));
                }
            }
        }
        let mean = pres_q.iter().sum::<f64>() / pres_q.len() as f64;
        let sd = (pres_q.iter().map(|q| (q - mean).powi(2)).sum::<f64>() / pres_q.len() as f64).sqrt();
        pres_spread += sd;
        pres_hit += pres_q.iter().filter(|&&q| q >= 0.5).count();
        pres_n += pres_q.len();
        for w in rec.windows(2) {
            let moved = (w[0].0 >= 0.5) != (w[1].0 >= 0.5);
            let label_moved = w[0].1 != w[1].1;
            if moved {
                rec_moves += 1;
            }
            if label_moved {
                rec_label_moves += 1;
            }
            if moved == label_moved {
                rec_agree += 1;
            }
        }
        rec_hit += rec.iter().filter(|(q, y)| (*q >= 0.5) == *y).count();
        rec_n += rec.len();
    }
    let pairs = sets * (perms - 1);
    println!("order spectrum: the same items permuted, each permutation asked afresh with memory frozen");
    println!("  {} training episodes; {} multisets of 8 items (T and U always present) x {} permutations\n", train_episodes, sets, perms);
    println!("  presence -- the right answer is the same in every permutation (YES)");
    println!("    accuracy {:.3}   mean spread of P(YES) across permutations {:.4}", pres_hit as f64 / pres_n as f64, pres_spread / sets as f64);
    println!("  recency -- the right answer changes with the permutation");
    println!("    accuracy {:.3}", rec_hit as f64 / rec_n as f64);
    println!(
        "    between consecutive permutations: the label moved {:.1}% of the time, the judgment moved {:.1}%, and they agreed on whether to move {:.1}%",
        100.0 * rec_label_moves as f64 / pairs as f64,
        100.0 * rec_moves as f64 / pairs as f64,
        100.0 * rec_agree as f64 / pairs as f64
    );
    println!("\n  a judge whose order dependence matches the world's has a presence spread near zero");
    println!("  and moves on recency exactly when the label does.");
}

/// Which silent-tick process moves an answer that has nothing under it?
///
/// With the episode wiped, the judgment the moment the question arrives is the
/// prior -- memory's view of the question with no situation -- and it scores at
/// the base rate. After six silent ticks it scores worse: on presence, 0.644
/// falls to 0.563. Something that runs during silence is moving an answer that
/// has nothing to move it. Four things run then: the walk on the operator graph
/// and three self-feedback channels into the fast rung. Turn each off in turn,
/// on the wiped arm, and see where the drift goes.
pub fn drift_diagnosis(episodes: usize, seed: u64) {
    let arms: [(&str, fn(&mut Config)); 7] = [
        ("all on", |_| {}),
        ("walk needs retrieval", |c| c.walk_needs_retrieval = true),
        ("no walk in the gap", |c| c.walk_during_gap = false),
        ("no covert feedback", |c| c.feedback_covert = false),
        ("no overt feedback", |c| c.feedback_overt = false),
        ("no write feedback", |c| c.feedback_write = false),
        (
            "no feedback at all",
            |c| {
                c.feedback_covert = false;
                c.feedback_overt = false;
                c.feedback_write = false;
            },
        ),
    ];
    println!("what moves an ungrounded answer during the silence? (history wiped before every question)");
    println!("  {} episodes, second half read; P(YES) on the declared set\n", episodes);
    println!(
        "{:>22} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "", "pres @ask", "pres after", "rec @ask", "rec after", "|drift|"
    );
    let rows: Vec<(&str, Vec<Rec>, Vec<Rec>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = arms
            .iter()
            .map(|(n, f)| {
                let (n, f) = (*n, *f);
                sc.spawn(move || {
                    (n, run_arm_with(Arm::Wiped, episodes, seed, f), run_arm_with(Arm::Full, episodes, seed, f))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (name, wiped, full) in rows.iter() {
        let t = &wiped[wiped.len() / 2..];
        let a = |task: u8, after: bool| {
            let r: Vec<&Rec> = t.iter().filter(|x| x.task == task).collect();
            let y: Vec<bool> = r.iter().map(|x| x.label).collect();
            let q: Vec<f64> = r.iter().map(|x| if after { x.q } else { x.q0 }).collect();
            acc(&q, &y)
        };
        let drift: f64 = t.iter().map(|x| (x.q - x.q0).abs()).sum::<f64>() / t.len() as f64;
        println!(
            "{:>22} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>10.4}",
            name,
            a(0, false),
            a(0, true),
            a(1, false),
            a(1, true),
            drift
        );
        // And the grounded judge under the same switch, so a fix that stops the
        // drift is not bought by breaking the thing that works.
        let tf = &full[full.len() / 2..];
        let af = |task: u8| {
            let r: Vec<&Rec> = tf.iter().filter(|x| x.task == task).collect();
            let y: Vec<bool> = r.iter().map(|x| x.label).collect();
            let q: Vec<f64> = r.iter().map(|x| x.q).collect();
            acc(&q, &y)
        };
        println!("{:>22}   grounded judge after the silence: presence {:.3}, recency {:.3}", "", af(0), af(1));
    }
    println!("\n  the wiped rows should not move between @ask and after: nothing was retrieved,");
    println!("  so nothing should change what is said.");
}

/// When memory fades, does confidence fade with it?
///
/// With the episode wiped the judge reads out 0.74 on its first choice and is
/// right 51% of the time: an unshaped state does not read out diffusely, as
/// `store.rs` supposes, it reads out confidently. But a wipe is an artificial
/// case, and it cannot test a remedy built on comparing the situation with the
/// prior, because a wiped situation *is* the prior.
///
/// The natural case is forgetting. Episodes here run 8 to 64 items, so T can
/// have occurred long enough ago that the state no longer holds it cleanly.
/// Bucket the presence questions by how many items ago T last occurred and ask,
/// in each bucket, whether accuracy falls, whether confidence falls with it,
/// and whether the situation's departure from what memory alone would say --
/// read by a frozen probe that asks the same question of a blank situation
/// through the same silence -- follows accuracy more closely than confidence
/// does.
pub fn fade(episodes: usize, seed: u64) {
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = V;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.derive();
    let mut m = Model::new(cfg);
    let blank = m.volatile();
    // (distance since T or None, label, q, q_counterfactual, q at the question)
    let mut rows: Vec<(Option<usize>, bool, f64, f64, f64)> = Vec::new();
    for e in 0..episodes {
        if e > 0 && e % (episodes / 10).max(1) == 0 {
            eprintln!("  fade: {}/{} episodes", e, episodes);
        }
        m.restore(blank.clone());
        let len = 8 + (cbrng(seed ^ 0xFAD, e as u64) % 57) as usize;
        // T is rarer here so that when it is present it is usually present
        // once, and its distance means something.
        let items: Vec<usize> = (0..len)
            .map(|i| {
                let r = cbrng(seed ^ 0x17F, (e * 128 + i) as u64);
                if r % 40 == 0 {
                    T_TOK
                } else {
                    1 + (r / 40 % (K as u64 - 1)) as usize
                }
            })
            .collect();
        let dist = items.iter().rposition(|&x| x == T_TOK).map(|p| len - 1 - p);
        let label = dist.is_some();
        for &it in items.iter() {
            m.tick(None, false);
            m.tick(Some(it), false);
        }
        m.tick(Some(Q_PRES), false);
        let q_ask = yes_given_declared(&m);
        for _ in 0..SILENCE {
            m.tick(None, false);
        }
        let q = yes_given_declared(&m);
        let test = e * 2 >= episodes;
        let mut qcf = 0.5;
        if test {
            // What memory alone says: the same question, a blank situation, the
            // same silence, nothing written, and the real situation put back.
            let saved = m.volatile();
            m.frozen = true;
            m.restore(blank.clone());
            m.tick(Some(Q_PRES), false);
            for _ in 0..SILENCE {
                m.tick(None, false);
            }
            qcf = yes_given_declared(&m);
            m.restore(saved);
            m.frozen = false;
        }
        m.tick(Some(if label { YES } else { NO }), false);
        if test {
            rows.push((dist, label, q, qcf, q_ask));
        }
    }
    let logit = |p: f64| {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    };
    let buckets: [(&str, Option<(usize, usize)>); 6] = [
        ("T absent", None),
        ("0-3 ago", Some((0, 3))),
        ("4-7 ago", Some((4, 7))),
        ("8-15 ago", Some((8, 15))),
        ("16-31 ago", Some((16, 31))),
        ("32+ ago", Some((32, usize::MAX))),
    ];
    let base = rows.iter().filter(|r| r.1).count() as f64 / rows.len().max(1) as f64;
    println!("when memory fades, does confidence fade with it?");
    println!(
        "  {} episodes of 8-64 items, presence only, second half read ({} questions, P(T present) {:.3})",
        episodes,
        rows.len(),
        base
    );
    println!("  confidence = max(P, 1-P) on the declared set; evidence = |logit P - logit P_memory-alone|\n");
    println!(
        "{:>12} {:>7} {:>10} {:>10} {:>12} {:>12} {:>14}",
        "", "n", "acc @ask", "accuracy", "confidence", "evidence", "memory alone"
    );
    for (name, rng) in buckets.iter() {
        let sel: Vec<&(Option<usize>, bool, f64, f64, f64)> = rows
            .iter()
            .filter(|r| match (rng, r.0) {
                (None, None) => true,
                (Some((a, b)), Some(d)) => d >= *a && d <= *b,
                _ => false,
            })
            .collect();
        if sel.is_empty() {
            continue;
        }
        let n = sel.len() as f64;
        let acc = sel.iter().filter(|r| (r.2 >= 0.5) == r.1).count() as f64 / n;
        let acc0 = sel.iter().filter(|r| (r.4 >= 0.5) == r.1).count() as f64 / n;
        let conf = sel.iter().map(|r| r.2.max(1.0 - r.2)).sum::<f64>() / n;
        let ev = sel.iter().map(|r| (logit(r.2) - logit(r.3)).abs()).sum::<f64>() / n;
        let mem = sel.iter().map(|r| r.3).sum::<f64>() / n;
        println!(
            "{:>12} {:>7} {:>10.3} {:>10.3} {:>12.3} {:>12.3} {:>14.3}",
            name, sel.len(), acc0, acc, conf, ev, mem
        );
    }
    // The selective-prediction comparison: accept the most confident, or the
    // most evidenced, fraction, and see which keeps accuracy up.
    let y: Vec<bool> = rows.iter().map(|r| r.1).collect();
    let q: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let ev: Vec<f64> = rows.iter().map(|r| (logit(r.2) - logit(r.3)).abs()).collect();
    let curve = |key: &dyn Fn(usize) -> f64| -> String {
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap());
        [0.2, 0.4, 0.6, 0.8, 1.0]
            .iter()
            .map(|&c| {
                let k = ((rows.len() as f64 * c) as usize).max(1);
                let hit = idx[..k].iter().filter(|&&i| (q[i] >= 0.5) == y[i]).count();
                format!("{:.0}%={:.3}", c * 100.0, hit as f64 / k as f64)
            })
            .collect::<Vec<_>>()
            .join("  ")
    };
    println!("\n  accept the most confident:  {}", curve(&|i| q[i].max(1.0 - q[i])));
    println!("  accept the most evidenced:  {}", curve(&|i| ev[i]));
    println!("\n  a calibrated judge loses confidence where it loses accuracy. evidence is");
    println!("  the situation's departure from what memory alone says, so it cannot be");
    println!("  inflated by the prior being confident about nothing.");
}

/// How close does a silent hop land to anything nameable?
///
/// Gating the silent walk on cleanup changed nothing measurable: the drift and
/// the walk-alone charge came out identical to gating on the bank's
/// retrievals, which means no silent hop ever cleared the cleanup threshold.
/// Before deciding that the walk cannot be gated without losing what it reads,
/// look at where hops actually land, in three situations: a deterministic
/// relation answered by the walk alone (hops there are doing useful reading),
/// the grounded judge, and the judge with its history wiped (hops there have
/// nothing under them). If the first sits above the third, some threshold
/// separates reading from drifting; if they overlap, the trade is real.
pub fn hop_cos(seed: u64) {
    let quant = |mut v: Vec<f32>| -> String {
        if v.is_empty() {
            return "no hops".to_string();
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
        format!(
            "n {:>7}   p10 {:.3}   p50 {:.3}   p90 {:.3}   max {:.3}",
            v.len(),
            q(0.1),
            q(0.5),
            q(0.9),
            v[v.len() - 1]
        )
    };

    // 1. The walk alone on a deterministic relation.
    let v = 64usize;
    let mut cfg = Config::local();
    cfg.seed = 0x0DE1;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.no_readout = true;
    cfg.derive();
    let floor = cfg.codebook_floor();
    let accept = cfg.cleanup_min_cos();
    let mut m = Model::new(cfg);
    let n = 60_000usize;
    let mut cue = 0usize;
    for i in 0..n {
        let tok = if i % 2 == 0 {
            cue = (cbrng(0xC0E, i as u64) % v as u64) as usize;
            cue
        } else {
            (cue * 7 + 3) % v
        };
        m.log_hop_cos = i * 5 >= n * 4;
        m.tick(None, false);
        m.tick(None, false);
        m.tick(Some(tok), false);
    }
    let walk_alone = std::mem::take(&mut m.hop_cos_log);

    // 2 and 3. The judge, grounded and wiped, over the silence after a question.
    let judge = |wiped: bool| -> Vec<f32> {
        let mut cfg = Config::local();
        cfg.seed = seed;
        cfg.vocab = V;
        cfg.d = 128;
        cfg.mem_banks = 4096;
        cfg.cleanup_floor_mult = 1.1;
        cfg.derive();
        let mut m = Model::new(cfg);
        let blank = m.volatile();
        let eps = 8000usize;
        for e in 0..eps {
            m.restore(blank.clone());
            let len = 4 + (cbrng(seed ^ 0x1E1, e as u64) % 13) as usize;
            let items: Vec<usize> = (0..len)
                .map(|i| (cbrng(seed ^ 0x17E, (e * 64 + i) as u64) % K as u64) as usize)
                .collect();
            let label = items.contains(&T_TOK);
            for &it in items.iter() {
                m.tick(None, false);
                m.tick(Some(it), false);
            }
            if wiped {
                m.restore(blank.clone());
            }
            m.log_hop_cos = e * 2 >= eps;
            m.tick(Some(Q_PRES), false);
            for _ in 0..SILENCE {
                m.tick(None, false);
            }
            m.log_hop_cos = false;
            m.tick(Some(if label { YES } else { NO }), false);
        }
        std::mem::take(&mut m.hop_cos_log)
    };
    let (g, w) = std::thread::scope(|sc| {
        let a = sc.spawn(|| judge(false));
        let b = sc.spawn(|| judge(true));
        (a.join().unwrap(), b.join().unwrap())
    });

    println!("where silent hops land: cosine to the nearest codebook entry");
    println!(
        "  chance floor sqrt(2 ln V / d) = {:.3} at V = 64; cleanup accepts at {:.3}
",
        floor, accept
    );
    println!("  walk alone, deterministic relation   {}", quant(walk_alone));
    println!("  judge, grounded                       {}", quant(g));
    println!("  judge, history wiped                  {}", quant(w));
    println!("
  if the first row sits above the last, a threshold separates a hop that reads");
    println!("  from one that drifts. if they overlap, gating the walk costs what it reads.");
}

/// Mean |P(YES) after the silence - P(YES) at the question| with the history
/// wiped, for a configuration: how far silence moves an answer with nothing
/// under it. Used by the facts matrix.
pub fn ungrounded_drift(episodes: usize, seed: u64, set: fn(&mut Config)) -> f64 {
    let recs = run_arm_with(Arm::Wiped, episodes, seed, set);
    let t = &recs[recs.len() / 2..];
    t.iter().map(|x| (x.q - x.q0).abs()).sum::<f64>() / t.len().max(1) as f64
}

/// Does the number of times an answer changes during the silence tell right
/// from wrong, when the judge is genuinely unsure?
///
/// On the easy tasks only 1.3% of answers changed at all, too few to lean on,
/// though those few were near chance and the stable ones 96% right. Here the
/// question is recency in episodes of 8 to 48 items, where order information
/// has had time to fade, so the judge is often unsure. Read, over the second
/// half: accuracy by change count; and three accept rules compared at the same
/// coverage -- by confidence, by stability (fewest changes first, confidence
/// breaking ties), and by stability alone. Stability costs no label and no
/// trained confidence head: it is read off the dynamics.
pub fn stability(episodes: usize, seed: u64) {
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = V;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.derive();
    let mut m = Model::new(cfg);
    let blank = m.volatile();
    // (label, q after the silence, changes)
    let mut rows: Vec<(bool, f64, u32)> = Vec::new();
    let mut e = 0usize;
    let mut asked = 0usize;
    while asked < episodes {
        e += 1;
        if asked > 0 && asked % (episodes / 10).max(1) == 0 {
            eprintln!("  stability: {}/{} questions", asked, episodes);
        }
        m.restore(blank.clone());
        let len = 8 + (cbrng(seed ^ 0x5AB, e as u64) % 41) as usize;
        let items: Vec<usize> = (0..len)
            .map(|i| {
                let r = cbrng(seed ^ 0x17A, (e * 128 + i) as u64);
                match r % 24 {
                    0 => T_TOK,
                    1 => U_TOK,
                    _ => 2 + (r / 24 % (K as u64 - 2)) as usize,
                }
            })
            .collect();
        let lt = items.iter().rposition(|&x| x == T_TOK);
        let lu = items.iter().rposition(|&x| x == U_TOK);
        if lt.is_none() && lu.is_none() {
            continue;
        }
        asked += 1;
        let label = match (lt, lu) {
            (Some(a), Some(b)) => a > b,
            (Some(_), None) => true,
            _ => false,
        };
        for &it in items.iter() {
            m.tick(None, false);
            m.tick(Some(it), false);
        }
        m.tick(Some(Q_REC), false);
        let mut prev = yes_given_declared(&m) >= 0.5;
        let mut changes = 0u32;
        for _ in 0..SILENCE {
            m.tick(None, false);
            let now = yes_given_declared(&m) >= 0.5;
            if now != prev {
                changes += 1;
                prev = now;
            }
        }
        let q = yes_given_declared(&m);
        m.tick(Some(if label { YES } else { NO }), false);
        if asked * 2 > episodes {
            rows.push((label, q, changes));
        }
    }
    let n = rows.len();
    let acc_all = rows.iter().filter(|r| (r.1 >= 0.5) == r.0).count() as f64 / n as f64;
    let y: Vec<bool> = rows.iter().map(|r| r.0).collect();
    let q: Vec<f64> = rows.iter().map(|r| r.1).collect();
    println!("does instability during the silence mark a wrong answer, when the judge is unsure?");
    println!(
        "  recency in episodes of 8-48 items; {} questions read; accuracy {:.3}, ECE {:.4}\n",
        n,
        acc_all,
        ece(&q, &y)
    );
    let mut by = std::collections::BTreeMap::<u32, (u64, u64, f64)>::new();
    for r in rows.iter() {
        let e = by.entry(r.2.min(3)).or_insert((0, 0, 0.0));
        e.0 += 1;
        if (r.1 >= 0.5) == r.0 {
            e.1 += 1;
        }
        e.2 += r.1.max(1.0 - r.1);
    }
    println!("{:>10} {:>8} {:>10} {:>12}", "changes", "n", "accuracy", "confidence");
    for (k, (cnt, hit, cf)) in by.iter() {
        println!(
            "{:>10} {:>8} {:>10.3} {:>12.3}",
            format!("{}{}", k, if *k == 3 { "+" } else { "" }),
            cnt,
            *hit as f64 / *cnt as f64,
            cf / *cnt as f64
        );
    }
    let curve = |order: &dyn Fn(usize, usize) -> std::cmp::Ordering| -> String {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| order(a, b));
        [0.5, 0.7, 0.8, 0.9, 1.0]
            .iter()
            .map(|&c| {
                let k = ((n as f64 * c) as usize).max(1);
                let hit = idx[..k].iter().filter(|&&i| (rows[i].1 >= 0.5) == rows[i].0).count();
                format!("{:.0}%={:.3}", c * 100.0, hit as f64 / k as f64)
            })
            .collect::<Vec<_>>()
            .join("  ")
    };
    let conf = |i: usize| rows[i].1.max(1.0 - rows[i].1);
    println!("\n  accept by confidence:             {}", curve(&|a, b| conf(b).partial_cmp(&conf(a)).unwrap()));
    println!(
        "  accept by stability, then conf:   {}",
        curve(&|a, b| rows[a].2.cmp(&rows[b].2).then(conf(b).partial_cmp(&conf(a)).unwrap()))
    );
    println!(
        "  accept by stability alone:        {}",
        curve(&|a, b| rows[a].2.cmp(&rows[b].2).then(a.cmp(&b)))
    );
    println!("\n  stability needs no label and no trained confidence: it is how often the");
    println!("  standing answer flipped while the world was quiet.");
}

/// When a silent hop collapses onto one token, which token?
///
/// The collapse gate refuses any hop that lands above 1/sqrt(2) on a single
/// codebook direction when nothing was retrieved. It stopped the ungrounded
/// drift and it also made composition 2.3 times worse (0.096 to 0.223 bits on
/// a new pair), because on the composition stream the walk converges onto the
/// answer -- which is also a collapse. The gate cannot tell those apart.
///
/// A guess worth testing before building on it: an ungrounded state is little
/// more than the question token, so its walk falls back onto the token just
/// said -- an echo -- while a grounded walk arrives somewhere else. Count, for
/// collapsing hops, whether they land on the last token the world said, on
/// the right answer, or elsewhere.
pub fn collapse_targets(seed: u64) {
    let judge = |wiped: bool| -> (u64, u64, u64, u64) {
        let mut cfg = Config::local();
        cfg.seed = seed;
        cfg.vocab = V;
        cfg.d = 128;
        cfg.mem_banks = 4096;
        cfg.cleanup_floor_mult = 1.1;
        cfg.walk_needs_retrieval = false;
        cfg.derive();
        let mut m = Model::new(cfg);
        let blank = m.volatile();
        let eps = 8000usize;
        let (mut total, mut echo, mut answer, mut other) = (0u64, 0u64, 0u64, 0u64);
        for e in 0..eps {
            m.restore(blank.clone());
            let len = 4 + (cbrng(seed ^ 0x1E1, e as u64) % 13) as usize;
            let items: Vec<usize> = (0..len)
                .map(|i| (cbrng(seed ^ 0x17E, (e * 64 + i) as u64) % K as u64) as usize)
                .collect();
            let label = items.contains(&T_TOK);
            for &it in items.iter() {
                m.tick(None, false);
                m.tick(Some(it), false);
            }
            if wiped {
                m.restore(blank.clone());
            }
            m.tick(Some(Q_PRES), false);
            m.log_hop_cos = e * 2 >= eps;
            for _ in 0..SILENCE {
                m.tick(None, false);
            }
            m.log_hop_cos = false;
            let ans = if label { YES } else { NO };
            for (cos, t, last) in m.hop_tok_log.drain(..) {
                if cos > std::f32::consts::FRAC_1_SQRT_2 {
                    total += 1;
                    if t == last {
                        echo += 1;
                    } else if t == ans {
                        answer += 1;
                    } else {
                        other += 1;
                    }
                }
            }
            m.tick(Some(ans), false);
        }
        (total, echo, answer, other)
    };
    let (g, w, c) = std::thread::scope(|sc| {
        let a = sc.spawn(|| judge(false));
        let b = sc.spawn(|| judge(true));
        let c = sc.spawn(|| crate::compose::collapse_targets(40_000));
        (a.join().unwrap(), b.join().unwrap(), c.join().unwrap())
    });
    println!("where collapsing silent hops land (cosine above 1/sqrt(2) to one token), gate off\n");
    println!(
        "{:>32} {:>9} {:>14} {:>14} {:>10}",
        "", "collapses", "the last token", "the answer", "elsewhere"
    );
    for (name, (t, e, a, o)) in [
        ("composition, after the cue", c),
        ("judge, grounded", g),
        ("judge, history wiped", w),
    ] {
        let f = |x: u64| 100.0 * x as f64 / t.max(1) as f64;
        println!(
            "{:>32} {:>9} {:>13.1}% {:>13.1}% {:>9.1}%",
            name,
            t,
            f(e),
            f(a),
            f(o)
        );
    }
    println!("\n  if ungrounded collapses land on the token just said and grounded ones on the");
    println!("  answer, refusing only the echo keeps the reading and stops the drift.");
}

/// (Brier at the question, Brier after the silence) with the history wiped,
/// over the second half: whether the silence makes an answer with nothing
/// under it better or worse under a proper score. Used by the structural
/// assertion and the facts matrix.
pub fn ungrounded_brier(episodes: usize, seed: u64, set: fn(&mut Config)) -> (f64, f64) {
    let recs = run_arm_with(Arm::Wiped, episodes, seed, set);
    let t = &recs[recs.len() / 2..];
    let y: Vec<bool> = t.iter().map(|x| x.label).collect();
    let q0: Vec<f64> = t.iter().map(|x| x.q0).collect();
    let q: Vec<f64> = t.iter().map(|x| x.q).collect();
    (brier(&q0, &y), brier(&q, &y))
}

/// Does an answer the world speaks rarely get learned less well, other things
/// equal?
///
/// At the bedside the architecture loses to a same-regime counter by 0.09,
/// while a linear probe on its own state comes within 0.02 of the counter. The
/// information is in the state and the readout does not take it out. One
/// guess: the outcome is one event in a thousand, and under the dense write
/// every other event pushes the outcome rows down as negatives, so the
/// association is learned against a flood of corrections.
///
/// Test it with the memory held easy and only the rarity moving. T can occur
/// only among the last three items of an episode, so whether it occurred is
/// always within easy reach of the state; the episode is lengthened with other
/// items in front of that, which does nothing to the question and makes the
/// answer tokens rarer in the stream. Same number of questions at every length.
pub fn dilution(questions: usize, seed: u64) {
    let lens = [8usize, 32, 128, 512];
    let run = |len: usize| -> (f64, f64, f64) {
        let mut cfg = Config::local();
        cfg.seed = seed;
        cfg.vocab = V;
        cfg.d = 128;
        cfg.mem_banks = 4096;
        cfg.cleanup_floor_mult = 1.1;
        cfg.derive();
        let mut m = Model::new(cfg);
        let blank = m.volatile();
        let mut q: Vec<f64> = Vec::new();
        let mut y: Vec<bool> = Vec::new();
        for e in 0..questions {
            m.restore(blank.clone());
            let present = cbrng(seed ^ 0xD11, e as u64) % 2 == 0;
            let slot = len - 1 - (cbrng(seed ^ 0xD12, e as u64) % 3) as usize;
            for i in 0..len {
                let it = if present && i == slot {
                    T_TOK
                } else {
                    1 + (cbrng(seed ^ 0xD13, (e * 1024 + i) as u64) % (K as u64 - 1)) as usize
                };
                m.tick(None, false);
                m.tick(Some(it), false);
            }
            m.tick(Some(Q_PRES), false);
            for _ in 0..SILENCE {
                m.tick(None, false);
            }
            if e * 2 >= questions {
                q.push(yes_given_declared(&m));
                y.push(present);
            }
            m.tick(Some(if present { YES } else { NO }), false);
        }
        (acc(&q, &y), auroc(&q, &y), brier(&q, &y))
    };
    let rows: Vec<(usize, (f64, f64, f64))> = std::thread::scope(|sc| {
        let hs: Vec<_> = lens.iter().map(|&l| { let r = &run; sc.spawn(move || (l, r(l))) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!("does a rarely spoken answer get learned less well? T only among the last 3 items");
    println!("  {} questions at every length, second half read\n", questions);
    println!("{:>8} {:>16} {:>10} {:>8} {:>8}", "length", "answers / event", "accuracy", "AUROC", "Brier");
    for (l, (a, u, b)) in rows.iter() {
        println!("{:>8} {:>16} {:>10.3} {:>8.3} {:>8.4}", l, format!("1 in {}", l + 2), a, u, b);
    }
    println!("\n  the question is equally easy at every length; only how rarely the world");
    println!("  says the answer changes. a falling column is dilution.");
}

/// A noisy label, with the ceiling known.
///
/// At the bedside the same state reaches 0.784 under a multi-pass probe and
/// 0.701 from the architecture's own readout; a same-regime online counter
/// reaches 0.794 at a step of 0.0003 to 0.001. The readout's step is 0.2 --
/// chosen on deterministic and i.i.d. streams and on next-token codelength,
/// never on a label that is noisy. A fixed step remembers about 1/eta
/// corrections, so at 0.2 an outcome row follows roughly the last five
/// patients, which on a deterministic label costs nothing and on a noisy one
/// means chasing chance.
///
/// Here the question is easy -- T only among the last three items -- and the
/// label spoken by the world is flipped with probability 0.2. The best any
/// judge can do is P(YES) = 0.8 or 0.2, which puts the AUROC ceiling at
/// exactly 0.80 and the Brier floor at exactly 0.16.
pub fn noisy(questions: usize, seed: u64) {
    let etas = [0.2f32, 0.05, 0.02, 0.005];
    let run = |eta: f32| -> (f64, f64, f64) {
        let mut cfg = Config::local();
        cfg.seed = seed;
        cfg.vocab = V;
        cfg.d = 128;
        cfg.mem_banks = 4096;
        cfg.cleanup_floor_mult = 1.1;
        cfg.eta = eta;
        cfg.derive();
        let mut m = Model::new(cfg);
        let blank = m.volatile();
        let mut q: Vec<f64> = Vec::new();
        let mut y: Vec<bool> = Vec::new();
        for e in 0..questions {
            m.restore(blank.clone());
            let present = cbrng(seed ^ 0xD11, e as u64) % 2 == 0;
            let slot = 7 - (cbrng(seed ^ 0xD12, e as u64) % 3) as usize;
            for i in 0..8usize {
                let it = if present && i == slot {
                    T_TOK
                } else {
                    1 + (cbrng(seed ^ 0xD13, (e * 1024 + i) as u64) % (K as u64 - 1)) as usize
                };
                m.tick(None, false);
                m.tick(Some(it), false);
            }
            let flip = cbrng(seed ^ 0xF11, e as u64) % 5 == 0;
            let spoken = present ^ flip;
            m.tick(Some(Q_PRES), false);
            for _ in 0..SILENCE {
                m.tick(None, false);
            }
            if e * 2 >= questions {
                q.push(yes_given_declared(&m));
                y.push(spoken);
            }
            m.tick(Some(if spoken { YES } else { NO }), false);
        }
        (auroc(&q, &y), brier(&q, &y), ece(&q, &y))
    };
    let rows: Vec<(f32, (f64, f64, f64))> = std::thread::scope(|sc| {
        let hs: Vec<_> = etas.iter().map(|&e| { let r = &run; sc.spawn(move || (e, r(e))) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!("a noisy label with a known ceiling: the spoken answer is flipped 20% of the time");
    println!("  {} questions, second half read. best possible: AUROC 0.800, Brier 0.160\n", questions);
    println!("{:>8} {:>8} {:>8} {:>8}", "eta", "AUROC", "Brier", "ECE");
    for (e, (a, b, c)) in rows.iter() {
        println!("{:>8} {:>8.3} {:>8.4} {:>8.4}", e, a, b, c);
    }
    println!("\n  a step that is fine on a clean label can chase the noise on this one.");
}

/// Does the echo gate keep history across a long silence, where history is
/// what the answer needs?
///
/// On PhysioNet the gate lowers codelength by 0.046 bits, most of it in novel
/// contexts; on the synthetic composition stream it costs 2.3 times over. The
/// reading that reconciles them is that a collapse back onto the last input is
/// the state forgetting everything else, which helps when the answer depends
/// on the last input alone and hurts when it depends on something earlier.
/// Codelength cannot settle that; a known answer can.
///
/// An episode is A, a short silence, B, a silence of variable length, then a
/// question and a short silence, and the world speaks the answer. Asked about
/// the first item, the answer is a fixed function of A, so the state has to
/// carry A across B and the silence after it. Asked about the last, it is a
/// fixed function of B, and forgetting A costs nothing.
pub fn history(episodes: usize, seed: u64) {
    let gaps = [0usize, 2, 5, 11];
    let run = |gate: bool, gap: usize| -> (f64, f64) {
        let mut cfg = Config::local();
        cfg.seed = seed;
        cfg.vocab = V;
        cfg.d = 128;
        cfg.mem_banks = 4096;
        cfg.cleanup_floor_mult = 1.1;
        cfg.walk_needs_retrieval = gate;
        cfg.derive();
        let mut m = Model::new(cfg);
        let blank = m.volatile();
        let (mut hf, mut nf, mut hl, mut nl) = (0usize, 0usize, 0usize, 0usize);
        for e in 0..episodes {
            m.restore(blank.clone());
            // A from tokens 2..9, B from 10..15; the answer about the first is
            // A < 6, about the last is B < 13.
            // Every item from the same eight tokens, so which one came first is
            // order information and nothing else: the bands sum what occurred
            // and cannot say which occurred first. With A drawn from a set the
            // fillers never used, the slow band alone gave A away and every
            // cell read 1.000.
            let a = 2 + (cbrng(seed ^ 0x4A, e as u64) % 8) as usize;
            let b = 2 + (cbrng(seed ^ 0x4B, e as u64) % 8) as usize;
            let about_first = cbrng(seed ^ 0x4C, e as u64) % 2 == 0;
            let label = if about_first { a < 6 } else { b < 6 };
            // A, then four fillers each followed by the silence under test,
            // the last of them being B. The first version had A and B alone
            // and scored 1.000 everywhere: the lag blocks and the bands still
            // held A whatever the walk did to the state, so it could not see
            // whether the gate keeps history. Four items on, they no longer do.
            m.tick(Some(a), false);
            for _ in 0..gap {
                m.tick(None, false);
            }
            for f in 0..3u64 {
                let filler = 2 + (cbrng(seed ^ 0x4D, e as u64 * 8 + f) % 8) as usize;
                m.tick(Some(filler), false);
                for _ in 0..gap {
                    m.tick(None, false);
                }
            }
            m.tick(Some(b), false);
            for _ in 0..gap {
                m.tick(None, false);
            }
            m.tick(Some(if about_first { Q_PRES } else { Q_REC }), false);
            for _ in 0..SILENCE {
                m.tick(None, false);
            }
            if e * 2 >= episodes {
                let right = (yes_given_declared(&m) >= 0.5) == label;
                if about_first {
                    nf += 1;
                    hf += right as usize;
                } else {
                    nl += 1;
                    hl += right as usize;
                }
            }
            m.tick(Some(if label { YES } else { NO }), false);
        }
        (hf as f64 / nf.max(1) as f64, hl as f64 / nl.max(1) as f64)
    };
    let mut jobs: Vec<(bool, usize)> = Vec::new();
    for &g in gaps.iter() {
        jobs.push((false, g));
        jobs.push((true, g));
    }
    let rows: Vec<((bool, usize), (f64, f64))> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .iter()
            .map(|&j| {
                let r = &run;
                sc.spawn(move || (j, r(j.0, j.1)))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!("does the echo gate keep history across a silence, where history is needed?");
    println!("  A, silence, B, silence of the length shown, question; {} episodes, second half read\n", episodes);
    println!(
        "{:>14} {:>22} {:>22}",
        "silence each", "about the first (A)", "about the last (B)"
    );
    println!("{:>14} {:>10} {:>11} {:>10} {:>11}", "", "gate off", "gate on", "gate off", "gate on");
    for &g in gaps.iter() {
        let off = rows.iter().find(|r| r.0 == (false, g)).unwrap().1;
        let on = rows.iter().find(|r| r.0 == (true, g)).unwrap().1;
        println!("{:>14} {:>10.3} {:>11.3} {:>10.3} {:>11.3}", g, off.0, on.0, off.1, on.1);
    }
    println!("\n  if the gate is what keeps history, the first column falls with the silence");
    println!("  when it is off and holds when it is on, and the last column does not care.");
}
