//! A stateful judge.
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
