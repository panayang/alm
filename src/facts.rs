//! Learners ordered by the facts they maintain.
//!
//! The foundations draft states the result this file implements. Order learners
//! by inclusion of the facts they maintain. A faithful scalar metric exists
//! only if that order is total; the order is the intersection of its linear
//! extensions; and the least number of scalars that represents it is its order
//! dimension. "Give one number" asks for a choice of linear extension, and the
//! facts do not determine that choice.
//!
//! So this does not score a configuration. It asks each one a fixed list of
//! questions whose answers are known by construction -- does it learn a
//! deterministic successor, does it avoid confident wrongness on noise, does
//! its state carry the previous event -- and records pass or fail. Then it
//! orders the configurations by inclusion and reports what that order is:
//! which pairs are comparable, which are not, and how many numbers it would
//! take to represent it without inventing a ranking the facts do not support.
//!
//! The thresholds are listed with each fact. Where one is a chance level it is
//! arithmetic; where it is a margin above chance it is mine, and it is printed
//! next to the measured value so a reader can move it.

use crate::config::Config;
use crate::model::Model;
use crate::num::cbrng;

struct Learner {
    name: &'static str,
    set: fn(&mut Config),
}

struct Fact {
    name: &'static str,
    rule: &'static str,
}

const FACTS: [Fact; 11] = [
    Fact { name: "learns a deterministic successor", rule: "charge < 0.5 bits (truth 0)" },
    Fact { name: "not confidently wrong on noise", rule: "i.i.d. charge < uniform 6.00" },
    Fact { name: "does not write off the rare", rule: "rare - common excess < 1 bit" },
    Fact { name: "state carries the previous event", rule: "decode > 0.50 (chance 0.25)" },
    Fact { name: "state carries the silence", rule: "decode > 0.50 (chance 0.25)" },
    Fact { name: "judges presence from history", rule: "AUROC > 0.90 (chance 0.50)" },
    Fact { name: "judges order from history", rule: "AUROC > 0.80 (chance 0.50)" },
    Fact { name: "answers a new pair from its seen part", rule: "novel charge < 0.25 bits (truth 0)" },
    Fact { name: "silence does not worsen the ungrounded", rule: "wiped Brier after <= at ask + 0.01" },
    Fact { name: "the walk alone answers a relation", rule: "rows off: charge < uniform 6.00" },
    Fact { name: "says back a key's value set this turn", rule: "accuracy > 0.50 (64 values, chance 0.015)" },
];

fn base(v: usize, seed: u64, set: fn(&mut Config)) -> Model {
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    set(&mut cfg);
    cfg.derive();
    Model::new(cfg)
}

fn zipf_cdf(v: usize) -> (Vec<f64>, Vec<f64>) {
    let w: Vec<f64> = (0..v).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let t: f64 = w.iter().sum();
    let p: Vec<f64> = w.iter().map(|x| x / t).collect();
    let mut c = Vec::with_capacity(v);
    let mut a = 0.0;
    for q in p.iter() {
        a += q;
        c.push(a);
    }
    (p, c)
}

fn draw(cdf: &[f64], key: u64, i: u64) -> usize {
    let u = (cbrng(key, i) >> 11) as f64 / (1u64 << 53) as f64;
    cdf.iter().position(|c| *c >= u).unwrap_or(cdf.len() - 1)
}

/// Deterministic successor: last fifth of the successor charges.
fn successor(set: fn(&mut Config)) -> f64 {
    let v = 64usize;
    let (_, cdf) = zipf_cdf(v);
    let mut m = base(v, 0x51D, set);
    let n = 40_000usize;
    let (mut b, mut c) = (0.0f64, 0u64);
    let mut cue = 0usize;
    for i in 0..n {
        let tok = if i % 2 == 0 {
            cue = draw(&cdf, 0xDA7B, i as u64);
            cue
        } else {
            (cue * 7 + 3) % v
        };
        m.tick(None, false);
        m.tick(None, false);
        let o = m.tick(Some(tok), false);
        if o.charged && i % 2 == 1 && i * 5 >= n * 4 {
            b += o.bits;
            c += 1;
        }
    }
    b / c.max(1) as f64
}

/// i.i.d. Zipf: mean charge over the second half.
fn noise(set: fn(&mut Config)) -> f64 {
    let v = 64usize;
    let (_, cdf) = zipf_cdf(v);
    let mut m = base(v, 0x51D, set);
    let n = 30_000usize;
    let (mut b, mut c) = (0.0f64, 0u64);
    for i in 0..n {
        let tok = draw(&cdf, 0xDA7B, i as u64);
        m.tick(None, false);
        m.tick(None, false);
        let o = m.tick(Some(tok), false);
        if o.charged && i * 2 >= n {
            b += o.bits;
            c += 1;
        }
    }
    b / c.max(1) as f64
}

/// Excess over -log2 p on the rarest tokens minus on the commonest.
fn rarity(set: fn(&mut Config)) -> f64 {
    let v = 296usize;
    let (p, cdf) = zipf_cdf(v);
    let mut m = base(v, 0x5A1E, set);
    let n = 30_000usize;
    let (mut cn, mut ce, mut rn, mut re) = (0u64, 0.0f64, 0u64, 0.0f64);
    for i in 0..n {
        let tok = draw(&cdf, 0xDA7B, i as u64);
        m.tick(None, false);
        m.tick(None, false);
        let o = m.tick(Some(tok), false);
        if o.charged && i * 2 >= n {
            let e = o.bits + p[tok].log2();
            if p[tok] >= 1.0 / 16.0 {
                cn += 1;
                ce += e;
            } else if p[tok] < 1.0 / 1024.0 {
                rn += 1;
                re += e;
            }
        }
    }
    re / rn.max(1) as f64 - ce / cn.max(1) as f64
}

fn centroid_acc(rows: &[(usize, Vec<f32>)], classes: usize) -> f64 {
    let cut = rows.len() / 2;
    let d = rows[0].1.len();
    let mut cents = vec![vec![0.0f32; d]; classes];
    for (l, v) in rows[..cut].iter() {
        for i in 0..d {
            cents[*l][i] += v[i];
        }
    }
    for c in cents.iter_mut() {
        crate::num::normalize(c);
    }
    let mut hit = 0usize;
    for (l, v) in rows[cut..].iter() {
        let mut u = v.clone();
        crate::num::normalize(&mut u);
        let best = (0..classes)
            .max_by(|&a, &b| {
                crate::num::dot(&cents[a], &u).partial_cmp(&crate::num::dot(&cents[b], &u)).unwrap()
            })
            .unwrap();
        if best == *l {
            hit += 1;
        }
    }
    hit as f64 / (rows.len() - cut).max(1) as f64
}

/// Decode the previous event from the state, conditioned on the present one.
fn previous(set: fn(&mut Config)) -> f64 {
    let np = 4usize;
    let mut m = base(64, 0x5747, set);
    let mut rows: Vec<(usize, usize, Vec<f32>)> = Vec::new();
    let events = 6000usize;
    let mut prev: Option<usize> = None;
    for i in 0..events {
        let k = (cbrng(0x517E, i as u64) % np as u64) as usize;
        m.tick(Some(3 + k), false);
        if let Some(p) = prev {
            if i * 4 >= events {
                rows.push((p, k, m.features_now()));
            }
        }
        prev = Some(k);
        for _ in 0..3 {
            m.tick(None, false);
        }
    }
    let mut acc = 0.0;
    for now in 0..np {
        let sub: Vec<(usize, Vec<f32>)> =
            rows.iter().filter(|(_, n, _)| *n == now).map(|(p, _, v)| (*p, v.clone())).collect();
        acc += centroid_acc(&sub, np);
    }
    acc / np as f64
}

/// Decode the silence length from the state, token stream held fixed.
fn silence(set: fn(&mut Config)) -> f64 {
    let gaps = [1usize, 3, 7, 15];
    let mut m = base(64, 0x6A9, set);
    let mut rows: Vec<(usize, Vec<f32>)> = Vec::new();
    let eps = 3000usize;
    for i in 0..eps {
        let gi = (cbrng(0x9A17, i as u64) % gaps.len() as u64) as usize;
        for _ in 0..gaps[gi] {
            m.tick(None, false);
        }
        m.tick(Some(21), false);
        if i * 4 >= eps {
            rows.push((gi, m.features_now()));
        }
    }
    centroid_acc(&rows, gaps.len())
}

/// The walk's own reading of memory: rows off, so only the state and the
/// cursors can name the successor of a cue.
fn walk_alone(set: fn(&mut Config)) -> f64 {
    let v = 64usize;
    let mut cfg = Config::local();
    cfg.seed = 0x0DE1;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    set(&mut cfg);
    cfg.no_readout = true;
    cfg.derive();
    let mut m = Model::new(cfg);
    let n = 40_000usize;
    let (mut b, mut c) = (0.0f64, 0u64);
    let mut cue = 0usize;
    for i in 0..n {
        let tok = if i % 2 == 0 {
            cue = (cbrng(0xC0E, i as u64) % v as u64) as usize;
            cue
        } else {
            (cue * 7 + 3) % v
        };
        m.tick(None, false);
        m.tick(None, false);
        let o = m.tick(Some(tok), false);
        if o.charged && i % 2 == 1 && i * 5 >= n * 4 {
            b += o.bits;
            c += 1;
        }
    }
    b / c.max(1) as f64
}

fn evaluate(l: &Learner) -> Vec<(bool, f64)> {
    let s = successor(l.set);
    let n = noise(l.set);
    let r = rarity(l.set);
    let p = previous(l.set);
    let q = silence(l.set);
    let (jp, jr) = crate::judge::auroc_pair(4000, 7, l.set);
    // Added after the first run of this matrix showed the self block and the
    // read-back gate indistinguishable from their absence on the other seven.
    // They are not indistinguishable -- they cut the charge on a never-seen
    // pair fourfold -- so the list was missing the fact they maintain, which is
    // what an order by inclusion is for finding.
    let (_, nv) = crate::compose::novel_charge(40_000, l.set);
    // Brier after the silence minus Brier at the question, history wiped.
    // Formerly the drift in P(YES), which measured stillness, not harm.
    let (ba, bb) = crate::judge::ungrounded_brier(4000, 7, l.set);
    let dr = bb - ba;
    let wa = walk_alone(l.set);
    // Added when MultiWOZ showed a value said in the same turn named 0.53 of
    // the time: can the learner say back what a key was just set to here?
    let tt = crate::slots::this_turn(2000, 7, l.set);
    vec![
        (s < 0.5, s),
        (n < 6.0, n),
        (r < 1.0, r),
        (p > 0.5, p),
        (q > 0.5, q),
        (jp > 0.9, jp),
        (jr > 0.8, jr),
        (nv < 0.25, nv),
        (dr <= 0.01, dr),
        (wa < 6.0, wa),
        (tt > 0.5, tt),
    ]
}

fn subset(a: &[bool], b: &[bool]) -> bool {
    a.iter().zip(b).all(|(x, y)| !*x || *y)
}

/// The configuration the paper measured (`experiments::closeout`) against the
/// default every later experiment started from, and each single step from the
/// one toward the other. Added 2026-09-28 when it was found that the two had
/// diverged: the paper bypassed the operator graph, froze it, routed by the
/// bound traces, entered reads by content and set the anchor to zero; the
/// default did none of these.
pub fn configs() {
    use crate::config::RouteQuery;
    fn paper(c: &mut Config) {
        c.bypass_graph = true;
        c.freeze_operator = true;
        c.route_query = RouteQuery::Bound;
        c.read_entry_by_content = true;
        c.anchor = 0.0;
    }
    run_with(vec![
        Learner { name: "default", set: |_| {} },
        Learner { name: "paper config", set: paper },
        Learner {
            name: "paper, graph kept",
            set: |c| {
                paper(c);
                c.bypass_graph = false;
            },
        },
        Learner { name: "graph bypassed", set: |c| c.bypass_graph = true },
        Learner { name: "operator frozen", set: |c| c.freeze_operator = true },
        Learner { name: "route by traces", set: |c| c.route_query = RouteQuery::Bound },
        Learner { name: "entry by content", set: |c| c.read_entry_by_content = true },
        Learner { name: "anchor 0", set: |c| c.anchor = 0.0 },
    ]);
}

pub fn run() {
    run_with(vec![
        Learner { name: "default", set: |_| {} },
        Learner { name: "no self block", set: |c| c.bind_self = false },
        // The read-back gate is off by default since 2026-09-28, so this
        // learner is the departure and the default is its absence.
        Learner { name: "read-back gate on", set: |c| c.verify_gate = true },
        Learner { name: "echo gate on", set: |c| c.walk_needs_retrieval = true },
        Learner { name: "said binds", set: |c| c.bind_overt = true },
        Learner { name: "situated memory", set: |c| c.ep_context = true },
        Learner {
            name: "episode trace",
            set: |c| {
                c.episodic = true;
                c.episodic_decay = 1.0;
            },
        },
        Learner { name: "top-16 negatives", set: |c| c.neg_samples = 16 },
        Learner { name: "no codebook term", set: |c| c.readout_codebook = 0.0 },
        Learner { name: "eta 0.5", set: |c| c.eta = 0.5 },
        Learner { name: "rows off", set: |c| c.no_readout = true },
    ]);
}

fn run_with(learners: Vec<Learner>) {
    println!("learners ordered by the facts they maintain, not by a score");
    println!("  every fact has a known answer; thresholds are listed and the measured value is printed.\n");
    for (i, f) in FACTS.iter().enumerate() {
        println!("  F{}  {:<36} {}", i + 1, f.name, f.rule);
    }

    let results: Vec<Vec<(bool, f64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = learners.iter().map(|l| sc.spawn(move || evaluate(l))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    println!("\n{:<20} {}", "learner", (1..=FACTS.len()).map(|i| format!("{:>10}", format!("F{}", i))).collect::<String>());
    for (l, r) in learners.iter().zip(results.iter()) {
        let cells: String = r
            .iter()
            .map(|(ok, v)| format!("{:>10}", format!("{} {:.3}", if *ok { "ok" } else { "NO" }, v)))
            .collect();
        println!("{:<20} {}", l.name, cells);
    }

    let passed: Vec<Vec<bool>> = results.iter().map(|r| r.iter().map(|(b, _)| *b).collect()).collect();
    let n = learners.len();
    println!("\n  the order: a >= b when a maintains every fact b maintains");
    let mut incomparable = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let ij = subset(&passed[j], &passed[i]);
            let ji = subset(&passed[i], &passed[j]);
            match (ij, ji) {
                (true, true) => println!("    {:<20} == {}", learners[i].name, learners[j].name),
                (true, false) => println!("    {:<20} >  {}", learners[i].name, learners[j].name),
                (false, true) => println!("    {:<20} <  {}", learners[i].name, learners[j].name),
                (false, false) => incomparable.push((i, j)),
            }
        }
    }
    for (i, j) in incomparable.iter() {
        let only_i: Vec<String> = (0..FACTS.len())
            .filter(|&f| passed[*i][f] && !passed[*j][f])
            .map(|f| format!("F{}", f + 1))
            .collect();
        let only_j: Vec<String> = (0..FACTS.len())
            .filter(|&f| passed[*j][f] && !passed[*i][f])
            .map(|f| format!("F{}", f + 1))
            .collect();
        println!(
            "    {:<20} || {:<20} ({} only: {}; {} only: {})",
            learners[*i].name,
            learners[*j].name,
            learners[*i].name,
            only_i.join(","),
            learners[*j].name,
            only_j.join(",")
        );
    }

    // Width by brute force over subsets: the largest antichain. The order
    // dimension is at least 2 whenever any pair is incomparable, and at most
    // the width (Hiraguchi), so the two together bound how many numbers a
    // faithful representation needs.
    let mut width = 1usize;
    for mask in 1u32..(1u32 << n) {
        let idx: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
        let anti = idx.iter().enumerate().all(|(a, &i)| {
            idx[a + 1..].iter().all(|&j| {
                !(subset(&passed[i], &passed[j]) || subset(&passed[j], &passed[i]))
            })
        });
        if anti && idx.len() > width {
            width = idx.len();
        }
    }
    let lower = if incomparable.is_empty() { 1 } else { 2 };
    println!(
        "\n  incomparable pairs: {}   width: {}   order dimension between {} and {}",
        incomparable.len(),
        width,
        lower,
        width.max(lower)
    );
    if incomparable.is_empty() {
        println!("  the order is total here, so one number would be faithful -- for these facts.");
    } else {
        println!("  no single number is faithful to this order: any score would decide the");
        println!("  incomparable pairs above, and the facts do not decide them.");
    }
}
