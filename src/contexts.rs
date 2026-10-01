//! Does the store recall what followed, in a situation like this one?
//!
//! A mechanism check for `Config::ep_context`, in the sense of the capacity
//! law's check: it asks whether the mechanism does what it claims, on a stream
//! whose structure we wrote. It is not a task for the model and not a measure
//! of the design's worth.
//!
//! Several situations recur in one continuous stream, never reset between
//! them. Each has its own background tokens, so the world cascade drifts to a
//! different place in each. A set of pairs is shared by every situation, and
//! each pair is followed by a different token in each situation -- the same
//! (entity, relation) pointing at a different target per regime, the case
//! `main.tex` called Mode B. A store keyed by the pair alone can only return
//! the mixture over situations. A store that also binds the situation should
//! return this situation's successor.
//!
//! Read at the tick the pair's second token arrives, before the successor is
//! said: what the bank for the pair recalls with the situation, and without.

use crate::config::Config;
use crate::model::Model;
use crate::num::cbrng;

const BG: usize = 8;
const PAIRS: usize = 8;
/// Rare pairs: shared by every situation like the others, but each
/// (situation, pair) comes round only a handful of times in a run, so the
/// readout rows cannot have learned it and a one-shot store can.
const RARE: usize = 32;

struct Rec {
    visit: usize,
    /// Events into the visit when the pair arrived.
    pos: usize,
    /// Times this (situation, pair) had been presented before.
    seen: usize,
    rare: bool,
    /// With redraw: visits of this situation since its successors last
    /// switched, and whether recall named the previous assignment instead.
    since_switch: usize,
    ctx_old: bool,
    /// Visits to other situations since this situation was last visited;
    /// None on its first visit.
    gap: Option<usize>,
    ctx_ok: bool,
    ctx_any: bool,
    /// Cosine of the situated unbinding with the right answer, and the best
    /// cosine with any other token (diagnostic; 0 with binding off).
    gold_cos: f32,
    rival_cos: f32,
    plain_ok: bool,
    bits: f64,
}

fn run_arm(situations: usize, d: usize, visits: usize, seed: u64, on: bool) -> Vec<Rec> {
    let bg0 = 0;
    let pair0 = bg0 + situations * BG;
    let y0 = pair0 + 2 * PAIRS;
    // With ALM_CTX_REDRAW=R, each situation's successors switch between two
    // assignments every R of its visits: does recall follow the newer one?
    let redraw: usize = std::env::var("ALM_CTX_REDRAW").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let rp0 = y0 + 2 * situations * PAIRS;
    let z0 = rp0 + 2 * RARE;
    let vocab = z0 + situations * RARE;
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = vocab;
    cfg.d = d;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.ep_context = on;
    cfg.diag_recall = true;
    cfg.apply_env();
    cfg.derive();
    let mut m = Model::new(cfg);
    let mut seen = vec![0usize; situations * PAIRS];
    let mut seen_rare = vec![0usize; situations * RARE];
    let mut last_visit: Vec<Option<usize>> = vec![None; situations];
    let mut visits_of = vec![0usize; situations];
    let mut out = Vec::new();
    for v in 0..visits {
        let s = (cbrng(seed ^ 0x5171, v as u64) % situations as u64) as usize;
        let gap = last_visit[s].map(|l| v - l - 1);
        last_visit[s] = Some(v);
        let (parity, since_switch) = if redraw > 0 {
            ((visits_of[s] / redraw) % 2, visits_of[s] % redraw)
        } else {
            (0, usize::MAX)
        };
        visits_of[s] += 1;
        let mut pos = 0usize;
        for e in 0..30u64 {
            let r = cbrng(seed ^ 0xE7E7, (v as u64) << 8 | e);
            m.tick(None, false);
            if r % 2 == 0 {
                m.tick(Some(bg0 + s * BG + (r >> 8) as usize % BG), false);
                pos += 1;
            } else {
                let rare = (r >> 20) % 6 == 0;
                let (a, b, y, cnt) = if rare {
                    let j = (r >> 8) as usize % RARE;
                    (rp0 + 2 * j, rp0 + 2 * j + 1, z0 + s * RARE + j, &mut seen_rare[s * RARE + j])
                } else {
                    let j = (r >> 8) as usize % PAIRS;
                    let y = y0 + 2 * (s * PAIRS + j) + parity;
                    (pair0 + 2 * j, pair0 + 2 * j + 1, y, &mut seen[s * PAIRS + j])
                };
                m.tick(Some(a), false);
                m.tick(Some(b), false);
                let ctx = m.ctx_recall;
                let plain = m.plain_recall;
                let (mut gold_cos, mut rival_cos) = (0.0f32, 0.0f32);
                if on {
                    if let Some(u) = m.situation_now() {
                        let d = m.emb.d;
                        let mut q = vec![0.0f32; d];
                        crate::num::circconv(m.emb.row(a), m.emb.row(b), &mut q);
                        crate::num::normalize(&mut q);
                        let mut cq = vec![0.0f32; d];
                        crate::num::circconv(&u, &q, &mut cq);
                        let mut r = vec![0.0f32; d];
                        crate::num::unbind(&m.mem_ctx[m.situated_bank_pub(a, b)], &cq, &mut r);
                        crate::num::normalize(&mut r);
                        for t in 0..m.emb.vocab {
                            let c = crate::num::dot(&r, m.emb.row(t));
                            if t == y {
                                gold_cos = c;
                            } else if c > rival_cos {
                                rival_cos = c;
                            }
                        }
                    }
                }
                let o = m.tick(Some(y), false);
                out.push(Rec {
                    visit: v,
                    pos,
                    seen: *cnt,
                    rare,
                    gap,
                    ctx_ok: ctx.map_or(false, |c| c.0 == y),
                    since_switch,
                    ctx_old: !rare && redraw > 0 && ctx.map_or(false, |c| c.0 == (y ^ 1)),
                    ctx_any: ctx.is_some(),
                    gold_cos,
                    rival_cos,
                    plain_ok: plain.map_or(false, |c| c.0 == y),
                    bits: o.bits,
                });
                *cnt += 1;
                pos += 3;
            }
        }
    }
    out
}

fn report(name: &str, recs: &[Rec], visits: usize) {
    let row = |label: &str, pool: &dyn Fn(&Rec) -> bool, sel: &dyn Fn(&Rec) -> bool| {
        let r: Vec<&Rec> = recs.iter().filter(|x| pool(x) && sel(x)).collect();
        if r.is_empty() {
            return;
        }
        let f = |g: &dyn Fn(&Rec) -> bool| r.iter().filter(|x| g(x)).count() as f64 / r.len() as f64;
        let bits = r.iter().map(|x| x.bits).sum::<f64>() / r.len() as f64;
        let gc = r.iter().map(|x| x.gold_cos as f64).sum::<f64>() / r.len() as f64;
        let rc = r.iter().map(|x| x.rival_cos as f64).sum::<f64>() / r.len() as f64;
        println!(
            "  {:<30} {:>6} {:>9.3} {:>9.3} {:>9.3} {:>8.3}   gold cos {:.3} rival {:.3}",
            label,
            r.len(),
            f(&|x| x.plain_ok),
            f(&|x| x.ctx_ok),
            f(&|x| x.ctx_any),
            bits,
            gc,
            rc
        );
    };
    let freq_late = |x: &Rec| !x.rare && x.visit * 2 >= visits;
    let rare_all = |x: &Rec| x.rare;
    println!("
{}", name);
    println!("  {:<30} {:>6} {:>9} {:>9} {:>9} {:>8}", "frequent pairs, second half", "n", "plain", "situated", "names", "bits");
    row("all", &freq_late, &|_| true);
    row("early in the visit (<9)", &freq_late, &|x| x.pos < 9);
    row("later in the visit", &freq_late, &|x| x.pos >= 9);
    row("  later, gap 0-1 visits", &freq_late, &|x| x.pos >= 9 && x.gap.map_or(false, |g| g <= 1));
    row("  later, gap 2-4", &freq_late, &|x| x.pos >= 9 && x.gap.map_or(false, |g| (2..=4).contains(&g)));
    row("  later, gap 5-9", &freq_late, &|x| x.pos >= 9 && x.gap.map_or(false, |g| (5..=9).contains(&g)));
    row("  later, gap 10+", &freq_late, &|x| x.pos >= 9 && x.gap.map_or(false, |g| g >= 10));
    if recs.iter().any(|x| x.since_switch != usize::MAX) {
        println!("  {:<30}", "after a switch (situated names the new / the old)");
        for (lo, hi, lab) in [(0usize, 0usize, "  first visit after"), (1, 2, "  1-2 visits after"), (3, 5, "  3-5 visits after"), (6, 1000, "  6+ visits after")] {
            let r: Vec<&Rec> = recs.iter().filter(|x| freq_late(x) && x.pos >= 9 && x.since_switch >= lo && x.since_switch <= hi).collect();
            if r.is_empty() {
                continue;
            }
            let new = r.iter().filter(|x| x.ctx_ok).count() as f64 / r.len() as f64;
            let old = r.iter().filter(|x| x.ctx_old).count() as f64 / r.len() as f64;
            let bits = r.iter().map(|x| x.bits).sum::<f64>() / r.len() as f64;
            println!("  {:<30} {:>6}   new {:.3}   old {:.3}   bits {:.3}", lab, r.len(), new, old, bits);
        }
    }
    println!("  {:<30}", "rare pairs, whole run, later in the visit");
    row("  first time here", &rare_all, &|x| x.pos >= 9 && x.seen == 0);
    row("  seen here once", &rare_all, &|x| x.pos >= 9 && x.seen == 1);
    row("  seen here twice", &rare_all, &|x| x.pos >= 9 && x.seen == 2);
    row("  seen here 3-5 times", &rare_all, &|x| x.pos >= 9 && (3..=5).contains(&x.seen));
    row("  seen here 6+ times", &rare_all, &|x| x.pos >= 9 && x.seen >= 6);
}

/// How alike the situation vector is at two pair events of the same
/// situation in different visits, against two of different situations. The
/// delta component of unbinding one by the other is this cosine (for
/// unitary vectors, the mean cosine of their phase differences), so it bounds
/// what a single write can recall.
fn similarity(situations: usize, d: usize, visits: usize, seed: u64) {
    let bg0 = 0;
    let pair0 = bg0 + situations * BG;
    let y0 = pair0 + 2 * PAIRS;
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = y0 + 2 * situations * PAIRS + 2 * RARE + situations * RARE;
    cfg.d = d;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.ep_context = true;
    cfg.apply_env();
    cfg.derive();
    let mut m = Model::new(cfg);
    // (visit, situation, position, vector) at each later-in-visit event
    let mut snaps: Vec<(usize, usize, Vec<f32>)> = Vec::new();
    for v in 0..visits {
        let s = (cbrng(seed ^ 0x5171, v as u64) % situations as u64) as usize;
        for e in 0..30u64 {
            let r = cbrng(seed ^ 0xE7E7, (v as u64) << 8 | e);
            m.tick(None, false);
            m.tick(Some(bg0 + s * BG + (r >> 8) as usize % BG), false);
            if e >= 10 && e % 5 == 0 {
                if let Some(u) = m.situation_now() {
                    snaps.push((v, s, u));
                }
            }
        }
    }
    let (mut same, mut ns, mut other, mut no, mut within, mut nw) = (0.0f64, 0u64, 0.0f64, 0u64, 0.0f64, 0u64);
    let half = snaps.len() / 2;
    for i in half..snaps.len() {
        for j in (i.saturating_sub(400))..i {
            let c = crate::num::dot(&snaps[i].2, &snaps[j].2) as f64;
            if snaps[i].0 == snaps[j].0 {
                within += c;
                nw += 1;
            } else if snaps[i].1 == snaps[j].1 {
                same += c;
                ns += 1;
            } else {
                other += c;
                no += 1;
            }
        }
    }
    println!(
        "  situation similarity, {} situations: same visit {:.3}   same situation, other visit {:.3}   other situation {:.3}",
        situations,
        within / nw.max(1) as f64,
        same / ns.max(1) as f64,
        other / no.max(1) as f64
    );
}

pub fn run(visits: usize, seed: u64) {
    if std::env::var("ALM_CTX_SIMILARITY").is_ok() {
        let d: usize = std::env::var("ALM_CTX_D").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        for s in [4usize, 8] {
            similarity(s, d, visits, seed);
        }
        return;
    }
    println!("what followed this pair, in a situation like this one?");
    println!("  the same {} pairs recur in every situation, each followed by a different", PAIRS);
    println!("  token per situation; situations recur in one stream, never reset.");
    println!("  'plain' unbinds the pair's bank with the pair alone; 'situated' with the");
    println!("  pair bound to the present situation. chance for either is 1/situations.");
    let sizes: Vec<usize> = std::env::var("ALM_CTX_SITUATIONS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 8, 16]);
    let d: usize = std::env::var("ALM_CTX_D").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    for &s in &sizes {
        for on in [false, true] {
            let recs = run_arm(s, d, visits, seed, on);
            report(
                &format!("{} situations, d = {}, situation binding {}", s, d, if on { "on" } else { "off" }),
                &recs,
                visits,
            );
        }
    }
}
