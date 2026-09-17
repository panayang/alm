//! What does the N+1-th thing cost, given the N already stored?
//!
//! Every metric used in this project so far has been the same object wearing
//! different clothes: the codelength of the next token. That is the
//! autoregressive target function, and borrowing it from somebody else's
//! benchmark does not change what it is -- it imports the frame along with the
//! number, and inside that frame a counter built for exactly that question wins
//! by construction. It also presupposes a fixed, enumerable sample space, which
//! is what the `ln V` in this project's own capacity law is the price of.
//!
//! The claim this architecture actually makes is different: that organisation
//! forms, and that what is already stored makes the next thing cheaper. That is
//! not a scalar to be optimised. It is a curve, and it cannot be turned into a
//! target function without becoming something else.
//!
//! # Why a learning curve is not enough
//!
//! Everything that counts gets cheaper with more data, PPM included. A falling
//! curve shows only that estimates are converging. So the measurement is split
//! by a property of each event that decides whether counting *can* help at all:
//!
//!   * **seen context** -- this (prev2, prev) pair has occurred before. An exact
//!     counter has a distribution for it and is very hard to beat.
//!   * **novel context** -- it has not. The counter has nothing and must escape
//!     to a shorter context or to the uniform. A memory that has organised
//!     should be able to place a never-seen combination of seen parts; a table
//!     structurally cannot.
//!
//! The second column is where "intelligence is the emergence of memory" makes a
//! prediction that frequency estimation cannot make, and it is the column this
//! project has never looked at. If we are no better there, the organisation
//! claim is empty and the whole architecture reduces to an expensive counter.
//!
//! # And the acceleration
//!
//! Both columns are reported in deciles of stream position. A counter's novel
//! column is flat by construction -- a context it has never seen costs what it
//! costs, no matter how much else it has stored. If ours falls as experience
//! accumulates, that is acquisition acceleration, and it is the thing the
//! founding note was about.

use std::collections::HashMap;

use crate::clinical::flat_streams_valued;
use crate::config::Config;
use crate::model::Model;

const DECILES: usize = 10;

#[derive(Default, Clone, Copy)]
struct Cell {
    bits: f64,
    n: u64,
}

impl Cell {
    fn push(&mut self, b: f64) {
        self.bits += b;
        self.n += 1;
    }
    fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.bits / self.n as f64
        }
    }
}

/// Interpolated PPM-C, per event, so the same split can be applied to it.
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
    fn observe(&mut self, sym: u32) -> f64 {
        let maxo = self.order.min(self.hist.len());
        let mut p = 0.0f64;
        let mut esc = 1.0f64;
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
        p += esc / self.vocab as f64;
        for o in 0..=maxo {
            *self.ctx.entry(self.key(o)).or_default().entry(sym).or_insert(0) += 1;
        }
        self.hist.push(sym);
        -p.max(f64::MIN_POSITIVE).log2()
    }
    fn reset_hist(&mut self) {
        self.hist.clear();
    }
}

#[inline]
fn silence_ticks(g: u32, cap: usize) -> usize {
    ((32 - g.max(1).leading_zeros()).saturating_sub(1) as usize).min(cap)
}

pub fn run(dir: &str, label: &str, d: usize, patients: usize, seed: u64, cap: usize, bins: usize, banks: usize) {
    let (mut streams, _ids, vp) = flat_streams_valued(dir, bins);
    if patients > 0 && streams.len() > patients {
        streams.truncate(patients);
    }
    let total: usize = streams.iter().map(|s| s.len()).sum();

    println!("==================== {} ====================", label);
    println!("patients {}   events {}   vocabulary {}   d = {}", streams.len(), total, vp, d);

    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = vp;
    cfg.d = d;
    cfg.cleanup_floor_mult = 1.1;
    cfg.mem_banks = banks;
    cfg.derive();

    // The capacity check this project derived and then never applied to this
    // source. Every PhysioNet number before this one was produced at 64 banks,
    // which puts k in the thousands and the required width in the tens of
    // thousands -- so the memory was saturated and the model was emitting
    // very nearly the uniform distribution. Print the verdict before the run,
    // so a saturated configuration can never again be read as a result.
    let mut tri: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for s in streams.iter() {
        let (mut p2, mut p1): (Option<u32>, Option<u32>) = (None, None);
        for &(_, t) in s.iter() {
            if let (Some(a), Some(b)) = (p2, p1) {
                tri.insert((a as u64) << 40 | (b as u64) << 20 | t as u64);
            }
            p2 = p1;
            p1 = Some(t);
        }
    }
    let k = tri.len() as f64 / banks as f64;
    let need = 2.0 * k * (vp.max(2) as f64).ln();
    println!(
        "distinct triples {}   banks {}   k = {:.1}   capacity law needs d > {:.0}   running d = {}   [{}]",
        tri.len(),
        banks,
        k,
        need,
        d,
        if (d as f64) > need { "INSIDE" } else { "OUTSIDE -- expect near-uniform" }
    );
    let mut model = Model::new(cfg);
    let blank = model.volatile();
    let mut ppm = Ppm::new(3, vp);

    // [novel?][decile]
    let mut ours = [[Cell::default(); DECILES]; 2];
    let mut theirs = [[Cell::default(); DECILES]; 2];
    // Which order-2 contexts have been seen, for the split only. This is a
    // property of the stream and is shared by both models, so neither is
    // scored on a different population than the other.
    let mut seen_ctx: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut idx = 0usize;

    for s in streams.iter() {
        model.restore(blank.clone());
        ppm.reset_hist();
        let mut p2: Option<u32> = None;
        let mut p1: Option<u32> = None;
        for &(gap, tok) in s.iter() {
            let ctx = match (p2, p1) {
                (Some(a), Some(b)) => {
                    Some((a as u64) << 32 | b as u64)
                }
                _ => None,
            };
            let novel = match ctx {
                Some(c) => !seen_ctx.contains(&c),
                None => true,
            };
            let dec = (idx * DECILES / total.max(1)).min(DECILES - 1);

            for _ in 0..silence_ticks(gap, cap) {
                model.tick(None, false);
            }
            let out = model.tick(Some(tok as usize), false);
            let b_ppm = ppm.observe(tok);
            if out.charged {
                ours[novel as usize][dec].push(out.bits);
                theirs[novel as usize][dec].push(b_ppm);
            }

            if let Some(c) = ctx {
                seen_ctx.insert(c);
            }
            p2 = p1;
            p1 = Some(tok);
            idx += 1;
        }
    }

    for (k, name) in [(1usize, "NOVEL context (counter has nothing)"), (0, "seen context")] {
        println!("\n-- {} --", name);
        println!("{:>8} {:>10} {:>12} {:>12} {:>10}", "decile", "events", "ours", "PPM-C o3", "margin");
        for t in 0..DECILES {
            let (o, p) = (ours[k][t], theirs[k][t]);
            if o.n == 0 {
                continue;
            }
            println!(
                "{:>8} {:>10} {:>12.4} {:>12.4} {:>+10.4}",
                t + 1,
                o.n,
                o.mean(),
                p.mean(),
                p.mean() - o.mean()
            );
        }
        let ot: Cell = ours[k].iter().fold(Cell::default(), |mut a, c| {
            a.bits += c.bits;
            a.n += c.n;
            a
        });
        let pt: Cell = theirs[k].iter().fold(Cell::default(), |mut a, c| {
            a.bits += c.bits;
            a.n += c.n;
            a
        });
        println!(
            "{:>8} {:>10} {:>12.4} {:>12.4} {:>+10.4}",
            "all",
            ot.n,
            ot.mean(),
            pt.mean(),
            pt.mean() - ot.mean()
        );
    }

    println!("\n  the novel column is the one that matters. a counter is flat there by");
    println!("  construction: a context it has never seen costs what it costs however much");
    println!("  else it holds. if our margin there grows down the deciles, that is");
    println!("  acquisition acceleration; if it is flat or negative, the organisation claim");
    println!("  is empty and this is an expensive counter.");
    println!();
}
