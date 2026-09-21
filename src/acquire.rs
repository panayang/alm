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
//!
//! # What it measures, on PhysioNet
//!
//! Read the following as what it is. The margin against a counter is a reading
//! taken inside the counter's own frame -- the codelength of the next token,
//! which is the autoregressive target function this file opens by refusing --
//! and it is not the objective. It is useful for one thing: a margin that moves
//! when a mechanism is repaired says the repair reached something real. It is
//! not evidence that the architecture's own claim holds, and it must never
//! become the thing being maximised, because a number maximised is a reward and
//! this design has none.
//!
//! 3996 patients, 1733518 events, V = 296, d = 256, inside the capacity law,
//! against PPM-C at its best order over the same stream (order 2, 4.0970 bits):
//!
//! ```text
//!                    ours      PPM      margin
//!   overall        3.3842   4.0970      +0.713
//!   seen context   3.3490   4.0698      +0.721
//!   novel context  5.6695   5.8333      +0.164
//! ```
//!
//! It is not acceleration. Both columns settle by the second decile and hold:
//! seen runs +0.76 +0.74 +0.73 +0.73 +0.73 +0.74 +0.69 +0.71 +0.71, novel rises
//! to +0.30 and comes back to +0.14 with no trend. What the measurement says is
//! that we converge 0.71 bits below a counter and stay there.
//!
//! A warning that cost two rounds to learn: on a 400-patient subset both columns
//! appear to grow monotonically to the last decile, twice, and both times the
//! full stream showed it flat. 175k events is entirely inside the learning
//! phase, so a trend read at that size is a statement about the subset.
//!
//! What the split below is for is different and does not depend on a baseline:
//! it says *where* a loss sits -- on rare targets, on new combinations, on
//! neither -- and that is a question about how memory is organised, which is
//! answerable without anyone keeping score.

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

/// One charged event, for the split that separates novelty from rarity.
struct Rec {
    /// The order-2 context has not occurred before.
    novel: bool,
    /// The order-1 context has: a counter escaping from order 2 lands on a
    /// conditional, not on the marginal.
    s1: bool,
    /// How often the target itself has been the target before, bucketed.
    tb: u8,
    /// How often this exact (order-1 context, target) association has occurred
    /// before. This is the evidence a counter escaping to order 1 actually has,
    /// and counting it once is enough for the counter; the delta rule needs
    /// several corrections to move the same mass.
    o1c: u8,
    ours: f32,
    ppm: [f32; ORDERS],
}

const ORDERS: usize = 6;
const RARITY: [&str; 5] = ["0", "1-9", "10-99", "100-999", "1000+"];
const ASSOC: [&str; 5] = ["0", "1", "2-4", "5-19", "20+"];

fn assoc_bucket(c: u32) -> usize {
    match c {
        0 => 0,
        1 => 1,
        2..=4 => 2,
        5..=19 => 3,
        _ => 4,
    }
}

fn rarity_bucket(c: u32) -> usize {
    match c {
        0 => 0,
        1..=9 => 1,
        10..=99 => 2,
        100..=999 => 3,
        _ => 4,
    }
}

/// Separate the two things the novel column mixes.
///
/// The raw novel column conflates "a new combination of seen parts" with "a
/// rare token", because new contexts are disproportionately made of rare
/// tokens, and a model without a counted prior is expected to pay on rare
/// tokens whatever it does with combinations. So compare novel against seen
/// *inside* a rarity bucket of the target. If the margin is the same in both,
/// the novel loss is rarity and says nothing about composition; if it is worse
/// on novel at matched rarity, composition is where we lose.
///
/// And by what a counter falls back to. When the order-2 context is new but the
/// order-1 context is not, PPM escapes to a conditional it has counted; only
/// when both are new does it fall to the marginal.
fn split_novelty(recs: &[Rec], best: usize) {
    println!("\n-- novelty vs rarity: margin = PPM-C o{} minus ours, bits per event --", best + 1);
    println!(
        "{:>10} | {:>26} | {:>26} | {:>26}",
        "target n",
        "seen o2 context",
        "novel o2, seen o1",
        "novel o2, novel o1"
    );
    println!(
        "{:>10} | {:>7} {:>8} {:>9} | {:>7} {:>8} {:>9} | {:>7} {:>8} {:>9}",
        "", "events", "ours", "margin", "events", "ours", "margin", "events", "ours", "margin"
    );
    for tb in 0..RARITY.len() {
        let mut line = format!("{:>10} |", RARITY[tb]);
        for (novel, s1) in [(false, true), (true, true), (true, false)] {
            let (mut n, mut o, mut p) = (0u64, 0.0f64, 0.0f64);
            for r in recs.iter() {
                if r.tb as usize == tb && r.novel == novel && (!novel || r.s1 == s1) {
                    n += 1;
                    o += r.ours as f64;
                    p += r.ppm[best] as f64;
                }
            }
            if n == 0 {
                line += &format!(" {:>7} {:>8} {:>9} |", 0, "-", "-");
            } else {
                line += &format!(
                    " {:>7} {:>8.3} {:>+9.3} |",
                    n,
                    o / n as f64,
                    (p - o) / n as f64
                );
            }
        }
        println!("{}", line.trim_end_matches('|'));
    }
    // The cell that stays negative, opened by the evidence a counter has.
    println!(
        "
-- the cell that stays negative: novel order-2, seen order-1, by how often --"
    );
    println!("-- this exact (order-1 context, target) association has occurred before --");
    println!(
        "{:>12} {:>10} {:>12} {:>12} {:>10}",
        "assoc count", "events", "ours", format!("PPM-C o{}", best + 1), "margin"
    );
    for ab in 0..ASSOC.len() {
        let (mut n, mut o, mut p) = (0u64, 0.0f64, 0.0f64);
        for r in recs.iter() {
            if r.novel && r.s1 && r.o1c as usize == ab {
                n += 1;
                o += r.ours as f64;
                p += r.ppm[best] as f64;
            }
        }
        if n == 0 {
            continue;
        }
        println!(
            "{:>12} {:>10} {:>12.3} {:>12.3} {:>+10.3}",
            ASSOC[ab],
            n,
            o / n as f64,
            p / n as f64,
            (p - o) / n as f64
        );
    }
    println!("  a counter needs one occurrence to hold an association; the delta rule");
    println!("  needs several corrections to move the same mass. if the loss is at the");
    println!("  low counts and gone by the high ones, that is what this cell is.");

    println!("  read across a row: same rarity of the target, different novelty of the");
    println!("  context. equal margins mean the novel loss is rarity, not composition.");
}

#[inline]
fn silence_ticks(g: u32, cap: usize) -> usize {
    ((32 - g.max(1).leading_zeros()).saturating_sub(1) as usize).min(cap)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    dir: &str,
    label: &str,
    d: usize,
    patients: usize,
    seed: u64,
    cap: usize,
    bins: usize,
    banks: usize,
    eta: Option<f32>,
    verify_gate: bool,
) {
    let (mut streams, _ids, vp) = flat_streams_valued(dir, bins);
    if patients > 0 && streams.len() > patients {
        streams.truncate(patients);
    }
    let total: usize = streams.iter().map(|s| s.len()).sum();

    println!("==================== {} ====================", label);
    println!(
        "patients {}   events {}   vocabulary {}   d = {}   eta = {}   verify_gate = {}",
        streams.len(),
        total,
        vp,
        d,
        eta.unwrap_or(Config::local().eta),
        verify_gate
    );

    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = vp;
    cfg.d = d;
    cfg.cleanup_floor_mult = 1.1;
    cfg.mem_banks = banks;
    if let Some(e) = eta {
        cfg.eta = e;
    }
    cfg.verify_gate = verify_gate;
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

    // Every order, not a pinned one.
    //
    // This used to run PPM-C at order 3 and report it as "the counter". That
    // breaks the rule the rest of this project runs on -- both sides sweep
    // their own hyperparameters or the number is not reported -- and it has
    // already reversed a result once: on the clinical subset the best order was
    // 1, and pinning 3 turned a 0.158-bit loss into a 0.010-bit win. So run all
    // of them and hand the baseline its best.
    let mut ppms: Vec<Ppm> = (1..=ORDERS).map(|o| Ppm::new(o, vp)).collect();

    // [novel?][decile]
    let mut ours = [[Cell::default(); DECILES]; 2];
    let mut theirs_all = vec![[[Cell::default(); DECILES]; 2]; ORDERS];
    // Which order-2 contexts have been seen, for the split only. This is a
    // property of the stream and is shared by both models, so neither is
    // scored on a different population than the other.
    let mut seen_ctx: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut idx = 0usize;

    // For the split below: which order-1 contexts have been seen, and how many
    // times each token has been the target so far. Properties of the stream,
    // shared by both models.
    let mut seen1: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut assoc: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    let mut tcount: Vec<u32> = vec![0; vp.max(1)];
    let mut recs: Vec<Rec> = Vec::with_capacity(total);

    for s in streams.iter() {
        model.restore(blank.clone());
        for p in ppms.iter_mut() {
            p.reset_hist();
        }
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
            let s1 = p1.map_or(false, |b| seen1.contains(&b));
            let tb = rarity_bucket(tcount.get(tok as usize).copied().unwrap_or(0));
            let akey = p1.map(|b| (b as u64) << 32 | tok as u64);
            let o1c = assoc_bucket(akey.and_then(|k| assoc.get(&k).copied()).unwrap_or(0));
            let mut pb = [0.0f32; ORDERS];

            for _ in 0..silence_ticks(gap, cap) {
                model.tick(None, false);
            }
            let out = model.tick(Some(tok as usize), false);
            for (oi, p) in ppms.iter_mut().enumerate() {
                let b = p.observe(tok);
                pb[oi] = b as f32;
                if out.charged {
                    theirs_all[oi][novel as usize][dec].push(b);
                }
            }
            if out.charged {
                ours[novel as usize][dec].push(out.bits);
                recs.push(Rec {
                    novel,
                    s1,
                    tb: tb as u8,
                    o1c: o1c as u8,
                    ours: out.bits as f32,
                    ppm: pb,
                });
            }

            if let Some(c) = ctx {
                seen_ctx.insert(c);
            }
            if let Some(b) = p1 {
                seen1.insert(b);
            }
            if let Some(k) = akey {
                *assoc.entry(k).or_insert(0) += 1;
            }
            if let Some(c) = tcount.get_mut(tok as usize) {
                *c += 1;
            }
            p2 = p1;
            p1 = Some(tok);
            idx += 1;
        }
    }

    // The baseline is the best order over the whole stream, chosen once and
    // used for both columns, so it cannot pick a different order per column.
    let mut best_order = 0usize;
    let mut best_bits = f64::INFINITY;
    for (oi, t) in theirs_all.iter().enumerate() {
        let (mut b, mut n) = (0.0f64, 0u64);
        for k in 0..2 {
            for d in 0..DECILES {
                b += t[k][d].bits;
                n += t[k][d].n;
            }
        }
        let m = b / n.max(1) as f64;
        println!("PPM-C order {}: {:.4} bits overall", oi + 1, m);
        if m < best_bits {
            best_bits = m;
            best_order = oi;
        }
    }
    println!("baseline: PPM-C order {} at {:.4} bits
", best_order + 1, best_bits);
    let theirs = theirs_all[best_order];

    for (k, name) in [(1usize, "NOVEL context (counter has nothing)"), (0, "seen context")] {
        println!("\n-- {} --", name);
        println!(
            "{:>8} {:>10} {:>12} {:>12} {:>10}",
            "decile",
            "events",
            "ours",
            format!("PPM-C o{}", best_order + 1),
            "margin"
        );
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

    split_novelty(&recs, best_order);

    println!("\n  the raw novel column mixes composition with rarity, and with what a");
    println!("  counter escapes to: new order-2 contexts are mostly made of rare tokens,");
    println!("  and a counter with a seen order-1 context escapes to a counted");
    println!("  conditional rather than to the marginal. read the table above, not the");
    println!("  raw column: the composition claim is the margin on novel contexts at");
    println!("  matched rarity, against the margin on seen ones.");
    println!();
}
