//! Is there anything left in this data that we have not taken?
//!
//! Nothing this project has measured has reached its own limit. On a stationary
//! stream whose truth is zero bits the charge is still descending at 600k
//! events at every step size tried, and on PhysioNet the band where a counter
//! beats us is the band where we have not arrived rather than a price we pay
//! forever. If every apparent floor is an unfinished descent, then "more
//! experience" and "a better mechanism" are two separate levers, and everything
//! done to this architecture lately has pulled the second one.
//!
//! So measure the first, without needing data we do not have. Three quarters of
//! the patients are a pool the model is walked through repeatedly; the last
//! quarter it sees once, at the end, and that is the only stretch anyone reads.
//! The counter runs the identical protocol, replays included, so neither side
//! is handed an advantage by the shape of the experiment.
//!
//! What this can answer: whether the mechanism has extracted what this data
//! holds. What it cannot answer: what genuinely new patients would be worth --
//! replaying the same records is not new experience, and a fall here is the
//! model finishing a descent rather than the model being given more world.

use std::collections::HashMap;

use crate::clinical::{build_vocab, flat_streams_under, flat_streams_valued};
use crate::config::Config;
use crate::model::Model;

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

const ORDERS: usize = 4;

/// (our bits/event on the held-out quarter, the counter's at its best order,
/// which order that was, events scored).
fn run_epochs(
    streams: &[Vec<(u32, u32)>],
    vp: usize,
    d: usize,
    banks: usize,
    cap: usize,
    seed: u64,
    epochs: usize,
) -> (f64, f64, usize, u64) {
    let split = streams.len() * 3 / 4;
    let (pool, held) = streams.split_at(split);
    run_split(pool, held, vp, d, banks, cap, seed, epochs)
}

#[allow(clippy::too_many_arguments)]
fn run_split(
    pool: &[Vec<(u32, u32)>],
    held: &[Vec<(u32, u32)>],
    vp: usize,
    d: usize,
    banks: usize,
    cap: usize,
    seed: u64,
    epochs: usize,
) -> (f64, f64, usize, u64) {
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = vp;
    cfg.d = d;
    cfg.cleanup_floor_mult = 1.1;
    cfg.mem_banks = banks;
    cfg.derive();
    let mut model = Model::new(cfg);
    let blank = model.volatile();
    let mut ppms: Vec<Ppm> = (1..=ORDERS).map(|o| Ppm::new(o, vp)).collect();

    let mut walk = |model: &mut Model, ppms: &mut Vec<Ppm>, s: &[Vec<(u32, u32)>], score: bool| {
        let mut ours = (0.0f64, 0u64);
        let mut theirs = vec![0.0f64; ORDERS];
        for st in s.iter() {
            model.restore(blank.clone());
            for p in ppms.iter_mut() {
                p.reset_hist();
            }
            for &(gap, tok) in st.iter() {
                for _ in 0..silence_ticks(gap, cap) {
                    model.tick(None, false);
                }
                let out = model.tick(Some(tok as usize), false);
                for (oi, p) in ppms.iter_mut().enumerate() {
                    let b = p.observe(tok);
                    if score && out.charged {
                        theirs[oi] += b;
                    }
                }
                if score && out.charged {
                    ours.0 += out.bits;
                    ours.1 += 1;
                }
            }
        }
        (ours, theirs)
    };

    for _ in 0..epochs {
        walk(&mut model, &mut ppms, pool, false);
    }
    let (ours, theirs) = walk(&mut model, &mut ppms, held, true);

    let n = ours.1.max(1);
    let mut best = 0usize;
    for oi in 1..ORDERS {
        if theirs[oi] < theirs[best] {
            best = oi;
        }
    }
    (ours.0 / n as f64, theirs[best] / n as f64, best + 1, ours.1)
}

/// Records the model has never met, tokenised under the vocabulary of the ones
/// it has.
///
/// The tokeniser hands out parameter ids in file order and puts the quantile
/// cuts where that directory's values fall, so a second set read on its own
/// terms produces ids that mean something else. Both sets go through set-a's
/// vocabulary or the comparison is between two different alphabets and nothing
/// in the numbers would say so.
pub fn run_held(
    dir: &str,
    held_dir: &str,
    d: usize,
    patients: usize,
    seed: u64,
    cap: usize,
    bins: usize,
    banks: usize,
) {
    let vocab = build_vocab(dir, bins);
    let (mut pool, _, _) = flat_streams_valued(dir, bins);
    let (mut held, _, dropped) = flat_streams_under(held_dir, &vocab);
    if patients > 0 && pool.len() > patients {
        pool.truncate(patients);
    }
    if patients > 0 && held.len() > patients / 2 {
        held.truncate(patients / 2);
    }
    let vp = vocab.size();
    println!("what is genuinely new experience worth?");
    println!(
        "  pool {} patients from {}, held {} patients from {} (never seen), V = {}",
        pool.len(),
        dir,
        held.len(),
        held_dir,
        vp
    );
    println!(
        "  {} readings in the held set were dropped: their parameter is not in the pool's vocabulary.",
        dropped
    );
    println!("  the counter runs the identical protocol, replays included.\n");
    println!("{:>8} {:>12} {:>14} {:>8} {:>10}", "epochs", "ours", "PPM-C best", "order", "margin");
    let counts = [0usize, 1, 2, 4];
    let rows: Vec<(usize, (f64, f64, usize, u64))> = std::thread::scope(|sc| {
        let (p, h) = (&pool, &held);
        let hs: Vec<_> = counts
            .iter()
            .map(|e| {
                let e = *e;
                sc.spawn(move || (e, run_split(p, h, vp, d, banks, cap, seed, e)))
            })
            .collect();
        hs.into_iter().map(|x| x.join().unwrap()).collect()
    });
    for (e, (o, p, ord, _n)) in rows.iter() {
        println!("{:>8} {:>12.4} {:>14.4} {:>8} {:>+10.4}", e, o, p, ord, p - o);
    }
    println!("\n  zero epochs is cold: neither side has seen anything before the held set.");
    println!("  a fall down the first column is what experience on other patients is");
    println!("  worth on patients nobody has met, which is the question replaying the");
    println!("  same records cannot answer.");
}

pub fn run(dir: &str, d: usize, patients: usize, seed: u64, cap: usize, bins: usize, banks: usize) {
    let (mut streams, _ids, vp) = flat_streams_valued(dir, bins);
    if patients > 0 && streams.len() > patients {
        streams.truncate(patients);
    }
    let split = streams.len() * 3 / 4;
    println!("replaying what we already have, and reading only what we have not seen");
    println!(
        "  {} patients: {} replayed, {} held out and walked once at the end",
        streams.len(),
        split,
        streams.len() - split
    );
    println!("  the counter runs the identical protocol, replays included.\n");
    println!(
        "{:>8} {:>12} {:>14} {:>8} {:>10}",
        "epochs", "ours", "PPM-C best", "order", "margin"
    );

    let counts = [1usize, 2, 4];
    let rows: Vec<(usize, (f64, f64, usize, u64))> = std::thread::scope(|sc| {
        let sref = &streams;
        let hs: Vec<_> = counts
            .iter()
            .map(|e| {
                let e = *e;
                sc.spawn(move || (e, run_epochs(sref, vp, d, banks, cap, seed, e)))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (e, (o, p, ord, _n)) in rows.iter() {
        println!("{:>8} {:>12.4} {:>14.4} {:>8} {:>+10.4}", e, o, p, ord, p - o);
    }
    println!("\n  a fall down the first column is experience the mechanism had not yet");
    println!("  taken out of records it already held. a flat column is a mechanism that");
    println!("  has finished with this data, and says the lever is the mechanism.");
    println!("  neither reading says what genuinely new patients would be worth.");
}
