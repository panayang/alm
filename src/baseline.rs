//! The baseline that actually matters: variable-order backoff.
//!
//! Structurally this design is close to PPM and to context-tree weighting -- a
//! variable-order context tree, per-node counts, and a fall back to a shorter
//! context when the longer one has not been seen. That is a defensible position,
//! but it settles what the comparison has to be. A transformer is the wrong
//! baseline here; it answers a different question and costs O(N) per token for
//! the privilege. PPM is the same family, is cheap, and is strong.
//!
//! If the mechanism cannot beat interpolated PPM-C on this stream, then neither
//! of the two things it claims over PPM -- a continuous, fuzzy, multi-timescale
//! context instead of an exact suffix, and a learned transform on the payload --
//! has paid for itself, and no amount of internal diagnostics changes that.
//!
//! PPM is fed the event tokens only, with the baseline ticks removed. That is
//! deliberately the *stronger* form of the baseline: it sees each cue adjacent
//! to its target and is never diluted by silence.

use std::collections::HashMap;

pub struct Ppm {
    order: usize,
    vocab: usize,
    /// Context hash -> (counts by symbol, total).
    ctx: HashMap<u64, (HashMap<u32, u64>, u64)>,
    history: Vec<u32>,
    pub total_bits: f64,
    pub events: u64,
    pub hits: u64,
}

#[inline]
fn hash_context(h: &[u32]) -> u64 {
    let mut x = 0xcbf2_9ce4_8422_2325u64;
    x ^= h.len() as u64;
    x = x.wrapping_mul(0x1000_0000_01b3);
    for &s in h {
        x ^= s as u64;
        x = x.wrapping_mul(0x1000_0000_01b3);
    }
    x
}

impl Ppm {
    pub fn new(order: usize, vocab: usize) -> Self {
        Ppm {
            order,
            vocab,
            ctx: HashMap::new(),
            history: Vec::new(),
            total_bits: 0.0,
            events: 0,
            hits: 0,
        }
    }

    /// Interpolated PPM-C: blend every order from longest to shortest, each
    /// contributing its remaining escape mass, terminating in a uniform.
    fn predict(&self, sym: u32) -> (f32, Option<u32>) {
        let mut p = 0.0f32;
        let mut esc = 1.0f32;
        let mut best: Option<(u32, f32)> = None;
        let n = self.history.len();
        let maxo = self.order.min(n);
        for o in (0..=maxo).rev() {
            let h = &self.history[n - o..];
            let key = hash_context(h);
            if let Some((counts, total)) = self.ctx.get(&key) {
                if *total == 0 {
                    continue;
                }
                let distinct = counts.len() as f32;
                let e = distinct / (distinct + *total as f32);
                let c = *counts.get(&sym).unwrap_or(&0) as f32;
                p += esc * (1.0 - e) * (c / *total as f32);
                if best.is_none() {
                    let mut b: Option<(u32, f32)> = None;
                    for (&s, &cc) in counts.iter() {
                        let v = cc as f32;
                        match b {
                            // Ties break to the lower symbol so the argmax is
                            // deterministic despite the hash map's ordering.
                            Some((bs, bv)) if bv > v || (bv == v && bs < s) => {}
                            _ => b = Some((s, v)),
                        }
                    }
                    best = b;
                }
                esc *= e;
            }
        }
        p += esc / self.vocab as f32;
        (p, best.map(|b| b.0))
    }

    fn update(&mut self, sym: u32) {
        let n = self.history.len();
        let maxo = self.order.min(n);
        for o in 0..=maxo {
            let key = hash_context(&self.history[n - o..]);
            let entry = self.ctx.entry(key).or_insert_with(|| (HashMap::new(), 0));
            *entry.0.entry(sym).or_insert(0) += 1;
            entry.1 += 1;
        }
        self.history.push(sym);
        if self.history.len() > self.order {
            let excess = self.history.len() - self.order;
            self.history.drain(0..excess);
        }
    }

    /// Predict, be charged, then write -- the same protocol the mechanism uses,
    /// so the two codelengths are comparable.
    pub fn observe(&mut self, sym: u32) -> (f64, bool) {
        let (p, top) = self.predict(sym);
        let bits = crate::code::charge_bits(p);
        self.total_bits += bits;
        self.events += 1;
        let hit = top == Some(sym);
        if hit {
            self.hits += 1;
        }
        self.update(sym);
        (bits, hit)
    }

    pub fn bits_per_event(&self) -> f64 {
        if self.events == 0 {
            0.0
        } else {
            self.total_bits / self.events as f64
        }
    }

    pub fn accuracy(&self) -> f64 {
        if self.events == 0 {
            0.0
        } else {
            self.hits as f64 / self.events as f64
        }
    }
}

pub struct PpmResult {
    pub ppm: Ppm,
    pub first: crate::metrics::Bucket,
    pub second: crate::metrics::Bucket,
    pub product: crate::metrics::Bucket,
}

/// Two variants, and the difference between them matters more than either.
///
/// * `skip_baseline = true` removes the silence, so PPM sees cue, cue, target as
///   consecutive symbols. It is handed the segmentation the mechanism refuses to
///   assume, and is therefore an *upper bound* on what a suffix model can do
///   here, not a like-for-like control.
/// * `skip_baseline = false` gives PPM the same stream the mechanism sees, with
///   silence as a symbol of its own. That is the like-for-like control, and the
///   gap between the two is a measure of how much the segmentation was worth.
pub fn run(stream: &crate::gen::Stream, order: usize, skip_baseline: bool) -> PpmResult {
    // One symbol above the vocabulary stands for silence.
    let mut ppm = Ppm::new(order, stream.vocab + 1);
    let silence = stream.vocab as u32;
    let mut second = crate::metrics::Bucket::default();
    let mut first = crate::metrics::Bucket::default();
    let mut product = crate::metrics::Bucket::default();
    for t in 0..stream.len() {
        let sym = match stream.observe(t) {
            Some(x) => x as u32,
            None => {
                if skip_baseline {
                    continue;
                }
                silence
            }
        };
        let (bits, hit) = ppm.observe(sym);
        if sym == silence {
            continue;
        }
        if let Some(i) = stream.ep_at[t] {
            match stream.episodes[i].kind {
                crate::gen::Kind::Second => second.push(bits, hit),
                crate::gen::Kind::Product => product.push(bits, hit),
                crate::gen::Kind::First => first.push(bits, hit),
                _ => {}
            }
        }
    }
    PpmResult { ppm, first, second, product }
}
