//! The emitted distribution and the charge.
//!
//! ```text
//! q(o) = exp(s(o)) / Z ,   s(o) = <R_o, phi> for a token with a row, 0 otherwise
//! Z    = sum over rows of exp(s(o))  +  (V - rows)
//! ```
//!
//! One softmax over the whole vocabulary, computed in O(rows) because every
//! token without a row scores exactly zero and those terms collapse into a
//! single count. Exact, not truncated, and normalised by construction -- there
//! is no candidate set to get wrong and no mixture whose weights could be
//! miscalibrated.
//!
//! There is no prior term. A counted prior degrades as the stream grows (it
//! converges on the corpus marginal), and the organisation that gives the data
//! meaning is the error-driven row rather than the frequency. See `store.rs`.
//!
//! What that costs: `q` is always positive but has no floor, so a confidently
//! wrong answer can be charged arbitrarily many bits. The tail of the charge
//! distribution is therefore an instrument, not a diagnostic.

use crate::store::Store;

/// exp(s(o)) for every token that has a row, and the normaliser.
///
/// Computed once per use so the charge, the argmax, the entropy and the write
/// all see the same numbers -- the write in particular must, or the rows would
/// be fitted against a distribution the ledger never charged.
pub struct Scored {
    pub rows: Vec<(u32, f32)>,
    pub z: f32,
    /// exp(0) = 1 for every token with no row.
    pub unseen_count: usize,
}

pub fn score(store: &Store, phi: &[f32], use_readout: bool) -> Scored {
    let mut rows = Vec::with_capacity(store.rows.len());
    let mut z = 0.0f32;
    if use_readout {
        let w = store.fw.min(phi.len());
        for (tok, row) in store.rows.iter() {
            let s = crate::num::dot(&row[..w], &phi[..w]);
            // Clamped so one large score cannot overflow the normaliser.
            let e = s.clamp(-30.0, 30.0).exp();
            z += e;
            rows.push((*tok, e));
        }
    }
    let unseen = store.vocab.saturating_sub(rows.len());
    z += unseen as f32;
    if z < 1e-20 {
        z = 1e-20;
    }
    Scored { rows, z, unseen_count: unseen }
}

impl Scored {
    #[inline]
    pub fn prob_of(&self, _store: &Store, tok: u32) -> f32 {
        for &(t, e) in self.rows.iter() {
            if t == tok {
                return e / self.z;
            }
        }
        1.0 / self.z
    }

    pub fn top(&self) -> Option<(u32, f32)> {
        let mut best: Option<(u32, f32)> = None;
        for &(t, e) in self.rows.iter() {
            let q = e / self.z;
            match best {
                Some((bt, bq)) if bq > q || (bq == q && bt < t) => {}
                _ => best = Some((t, q)),
            }
        }
        best
    }

    pub fn entropy_bits(&self) -> f64 {
        let mut h = 0.0f64;
        for &(_, e) in self.rows.iter() {
            let q = (e / self.z) as f64;
            if q > 1e-30 {
                h -= q * q.log2();
            }
        }
        let u = (1.0 / self.z) as f64;
        if u > 1e-30 {
            h -= self.unseen_count as f64 * u * u.log2();
        }
        h
    }

    pub fn mass(&self) -> f64 {
        let mut m = 0.0f64;
        for &(_, e) in self.rows.iter() {
            m += (e / self.z) as f64;
        }
        m + self.unseen_count as f64 * (1.0 / self.z) as f64
    }
}

/// Codelength in bits. Always finite because `Z` is bounded below, but not
/// bounded above: without a prior floor a confident miss is expensive, which is
/// exactly what the charge-tail instruments exist to catch.
pub fn charge_bits(prob: f32) -> f64 {
    let p = prob.max(1e-30) as f64;
    -p.log2()
}
