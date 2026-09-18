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
    pub unseen_count: usize,
    /// The weight each row-less token carries: exp(0 - max score), not 1.
    pub unseen_w: f32,
}

pub fn score(store: &Store, phi: &[f32], use_readout: bool) -> Scored {
    score_with(store, phi, use_readout, None, 0.0, None)
}

/// As `score`, plus the cleanup term: every token is also compared directly to
/// the state, so a retrieved cursor can be named.
pub fn score_with(
    store: &Store,
    phi: &[f32],
    use_readout: bool,
    state: Option<&[f32]>,
    codebook: f32,
    emb: Option<&crate::embed::Embeddings>,
) -> Scored {
    // Softmax by subtracting the maximum, not by clamping.
    //
    // This used to be `s.clamp(-30.0, 30.0).exp()`, with the reasoning that one
    // large score must not overflow the normaliser. It does prevent the
    // overflow, and it also destroys the distribution: the delta rule is
    // unregularised, so rows and biases grow without bound -- measured at row
    // norms of 350 against a feature norm of 3, and biases up to 4876 -- and
    // once several tokens sit above +30 they all become exp(30) and are exactly
    // indistinguishable. The readout then charges log2(V) no matter what it has
    // learned, which is what it was doing: 6.0095 bits on a 64-token stream,
    // 8.04 on a 296-token one, and *worse* as more was written, because more
    // writing means more saturation.
    //
    // Subtracting the maximum is shift-invariant, so it changes no probability
    // this was supposed to produce; it only stops them from being crushed
    // together. Unseen tokens carry an implicit score of zero and so enter the
    // maximum as well.
    let mut raw = Vec::with_capacity(store.rows.len());
    let mut rows = Vec::with_capacity(store.rows.len());
    let mut hi = f32::NEG_INFINITY;
    // The two terms are gated separately, and they range over different sets.
    //
    // The codebook comparison used to sit inside `if use_readout` and to iterate
    // `store.rows`. Both are wrong, and together they made the walk
    // unmeasurable: with `no_readout` the row write is skipped too, so there
    // were no rows, so nothing was scored, so the emitted distribution was
    // *exactly* uniform -- 6.0000 bits against a uniform of 6.0000 on a stream
    // whose conditional is deterministic and whose answer the walk is meant to
    // be able to name by itself. Every ablation that set `no_readout` was
    // measuring the absence of scoring rather than the absence of rows.
    //
    // The row term is a claim about what memory has learned to say, so it
    // ranges over the rows that exist. The codebook term is cleanup -- naming
    // the token a cursor has arrived at -- and naming cannot be conditional on
    // that token having been written before, because the first time the walk
    // reaches something is exactly when it has not been. So it ranges over the
    // whole codebook, and the O(rows) fast path stays for callers that do not
    // ask for it.
    let use_cb = codebook != 0.0 && state.is_some() && emb.is_some();
    let w = store.fw.min(phi.len());
    let unseen;
    if use_cb {
        let (p, tab) = (state.unwrap(), emb.unwrap());
        // Row contributions first, densely, so the codebook pass needs no
        // hashing per token.
        let mut acc = vec![0.0f32; store.vocab];
        if use_readout {
            for (tok, row) in store.rows.iter() {
                if let Some(a) = acc.get_mut(*tok as usize) {
                    *a = crate::num::dot(&row[..w], &phi[..w]);
                }
            }
            if store.use_bias {
                for (tok, a) in acc.iter_mut().enumerate() {
                    *a += store.bias.get(tok).copied().unwrap_or(0.0);
                }
            }
        }
        for tok in 0..store.vocab {
            let s = acc[tok] + codebook * crate::num::dot(tab.row(tok), p);
            if s > hi {
                hi = s;
            }
            raw.push((tok as u32, s));
        }
        unseen = 0;
    } else if use_readout {
        for (tok, row) in store.rows.iter() {
            let mut s = crate::num::dot(&row[..w], &phi[..w]);
            if store.use_bias {
                s += store.bias.get(*tok as usize).copied().unwrap_or(0.0);
            }
            if s > hi {
                hi = s;
            }
            raw.push((*tok, s));
        }
        unseen = store.vocab.saturating_sub(store.rows.len());
    } else {
        unseen = store.vocab;
    }
    if unseen > 0 && hi < 0.0 {
        hi = 0.0;
    }
    if !hi.is_finite() {
        hi = 0.0;
    }
    let mut z = 0.0f32;
    for (tok, s) in raw {
        let e = (s - hi).exp();
        z += e;
        rows.push((tok, e));
    }
    // The weight of a token with no row. Under max-subtraction this is
    // exp(0 - hi), not 1, and it is carried rather than re-derived because two
    // places need it and one of them silently assumed 1.
    let unseen_w = (-hi).exp();
    z += unseen as f32 * unseen_w;
    if z < 1e-20 {
        z = 1e-20;
    }
    Scored { rows, z, unseen_count: unseen, unseen_w }
}

impl Scored {
    #[inline]
    pub fn prob_of(&self, _store: &Store, tok: u32) -> f32 {
        // The codebook pass pushes every token in order, so the token's own
        // index is the answer. Worth the check: `write` calls this once per
        // touched token and the scan behind it is over the whole vocabulary.
        if let Some(&(t, e)) = self.rows.get(tok as usize) {
            if t == tok {
                return e / self.z;
            }
        }
        for &(t, e) in self.rows.iter() {
            if t == tok {
                return e / self.z;
            }
        }
        self.unseen_w / self.z
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
        // `unseen_w / z`, not `1 / z`. Under max-subtraction a token with no
        // row weighs exp(0 - max), which is 1 only when the maximum score is
        // zero. `mass()` had exactly this bug and the normalisation assertion
        // caught it there; nothing checks the entropy against a known value, so
        // here it survived the same fix.
        let u = (self.unseen_w / self.z) as f64;
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
        m + self.unseen_count as f64 * (self.unseen_w / self.z) as f64
    }
}

/// Codelength in bits. Always finite because `Z` is bounded below, but not
/// bounded above: without a prior floor a confident miss is expensive, which is
/// exactly what the charge-tail instruments exist to catch.
pub fn charge_bits(prob: f32) -> f64 {
    let p = prob.max(1e-30) as f64;
    -p.log2()
}
