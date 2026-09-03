//! The emitted distribution and the charge.
//!
//! ```text
//! P(o) = sum_u vis[u] . prior(o | u)          the walk's prior
//! q(o) = P(o) . exp(<W(o), phi>) / Z          times the shared likelihood
//! Z    = 1 + sum over candidates of P(o) . (exp(s(o)) - 1)
//! ```
//!
//! A product of experts, and the prior half is a *mixture over the nodes the
//! response has visited* rather than the counts of one chosen node.
//!
//! That mixture is the point. Choosing a node is a commitment, and a commitment
//! made by a static structure against a drifting world is wrong often enough for
//! the asymmetry to bite: a sharp prior on the wrong candidate set removes the
//! right answer entirely, which costs logarithmically, while being right only
//! pays linearly. A trajectory does not commit -- it accumulates, and a wrong
//! hop is one term in a mixture rather than the whole prior.
//!
//! `Z` telescopes because `s(o)` is zero wherever there is no row, so the
//! normaliser costs O(candidates) and is exact rather than truncated. The
//! candidate set has to be a function of the visit distribution and the features
//! alone, never of the token being asked about: an exception for the observed
//! token would leave the distribution unnormalised, and a positional cap would
//! leave a target outside it with an error stuck at one and a row growing in one
//! direction forever.

use crate::store::Store;

/// The nodes a response has touched, with weights summing to one.
#[derive(Clone, Default)]
pub struct Visit {
    pub w: Vec<(usize, f32)>,
}

impl Visit {
    pub fn single(node: usize) -> Self {
        Visit { w: vec![(node, 1.0)] }
    }

    pub fn is_empty(&self) -> bool {
        self.w.is_empty()
    }

    /// The node carrying the most weight. Used for the write's own bookkeeping
    /// and nowhere in the scoring.
    pub fn argmax(&self) -> usize {
        let mut best = (0usize, f32::NEG_INFINITY);
        for &(u, w) in self.w.iter() {
            if w > best.1 {
                best = (u, w);
            }
        }
        best.0
    }

    /// Entropy of the visit distribution. Saturating toward uniform over the
    /// graph is the failure mode this design has to be watched for: a walk that
    /// diffuses has a prior that says nothing.
    pub fn entropy_bits(&self) -> f64 {
        let mut h = 0.0f64;
        for &(_, w) in self.w.iter() {
            let w = w as f64;
            if w > 1e-12 {
                h -= w * w.log2();
            }
        }
        h
    }

    /// prior(o) under the mixture. Normalised over the vocabulary because each
    /// term is and the weights sum to one.
    pub fn prior(&self, store: &Store, tok: u32) -> f32 {
        let mut p = 0.0f32;
        for &(u, w) in self.w.iter() {
            p += w * store.prior_of(u, tok);
        }
        p
    }

    pub fn prior_unseen(&self, store: &Store) -> f32 {
        let mut p = 0.0f32;
        for &(u, w) in self.w.iter() {
            p += w * store.prior_unseen(u);
        }
        p
    }

    /// Every token any visited node has seen: the candidate set.
    pub fn candidates(&self, store: &Store) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for &(u, _) in self.w.iter() {
            for &(t, _) in store.nodes[u].counts.iter() {
                if !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        out
    }
}

/// exp(s(o)) for every candidate, and the normaliser, in one pass.
///
/// Returned rather than recomputed per token so the charge, the argmax, the
/// entropy and the delta rule all see the same numbers.
pub struct Scored {
    /// (token, P(o), exp(s(o))).
    pub rows: Vec<(u32, f32, f32)>,
    pub z: f32,
}

pub fn score(store: &Store, visit: &Visit, phi: &[f32], use_readout: bool) -> Scored {
    let mut rows = Vec::new();
    let mut z = 1.0f32;
    if use_readout && !visit.is_empty() {
        let w = store.fw.min(phi.len());
        for tok in visit.candidates(store) {
            let row = match store.row_of(tok) {
                None => continue,
                Some(r) => r,
            };
            let s = crate::num::dot(&row[..w], &phi[..w]);
            // Clamped so one large score cannot overflow the normaliser.
            let e = s.clamp(-30.0, 30.0).exp();
            let p = visit.prior(store, tok);
            z += p * (e - 1.0);
            rows.push((tok, p, e));
        }
    }
    if z < 1e-20 {
        z = 1e-20;
    }
    Scored { rows, z }
}

impl Scored {
    #[inline]
    pub fn prob_of(&self, store: &Store, visit: &Visit, tok: u32) -> f32 {
        for &(t, p, e) in self.rows.iter() {
            if t == tok {
                return p * e / self.z;
            }
        }
        // No row: the likelihood term is one, so the prior stands.
        visit.prior(store, tok) / self.z
    }

    pub fn top(&self) -> Option<(u32, f32)> {
        let mut best: Option<(u32, f32)> = None;
        for &(t, p, e) in self.rows.iter() {
            let q = p * e / self.z;
            match best {
                Some((bt, bq)) if bq > q || (bq == q && bt < t) => {}
                _ => best = Some((t, q)),
            }
        }
        best
    }
}

pub fn prob(store: &Store, visit: &Visit, phi: &[f32], tok: u32, use_readout: bool) -> f32 {
    score(store, visit, phi, use_readout).prob_of(store, visit, tok)
}

/// Codelength in bits. Always finite: the escape chain terminates in a uniform
/// and `Z` is bounded below.
pub fn charge_bits(prob: f32) -> f64 {
    let p = prob.max(1e-30) as f64;
    -p.log2()
}

// ---------------------------------------------------------------------------
// The full distribution, for the sharpening curve. Every token nothing has ever
// seen receives the same probability, so the spread is exact over the observed
// support plus one constant.
// ---------------------------------------------------------------------------

pub struct Spread {
    pub support: Vec<f32>,
    pub unseen_each: f32,
    pub unseen_count: usize,
}

impl Spread {
    pub fn entropy_bits(&self) -> f64 {
        let mut h = 0.0f64;
        for &q in self.support.iter() {
            let q = q as f64;
            if q > 1e-30 {
                h -= q * q.log2();
            }
        }
        let u = self.unseen_each as f64;
        if u > 1e-30 {
            h -= (self.unseen_count as f64) * u * u.log2();
        }
        h
    }
    pub fn mass(&self) -> f64 {
        let mut m = 0.0f64;
        for &q in self.support.iter() {
            m += q as f64;
        }
        m + self.unseen_count as f64 * self.unseen_each as f64
    }
}

pub fn spread(store: &Store, visit: &Visit, phi: &[f32], use_readout: bool) -> Spread {
    let sc = score(store, visit, phi, use_readout);
    let mut boost: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
    for &(t, _, e) in sc.rows.iter() {
        boost.insert(t, e);
    }
    let n = store.global.counts.len();
    let mut support = Vec::with_capacity(n);
    for &(tok, _) in store.global.counts.iter() {
        let p = visit.prior(store, tok);
        let e = *boost.get(&tok).unwrap_or(&1.0);
        support.push(p * e / sc.z);
    }
    let unseen_each = visit.prior_unseen(store) / sc.z;
    Spread { support, unseen_each, unseen_count: store.vocab.saturating_sub(n) }
}
