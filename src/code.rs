//! The emitted distribution and the charge.
//!
//! One expression:
//!
//! ```text
//! q(o) = P(o) . exp(s(o)) / Z ,      P = prior_of(u, .),  s(o) = <W(o), phi>
//! Z    = 1 + sum over rows of  P(o) . (exp(s(o)) - 1)
//! ```
//!
//! A product of experts. The address supplies a prior over what this regime
//! emits -- private, exact, counted, never trained. The features supply a
//! likelihood over what the current situation looks like -- global, shared,
//! learned from every write anywhere. `Z` telescopes because `s(o)` is zero
//! wherever there is no row, so the normaliser costs O(rows), not O(V), and the
//! sum is exact rather than truncated.
//!
//! # What this replaced, and why
//!
//! The previous form was a per-level factorisation: one branch probability per
//! tree level, multiplied together, with a mixture over the siblings the walk
//! did not enter at every level and a separate combination rule at the tail. It
//! was elegant -- the code was a product of factors, one committed per tick, and
//! it telescoped to one at every depth.
//!
//! It was also the reason depth hurt. Those branch probabilities are softmaxes
//! over inner products against *placed* prototypes, at an inherited constant
//! temperature, with nothing anywhere calibrating them; measured, their expected
//! calibration error ran 0.415 at level one, 0.291 at level two, 0.151 at level
//! three. The code multiplied one such number in per level, so error compounded
//! with depth. The address was being asked to supply probabilities and it
//! structurally cannot: nothing in the design fits it to produce them.
//!
//! So the address no longer supplies probabilities. It supplies a prior and a
//! candidate set, which is what a placed, untrained structure can honestly
//! provide, and the branch scores go back to their one competent job -- deciding
//! where to walk.
//!
//! The cost is stated where it belongs: sharing `W` means a departed regime's
//! *function* keeps being rewritten by everything that arrives after it, which
//! is interference, the failure the reference mechanism's allocation exists to
//! convert away. The claim here is not that the trade was wrong but that it was
//! applied to the wrong object: allocate what differs between regimes -- the
//! counts -- and share what does not -- the map from features to tokens.

use crate::tree::Tree;

/// The path a particle has committed to. Only the path: the per-level branch
/// probabilities it used to carry are no longer part of the code, because they
/// were never probabilities.
#[derive(Clone)]
pub struct PathCode {
    /// Node ids from the root, inclusive. `path[0]` is the root.
    pub path: Vec<usize>,
}

impl PathCode {
    pub fn root() -> Self {
        PathCode { path: vec![0] }
    }
    pub fn leaf(&self) -> usize {
        *self.path.last().unwrap()
    }
    pub fn depth(&self) -> usize {
        self.path.len() - 1
    }
    pub fn push(&mut self, child: usize) {
        self.path.push(child);
    }
    /// Drop the deepest commitment. Used when a particle backtracks rather than
    /// dying.
    pub fn pop(&mut self) {
        if self.path.len() > 1 {
            self.path.pop();
        }
    }
}

/// exp(s(o)) for every row, and the normaliser, in one pass over the rows.
///
/// Returned rather than recomputed per token so that the charge, the argmax and
/// the entropy all see the same numbers.
pub struct Scored {
    /// (token, P(o), exp(s(o))) for every token that has a row.
    pub rows: Vec<(u32, f32, f32)>,
    pub z: f32,
}

/// The candidate set is the node's *own* emitted tokens, not the whole shared
/// table.
///
/// Sharing `W` shares the function, not the candidates: what the address
/// contributes is precisely which tokens are in play here, and scoring every
/// token the system has ever emitted anywhere would throw that away -- along
/// with the property that retrieval cost follows the content stored at a node
/// rather than the content stored anywhere. It is also what keeps the
/// normaliser O(tokens at this node) instead of O(everything).
pub fn score(tree: &Tree, u: usize, phi: &[f32], use_readout: bool) -> Scored {
    let node = &tree.arena[u];
    let mut rows = Vec::with_capacity(node.counts.len());
    let mut z = 1.0f32;
    if use_readout {
        let w = tree.fw.min(phi.len());
        // Every token this node has emitted, uncapped.
        //
        // A positional cap looks like the reference mechanism's "bound the
        // scored set" and is not: the scored set has to be a function of the
        // node and the features alone, never of the token being asked about, or
        // the distribution stops being normalised. Cap it and a target outside
        // the cap gets no boost, its error stays at one forever, and its row
        // grows in one direction without ever converging -- which took every
        // accuracy in this suite below chance. If a shallow node's candidate set
        // is large, that is a cost to measure, not to truncate away.
        for &(tok, _) in node.counts.iter() {
            let row = match tree.row_of(tok) {
                None => continue,
                Some(r) => r,
            };
            let s = crate::num::dot(&row[..w], &phi[..w]);
            // Clamped so a large score cannot overflow the normaliser. The
            // bound is generous relative to any score the delta rule produces.
            let e = s.clamp(-30.0, 30.0).exp();
            let p = tree.prior_of(u, tok);
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
    pub fn prob_of(&self, tree: &Tree, u: usize, tok: u32) -> f32 {
        for &(t, p, e) in self.rows.iter() {
            if t == tok {
                return p * e / self.z;
            }
        }
        // No row: the likelihood term is one, so the prior stands.
        tree.prior_of(u, tok) / self.z
    }

    /// The most probable token among those with rows, and its probability.
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

/// q(o) for one token at one node.
pub fn prob(tree: &Tree, u: usize, phi: &[f32], tok: u32, use_readout: bool) -> f32 {
    score(tree, u, phi, use_readout).prob_of(tree, u, tok)
}

/// Codelength in bits of the observed token. Always finite: the escape chain
/// terminates in a uniform over the vocabulary and `Z` is bounded below.
pub fn charge_bits(prob: f32) -> f64 {
    let p = prob.max(1e-30) as f64;
    -p.log2()
}

// ---------------------------------------------------------------------------
// The full distribution, for the sharpening curve.
//
// Every token never seen anywhere receives the same probability, because its
// path through the escape chain is identical and it has no row. So the spread is
// computed exactly over the observed support plus one constant.
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

pub fn spread(tree: &Tree, u: usize, phi: &[f32], use_readout: bool) -> Spread {
    let root = &tree.arena[0];
    let n = root.counts.len();
    let sc = score(tree, u, phi, use_readout);
    let mut boost: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
    for &(t, _, e) in sc.rows.iter() {
        boost.insert(t, e);
    }
    let mut support = Vec::with_capacity(n);
    for &(tok, _) in root.counts.iter() {
        let p = tree.prior_of(u, tok);
        let e = *boost.get(&tok).unwrap_or(&1.0);
        support.push(p * e / sc.z);
    }
    // A token never emitted anywhere has no row and no count at any node, so its
    // prior is the escape product down to the uniform, identical for all of
    // them.
    let unseen_each = tree.prior_unseen(u) / sc.z;
    Spread { support, unseen_each, unseen_count: tree.vocab.saturating_sub(n) }
}
