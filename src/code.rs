//! The emitted distribution and the charge.
//!
//! After committing `l` factors, the system's output is
//!
//!   q_l(o) = prod_{j<=j*(o)} q(u_j | u_{j-1}) . prior(o | u_{j*})
//!
//! where j*(o) is the deepest level at which o's ancestor is still on the walked
//! path. Mass that went to branches the walk did not enter is filled by those
//! branches' own priors. Summing over the vocabulary telescopes to one, so this
//! is a proper normalised distribution at *every* l -- which is the whole point:
//! the code can be settled at any moment, and stopping early costs exactly the
//! prior entropy of the subtree that was never descended.
//!
//! A particle set is a bounded posterior over paths, so the emitted
//! distribution is the weight-mixture of the particles' own distributions. With
//! one particle it reduces to the single-path formula.

use crate::tree::Tree;

/// One particle's committed factorisation of the address.
#[derive(Clone)]
pub struct PathCode {
    /// Node ids from the root, inclusive. `path[0]` is the root.
    pub path: Vec<usize>,
    /// `taken[j]` is q(path[j+1] | path[j]).
    pub taken: Vec<f32>,
    /// `others[j]` holds the siblings the walk did not enter at level j, with
    /// their branch probabilities.
    pub others: Vec<Vec<(usize, f32)>>,
}

impl PathCode {
    pub fn root() -> Self {
        PathCode { path: vec![0], taken: Vec::new(), others: Vec::new() }
    }
    pub fn leaf(&self) -> usize {
        *self.path.last().unwrap()
    }
    pub fn depth(&self) -> usize {
        self.path.len() - 1
    }
    pub fn push(&mut self, child: usize, q_taken: f32, others: Vec<(usize, f32)>) {
        self.path.push(child);
        self.taken.push(q_taken);
        self.others.push(others);
    }
    /// Drop the deepest commitment. Used when a particle backtracks rather than
    /// dying.
    pub fn pop(&mut self) {
        if self.path.len() > 1 {
            self.path.pop();
            self.taken.pop();
            self.others.pop();
        }
    }
}

/// Probability the tail node assigns to one token, given the payload.
///
/// If the node has emitted rows and the readout is enabled, the count term is
/// replaced by the readout softmax, keeping the same escape structure. Turning
/// the readout off recovers pure count-based backoff, which is exactly the
/// ablation that says what the learned rows buy.
fn tail_prob(tree: &Tree, u: usize, p: &[f32], tok: u32, use_readout: bool) -> f32 {
    let node = &tree.arena[u];
    if !use_readout || node.rows.is_empty() {
        return tree.prior_of(u, tok);
    }
    let e = node.escape();
    let base = match node.parent {
        None => 1.0 / tree.vocab as f32,
        Some(par) => tree.prior_of(par, tok),
    };
    let mut own = 0.0f32;
    for (t, q) in tree.readout_dist(u, p) {
        if t == tok {
            own = q;
            break;
        }
    }
    e * base + (1.0 - e) * own
}

/// q_l(o) for one particle.
pub fn path_prob(tree: &Tree, code: &PathCode, p: &[f32], tok: u32, use_readout: bool) -> f32 {
    let mut acc = 0.0f32;
    let mut prefix = 1.0f32;
    for j in 0..code.taken.len() {
        for &(v, qv) in code.others[j].iter() {
            acc += prefix * qv * tree.prior_of(v, tok);
        }
        prefix *= code.taken[j];
    }
    acc + prefix * tail_prob(tree, code.leaf(), p, tok, use_readout)
}

/// The mixture over particles. `weights` must sum to one.
pub fn mixture_prob(
    tree: &Tree,
    codes: &[PathCode],
    payloads: &[Vec<f32>],
    weights: &[f32],
    tok: u32,
    use_readout: bool,
) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..codes.len() {
        if weights[i] <= 0.0 {
            continue;
        }
        acc += weights[i] * path_prob(tree, &codes[i], &payloads[i], tok, use_readout);
    }
    acc
}

/// Codelength in bits of the observed token under the emitted distribution.
/// Always finite: the escape chain terminates in a uniform over the vocabulary.
pub fn charge_bits(prob: f32) -> f64 {
    let p = prob.max(1e-30) as f64;
    -p.log2()
}

// ---------------------------------------------------------------------------
// Full distribution, for the sharpening curve.
//
// Materialising V floats every tick would dominate the run, and it is also
// unnecessary: every token the system has never seen anywhere receives exactly
// the same probability, because its path through the escape chain is identical.
// So the distribution is computed exactly over the observed support plus one
// constant covering the rest.
// ---------------------------------------------------------------------------

pub struct Spread {
    /// Probability of each token in the observed support, in the root's
    /// insertion order.
    pub support: Vec<f32>,
    /// Probability of any single token never observed anywhere.
    pub unseen_each: f32,
    /// How many such tokens there are.
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

/// prior(. | u) restricted to the support, plus the unseen constant.
fn prior_spread(tree: &Tree, u: usize, pos: &std::collections::HashMap<u32, usize>, out: &mut Vec<f32>, unseen: &mut f32) {
    let node = &tree.arena[u];
    match node.parent {
        None => {
            let p = 1.0 / tree.vocab as f32;
            for v in out.iter_mut() {
                *v = p;
            }
            *unseen = p;
        }
        Some(par) => prior_spread(tree, par, pos, out, unseen),
    }
    let e = node.escape();
    for v in out.iter_mut() {
        *v *= e;
    }
    *unseen *= e;
    if node.total > 0 {
        let inv = (1.0 - e) / node.total as f32;
        for &(tok, c) in node.counts.iter() {
            if let Some(&i) = pos.get(&tok) {
                out[i] += inv * c as f32;
            }
        }
    }
}

pub fn spread(
    tree: &Tree,
    codes: &[PathCode],
    payloads: &[Vec<f32>],
    weights: &[f32],
    use_readout: bool,
) -> Spread {
    let root = &tree.arena[0];
    let n = root.counts.len();
    let mut pos = std::collections::HashMap::with_capacity(n);
    for (i, &(tok, _)) in root.counts.iter().enumerate() {
        pos.insert(tok, i);
    }
    let mut acc = vec![0.0f32; n];
    let mut acc_unseen = 0.0f32;
    let mut buf = vec![0.0f32; n];
    let mut buf_unseen;

    for i in 0..codes.len() {
        let w = weights[i];
        if w <= 0.0 {
            continue;
        }
        let code = &codes[i];
        let mut prefix = 1.0f32;
        for j in 0..code.taken.len() {
            for &(v, qv) in code.others[j].iter() {
                buf_unseen = 0.0;
                prior_spread(tree, v, &pos, &mut buf, &mut buf_unseen);
                let s = w * prefix * qv;
                for k in 0..n {
                    acc[k] += s * buf[k];
                }
                acc_unseen += s * buf_unseen;
            }
            prefix *= code.taken[j];
        }
        // Tail.
        let u = code.leaf();
        buf_unseen = 0.0;
        let node = &tree.arena[u];
        if use_readout && !node.rows.is_empty() {
            let e = node.escape();
            match node.parent {
                None => {
                    let p = 1.0 / tree.vocab as f32;
                    for v in buf.iter_mut() {
                        *v = p * e;
                    }
                    buf_unseen = p * e;
                }
                Some(par) => {
                    prior_spread(tree, par, &pos, &mut buf, &mut buf_unseen);
                    for v in buf.iter_mut() {
                        *v *= e;
                    }
                    buf_unseen *= e;
                }
            }
            for (t, q) in tree.readout_dist(u, &payloads[i]) {
                if let Some(&k) = pos.get(&t) {
                    buf[k] += (1.0 - e) * q;
                }
            }
        } else {
            prior_spread(tree, u, &pos, &mut buf, &mut buf_unseen);
        }
        let s = w * prefix;
        for k in 0..n {
            acc[k] += s * buf[k];
        }
        acc_unseen += s * buf_unseen;
    }

    Spread { support: acc, unseen_each: acc_unseen, unseen_count: tree.vocab.saturating_sub(n) }
}
