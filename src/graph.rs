//! Content-addressed edge memory: a small-world ring whose edges carry a fixed
//! key and a learned transform.
//!
//! The addressing half is taken unchanged from the reference mechanism. Keys are
//! drawn once and never updated, so at a fixed payload the walked path is a
//! deterministic function of content and no amount of later learning can move
//! where a fact lives. `tests/structural.rs` asserts that as an equality, not a
//! statistic.
//!
//! What is new here is that reads and writes no longer share a path. The write
//! walk is the deterministic one and keeps the consistency guarantee. Read
//! particles walk wherever the evidence takes them; consistency is simply not a
//! requirement on them, which is what lets composition back into the design.

use crate::config::Config;
use crate::num::{argmax, axpy, dot, normalize, unit_vector, Mat};

#[derive(Clone)]
pub struct WalkStep {
    pub edge: usize,
    pub p_in: Vec<f32>,
    /// tanh(W p_in), cached so the backward pass needs no recomputation.
    pub th: Vec<f32>,
    pub lambda: f32,
    pub p_out: Vec<f32>,
}

/// What an edge remembers about the last read that crossed it, so that a
/// settlement arriving later can land credit on it without unrolling anything.
#[derive(Clone)]
struct ReadCache {
    p_in: Vec<f32>,
    th: Vec<f32>,
    lambda: f32,
    p_out: Vec<f32>,
}

pub struct Graph {
    pub d: usize,
    pub nodes: usize,
    /// Edge ids leaving each node.
    out: Vec<Vec<usize>>,
    /// Head node of each edge.
    head: Vec<usize>,
    /// Fixed addressing keys, never updated.
    keys: Vec<Vec<f32>>,
    /// Learned transforms, shared across regimes.
    pub w: Vec<Mat>,
    /// Eligibility trace per edge.
    pub trace: Vec<f32>,
    read_cache: Vec<Option<ReadCache>>,
    /// While frozen the graph records nothing. A probe still walks the memory
    /// to be scored, but the trace it would leave must not survive it: the next
    /// real settlement credits every traced edge, so a probe's footprints would
    /// end up steering real weight updates.
    pub frozen: bool,
    rank_clamped: std::cell::Cell<u64>,
    rank_selected: std::cell::Cell<u64>,
}

const KEY_SHORTCUT: u64 = 0x0000_0000_0000_0021;
const KEY_EDGE: u64 = 0x0000_0000_0000_0022;
const KEY_WMAT: u64 = 0x0000_0000_0000_0023;

impl Graph {
    pub fn new(cfg: &Config) -> Self {
        let d = cfg.d;
        let n = cfg.nodes;
        let key = cfg.seed ^ 0x6_9A97;
        let mut out = vec![Vec::new(); n];
        let mut head = Vec::new();
        for u in 0..n {
            // Two ring neighbours plus fixed random shortcuts: the small-world
            // structure of the reference mechanism.
            let mut targets = vec![(u + 1) % n, (u + n - 1) % n];
            for s in 0..cfg.shortcuts {
                let t = crate::num::uniform_below(key ^ KEY_SHORTCUT, (u * 16 + s) as u64, n as u64)
                    as usize;
                if t != u && !targets.contains(&t) {
                    targets.push(t);
                }
            }
            for t in targets {
                let id = head.len();
                head.push(t);
                out[u].push(id);
            }
        }
        let m = head.len();
        let keys = (0..m).map(|a| unit_vector(key ^ KEY_EDGE, a as u64, d)).collect();
        // The initialisation scale decides whether the payload chain does any
        // work at all. At 0.05 the argument of the tanh is small, the tanh is in
        // its linear regime, and the residual keeps the input dominant, so the
        // "learned continuous transform" -- one of the two things this design
        // claims over a suffix model -- arrives at the readout as very nearly
        // the identity. It is a config field so that it is swept rather than
        // silently chosen.
        let w = (0..m).map(|a| Mat::random(key ^ KEY_WMAT, a as u64, d, d, cfg.w_init)).collect();
        Graph {
            d,
            nodes: n,
            out,
            head,
            keys,
            w,
            trace: vec![0.0; m],
            read_cache: vec![None; m],
            frozen: false,
            rank_clamped: std::cell::Cell::new(0),
            rank_selected: std::cell::Cell::new(0),
        }
    }

    pub fn edges(&self) -> usize {
        self.head.len()
    }

    /// Entry node: argmax over the first out-edge key of each node.
    pub fn entry(&self, q: &[f32]) -> usize {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for u in 0..self.nodes {
            let a = self.out[u][0];
            let s = dot(&self.keys[a], q);
            if s > bv {
                bv = s;
                best = u;
            }
        }
        best
    }

    /// Argmax over the out-edges of `u`. Ties break to the lowest edge id.
    pub fn select(&self, u: usize, q: &[f32]) -> usize {
        let outs = &self.out[u];
        let mut scores = Vec::with_capacity(outs.len());
        for &a in outs {
            scores.push(dot(&self.keys[a], q));
        }
        outs[argmax(&scores)]
    }

    /// The `rank`-th best out-edge, zero being the argmax.
    ///
    /// The near-miss instrument. Taking the runner-up on every hop is what a
    /// crowded address space does to a competition, and the design's central
    /// claim is that in an operator set this costs a small perturbation rather
    /// than a cliff.
    pub fn select_rank(&self, u: usize, q: &[f32], rank: usize) -> usize {
        let outs = &self.out[u];
        if rank == 0 || outs.len() == 1 {
            return self.select(u, q);
        }
        let mut scored: Vec<(usize, f32)> =
            outs.iter().map(|&a| (a, dot(&self.keys[a], q))).collect();
        scored.sort_by(|x, y| {
            y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal).then(x.0.cmp(&y.0))
        });
        if rank >= scored.len() {
            self.rank_clamped.set(self.rank_clamped.get() + 1);
        }
        self.rank_selected.set(self.rank_selected.get() + 1);
        scored[rank.min(scored.len() - 1)].0
    }

    /// Fraction of perturbed selections that fell back to a lower rank because
    /// the node did not have that many out-edges. A near-miss arm with a high
    /// value here is partly the arm below it, and the two are not independent.
    pub fn clamp_rate(&self) -> f64 {
        let n = self.rank_selected.get();
        if n == 0 { 0.0 } else { self.rank_clamped.get() as f64 / n as f64 }
    }

    #[inline]
    pub fn head_of(&self, a: usize) -> usize {
        self.head[a]
    }

    /// A node's identity as a vector: the fixed key of its first out-edge.
    ///
    /// This is what the write channel reports -- "the memory I am touching" --
    /// and feeding it back into the fast rung is what keeps a response moving
    /// through the gap. Without it the walk is autonomous on a decaying
    /// background and settles into a fixed point or a short cycle; with it,
    /// visiting a node pushes the next query away from that node, so
    /// inhibition of return is a consequence rather than an added mechanism.
    #[inline]
    pub fn node_key(&self, u: usize) -> &[f32] {
        &self.keys[self.out[u][0]]
    }

    /// Out-degree, reported so the branching a walk actually has is not assumed.
    pub fn out_degree(&self, u: usize) -> usize {
        self.out[u].len()
    }

    /// One residual hop: p_out = nu( p_in + tanh(W_a p_in) ).
    pub fn hop(&mut self, a: usize, p_in: &[f32]) -> WalkStep {
        let mut th = vec![0.0f32; self.d];
        self.w[a].matvec(p_in, &mut th);
        for v in th.iter_mut() {
            *v = v.tanh();
        }
        let mut p_out = vec![0.0f32; self.d];
        for i in 0..self.d {
            p_out[i] = p_in[i] + th[i];
        }
        let lambda = normalize(&mut p_out);
        WalkStep { edge: a, p_in: p_in.to_vec(), th, lambda, p_out }
    }

    /// The deterministic write walk: `hops` steps from the entry node, routed by
    /// the fixed keys on a payload that does not change between calls.
    pub fn write_walk(&mut self, q: &[f32], p0: &[f32], hops: usize) -> Vec<WalkStep> {
        let mut u = self.entry(q);
        let mut p = p0.to_vec();
        let mut steps = Vec::with_capacity(hops);
        for _ in 0..hops {
            let a = self.select(u, q);
            let st = self.hop(a, &p);
            p = st.p_out.clone();
            u = self.head[a];
            steps.push(st);
        }
        steps
    }

    /// Just the edge sequence, for the routing-consistency assertion.
    pub fn write_path(&self, q: &[f32], hops: usize) -> Vec<usize> {
        let mut u = self.entry(q);
        let mut path = Vec::with_capacity(hops);
        for _ in 0..hops {
            let a = self.select(u, q);
            path.push(a);
            u = self.head[a];
        }
        path
    }

    /// One backward step through the normalisation and the residual tanh block,
    /// applying the weight update in place before forming the incoming gradient
    /// -- the same order as the reference mechanism, so the backward signal
    /// passes through the post-update matrix.
    pub fn backward_step(&mut self, st: &WalkStep, grad_out: &[f32], eta: f32) -> Vec<f32> {
        let d = self.d;
        // grad_u = (1/lambda) (I - p_out p_out^T) grad_out
        let proj = dot(&st.p_out, grad_out);
        let inv = if st.lambda > 1e-9 { 1.0 / st.lambda } else { 0.0 };
        let mut grad_u = vec![0.0f32; d];
        for i in 0..d {
            grad_u[i] = inv * (grad_out[i] - st.p_out[i] * proj);
        }
        // grad_b = (1 - tanh^2) . grad_u
        let mut grad_b = vec![0.0f32; d];
        for i in 0..d {
            grad_b[i] = (1.0 - st.th[i] * st.th[i]) * grad_u[i];
        }
        // W -= eta grad_b p_in^T
        self.w[st.edge].sub_outer(eta, &grad_b, &st.p_in);
        // grad_in = grad_u + W^T grad_b, with the post-update W.
        let mut grad_in = vec![0.0f32; d];
        self.w[st.edge].matvec_t(&grad_b, &mut grad_in);
        axpy(1.0, &grad_u, &mut grad_in);
        grad_in
    }

    /// Backpropagate along the walked path and along no other.
    pub fn backprop(&mut self, steps: &[WalkStep], grad_last: &[f32], eta: f32) -> Vec<f32> {
        let mut g = grad_last.to_vec();
        for st in steps.iter().rev() {
            g = self.backward_step(st, &g, eta);
        }
        g
    }

    // ---- eligibility ---------------------------------------------------

    pub fn decay_traces(&mut self, lambda: f32) {
        if self.frozen {
            return;
        }
        for t in self.trace.iter_mut() {
            *t *= lambda;
        }
    }

    /// A read particle crossed this edge: leave a trace and remember the state,
    /// so a settlement arriving later can credit it in O(1).
    pub fn touch_read(&mut self, st: &WalkStep) {
        if self.frozen {
            return;
        }
        self.trace[st.edge] = (self.trace[st.edge] + 1.0).min(4.0);
        self.read_cache[st.edge] = Some(ReadCache {
            p_in: st.p_in.clone(),
            th: st.th.clone(),
            lambda: st.lambda,
            p_out: st.p_out.clone(),
        });
    }

    /// Land settlement credit on every traced edge, using the state each edge
    /// remembers. This is what lets gap-time computation be learned at all,
    /// without unrolling the gap.
    pub fn credit_traces(&mut self, grad_seed: &[f32], eta: f32, floor: f32) -> usize {
        let mut touched = 0usize;
        for a in 0..self.edges() {
            let tr = self.trace[a];
            if tr <= floor {
                continue;
            }
            let cache = match &self.read_cache[a] {
                None => continue,
                Some(c) => c.clone(),
            };
            let st = WalkStep {
                edge: a,
                p_in: cache.p_in,
                th: cache.th,
                lambda: cache.lambda,
                p_out: cache.p_out,
            };
            let _ = self.backward_step(&st, grad_seed, eta * tr);
            touched += 1;
        }
        touched
    }

}
