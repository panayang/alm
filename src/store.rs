//! What is stored: per-node counts, and one shared readout table.
//!
//! Flat, and fixed. There is one store node per memory-graph node and nothing
//! grows: no prototypes, no hierarchy, no allocation criterion, no split
//! threshold. All of that was measured and all of it cost more than it bought --
//! depth was monotonically worse on every axis including the retention it was
//! supposed to provide.
//!
//! The reason is worth keeping written down, because it is not "addressing is
//! expensive". A placed prototype had to win an argmax against hundreds of
//! competitors under a drifted context, and the loss from getting that wrong is
//! asymmetric: a sharp prior on the wrong candidate set removes the right answer
//! from consideration entirely, which costs logarithmically, while being right
//! only pays linearly. Depth raised both the sharpness and the chance of being
//! wrong. Retention fell hardest because an old regime is exactly where the read
//! is least able to reproduce the routing its write took.
//!
//! So the address is no longer a competition. A response walks the memory graph
//! and the prior is what the walk has touched; this file holds what each node
//! has seen.
//!
//! Two things live here and they differ in kind:
//!
//! * **Counts, per node.** Which tokens reach this node and how often. Genuinely
//!   node-specific, exactly known, never trained, and the only thing that tells
//!   one part of memory from another.
//! * **Rows, one per token, shared.** The map from features to a token is the
//!   same function everywhere -- "if the bound trace is E_a (*) E_b the answer is
//!   T[a][b]" does not depend on where the walk happens to be. Per-node rows made
//!   the system relearn one function independently in every cell, from that
//!   cell's fraction of the evidence.

use std::collections::HashMap;

use crate::config::Config;
use crate::num::Running;

pub struct Node {
    /// Occupancy: how many times each token was written here. Integers, not
    /// parameters. Kept approximately frequency-ordered by one swap per write,
    /// which costs O(1) and is all the candidate enumeration needs.
    pub counts: Vec<(u32, u64)>,
    index: HashMap<u32, usize>,
    pub total: u64,
    /// Per-write surprise in bits. Diagnostic: nothing splits any more.
    pub surprise: Running,
}

impl Node {
    fn new() -> Self {
        Node { counts: Vec::new(), index: HashMap::new(), total: 0, surprise: Running::default() }
    }

    #[inline]
    pub fn count_of(&self, tok: u32) -> u64 {
        match self.index.get(&tok) {
            None => 0,
            Some(&i) => self.counts[i].1,
        }
    }

    fn bump(&mut self, tok: u32) {
        match self.index.get(&tok) {
            Some(&i) => {
                self.counts[i].1 += 1;
                if i > 0 && self.counts[i].1 > self.counts[i - 1].1 {
                    self.counts.swap(i - 1, i);
                    let a = self.counts[i - 1].0;
                    let b = self.counts[i].0;
                    self.index.insert(a, i - 1);
                    self.index.insert(b, i);
                }
            }
            None => {
                self.index.insert(tok, self.counts.len());
                self.counts.push((tok, 1));
            }
        }
        self.total += 1;
    }

    #[inline]
    pub fn distinct(&self) -> usize {
        self.counts.len()
    }

    /// PPM-C escape mass: distinct / (distinct + total). Derived from the counts
    /// themselves, so the backoff carries no tuned parameter and the codelength
    /// is finite for every token, including ones this node has never seen.
    #[inline]
    pub fn escape(&self) -> f32 {
        let d = self.distinct() as f32;
        if self.total == 0 {
            1.0
        } else {
            d / (d + self.total as f32)
        }
    }
}

/// Per-bin reliability counters: (observations, correct).
#[derive(Clone)]
pub struct Calibration {
    pub bins: Vec<(u64, u64)>,
}

impl Calibration {
    fn new(nbins: usize) -> Self {
        Calibration { bins: vec![(0, 0); nbins] }
    }
    #[inline]
    fn bin_of(&self, conf: f32) -> usize {
        let n = self.bins.len();
        ((conf * n as f32) as usize).min(n - 1)
    }
    pub fn push(&mut self, conf: f32, correct: bool) {
        let b = self.bin_of(conf);
        self.bins[b].0 += 1;
        if correct {
            self.bins[b].1 += 1;
        }
    }
    pub fn accuracy(&self, b: usize) -> f32 {
        let (n, c) = self.bins[b];
        if n == 0 {
            0.0
        } else {
            c as f32 / n as f32
        }
    }
    pub fn observations(&self) -> u64 {
        self.bins.iter().map(|x| x.0).sum()
    }
    /// Expected calibration error. Reported, and gating nothing: both thresholds
    /// that once read out of these counters deadlocked, and both are now "the
    /// evidence has stopped moving".
    pub fn ece(&self) -> f32 {
        let total = self.observations();
        if total == 0 {
            return 0.0;
        }
        let n = self.bins.len();
        let mut e = 0.0f32;
        for b in 0..n {
            let (cnt, _) = self.bins[b];
            if cnt == 0 {
                continue;
            }
            let centre = (b as f32 + 0.5) / n as f32;
            e += (cnt as f32 / total as f32) * (self.accuracy(b) - centre).abs();
        }
        e
    }
}

pub struct Store {
    pub d: usize,
    /// Width of a readout row: the payload block plus one per bound trace.
    pub fw: usize,
    pub vocab: usize,
    /// One node per memory-graph node.
    pub nodes: Vec<Node>,
    /// Every write is also counted here, so this is the backoff every node
    /// escapes through.
    pub global: Node,
    /// One row per token, shared by every node.
    pub rows: Vec<(u32, Vec<f32>)>,
    row_index: HashMap<u32, usize>,
    pub answer_calib: Calibration,
    pub write_surprise: Running,
}

impl Store {
    pub fn new(cfg: &Config, graph_nodes: usize) -> Self {
        Store {
            d: cfg.d,
            fw: cfg.feature_blocks() * cfg.d,
            vocab: cfg.vocab,
            nodes: (0..graph_nodes).map(|_| Node::new()).collect(),
            global: Node::new(),
            rows: Vec::new(),
            row_index: HashMap::new(),
            answer_calib: Calibration::new(cfg.calib_bins),
            write_surprise: Running::default(),
        }
    }

    pub fn occupied_rows(&self) -> usize {
        self.rows.len()
    }

    /// Nodes that have taken at least one write: how much of the memory the
    /// stream has actually used.
    pub fn live_nodes(&self) -> usize {
        self.nodes.iter().filter(|n| n.total > 0).count()
    }

    #[inline]
    pub fn row_of(&self, tok: u32) -> Option<&[f32]> {
        self.row_index.get(&tok).map(|&i| self.rows[i].1.as_slice())
    }

    fn row_mut_or_insert(&mut self, tok: u32) -> &mut Vec<f32> {
        if let Some(&i) = self.row_index.get(&tok) {
            return &mut self.rows[i].1;
        }
        let fw = self.fw;
        self.row_index.insert(tok, self.rows.len());
        self.rows.push((tok, vec![0.0; fw]));
        let i = self.rows.len() - 1;
        &mut self.rows[i].1
    }

    pub fn record(&mut self, node: usize, tok: u32) {
        self.nodes[node].bump(tok);
        self.global.bump(tok);
    }

    /// prior(o | node), escaping through the global counts to a uniform.
    pub fn prior_of(&self, node: usize, tok: u32) -> f32 {
        let n = &self.nodes[node];
        let e = n.escape();
        let own = if n.total == 0 { 0.0 } else { n.count_of(tok) as f32 / n.total as f32 };
        e * self.prior_global(tok) + (1.0 - e) * own
    }

    pub fn prior_global(&self, tok: u32) -> f32 {
        let g = &self.global;
        let e = g.escape();
        let own = if g.total == 0 { 0.0 } else { g.count_of(tok) as f32 / g.total as f32 };
        e * (1.0 / self.vocab as f32) + (1.0 - e) * own
    }

    /// The probability a token nothing has ever seen receives at a node.
    /// Identical for all of them, which is what makes the full spread computable
    /// in O(support) rather than O(V).
    pub fn prior_unseen(&self, node: usize) -> f32 {
        self.nodes[node].escape() * self.global.escape() / self.vocab as f32
    }

    /// The delta-rule step, against the *same* distribution the ledger charges.
    ///
    /// The count prior enters as a fixed offset inside the score, so the rows
    /// learn the residual on top of it. That is the offset trick from generalised
    /// linear models and it is why a counted prior and a learned likelihood can
    /// multiply without using the evidence twice -- but only if the offset is
    /// present *during training*, which is why this calls the same `score` the
    /// charge does instead of computing its own softmax.
    pub fn readout_update(
        &mut self,
        visit: &crate::code::Visit,
        phi: &[f32],
        target: u32,
        negatives: &[u32],
        eta: f32,
    ) -> Vec<f32> {
        let d = self.fw.min(phi.len());
        let sc = crate::code::score(self, visit, phi, true);

        let mut touched: Vec<u32> = Vec::with_capacity(negatives.len() + 1);
        touched.push(target);
        for &n in negatives {
            if n != target && !touched.contains(&n) {
                touched.push(n);
            }
        }

        let mut grad = vec![0.0f32; d];
        for &t in touched.iter() {
            let q = sc.prob_of(self, visit, t);
            let err = if t == target { 1.0 - q } else { -q };
            let row = self.row_mut_or_insert(t);
            for i in 0..d {
                // Read before updating: taking the row back afterwards adds
                // -eta * phi * sum(err^2), the same order as the gradient.
                grad[i] -= err * row[i];
                row[i] += eta * err * phi[i];
            }
        }
        grad
    }

    /// Tokens a node has seen, which is where a write samples its negatives.
    pub fn emitted(&self, node: usize) -> Vec<u32> {
        self.nodes[node].counts.iter().map(|(t, _)| *t).collect()
    }
}
