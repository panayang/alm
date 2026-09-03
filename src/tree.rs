//! The allocation tree: the nested address, its occupancy counts, and its
//! per-level calibration counters.
//!
//! Rung k of the ladder feeds level k of this tree, so the ladder is not a
//! separate structure -- it is this tree's time axis. Depth and branching are
//! emergent: both grow by *appending*, which leaves every existing node at its
//! offset with its values untouched, so adding capacity cannot disturb what is
//! already stored.
//!
//! Every learned quantity in the system is either an edge transform (in
//! `graph.rs`) or one of the sparse readout rows here. Prototypes are placed and
//! never receive a gradient; priors are counted, not fitted.

use std::collections::HashMap;

use crate::config::{Config, SplitRule};
use crate::num::{dot, normalize, softmax, Running};

pub struct Node {
    pub level: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Placed at the query that was live when this node was appended. Never
    /// trained.
    pub proto: Vec<f32>,

    /// Occupancy: how many times each token was emitted anywhere under this
    /// node. Integers, not parameters.
    pub counts: Vec<(u32, u64)>,
    index: HashMap<u32, usize>,
    pub total: u64,


    /// Dispersion statistics for the widen criterion.
    pub sim: Running,
    /// Per-write surprise in bits, for the predictive split criterion.
    pub surprise: Running,
    pub hold: u32,
    /// The very first child of the tree has nowhere meaningful to be placed at
    /// construction, so it is placed at the first query that reaches it. Every
    /// later node is placed at the query that caused it to be appended.
    pub placed: bool,
}

impl Node {
    fn new(level: usize, parent: Option<usize>, proto: Vec<f32>) -> Self {
        Node {
            level,
            parent,
            children: Vec::new(),
            proto,
            counts: Vec::new(),
            index: HashMap::new(),
            total: 0,
            sim: Running::default(),
            surprise: Running::default(),
            hold: 0,
            placed: true,
        }
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
                // One swap forward if this token has overtaken its neighbour.
                // Over many writes the vector becomes approximately ordered by
                // frequency, which is all the scored-set cap needs, and it costs
                // O(1) rather than a sort.
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
    /// is finite for every token, including ones this node has never emitted.
    #[inline]
    pub fn escape(&self) -> f32 {
        let dcount = self.distinct() as f32;
        if self.total == 0 {
            1.0
        } else {
            dcount / (dcount + self.total as f32)
        }
    }

}

/// Per-level reliability counters: (observations, correct) per confidence bin.
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
        let i = (conf * n as f32) as usize;
        i.min(n - 1)
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
    /// Expected calibration error, the summary number for the reliability plot.
    pub fn ece(&self) -> f32 {
        let total: u64 = self.observations();
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
    // Note on what is *not* here. Both the branch commit threshold and the
    // speak threshold were once read out of these counters, under a rule that
    // could only raise the bar. Both deadlocked: a model over-confident in any
    // bin had its bar pinned at the ceiling, so a level never matured on
    // confidence and the channel never spoke, and neither then generated the
    // data that would have calibrated it. Both are now "the evidence has
    // stopped moving", in `descent.rs::step` and `model.rs::emit`. These
    // counters remain because the reliability curve is one of the reported
    // measurements -- they no longer gate anything, and the module docs should
    // not claim they do.
}

pub struct Tree {
    /// One readout row per token, shared by every node.
    ///
    /// The map from a feature vector to a token is the same function in every
    /// regime -- "if the bound trace is E_a (*) E_b the answer is T[a][b]" does
    /// not depend on which regime is live. Giving each node its own rows made
    /// the system relearn that one function independently in every cell, from
    /// that cell's fraction of the evidence. What genuinely differs between
    /// regimes is *which* tokens they emit and how often, and that is the
    /// counts, which stay private and exact.
    pub rows: Vec<(u32, Vec<f32>)>,
    row_index: HashMap<u32, usize>,
    /// Reliability of the *answer*, as distinct from the reliability of a
    /// branch. Speaking is gated by this one.
    pub answer_calib: Calibration,
    pub d: usize,
    /// Width of a readout row. Equal to `d` when the readout sees only the
    /// payload, and `2 * d` when the bound trace is concatenated onto it.
    pub fw: usize,
    pub vocab: usize,
    pub arena: Vec<Node>,
    pub calib: Vec<Calibration>,
    pub depth_cap: usize,
    max_children: usize,
    max_nodes: usize,
    grow_theta: f64,
    split_rule: SplitRule,
    split_bits: f64,
    hybrid_coarse_levels: usize,
    grow_hold: u32,
    grow_min_obs: u64,
    deepen_min_obs: u64,
    pub widen_events: u64,
    pub deepen_events: u64,
    /// How many write descents consulted each ladder rung. A rung with zero
    /// visits is a band the model does not have, whatever the configuration
    /// says, and every measurement that varies the rung count depends on this
    /// being non-degenerate.
    pub rung_visits: Vec<u64>,
    /// Mean per-write surprise at the destination, in bits. The split threshold
    /// has to be chosen against this and not guessed: a threshold below the
    /// observed range makes every node split always, and a sweep over such
    /// thresholds returns identical arms.
    pub leaf_surprise: Running,
}

impl Tree {
    pub fn new(cfg: &Config) -> Self {
        let d = cfg.d;
        let cap = cfg.effective_depth_cap();
        // The root is level 0 and holds no prototype of its own. It starts with
        // exactly one child, so the mechanism is not told how many regimes
        // exist.
        let mut arena = vec![Node::new(0, None, vec![0.0; d])];
        let mut first = Node::new(1, Some(0), vec![0.0; d]);
        first.proto[0] = 1.0;
        first.placed = false;
        arena.push(first);
        arena[0].children.push(1);
        Tree {
            rows: Vec::new(),
            row_index: HashMap::new(),
            answer_calib: Calibration::new(cfg.calib_bins),
            d,
            fw: cfg.feature_blocks() * d,
            vocab: cfg.vocab,
            arena,
            calib: (0..cap + 1).map(|_| Calibration::new(cfg.calib_bins)).collect(),
            depth_cap: cap,
            max_children: cfg.max_children,
            max_nodes: cfg.max_nodes,
            grow_theta: cfg.grow_theta,
            split_rule: cfg.split_rule,
            split_bits: cfg.split_bits,
            hybrid_coarse_levels: cfg.hybrid_coarse_levels,
            grow_hold: cfg.grow_hold,
            grow_min_obs: cfg.grow_min_obs,
            deepen_min_obs: cfg.deepen_min_obs,
            widen_events: 0,
            deepen_events: 0,
            rung_visits: vec![0; cfg.rungs],
            leaf_surprise: Running::default(),
        }
    }

    /// Fraction of rungs the address genuinely uses.
    ///
    /// Counting any rung with a single visit as "used" is too generous: a band
    /// read twice in fourteen thousand descents is not a band the model has.
    /// The threshold is one per cent of the busiest rung.
    pub fn rung_coverage(&self) -> f64 {
        let max = *self.rung_visits.iter().max().unwrap_or(&0);
        if max == 0 {
            return 0.0;
        }
        let floor = (max as f64 * 0.01).ceil() as u64;
        let used = self.rung_visits.iter().filter(|&&v| v >= floor).count();
        used as f64 / self.rung_visits.len().max(1) as f64
    }

    pub fn nodes(&self) -> usize {
        self.arena.len()
    }

    pub fn realised_depth(&self) -> usize {
        self.arena.iter().map(|n| n.level).max().unwrap_or(0)
    }

    pub fn leaves(&self) -> usize {
        self.arena.iter().filter(|n| n.children.is_empty()).count()
    }

    pub fn occupied_rows(&self) -> usize {
        self.rows.len()
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

    /// Probability the escape chain gives a token no node has ever emitted.
    /// Identical for all of them, which is what makes the spread computable in
    /// O(support) rather than O(V).
    pub fn prior_unseen(&self, u: usize) -> f32 {
        let node = &self.arena[u];
        let base = match node.parent {
            None => 1.0 / self.vocab as f32,
            Some(p) => self.prior_unseen(p),
        };
        node.escape() * base
    }

    /// Place any unplaced child of `u` at the current query, so the first class
    /// starts where the stream actually is rather than on an arbitrary axis.
    pub fn place_pending(&mut self, u: usize, query: &[f32]) {
        let kids = self.arena[u].children.clone();
        for c in kids {
            if !self.arena[c].placed {
                let mut proto = query.to_vec();
                normalize(&mut proto);
                self.arena[c].proto = proto;
                self.arena[c].placed = true;
            }
        }
    }

    /// Scores of `u`'s children against a query. Unnormalised inner products
    /// against placed prototypes; the caller turns them into a distribution.
    pub fn branch_scores(&self, u: usize, query: &[f32], temp: f32, out: &mut Vec<f32>) {
        out.clear();
        for &c in self.arena[u].children.iter() {
            out.push(temp * dot(&self.arena[c].proto, query));
        }
    }

    /// Branch distribution over `u`'s children.
    pub fn branch_dist(&self, u: usize, query: &[f32], temp: f32) -> Vec<f32> {
        let mut s = Vec::new();
        self.branch_scores(u, query, temp, &mut s);
        if s.is_empty() {
            return s;
        }
        softmax(&mut s);
        s
    }

    // ---- growth: both operations append ---------------------------------

    /// Append a sibling under `u`, placed at the current query. Existing
    /// children keep their ids, their prototypes, and their rows.
    pub fn widen(&mut self, u: usize, query: &[f32]) -> Option<usize> {
        if self.arena.len() >= self.max_nodes || self.arena[u].children.len() >= self.max_children {
            return None;
        }
        let level = self.arena[u].level + 1;
        if level > self.depth_cap {
            return None;
        }
        let mut proto = query.to_vec();
        normalize(&mut proto);
        let id = self.arena.len();
        self.arena.push(Node::new(level, Some(u), proto));
        self.arena[u].children.push(id);
        Some(id)
    }

    /// Give `u` its first child, deepening the address by one level.
    pub fn deepen(&mut self, u: usize, query: &[f32]) -> Option<usize> {
        if !self.arena[u].children.is_empty() {
            return None;
        }
        if self.arena[u].total < self.deepen_min_obs {
            return None;
        }
        let r = self.widen(u, query);
        if r.is_some() {
            self.deepen_events += 1;
        }
        r
    }

    /// The reference mechanism's criterion, run per parent: a child is split
    /// when its own similarity distribution is too dispersed for `hold`
    /// consecutive steps. Dispersion rather than an outlier test, because an
    /// outlier test against a class's own distribution fails exactly when
    /// splitting becomes necessary.
    pub fn observe_and_maybe_grow(
        &mut self,
        parent: usize,
        winner: usize,
        sim: f32,
        query: &[f32],
    ) -> bool {
        let dispersion_here = match self.split_rule {
            SplitRule::Dispersion => true,
            SplitRule::Surprise => false,
            SplitRule::Hybrid => self.arena[winner].level <= self.hybrid_coarse_levels,
        };
        if !dispersion_here {
            // Under a surprise-governed level the statistic arrives after the
            // write, so the similarity is still recorded but decides nothing.
            self.arena[winner].sim.push(sim as f64);
            return false;
        }
        {
            let n = &mut self.arena[winner];
            n.sim.push(sim as f64);
            if n.sim.n < self.grow_min_obs {
                n.hold = 0;
                return false;
            }
            let disp = n.sim.std() / n.sim.mean.abs().max(1e-6);
            if disp > self.grow_theta {
                n.hold += 1;
            } else {
                n.hold = 0;
            }
            if n.hold < self.grow_hold {
                return false;
            }
            n.hold = 0;
        }
        let grew = self.widen(parent, query).is_some();
        if grew {
            self.widen_events += 1;
        }
        grew
    }

    /// Record how surprised a node was by what it just had to store, and split
    /// it if it is still not predicting its own content.
    ///
    /// Called after the write, with the surprise measured *before* the token was
    /// counted, so a node is never rewarded for the observation that is about to
    /// be added to it.
    pub fn observe_surprise(
        &mut self,
        parent: usize,
        node: usize,
        bits: f64,
        query: &[f32],
    ) -> bool {
        let surprise_here = match self.split_rule {
            SplitRule::Surprise => true,
            SplitRule::Dispersion => false,
            SplitRule::Hybrid => self.arena[node].level > self.hybrid_coarse_levels,
        };
        if !surprise_here {
            // Still record it: the threshold for a later sweep has to be read
            // off the observed range, and a run that does not use the rule is
            // exactly where an unbiased range comes from.
            self.leaf_surprise.push(bits);
            return false;
        }
        self.leaf_surprise.push(bits);
        {
            let n = &mut self.arena[node];
            n.surprise.push(bits);
            if n.surprise.n < self.grow_min_obs {
                n.hold = 0;
                return false;
            }
            if n.surprise.mean > self.split_bits {
                n.hold += 1;
            } else {
                n.hold = 0;
            }
            if n.hold < self.grow_hold {
                return false;
            }
            n.hold = 0;
            // Start the estimate again, so a node that has just been given a
            // sibling is judged on what it holds afterwards rather than on the
            // backlog that caused the split.
            n.surprise = Running::default();
        }
        let grew = self.widen(parent, query).is_some();
        if grew {
            self.widen_events += 1;
        }
        grew
    }

    // ---- counts and priors ------------------------------------------------

    /// Record an emission at `leaf`, incrementing every ancestor. This is what
    /// makes `prior` well defined at every node for every token, including
    /// tokens that several leaves have emitted.
    pub fn record(&mut self, leaf: usize, tok: u32) {
        let mut u = Some(leaf);
        while let Some(id) = u {
            self.arena[id].bump(tok);
            u = self.arena[id].parent;
        }
    }

    /// prior(o | u), with PPM-C escape up the ancestor chain to a uniform.
    pub fn prior_of(&self, u: usize, tok: u32) -> f32 {
        let node = &self.arena[u];
        let base = match node.parent {
            None => 1.0 / self.vocab as f32,
            Some(p) => self.prior_of(p, tok),
        };
        let e = node.escape();
        let own = if node.total == 0 {
            0.0
        } else {
            node.count_of(tok) as f32 / node.total as f32
        };
        e * base + (1.0 - e) * own
    }

    /// The whole prior over the vocabulary at `u`. Only used on the ticks where
    /// the full emitted distribution is materialised for its entropy.
    pub fn prior_into(&self, u: usize, out: &mut [f32]) {
        let node = &self.arena[u];
        match node.parent {
            None => {
                let p = 1.0 / self.vocab as f32;
                for v in out.iter_mut() {
                    *v = p;
                }
            }
            Some(p) => self.prior_into(p, out),
        }
        let e = node.escape();
        for v in out.iter_mut() {
            *v *= e;
        }
        if node.total > 0 {
            let inv = (1.0 - e) / node.total as f32;
            for &(tok, c) in node.counts.iter() {
                out[tok as usize] += inv * c as f32;
            }
        }
    }

    // ---- readout ------------------------------------------------------------

    /// Softmax over the node's own emitted rows. Returns the (token, prob)
    /// pairs; tokens outside the row set are covered by the escape mass.
    /// The delta-rule step, against the *same* distribution the ledger charges.
    ///
    /// The count prior enters as a fixed offset inside the softmax, so what the
    /// rows learn is the residual on top of it. That is the offset trick from
    /// generalised linear models and it is the whole reason a count prior and a
    /// learned likelihood can be multiplied without using the evidence twice --
    /// but only if the offset is present *during training*. Scoring with the
    /// prior after fitting without it would double-count.
pub fn readout_update(
        &mut self,
        u: usize,
        phi: &[f32],
        target: u32,
        negatives: &[u32],
        eta: f32,
    ) -> Vec<f32> {
        let d = self.fw.min(phi.len());
        let sc = crate::code::score(self, u, phi, true);

        let mut touched: Vec<u32> = Vec::with_capacity(negatives.len() + 1);
        touched.push(target);
        for &n in negatives {
            if n != target && !touched.contains(&n) {
                touched.push(n);
            }
        }

        let mut grad = vec![0.0f32; d];
        for &t in touched.iter() {
            let q = sc.prob_of(self, u, t);
            let err = if t == target { 1.0 - q } else { -q };
            // Read the row before updating it: taking it back afterwards adds
            // -eta * phi * sum(err^2), which is the same order as the gradient.
            {
                let row = self.row_mut_or_insert(t);
                for i in 0..d {
                    grad[i] -= err * row[i];
                    row[i] += eta * err * phi[i];
                }
            }
        }
        grad
    }

    /// Tokens this node has emitted, which is the set a write samples its
    /// negatives from.
    pub fn emitted(&self, u: usize) -> Vec<u32> {
        self.arena[u].counts.iter().map(|(t, _)| *t).collect()
    }
}
