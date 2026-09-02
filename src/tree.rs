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

use crate::config::Config;
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

    /// Sparse readout rows, one per token this node has actually emitted. The
    /// candidate set at retrieval is therefore the emitted content, not the
    /// vocabulary.
    pub rows: Vec<(u32, Vec<f32>)>,
    row_index: HashMap<u32, usize>,

    /// Dispersion statistics for the widen criterion.
    pub sim: Running,
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
            rows: Vec::new(),
            row_index: HashMap::new(),
            sim: Running::default(),
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
            Some(&i) => self.counts[i].1 += 1,
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

    #[inline]
    pub fn row_of(&self, tok: u32) -> Option<&[f32]> {
        self.row_index.get(&tok).map(|&i| self.rows[i].1.as_slice())
    }

    fn row_mut_or_insert(&mut self, tok: u32, d: usize) -> &mut Vec<f32> {
        if let Some(&i) = self.row_index.get(&tok) {
            return &mut self.rows[i].1;
        }
        self.row_index.insert(tok, self.rows.len());
        self.rows.push((tok, vec![0.0; d]));
        let i = self.rows.len() - 1;
        &mut self.rows[i].1
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
    /// The confidence a branch must reach before this level commits to it.
    ///
    /// Calibration may only *raise* the bar, never lower it. The lowest bin
    /// whose accuracy happens to match its own centre is not a reason to commit
    /// at fifteen per cent confidence -- a badly calibrated level would then
    /// commit instantly on no evidence, which is exactly the failure that made
    /// the evidence accumulation inert the first time this ran.
    ///
    /// This stands in for the optimal-stopping rule, which is not closed. It is
    /// read out of counters rather than set, so it adds no free parameter, but
    /// it is provisional and is labelled as such in the design note.
    pub fn crossing(&self, min_obs: u64, fallback: f32) -> f32 {
        if self.observations() < min_obs {
            return fallback;
        }
        let n = self.bins.len();
        let mut needed = fallback;
        for b in 0..n {
            let centre = (b as f32 + 0.5) / n as f32;
            if self.bins[b].0 > 0 && self.accuracy(b) < centre {
                // Over-confident here: do not trust this level at this level of
                // confidence, and require more.
                needed = needed.max(centre);
            }
        }
        needed.min(0.95)
    }
}

pub struct Tree {
    pub d: usize,
    pub vocab: usize,
    pub arena: Vec<Node>,
    pub calib: Vec<Calibration>,
    pub depth_cap: usize,
    max_children: usize,
    max_nodes: usize,
    grow_theta: f64,
    grow_hold: u32,
    grow_min_obs: u64,
    deepen_min_obs: u64,
    pub widen_events: u64,
    pub deepen_events: u64,
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
            d,
            vocab: cfg.vocab,
            arena,
            calib: (0..cap + 1).map(|_| Calibration::new(cfg.calib_bins)).collect(),
            depth_cap: cap,
            max_children: cfg.max_children,
            max_nodes: cfg.max_nodes,
            grow_theta: cfg.grow_theta,
            grow_hold: cfg.grow_hold,
            grow_min_obs: cfg.grow_min_obs,
            deepen_min_obs: cfg.deepen_min_obs,
            widen_events: 0,
            deepen_events: 0,
        }
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
        self.arena.iter().map(|n| n.rows.len()).sum()
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
    pub fn readout_dist(&self, u: usize, p: &[f32]) -> Vec<(u32, f32)> {
        let node = &self.arena[u];
        if node.rows.is_empty() {
            return Vec::new();
        }
        let mut s: Vec<f32> = node.rows.iter().map(|(_, r)| dot(r, p)).collect();
        softmax(&mut s);
        node.rows.iter().map(|(t, _)| *t).zip(s).collect()
    }

    /// The delta-rule step, restricted to a touched set: the observed token plus
    /// negatives sampled from this node's own emitted targets. Rows outside the
    /// touched set are left exactly zero rather than nearly zero, which is what
    /// makes occupancy a measure of stored content.
    pub fn readout_update(
        &mut self,
        u: usize,
        p: &[f32],
        target: u32,
        negatives: &[u32],
        eta: f32,
    ) -> Vec<f32> {
        let d = self.d;
        let dist = self.readout_dist(u, p);
        let mut prob: HashMap<u32, f32> = HashMap::new();
        for (t, q) in dist.iter() {
            prob.insert(*t, *q);
        }
        // grad wrt payload, accumulated over the touched rows only.
        let mut grad = vec![0.0f32; d];
        let mut touched: Vec<u32> = Vec::with_capacity(negatives.len() + 1);
        touched.push(target);
        for &n in negatives {
            if n != target && !touched.contains(&n) {
                touched.push(n);
            }
        }
        for &t in touched.iter() {
            let q = *prob.get(&t).unwrap_or(&0.0);
            let err = if t == target { 1.0 - q } else { -q };
            {
                let row = self.arena[u].row_mut_or_insert(t, d);
                for i in 0..d {
                    row[i] += eta * err * p[i];
                }
            }
            if let Some(row) = self.arena[u].row_of(t) {
                for i in 0..d {
                    grad[i] -= err * row[i];
                }
            }
        }
        grad
    }
}
