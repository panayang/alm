//! Reading the address: a bounded posterior over paths, advanced one evidence
//! increment per tick.
//!
//! A top-k beam degenerates on a tree -- all the cursors end up inside the same
//! subtree a level or two down, which is exactly the failure a coarse routing
//! error causes. Particles resampled by branch mass do not: as long as one
//! particle is alive in the true subtree, evidence turning that way multiplies
//! it. Each particle also carries its ancestor stack, so a collapsing particle
//! backtracks to its deepest still-supported ancestor rather than dying, which
//! turns "spread the bets" into "take a bad bet back".
//!
//! The commit rule is the same machinery in both of its jobs: a level keeps
//! accumulating evidence until its branch distribution is confident enough by
//! the level's own calibration counters, and the same counters decide whether to
//! speak. Difficulty therefore shows up as time -- a hard discrimination is a
//! wide step, not a wrong answer.

use crate::code::PathCode;
use crate::config::Config;
use crate::embed::Embeddings;
use crate::graph::Graph;
use crate::ladder::Ladder;
use crate::num::{argmax, normalize, softmax};
use crate::tree::Tree;

/// Which ladder rung feeds which tree level. Level 1 -- the coarsest decision --
/// reads the slowest rung; the deepest level reads the fastest. This alignment
/// is the whole reason the tree encodes an ordered conjunction across
/// timescales rather than a re-encoding of the same information L times.
#[inline]
pub fn rung_for_level(rungs: usize, level: usize) -> usize {
    rungs.saturating_sub(level).min(rungs - 1)
}

pub struct Particle {
    pub code: PathCode,
    pub p: Vec<f32>,
    /// The payload the walk is *routed* by, held fixed for the whole response.
    ///
    /// The readout rows are learned at the payload the write walk ends on, so a
    /// read has to arrive at the same vector for those rows to be scored
    /// against anything meaningful. Routing each hop by the running payload
    /// instead makes the read walk diverge from the write walk and the readout
    /// is then evaluated at a point it was never trained at.
    pub q0: Vec<f32>,
    pub hops_taken: usize,
    pub gnode: usize,
    pub logw: f32,
    pub ticks_here: u32,
    evidence: Vec<f32>,
}

/// One commit made by the leading particle, held until the world reveals which
/// branch the write actually took. This is the calibration signal, and it is
/// entirely self-supervised: the prediction is made before the resolution
/// arrives, and the target is the write's own descent, not a label.
#[derive(Clone, Copy)]
pub struct PendingCommit {
    pub level: usize,
    pub confidence: f32,
    pub chosen: usize,
}

pub struct Swarm {
    pub parts: Vec<Particle>,
    pub weights: Vec<f32>,
    pub pending: Vec<PendingCommit>,
    pub resamples: u64,
    pub backtracks: u64,
    pub commits: u64,
    /// Histogram of how many ticks each level took to commit.
    pub level_ticks: Vec<u64>,
    pub level_commits: Vec<u64>,
    d: usize,
    rungs: usize,
    hops: usize,
    temp: f32,
    max_ticks: u32,
    calib_min_obs: u64,
    fallback: f32,
    step_counter: u64,
    seed: u64,
}

impl Swarm {
    pub fn new(cfg: &Config) -> Self {
        let cap = cfg.effective_depth_cap();
        Swarm {
            parts: Vec::new(),
            weights: Vec::new(),
            pending: Vec::new(),
            resamples: 0,
            backtracks: 0,
            commits: 0,
            level_ticks: vec![0; cap + 2],
            level_commits: vec![0; cap + 2],
            d: cfg.d,
            rungs: cfg.rungs,
            hops: cfg.hops,
            temp: cfg.branch_temp,
            max_ticks: cfg.max_ticks_per_level,
            calib_min_obs: cfg.calib_min_obs,
            fallback: cfg.commit_fallback_slack,
            step_counter: 0,
            seed: cfg.seed ^ 0x7A17,
        }
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Start a fresh response from a payload. Every particle begins at the root
    /// and they diverge through sampled branches.
    ///
    /// The memory walk is *not* done here. One hop happens per tick, so how far
    /// into the memory a payload is carried is a temporal resource: a response
    /// gets the depth the world gave it time for. Routing is by the fixed `q0`,
    /// the same way the write walks, so a read that has taken as many hops as
    /// the write arrives at the same vector and the readout rows mean something.
    ///
    /// Hop count is deliberately not tied to tree depth. They are different
    /// resources -- how finely the address is resolved, and how far the payload
    /// has been carried -- and coupling them was what made the readout get
    /// evaluated at points it was never trained at.
    pub fn seed_response(&mut self, n: usize, p0: &[f32], graph: &mut Graph, _hops: usize) {
        let gn = graph.entry(p0);
        let p_end = p0.to_vec();
        self.parts.clear();
        self.weights.clear();
        self.pending.clear();
        for _ in 0..n {
            self.parts.push(Particle {
                code: PathCode::root(),
                p: p_end.clone(),
                q0: p0.to_vec(),
                hops_taken: 0,
                gnode: gn,
                logw: 0.0,
                ticks_here: 0,
                evidence: Vec::new(),
            });
            self.weights.push(1.0 / n as f32);
        }
    }

    /// Drive the live particles with an incoming token, without restarting them.
    /// This is the "he is still talking and you are revising" case: the payload
    /// is composed multiplicatively, so order matters, and no separate re-drive
    /// rule is needed -- the particles are rescored against the new background
    /// on the very next tick and the ones that no longer fit lose their weight.
    pub fn drive(&mut self, tok: usize, emb: &Embeddings, graph: &mut Graph, _hops: usize) {
        for i in 0..self.parts.len() {
            emb.apply_operator(Some(tok), &mut self.parts[i].q0);
            let q0 = self.parts[i].q0.clone();
            self.parts[i].p = q0.clone();
            self.parts[i].gnode = graph.entry(&q0);
            self.parts[i].hops_taken = 0;
            self.parts[i].evidence.clear();
            self.parts[i].ticks_here = 0;
        }
    }

    /// One memory hop per particle per tick, until the walk is as deep as the
    /// memory allows. This is the rate limit doing its work: a response reaches
    /// only as far into the memory as it had ticks for.
    pub fn advance_walk(&mut self, graph: &mut Graph) {
        for i in 0..self.parts.len() {
            if self.parts[i].hops_taken >= self.hops {
                continue;
            }
            let a = graph.select(self.parts[i].gnode, &self.parts[i].q0);
            let st = graph.hop(a, &self.parts[i].p);
            self.parts[i].p = st.p_out.clone();
            self.parts[i].gnode = graph.head_of(a);
            self.parts[i].hops_taken += 1;
            graph.touch_read(&st);
        }
    }

    /// Mean hops taken, so the depth a response actually reached is reported
    /// rather than assumed.
    pub fn mean_hops(&self) -> f32 {
        if self.parts.is_empty() {
            return 0.0;
        }
        self.parts.iter().map(|p| p.hops_taken as f32).sum::<f32>() / self.parts.len() as f32
    }

    fn query_for(&self, level: usize, p: &[f32], ladder: &Ladder, out: &mut Vec<f32>) {
        let k = rung_for_level(self.rungs, level);
        let delta = ladder.delta(k);
        out.clear();
        out.extend_from_slice(delta);
        for i in 0..self.d {
            out[i] += p[i];
        }
        normalize(out);
    }

    /// One tick: accumulate evidence, commit matured factors, rescore, resample.
    ///
    /// Reads never allocate. Growth belongs to the write path, where it follows
    /// content that was actually stored; letting exploratory particles append
    /// nodes would make the tree a function of the search rather than of the
    /// stream.
    pub fn step(&mut self, tree: &mut Tree, graph: &mut Graph, ladder: &Ladder) {
        self.step_counter += 1;
        self.advance_walk(graph);
        let mut query = Vec::with_capacity(self.d);

        for i in 0..self.parts.len() {
            let node = self.parts[i].code.leaf();
            let level = self.parts[i].code.depth();
            if tree.arena[node].children.is_empty() {
                // Nothing to commit here; the particle rests at this node and
                // sharpens only through its readout.
                self.parts[i].ticks_here += 1;
                continue;
            }
            let p = self.parts[i].p.clone();
            self.query_for(level + 1, &p, ladder, &mut query);

            let mut scores = Vec::new();
            tree.branch_scores(node, &query, self.temp, &mut scores);
            if self.parts[i].evidence.len() != scores.len() {
                self.parts[i].evidence = vec![0.0; scores.len()];
            }
            for (e, s) in self.parts[i].evidence.iter_mut().zip(scores.iter()) {
                *e += *s;
            }
            self.parts[i].ticks_here += 1;

            let ticks = self.parts[i].ticks_here as f32;
            let mut q: Vec<f32> = self.parts[i].evidence.iter().map(|e| e / ticks).collect();
            softmax(&mut q);
            let best = argmax(&q);
            let conf = q[best];

            let threshold = tree.calib[level.min(tree.calib.len() - 1)]
                .crossing(self.calib_min_obs, self.fallback);
            let mature = conf >= threshold || self.parts[i].ticks_here >= self.max_ticks;

            if !mature {
                continue;
            }

            // Particle 0 exploits; the rest sample, which is what keeps the
            // posterior covered when the coarse decision is wrong.
            let pick = if i == 0 {
                best
            } else {
                let u = crate::num::uniform(
                    self.seed ^ (i as u64) << 32,
                    self.step_counter.wrapping_mul(31),
                );
                let mut acc = 0.0f32;
                let mut k = q.len() - 1;
                for (j, &qq) in q.iter().enumerate() {
                    acc += qq;
                    if u < acc {
                        k = j;
                        break;
                    }
                }
                k
            };

            let children = tree.arena[node].children.clone();
            let chosen = children[pick];
            let others: Vec<(usize, f32)> = children
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != pick)
                .map(|(j, &c)| (c, q[j]))
                .collect();

            self.parts[i].code.push(chosen, q[pick], others);
            self.parts[i].ticks_here = 0;
            self.parts[i].evidence.clear();
            self.commits += 1;
            let lv = (level + 1).min(self.level_commits.len() - 1);
            self.level_commits[lv] += 1;
            self.level_ticks[lv] += ticks as u64;

            if i == 0 {
                self.pending.push(PendingCommit { level: level + 1, confidence: conf, chosen });
            }

        }

        self.rescore(tree, ladder);
        self.maybe_resample(tree, ladder);
    }

    /// Rescore every particle's whole ancestor path against the *current*
    /// pyramid. A coarse choice made earlier can become inconsistent as the
    /// background moves, and this is where that shows up.
    fn rescore(&mut self, tree: &Tree, ladder: &Ladder) {
        let mut query = Vec::with_capacity(self.d);
        for i in 0..self.parts.len() {
            let p = self.parts[i].p.clone();
            let mut lw = 0.0f32;
            for j in 1..self.parts[i].code.path.len() {
                let node = self.parts[i].code.path[j];
                self.query_for(j, &p, ladder, &mut query);
                lw += self.temp * crate::num::dot(&tree.arena[node].proto, &query);
            }
            self.parts[i].logw = lw;
        }
        let n = self.parts.len();
        if n == 0 {
            return;
        }
        let mut w: Vec<f32> = self.parts.iter().map(|p| p.logw).collect();
        softmax(&mut w);
        self.weights = w;
    }

    /// Effective sample size. Below half the particle count the swarm has
    /// collapsed onto too few paths and is resampled.
    pub fn ess(&self) -> f32 {
        let s: f32 = self.weights.iter().map(|w| w * w).sum();
        if s <= 0.0 {
            0.0
        } else {
            1.0 / s
        }
    }

    fn maybe_resample(&mut self, tree: &Tree, ladder: &Ladder) {
        let n = self.parts.len();
        if n < 2 {
            return;
        }
        if self.ess() >= n as f32 / 2.0 {
            return;
        }
        self.resamples += 1;
        // Systematic resampling from one counter-based uniform: reproducible,
        // and it introduces no free parameter.
        let u0 = crate::num::uniform(self.seed ^ 0xE55, self.step_counter) / n as f32;
        let mut cum = 0.0f32;
        let mut idx = 0usize;
        let mut picks = Vec::with_capacity(n);
        for k in 0..n {
            let target = u0 + k as f32 / n as f32;
            while idx < n - 1 && cum + self.weights[idx] < target {
                cum += self.weights[idx];
                idx += 1;
            }
            picks.push(idx);
        }
        let mut next: Vec<Particle> = Vec::with_capacity(n);
        for (slot, &src) in picks.iter().enumerate() {
            let mut part = Particle {
                code: self.parts[src].code.clone(),
                p: self.parts[src].p.clone(),
                q0: self.parts[src].q0.clone(),
                hops_taken: self.parts[src].hops_taken,
                gnode: self.parts[src].gnode,
                logw: self.parts[src].logw,
                ticks_here: 0,
                evidence: Vec::new(),
            };
            // A duplicate that keeps the whole path adds nothing. Each copy
            // backtracks to its deepest still-supported ancestor -- the deepest
            // level at which its branch is still the argmax under the current
            // pyramid -- and re-diverges from there.
            if slot > 0 {
                let popped = Self::backtrack(&mut part, tree, ladder, self.rungs, self.temp);
                self.backtracks += popped as u64;
            }
            next.push(part);
        }
        self.parts = next;
        let w = 1.0 / n as f32;
        for x in self.weights.iter_mut() {
            *x = w;
        }
    }

    /// Truncate a particle's path to its deepest still-supported prefix.
    ///
    /// The scan runs from the root *down*, not from the leaf up. Checking
    /// upward and stopping at the first level that still agrees looks
    /// equivalent and is not: deepening appends single-child nodes, and at a
    /// node with one child the recorded branch is trivially the argmax, so an
    /// upward scan halts immediately and a stale coarse commitment is never
    /// revisited. That is how a particle ends up pinned to a leaf it chose in
    /// the first few events and never moves again.
    fn backtrack(
        part: &mut Particle,
        tree: &Tree,
        ladder: &Ladder,
        rungs: usize,
        temp: f32,
    ) -> usize {
        let depth = part.code.depth();
        let mut query = vec![0.0f32; part.p.len()];
        let mut keep = depth;
        for j in 1..=depth {
            let node = part.code.path[j];
            let parent = part.code.path[j - 1];
            let k = rung_for_level(rungs, j);
            query.copy_from_slice(ladder.delta(k));
            for i in 0..query.len() {
                query[i] += part.p[i];
            }
            normalize(&mut query);
            let mut scores = Vec::new();
            tree.branch_scores(parent, &query, temp, &mut scores);
            if scores.is_empty() {
                keep = j - 1;
                break;
            }
            if tree.arena[parent].children[argmax(&scores)] != node {
                keep = j - 1;
                break;
            }
        }
        let popped = depth - keep;
        for _ in 0..popped {
            part.code.pop();
        }
        if popped > 0 {
            part.ticks_here = 0;
            part.evidence.clear();
        }
        popped
    }

    /// Force every particle back to its deepest still-supported ancestor.
    ///
    /// Called when the world drives, which is the only moment a committed
    /// coarse factor can become wrong. There is no separate re-drive rule: a
    /// path that still fits the background keeps its depth, and one that does
    /// not loses exactly the levels that no longer hold.
    pub fn recheck(&mut self, tree: &Tree, ladder: &Ladder) {
        for i in 0..self.parts.len() {
            let popped =
                Self::backtrack(&mut self.parts[i], tree, ladder, self.rungs, self.temp);
            self.backtracks += popped as u64;
        }
    }

    pub fn payloads(&self) -> Vec<Vec<f32>> {
        self.parts.iter().map(|p| p.p.clone()).collect()
    }

    pub fn codes(&self) -> Vec<PathCode> {
        self.parts.iter().map(|p| p.code.clone()).collect()
    }

    pub fn leader(&self) -> usize {
        argmax(&self.weights)
    }

    pub fn mean_depth(&self) -> f32 {
        if self.parts.is_empty() {
            return 0.0;
        }
        self.parts.iter().map(|p| p.code.depth() as f32).sum::<f32>() / self.parts.len() as f32
    }

    /// Fraction of particles that sit in different subtrees at level 1. If this
    /// is near zero the particle set has collapsed and the resampling is not
    /// earning its budget.
    pub fn spread_at_root(&self) -> f32 {
        if self.parts.len() < 2 {
            return 0.0;
        }
        let mut seen: Vec<usize> = Vec::new();
        for p in self.parts.iter() {
            if p.code.path.len() > 1 {
                let v = p.code.path[1];
                if !seen.contains(&v) {
                    seen.push(v);
                }
            }
        }
        seen.len() as f32 / self.parts.len() as f32
    }

}
