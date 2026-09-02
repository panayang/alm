//! The tick loop.
//!
//! One tick is one bounded reaction: advance the background, advance the
//! particles by one evidence increment, emit. When the world speaks, the code
//! standing on the output is settled first and only then is anything written --
//! predict, be charged, then write, with no example ever scored after it has
//! been trained on.
//!
//! Two facts about this loop are worth stating because they are what make the
//! accounting event-driven rather than per-tick. A baseline tick performs no
//! content write at all, because the token's operator is the identity and the
//! write gate reads its norm; and a baseline tick is not charged, because there
//! is nothing there to encode. Those are the same fact seen from the write side
//! and the ledger side, which is why the design has one axiom where it used to
//! have three.

use crate::code::{self, PathCode};
use crate::config::Config;
use crate::descent::{rung_for_level, Swarm};
use crate::embed::{Channel, Embeddings};
use crate::graph::Graph;
use crate::ladder::Ladder;
use crate::num::{argmax, normalize};
use crate::tree::Tree;

const KEY_NEG: u64 = 0x0000_0000_0000_0031;

pub struct TickOutcome {
    /// Whether the world spoke on this tick, and was therefore charged.
    pub charged: bool,
    pub bits: f64,
    /// Entropy of the emitted distribution, on the ticks where it is evaluated.
    pub entropy_bits: Option<f64>,
    pub ticks_since_event: u32,
    pub depth: f32,
    pub top1: Option<usize>,
    pub correct: bool,
    /// The token spoken out loud, if the leader was confident enough.
    pub overt: Option<usize>,
    /// Whether any content write happened. Must be false on baseline ticks.
    pub wrote: bool,
}

pub struct Model {
    pub cfg: Config,
    pub emb: Embeddings,
    pub ladder: Ladder,
    pub graph: Graph,
    pub tree: Tree,
    pub swarm: Swarm,

    /// The payload the current challenge has composed so far. Multiplicative and
    /// order-sensitive; a baseline tick applies the identity and leaves it be.
    cue: Vec<f32>,
    covert: Option<usize>,
    overt: Option<usize>,
    ticks_since_event: u32,
    tick_index: u64,

    pub events: u64,
    pub total_bits: f64,
    pub content_writes: u64,
    pub baseline_ticks: u64,
    pub overt_emissions: u64,
    /// The leaf the last write landed in. Exposed only so an experiment can ask
    /// whether the address separates regimes; the model never reads it.
    pub last_write_leaf: usize,
    scratch: Vec<f32>,
}

impl Model {
    pub fn new(cfg: Config) -> Self {
        let emb = Embeddings::new(&cfg);
        let ladder = Ladder::new(&cfg);
        let graph = Graph::new(&cfg);
        let tree = Tree::new(&cfg);
        let swarm = Swarm::new(&cfg);
        let d = cfg.d;
        let mut cue = vec![0.0f32; d];
        cue[0] = 1.0;
        Model {
            cfg,
            emb,
            ladder,
            graph,
            tree,
            swarm,
            cue,
            covert: None,
            overt: None,
            ticks_since_event: 0,
            tick_index: 0,
            events: 0,
            total_bits: 0.0,
            content_writes: 0,
            baseline_ticks: 0,
            overt_emissions: 0,
            last_write_leaf: 0,
            scratch: vec![0.0; d],
        }
    }

    /// The deterministic descent the write takes: argmax at every level, one
    /// pass, no evidence accumulation. Its path is the target the read
    /// particles' branch predictions are calibrated against -- a prediction made
    /// before the resolution arrived, scored against the stream itself.
    fn write_descent(&mut self, p: &[f32], grow: bool) -> Vec<usize> {
        let mut path = vec![0usize];
        let mut u = 0usize;
        let mut query = vec![0.0f32; self.cfg.d];
        loop {
            if self.tree.arena[u].children.is_empty() {
                break;
            }
            let level = self.tree.arena[u].level + 1;
            if level > self.tree.depth_cap {
                break;
            }
            let k = rung_for_level(self.cfg.rungs, level);
            query.copy_from_slice(self.ladder.delta(k));
            for i in 0..self.cfg.d {
                query[i] += p[i];
            }
            normalize(&mut query);
            if grow {
                self.tree.place_pending(u, &query);
            }
            let mut scores = Vec::new();
            self.tree.branch_scores(u, &query, 1.0, &mut scores);
            if scores.is_empty() {
                break;
            }
            let best = argmax(&scores);
            let winner = self.tree.arena[u].children[best];
            if grow {
                // The dispersion criterion reads the *raw* similarity, not a
                // softmax over siblings: with a single child the softmax is
                // identically one and its variance identically zero, so a class
                // could never be found dispersed enough to split.
                self.tree.observe_and_maybe_grow(u, winner, scores[best], &query);
            }
            u = winner;
            path.push(u);
        }
        if grow {
            // Deepening is an append like widening, gated by observations, and
            // it happens where content lands rather than where a particle
            // happened to wander.
            let leaf = *path.last().unwrap();
            if self.tree.arena[leaf].children.is_empty()
                && self.tree.arena[leaf].level < self.tree.depth_cap
            {
                let level = self.tree.arena[leaf].level + 1;
                let k = rung_for_level(self.cfg.rungs, level);
                query.copy_from_slice(self.ladder.delta(k));
                for i in 0..self.cfg.d {
                    query[i] += p[i];
                }
                normalize(&mut query);
                self.tree.deepen(leaf, &query);
            }
        }
        path
    }

    fn sample_negatives(&self, leaf: usize, target: u32, k: usize) -> Vec<u32> {
        let rows = &self.tree.arena[leaf].rows;
        if rows.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            let j = crate::num::uniform_below(
                self.cfg.seed ^ KEY_NEG,
                self.tick_index.wrapping_mul(97).wrapping_add(i as u64),
                rows.len() as u64,
            ) as usize;
            let t = rows[j].0;
            if t != target {
                out.push(t);
            }
        }
        out
    }

    /// The leader's current best token and its probability, from the tail
    /// distribution alone -- cheap enough to run every tick.
    fn leader_top1(&self) -> Option<(usize, f32)> {
        if self.swarm.is_empty() {
            return None;
        }
        let i = self.swarm.leader();
        let code = &self.swarm.parts[i].code;
        let u = code.leaf();
        let dist = self.tree.readout_dist(u, &self.swarm.parts[i].p);
        if !dist.is_empty() && !self.cfg.no_readout {
            let mut best = dist[0];
            for &(t, q) in dist.iter() {
                if q > best.1 {
                    best = (t, q);
                }
            }
            return Some((best.0 as usize, best.1));
        }
        let node = &self.tree.arena[u];
        if node.counts.is_empty() {
            return None;
        }
        let mut best = node.counts[0];
        for &(t, c) in node.counts.iter() {
            if c > best.1 {
                best = (t, c);
            }
        }
        Some((best.0 as usize, best.1 as f32 / node.total.max(1) as f32))
    }

    /// Feed the self channels back into the fast end of the background.
    fn feedback(&mut self) {
        let d = self.cfg.d;
        if self.cfg.feedback_overt {
            if let Some(t) = self.covert {
                let mut v = vec![0.0f32; d];
                self.emb.rotated(t, Channel::Covert, &mut v);
                self.ladder.observe_self(&v);
            }
            if let Some(t) = self.overt {
                let mut v = vec![0.0f32; d];
                self.emb.rotated(t, Channel::Overt, &mut v);
                self.ladder.observe_self(&v);
            }
        }
        if self.cfg.feedback_write && !self.swarm.is_empty() {
            // The update reporting on itself: which memory is being touched,
            // squashed by how much activity there is. During a gap this is the
            // only thing still moving, and it is what keeps the response
            // unfolding when the world has gone quiet.
            let i = self.swarm.leader();
            let act: f32 = self.graph.trace.iter().sum::<f32>() / self.graph.edges() as f32;
            let mut v = vec![0.0f32; d];
            self.emb.rotate_vec(&self.swarm.parts[i].p.clone(), Channel::Write, &mut v);
            let g = act.tanh();
            for x in v.iter_mut() {
                *x *= g;
            }
            self.ladder.observe_self(&v);
        }
    }

    pub fn tick(&mut self, obs: Option<usize>, want_entropy: bool) -> TickOutcome {
        self.tick_index += 1;
        let mut out = TickOutcome {
            charged: false,
            bits: 0.0,
            entropy_bits: None,
            ticks_since_event: self.ticks_since_event,
            depth: self.swarm.mean_depth(),
            top1: None,
            correct: false,
            overt: None,
            wrote: false,
        };

        match obs {
            // ------------------------------------------------ baseline
            None => {
                self.baseline_ticks += 1;
                self.ticks_since_event += 1;
                self.ladder.tick_baseline();
                self.feedback();
                self.ladder.refresh();
                self.graph.decay_traces(self.cfg.trace_lambda);
                if !self.swarm.is_empty() {
                    self.swarm.step(&mut self.tree, &mut self.graph, &self.ladder);
                }
                if want_entropy && !self.swarm.is_empty() {
                    let sp = code::spread(
                        &self.tree,
                        &self.swarm.codes(),
                        &self.swarm.payloads(),
                        &self.swarm.weights,
                        !self.cfg.no_readout,
                    );
                    out.entropy_bits = Some(sp.entropy_bits());
                }
                if let Some((t, q)) = self.leader_top1() {
                    self.covert = Some(t);
                    out.top1 = Some(t);
                    let depth = self.swarm.parts[self.swarm.leader()].code.depth();
                    let thr = self.tree.calib[depth.min(self.tree.calib.len() - 1)]
                        .crossing(self.cfg.calib_min_obs, self.cfg.commit_fallback_slack);
                    if q >= thr {
                        self.overt = Some(t);
                        out.overt = Some(t);
                        self.overt_emissions += 1;
                    } else {
                        self.overt = None;
                    }
                }
                out.depth = self.swarm.mean_depth();
                out
            }

            // ------------------------------------------------ the world speaks
            Some(x) => {
                self.events += 1;
                out.charged = true;
                out.ticks_since_event = self.ticks_since_event;

                // 1. Settle the standing code before anything is written.
                if self.swarm.is_empty() {
                    let c = self.cue.clone();
                    let (n, h) = (self.cfg.particles, self.cfg.hops);
                    self.swarm.seed_response(n, &c, &mut self.graph, h);
                }
                let codes: Vec<PathCode> = self.swarm.codes();
                let payloads = self.swarm.payloads();
                let prob = code::mixture_prob(
                    &self.tree,
                    &codes,
                    &payloads,
                    &self.swarm.weights,
                    x as u32,
                    !self.cfg.no_readout,
                );
                out.bits = code::charge_bits(prob);
                self.total_bits += out.bits;
                if let Some((t, _)) = self.leader_top1() {
                    out.top1 = Some(t);
                    out.correct = t == x;
                }
                if want_entropy {
                    let sp = code::spread(
                        &self.tree,
                        &codes,
                        &payloads,
                        &self.swarm.weights,
                        !self.cfg.no_readout,
                    );
                    out.entropy_bits = Some(sp.entropy_bits());
                }

                // 2. Calibrate: the write's own descent reveals which branch was
                //    right, and the pending read commitments are scored against
                //    it. No labels enter here; the target is the stream.
                let cue = self.cue.clone();
                let truth = self.write_descent(&cue, true);
                for pc in self.swarm.pending.clone() {
                    if pc.level < truth.len() {
                        let correct = truth[pc.level] == pc.chosen;
                        let lv = pc.level.min(self.tree.calib.len() - 1);
                        self.tree.calib[lv].push(pc.confidence, correct);
                    }
                }
                self.swarm.pending.clear();

                // 3. Write. Gated by the operator norm, which is exactly zero at
                //    baseline and so cannot reach this branch from there.
                let gate = self.emb.drive_norm(Some(x));
                if gate > 0.0 {
                    let leaf = *truth.last().unwrap();
                    let steps = self.graph.write_walk(&cue, &cue, self.cfg.hops);
                    let p_end =
                        steps.last().map(|s| s.p_out.clone()).unwrap_or_else(|| cue.clone());
                    let negs = self.sample_negatives(leaf, x as u32, self.cfg.neg_samples);
                    let grad = self.tree.readout_update(
                        leaf,
                        &p_end,
                        x as u32,
                        &negs,
                        self.cfg.eta * gate,
                    );
                    self.graph.backprop(&steps, &grad, self.cfg.eta * gate);

                    // Gap-time particle activity gets its share of the
                    // settlement through the eligibility traces, in O(1) per
                    // edge and without unrolling the gap.
                    if !self.cfg.no_eligibility {
                        self.graph.credit_traces(&grad, self.cfg.eta * gate * 0.25, 0.05);
                    }

                    self.tree.record(leaf, x as u32);
                    self.last_write_leaf = leaf;
                    self.content_writes += 1;
                    out.wrote = true;
                }

                // 4. Fold the observation into the background and the cue, then
                //    drive the live particles rather than restarting them.
                self.emb.rotated(x, Channel::In, &mut self.scratch);
                let v = self.scratch.clone();
                self.ladder.observe_world(&v);
                self.feedback();
                self.ladder.refresh();
                self.emb.apply_operator(Some(x), &mut self.cue);
                self.graph.decay_traces(self.cfg.trace_lambda);

                let (n, h) = (self.cfg.particles, self.cfg.hops);
                if self.swarm.is_empty() {
                    let c = self.cue.clone();
                    self.swarm.seed_response(n, &c, &mut self.graph, h);
                } else {
                    self.swarm.drive(x, &self.emb, &mut self.graph, h);
                    self.swarm.recheck(&self.tree, &self.ladder);
                }
                self.swarm.step(&mut self.tree, &mut self.graph, &self.ladder);

                self.ticks_since_event = 0;
                self.overt = None;
                out.depth = self.swarm.mean_depth();
                out
            }
        }
    }
}
