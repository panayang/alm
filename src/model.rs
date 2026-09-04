//! The tick loop.
//!
//! One tick is one bounded reaction, and it is three things: **one read, one
//! emission, one write slot** (A1). The read is a single hop through the memory
//! graph; the emission is the distribution the response currently stands on; the
//! write slot is the identity at baseline and a real write when the world speaks.
//!
//! # The response is a walk, not a lookup
//!
//! A challenge does not select a place in memory and read it. It starts a
//! trajectory, and the trajectory keeps moving for as long as the world gives it
//! ticks. The prior is what the walk has touched -- a mixture, not a choice -- so
//! there is no commitment to be wrong about, and a wrong hop is one term among
//! several rather than the whole answer.
//!
//! This is what the gap is *for*. Previously the payload took its hops in the
//! first two ticks and then froze, so after tick two the response state was
//! constant and the only thing still moving was a tree descent that measurement
//! showed was harmful. The response unfolding through the gap had never actually
//! been implemented.
//!
//! # The three streams are the drive
//!
//! A walk on a decaying background is autonomous: it reaches a fixed point or a
//! short cycle and stops being informative. What keeps it moving is that the
//! system hears itself. Each tick feeds back, into the fastest rung only (A5):
//!
//! * **Covert** -- what it currently thinks the answer is, emitted every tick,
//!   because the system is always outputting.
//! * **Overt** -- what it has said out loud, once the evidence stops moving.
//! * **Write** -- which memory it is touching, i.e. where the walk now is.
//!
//! The write channel is not bookkeeping here. It is the drive: visiting a node
//! pushes the next query away from that node, so inhibition of return falls out
//! rather than being added, and "the update is itself part of the context" is
//! load-bearing rather than philosophical. If ablating it changes nothing, the
//! walk was never being driven and this design is wrong.

use crate::code::{self};
use crate::config::{BindMode, Config};
use crate::embed::{Channel, Embeddings};
use crate::graph::Graph;
use crate::ladder::Ladder;
use crate::num::normalize;
use crate::store::Store;

const KEY_NEG: u64 = 0x0000_0000_0000_0031;

pub struct TickOutcome {
    /// Ticks after the last event at which the right answer first appeared in
    /// inner speech, and at which it was first said out loud. Two reaction
    /// times: having the idea and committing to it are separate events.
    pub idea_onset: Option<u32>,
    pub speech_onset: Option<u32>,
    /// Whether the world spoke on this tick, and was therefore charged.
    pub charged: bool,
    pub bits: f64,
    pub entropy_bits: Option<f64>,
    pub ticks_since_event: u32,
    /// How many hops this response has taken, and how spread its visit
    /// distribution is.
    pub hops: u32,
    pub visit_entropy: f64,
    pub top1: Option<usize>,
    pub correct: bool,
    pub overt: Option<usize>,
    /// Whether any content write happened. Must be false on baseline ticks.
    pub wrote: bool,
}

/// Everything a probe displaces and has to give back.
///
/// The memory itself is not here: a probe runs frozen, so nothing stored may
/// change. What a probe unavoidably disturbs is the *situation* -- the
/// background, the state the response has composed, the bound traces, and where
/// the walk is -- and that is saved and restored, which is what makes a
/// retention number a statement about what was stored rather than about what the
/// probe left behind.
#[derive(Clone)]
pub struct Volatile {
    ladder: Ladder,
    p: Vec<f32>,
    binds: Vec<Vec<f32>>,
    event_hist: Vec<usize>,
    gnode: usize,
    visit: Vec<f32>,
    hops: u32,
    covert: Option<usize>,
    overt: Option<usize>,
    ticks_since_event: u32,
    prev_answer_conf: f32,
    /// A probe runs ~15 ticks and there are 24 per batch, so leaving this out
    /// advanced the live counter by ~400 ticks per batch. It keys
    /// `select_random`'s nonce and `sample_negatives`, so a probe batch shifted
    /// the random-routing arm's edge sequence and every arm's negative samples.
    tick_index: u64,
    committed: Option<Vec<f32>>,
    committed_token: Option<usize>,
}

pub struct Model {
    /// While frozen the model reads and emits but stores nothing: no content
    /// write, no calibration, no counts. A probe that wrote would be retraining
    /// on the fact it is testing.
    pub frozen: bool,
    pub cfg: Config,
    pub emb: Embeddings,
    pub ladder: Ladder,
    pub graph: Graph,
    pub store: Store,

    /// The one state vector. Events fold a token into it; every tick takes one
    /// hop with it. There is no separate frozen cue: the state *is* the response.
    p: Vec<f32>,
    /// Where the walk is, and where it has been.
    gnode: usize,
    visit: Vec<f32>,
    hops: u32,
    /// The transform applied on the last tick, so the operator write is local
    /// to it and nothing has to cross a hop.
    last_step: Option<crate::graph::WalkStep>,

    /// Bound traces, one per block the mode carries.
    binds: Vec<Vec<f32>>,
    event_hist: Vec<usize>,

    covert: Option<usize>,
    overt: Option<usize>,
    covert_log: Vec<(u32, usize)>,
    overt_log: Vec<(u32, usize)>,
    /// The distribution as it stood when this response first spoke.
    committed: Option<Vec<f32>>,
    committed_token: Option<usize>,
    prev_answer_conf: f32,

    ticks_since_event: u32,
    tick_index: u64,

    pub events: u64,
    pub total_bits: f64,
    pub content_writes: u64,
    pub baseline_ticks: u64,
    pub overt_emissions: u64,
    pub commitments: u64,
    pub silent_settlements: u64,
    /// The node the last write landed in. Exposed only so an experiment can ask
    /// whether the address separates regimes; the model never reads it.
    pub last_write_node: usize,
    /// The node a read actually settled on -- what an address instrument must
    /// look at. `last_write_node` describes the write walk and is unaffected by
    /// `route_query`, `read_entry_by_content` or `route_random`, so pointing the
    /// consistency statistic at it measured a walk none of those arms touch.
    pub last_read_node: usize,
    scratch: Vec<f32>,
}

impl Model {
    pub fn new(cfg: Config) -> Self {
        let emb = Embeddings::new(&cfg);
        let ladder = Ladder::new(&cfg);
        let graph = Graph::new(&cfg);
        let store = Store::new(&cfg);
        let d = cfg.d;
        let blocks = cfg.bind_blocks();
        let mut p = vec![0.0f32; d];
        p[0] = 1.0;
        let n = graph.nodes;
        Model {
            frozen: false,
            cfg,
            emb,
            ladder,
            graph,
            store,
            p,
            gnode: 0,
            visit: vec![0.0; n],
            hops: 0,
            last_step: None,
            binds: vec![vec![0.0; d]; blocks],
            event_hist: Vec::new(),
            covert: None,
            overt: None,
            covert_log: Vec::new(),
            overt_log: Vec::new(),
            committed: None,
            committed_token: None,
            prev_answer_conf: 0.0,
            ticks_since_event: 0,
            tick_index: 0,
            events: 0,
            total_bits: 0.0,
            content_writes: 0,
            baseline_ticks: 0,
            overt_emissions: 0,
            commitments: 0,
            silent_settlements: 0,
            last_write_node: 0,
            last_read_node: 0,
            scratch: vec![0.0; d],
        }
    }

    // ---- state a probe borrows and gives back -------------------------

    pub fn volatile(&self) -> Volatile {
        Volatile {
            ladder: self.ladder.clone(),
            p: self.p.clone(),
            binds: self.binds.clone(),
            event_hist: self.event_hist.clone(),
            gnode: self.gnode,
            visit: self.visit.clone(),
            hops: self.hops,
            covert: self.covert,
            overt: self.overt,
            ticks_since_event: self.ticks_since_event,
            prev_answer_conf: self.prev_answer_conf,
            tick_index: self.tick_index,
            committed: self.committed.clone(),
            committed_token: self.committed_token,
        }
    }

    pub fn restore(&mut self, v: Volatile) {
        self.ladder = v.ladder;
        self.p = v.p;
        self.binds = v.binds;
        self.event_hist = v.event_hist;
        self.gnode = v.gnode;
        self.visit = v.visit;
        self.hops = v.hops;
        self.covert = v.covert;
        self.overt = v.overt;
        self.ticks_since_event = v.ticks_since_event;
        self.prev_answer_conf = v.prev_answer_conf;
        self.tick_index = v.tick_index;
        self.committed = v.committed;
        self.committed_token = v.committed_token;
        self.covert_log.clear();
        self.overt_log.clear();
    }

    /// One retention probe under frozen memory: the codelength it was charged
    /// and whether the answer was right. The displaced state is put back here,
    /// so a probe cannot leak into the stream that follows it.
    pub fn probe(&mut self, spec: &crate::gen::ProbeSpec, answer_gap: u32) -> (f64, bool) {
        let saved = self.volatile();
        let was_frozen = self.frozen;
        self.frozen = true;
        self.graph.frozen = true;
        let counts = (
            self.events,
            self.total_bits,
            self.baseline_ticks,
            self.overt_emissions,
            self.commitments,
            self.silent_settlements,
        );

        for &c in spec.context.iter() {
            self.tick(Some(c), false);
            self.tick(None, false);
        }
        for (i, &c) in spec.cue.iter().enumerate() {
            self.tick(Some(c), false);
            let gap = if i + 1 == spec.cue.len() { answer_gap } else { 1 };
            for _ in 0..gap {
                self.tick(None, false);
            }
        }
        let out = self.tick(Some(spec.target), false);

        self.frozen = was_frozen;
        self.graph.frozen = was_frozen;
        self.events = counts.0;
        self.total_bits = counts.1;
        self.baseline_ticks = counts.2;
        self.overt_emissions = counts.3;
        self.commitments = counts.4;
        self.silent_settlements = counts.5;
        self.restore(saved);
        (out.bits, out.correct)
    }

    /// The response's current state vector, for assertions about what a probe
    /// displaced.
    pub fn state_now(&self) -> Vec<f32> {
        self.p.clone()
    }

    /// The current feature vector, for the block-agreement assertion.
    pub fn features_now(&self) -> Vec<f32> {
        self.features(&self.p.clone())
    }

    /// How big the operator term is against the state, and how much of the
    /// state one hop preserves.
    pub fn hop_scale(&self) -> (f32, f32) {
        let p = crate::num::unit_vector(0xA11, 3, self.cfg.d);
        let mut b = vec![0.0f32; self.cfg.d];
        self.graph.w[0].matvec(&p, &mut b);
        let th: Vec<f32> = b.iter().map(|x| x.tanh()).collect();
        let mut out: Vec<f32> = (0..self.cfg.d).map(|i| p[i] + th[i]).collect();
        normalize(&mut out);
        (crate::num::norm(&th), crate::num::dot(&out, &p))
    }

    /// Where the walk is, for the route-perturbation sanity check.
    pub fn node_now(&self) -> usize {
        self.gnode
    }

    /// Cosine between the state and the summed background. Near one means the
    /// anchor has swamped the operator term and the state carries nothing the
    /// bands do not already carry.
    pub fn state_vs_background(&self) -> f32 {
        let mut b = vec![0.0f32; self.cfg.d];
        for k in 0..self.cfg.rungs {
            let d = self.ladder.delta(k);
            for i in 0..self.cfg.d {
                b[i] += d[i];
            }
        }
        normalize(&mut b);
        crate::num::dot(&b, &self.p).abs()
    }

    /// The emitted distribution as it stands, for the normalisation assertion.
    pub fn spread_now(&self) -> code::Scored {
        let phi = self.features(&self.p.clone());
        code::score(&self.store, &phi, !self.cfg.no_readout)
    }

    // ---- features -------------------------------------------------------

    /// The state, the bound traces, and the background bands.
    ///
    /// The bands belong here. They used to feed only the routing query, so the
    /// three self channels -- which write into the ladder -- had no path to the
    /// prediction, and every ablation of them read as no effect. This closes
    /// that loop, and it is also where the fuzzy multi-level background does its
    /// work now that there is no counted prior to carry it.
    fn features(&self, p: &[f32]) -> Vec<f32> {
        let mut f = Vec::with_capacity(self.cfg.feature_blocks() * self.cfg.d);
        f.extend_from_slice(p);
        for b in self.binds.iter() {
            f.extend_from_slice(b);
        }
        for k in 0..self.cfg.rungs {
            f.extend_from_slice(self.ladder.delta(k));
        }
        f
    }

    /// Refresh the bound traces from the arriving token. Called after the
    /// charge, so a response is scored against the binding as it stood when the
    /// last cue arrived -- at the target of a two-cue item, exactly
    /// `E_cueA (*) E_cueB` with nothing else in it.
    fn rebind(&mut self, x: usize) {
        if self.binds.is_empty() {
            return;
        }
        let d = self.cfg.d;
        let ex = self.emb.row(x).to_vec();
        let mut slot = 0usize;
        if matches!(self.cfg.bind_mode, BindMode::EventLag | BindMode::Both) {
            for j in 0..self.cfg.bind_lags {
                if let Some(&prev) = self.event_hist.get(j) {
                    let ep = self.emb.row(prev).to_vec();
                    let mut b = vec![0.0f32; d];
                    crate::num::circconv(&ep, &ex, &mut b);
                    normalize(&mut b);
                    self.binds[slot] = b;
                }
                slot += 1;
            }
        }
        if matches!(self.cfg.bind_mode, BindMode::Band | BindMode::Both) {
            for k in 0..self.cfg.rungs {
                let band = self.ladder.delta(k).to_vec();
                let mut b = vec![0.0f32; d];
                crate::num::circconv(&band, &ex, &mut b);
                normalize(&mut b);
                self.binds[slot] = b;
                slot += 1;
            }
        }
        self.event_hist.insert(0, x);
        self.event_hist.truncate(self.cfg.bind_lags.max(1));
    }

    // ---- the walk --------------------------------------------------------

    /// The query a hop is routed by: the fuzzy multi-level background summed
    /// with the current state. Both halves matter -- the background is what
    /// makes the walk depend on the situation, the state is what makes it depend
    /// on the challenge.
    fn query(&self, out: &mut Vec<f32>) {
        use crate::config::RouteQuery;
        if matches!(self.cfg.route_query, RouteQuery::Bound | RouteQuery::BoundState) {
            out.clear();
            out.resize(self.cfg.d, 0.0);
            for b in self.binds.iter() {
                for i in 0..self.cfg.d {
                    out[i] += b[i];
                }
            }
            if matches!(self.cfg.route_query, RouteQuery::BoundState) {
                for i in 0..self.cfg.d {
                    out[i] += self.p[i];
                }
            }
            normalize(out);
            return;
        }
        out.clear();
        out.extend_from_slice(self.ladder.delta(0));
        for k in 1..self.cfg.rungs {
            let b = self.ladder.delta(k);
            for i in 0..self.cfg.d {
                out[i] += b[i];
            }
        }
        for i in 0..self.cfg.d {
            out[i] += self.p[i];
        }
        normalize(out);
    }

    /// One read: the memory transforms the state, and the background pulls it
    /// back toward the challenge.
    ///
    /// `p <- nu( p + tanh(W_a p) + anchor * Delta )`. The operator term is the
    /// computation; the anchor term is what makes the iteration driven rather
    /// than autonomous, and therefore what makes it converge instead of drifting
    /// into a fixed point or a cycle.
    fn step_walk(&mut self) {
        if self.cfg.bypass_graph {
            // The monolith still integrates the background -- the ladder is
            // part of the single body, not part of the extension. What is gone
            // is the routed transform.
            if self.cfg.anchor > 0.0 {
                for k in 0..self.cfg.rungs {
                    let b = self.ladder.delta(k);
                    for i in 0..self.cfg.d {
                        self.p[i] += self.cfg.anchor * b[i];
                    }
                }
            }
            crate::num::normalize(&mut self.p);
            self.hops += 1;
            return;
        }
        let mut q = Vec::with_capacity(self.cfg.d);
        self.query(&mut q);
        // The random control has to differ from the treatment in exactly one
        // way: it ignores content. It must not also be pinned to a local walk
        // while the treatment gets a global jump, or "content routing beats
        // random routing" is confounded with "global re-entry beats a local
        // wander" -- which the logs said was the whole effect.
        let from = if self.cfg.read_entry_by_content {
            if self.cfg.route_random {
                self.graph.random_node(self.tick_index as u64)
            } else {
                self.graph.entry(&q)
            }
        } else {
            self.gnode
        };
        let a = if self.cfg.route_random {
            self.graph.select_random(from, self.tick_index as u64)
        } else {
            self.graph.select_rank(from, &q, self.cfg.route_perturb)
        };
        let cur = self.p.clone();
        let st = self.graph.hop(a, &cur);
        self.p = st.p_out.clone();

        if self.cfg.anchor > 0.0 {
            let b = self.cfg.anchor;
            for k in 0..self.cfg.rungs {
                let band = self.ladder.delta(k);
                for i in 0..self.cfg.d {
                    self.p[i] += b * band[i];
                }
            }
            normalize(&mut self.p);
        }

        self.gnode = self.graph.head_of(a);
        self.last_read_node = self.gnode;
        self.graph.touch_read(&st);
        self.last_step = Some(st);
        self.hops += 1;

        let lam = self.cfg.visit_decay;
        for v in self.visit.iter_mut() {
            *v *= lam;
        }
        self.visit[self.gnode] += 1.0;
    }

    /// The local write into the operator that was just applied.
    ///
    /// No gradient crosses a hop. The founding requirement is that a node
    /// performs a memory write rather than an autoregressive fit, and this is
    /// that write: move the transform so that, from the state it was applied to,
    /// it carries the state toward what the world then said. Because it is
    /// local, "an identity input does not strongly change a node" is a property
    /// of the layout -- nothing can flow in from elsewhere -- rather than an
    /// approximation.
    fn write_operator_on(&mut self, st: &crate::graph::WalkStep, x: usize, eta: f32) {
        if self.cfg.freeze_operator {
            return;
        }
        let target = if self.cfg.write_toward_embedding {
            self.emb.row(x).to_vec()
        } else {
            match self.store.row_of(x as u32) {
                None => self.emb.row(x).to_vec(),
                Some(r) => r[..self.cfg.d].to_vec(),
            }
        };
        let mut delta = vec![0.0f32; self.cfg.d];
        for i in 0..self.cfg.d {
            delta[i] = target[i] - st.p_out[i];
        }
        // Through the tanh, so a saturated unit is not asked to move.
        for i in 0..self.cfg.d {
            delta[i] *= 1.0 - st.th[i] * st.th[i];
        }
        self.graph.w[st.edge].sub_outer(-eta, &delta, &st.p_in);
    }

    /// The nodes this response has touched, normalised. Nodes below a hundredth
    /// of the peak are dropped: they contribute nothing to the prior and each
    /// one costs a pass over its candidate list.
    /// Spread of the visit accumulator. Diagnostic only now: nothing reads the
    /// visit distribution, but a walk that covers the whole graph uniformly is
    /// still worth seeing.
    fn visit_entropy(&self) -> f64 {
        let total: f32 = self.visit.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let mut h = 0.0f64;
        for &v in self.visit.iter() {
            let q = (v / total) as f64;
            if q > 1e-12 {
                h -= q * q.log2();
            }
        }
        h
    }

    // ---- emission --------------------------------------------------------

    fn best_answer(&self, phi: &[f32]) -> Option<(usize, f32)> {
        code::score(&self.store, phi, !self.cfg.no_readout).top().map(|(t, q)| (t as usize, q))
    }

    /// Decide what to think and what to say, and log both. Runs on every tick,
    /// driven or not: the system is always outputting.
    fn emit(&mut self, out: &mut TickOutcome, phi: &[f32]) {
        let (t, q) = match self.best_answer(phi) {
            None => {
                self.covert = None;
                self.overt = None;
                return;
            }
            Some(v) => v,
        };
        self.covert = Some(t);
        out.top1 = Some(t);
        self.covert_log.push((self.ticks_since_event, t));

        // Speak when the answer has stopped improving. A calibrated-confidence
        // threshold cannot work here: it pins at its ceiling whenever the model
        // is over-confident anywhere, and then the channel never fires, never
        // generates calibration data, and stays shut.
        let rising = q > self.prev_answer_conf + 1e-4;
        let settled = !rising && q >= self.cfg.speak_fallback;
        self.prev_answer_conf = q;

        if settled && self.committed.is_none() {
            self.overt = Some(t);
            out.overt = Some(t);
            self.overt_emissions += 1;
            self.overt_log.push((self.ticks_since_event, t));
            if self.cfg.commit_locks_charge {
                self.committed = Some(phi.to_vec());
                self.committed_token = Some(t);
                self.commitments += 1;
            }
        } else if self.committed.is_none() {
            self.overt = None;
        }
    }

    /// Feed the self channels back into the fast end of the background.
    fn feedback(&mut self) {
        let d = self.cfg.d;
        if self.cfg.feedback_covert {
            if let Some(t) = self.covert {
                let mut v = vec![0.0f32; d];
                self.emb.rotated(t, Channel::Covert, &mut v);
                self.ladder.observe_self(&v);
            }
        }
        if self.cfg.feedback_overt {
            if let Some(t) = self.overt {
                let mut v = vec![0.0f32; d];
                self.emb.rotated(t, Channel::Overt, &mut v);
                self.ladder.observe_self(&v);
            }
        }
        if self.cfg.feedback_write {
            // Where the walk is. This is the drive: it is what keeps the query
            // moving through a gap, and it is why revisiting a node becomes less
            // likely without any fatigue term.
            let key = self.graph.node_key(self.gnode).to_vec();
            let mut v = vec![0.0f32; d];
            self.emb.rotate_vec(&key, Channel::Write, &mut v);
            self.ladder.observe_self(&v);
        }
    }

    fn sample_negatives(&self, target: u32, k: usize) -> Vec<u32> {
        let toks = self.store.known();
        if toks.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            let j = crate::num::uniform_below(
                self.cfg.seed ^ KEY_NEG,
                self.tick_index.wrapping_mul(97).wrapping_add(i as u64),
                toks.len() as u64,
            ) as usize;
            let t = toks[j];
            if t != target {
                out.push(t);
            }
        }
        out
    }

    // ---- the tick ---------------------------------------------------------

    pub fn tick(&mut self, obs: Option<usize>, want_entropy: bool) -> TickOutcome {
        self.tick_index += 1;
        let mut out = TickOutcome {
            idea_onset: None,
            speech_onset: None,
            charged: false,
            bits: 0.0,
            entropy_bits: None,
            ticks_since_event: self.ticks_since_event,
            hops: self.hops,
            visit_entropy: 0.0,
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
                // A bound trace has to decay or it is not a trace. `bind_decay`
                // was declared, documented as "decay of the binding trace across
                // a response", and referenced nowhere -- so a conjunction stood
                // at full strength from one episode into the next until an event
                // overwrote it, and the lag-1 block at a target carried a
                // cross-episode pair.
                for b in self.binds.iter_mut() {
                    for v in b.iter_mut() {
                        *v *= self.cfg.bind_decay;
                    }
                }
                self.ladder.tick_baseline();
                self.feedback();
                self.ladder.refresh();
                self.graph.decay_traces(self.cfg.trace_lambda);

                // The read. The response keeps unfolding for as long as the
                // world gives it ticks; this is what the gap is for. Switched
                // off, the response state freezes after the write walk and the
                // gap does nothing -- which is the control that says whether any
                // of the gain is the walk.
                if self.cfg.walk_during_gap {
                    self.step_walk();
                }

                let p = self.p.clone();
                let phi = self.features(&p);
                if want_entropy {
                    let sc = code::score(&self.store, &phi, !self.cfg.no_readout);
                    out.entropy_bits = Some(sc.entropy_bits());
                }
                self.emit(&mut out, &phi);
                out.hops = self.hops;
                out.visit_entropy = self.visit_entropy();
                out
            }

            // ------------------------------------------------ the world speaks
            Some(x) => {
                self.events += 1;
                out.charged = true;
                out.ticks_since_event = self.ticks_since_event;

                // 1. Settle the standing code before anything is written.
                let p0 = self.p.clone();
                let live_phi = self.features(&p0);
                let (phi, silent) = match (self.cfg.commit_locks_charge, self.committed.clone()) {
                    (true, Some(f)) => (f, false),
                    (true, None) => (live_phi.clone(), true),
                    (false, _) => (live_phi.clone(), false),
                };
                if silent {
                    self.silent_settlements += 1;
                }
                let sc = code::score(&self.store, &phi, !self.cfg.no_readout && !silent);
                let prob = sc.prob_of(&self.store, x as u32);
                out.bits = code::charge_bits(prob);
                self.total_bits += out.bits;

                if let Some((t, q)) = self.best_answer(&live_phi) {
                    out.top1 = Some(t);
                    out.correct = if self.cfg.commit_locks_charge {
                        self.committed_token == Some(x)
                    } else {
                        t == x
                    };
                    if !self.frozen {
                        self.store.answer_calib.push(q, t == x);
                    }
                }
                if want_entropy {
                    out.entropy_bits = Some(sc.entropy_bits());
                }

                out.idea_onset = self.covert_log.iter().find(|(_, t)| *t == x).map(|(k, _)| *k);
                out.speech_onset = self.overt_log.iter().find(|(_, t)| *t == x).map(|(k, _)| *k);
                self.covert_log.clear();
                self.overt_log.clear();

                // 2. Write. The write walk is deterministic and content-addressed
                //    -- a fixed number of hops routed by the state alone -- so it
                //    keeps the consistency the read path does not need.
                let gate = if self.frozen { 0.0 } else { self.emb.drive_norm(Some(x)) };
                if gate > 0.0 {
                    // The write takes its own deterministic, content-addressed
                    // walk: `hops` steps routed by the state alone, never
                    // perturbed. A3 is the whole point -- reads may wander,
                    // writes may not -- and writing into the read's own
                    // (perturbable) step instead meant reads and writes always
                    // landed on the same edge together, so the near-miss
                    // experiment had no inconsistency left to detect and its
                    // flat curve measured nothing.
                    // The write walk takes the *read's* query and the read's
                    // entry rule, differing only in that it is never perturbed.
                    //
                    // It used to route on `p0` while reads routed on `query()`.
                    // Under `RouteQuery::Bound` those are unrelated vectors, so
                    // the write landed on edges reads never traversed -- measured
                    // at chance coincidence -- and freezing the transforms could
                    // not cost anything whatever operator memory was worth. A3
                    // asks that writes be unperturbed, not that they go
                    // somewhere else.
                    let steps = if self.cfg.bypass_graph {
                        Vec::new()
                    } else {
                        let mut wq = Vec::with_capacity(self.cfg.d);
                        self.query(&mut wq);
                        let from = if self.cfg.read_entry_by_content {
                            None
                        } else {
                            Some(self.gnode)
                        };
                        self.graph.write_walk(&wq, &p0, self.cfg.hops, from)
                    };
                    self.store.write_surprise.push(out.bits);
                    let eta = self.cfg.eta * gate;
                    if !self.cfg.no_readout {
                        // The associative write: the same `Scored` the ledger
                        // charged, so the rows are fitted against the
                        // distribution that was actually settled.
                        let negs = self.sample_negatives(x as u32, self.cfg.neg_samples);
                        // The settled features, not the live ones. Under
                        // `commit_locks_charge` these differ: the charge came
                        // from the committed snapshot while the write used the
                        // live state, so the rows were fitted against a
                        // distribution the ledger never charged -- exactly what
                        // `code.rs` says must not happen. Identical when the
                        // flag is off, which is why it stayed hidden.
                        let sc_write = code::score(&self.store, &phi, !self.cfg.no_readout && !silent);
                        self.store.write(&sc_write, &phi, x as u32, &negs, eta);
                    }
                    if !self.cfg.no_eligibility {
                        // Gap-time reads get their share of the settlement
                        // through the traces they left, in O(1) per edge.
                        let seed: Vec<f32> = (0..self.cfg.d)
                            .map(|i| self.emb.row(x)[i] - p0[i])
                            .collect();
                        self.graph.credit_traces(&seed, eta * 0.25, 0.05);
                    }

                    // The operator write, on the write walk's own last step.
                    // Local: nothing crosses a hop.
                    // Every step, not only the last. With `hops = 2` only the
                    // second edge ever learned, and its `p_in` was the output of
                    // an untrained first hop -- a region of state space the read
                    // path, which takes one hop per tick, never produces.
                    for st in steps.iter() {
                        self.write_operator_on(st, x, eta);
                    }
                    if let Some(st) = steps.last() {
                        self.last_write_node = self.graph.head_of(st.edge);
                    }
                    self.content_writes += 1;
                    out.wrote = true;
                }

                // 3. Fold the observation into the background, the bound traces
                //    and the state, then take this tick's read.
                self.emb.rotated(x, Channel::In, &mut self.scratch);
                let v = self.scratch.clone();
                // Bind first. `observe_world` folds `E_x` into every band, so
                // binding afterwards convolved bands that already contained
                // `E_x` with `E_x` again -- an autocorrelation term in three of
                // the five bound blocks, which are also three fifths of the
                // routing query under `RouteQuery::Bound`. The trace should be
                // "the background as it stood, bound to what just arrived".
                self.rebind(x);
                self.ladder.observe_world(&v);
                self.feedback();
                self.ladder.refresh();
                self.emb.apply_operator(Some(x), &mut self.p);
                self.graph.decay_traces(self.cfg.trace_lambda);
                self.step_walk();

                self.ticks_since_event = 0;
                self.committed = None;
                self.committed_token = None;
                self.prev_answer_conf = 0.0;
                self.hops = 0;

                let scored = (out.top1, out.correct);
                let np = self.p.clone();
                let nphi = self.features(&np);
                self.emit(&mut out, &nphi);
                out.top1 = scored.0;
                out.correct = scored.1;
                // out.hops keeps the value captured at entry: the hops this
                // response had taken when the world spoke. Overwriting it with
                // zero here filed every charged event in the hops=0 bucket and
                // left the speed-accuracy curve -- the architecture's reason to
                // exist -- unmeasured while looking like a flat result.
                out.visit_entropy = self.visit_entropy();
                out
            }
        }
    }
}
