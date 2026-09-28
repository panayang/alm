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
    /// Responses in flight are situation, not memory, and a probe must leave
    /// them exactly where it found them -- the same class of omission as
    /// `tick_index`, which an audit had to find.
    cursors: Vec<Vec<f32>>,
    cursor_age: Vec<u32>,
    cursor_w: Vec<f32>,
    cursor_steps: Vec<u32>,
    cursor_tok: Vec<usize>,
    cursor_hit: Vec<u64>,
    prev_answer: Option<(Vec<f32>, usize)>,
    last_self_token: Option<usize>,
    prev2: Option<usize>,
    /// The episode trace is situation: see `Config::episodic`.
    ep_trace: Vec<f32>,
    ep_hist: Vec<usize>,
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
    /// The superposed memory: bound triples, written on speech, read on silence.
    pub mem: Vec<Vec<f32>>,
    /// The token before last, so a triple can be formed without a boundary.
    prev2: Option<usize>,
    /// Learned gain on naming the episode trace's recall; memory, not situation.
    naming_gain: f32,
    naming_gate: Vec<f32>,
    /// This context's bindings, decayed per event and never renormalised.
    ep_trace: Vec<f32>,
    /// The two latest events, the trace's key.
    ep_hist: Vec<usize>,
    /// Responses in flight. Each stepped every tick; the oldest is re-seeded
    /// whenever the world speaks.
    cursors: Vec<Vec<f32>>,
    cursor_age: Vec<u32>,
    /// How live each response is. A step whose cleanup found something sets it
    /// to one; a step that found nothing lets it fade.
    ///
    /// Equal weights let a stalled response swamp a moving one: seeding happens
    /// on every world token, so a relation token becomes a cursor of its own,
    /// `r (o) (r (*) r)` retrieves nothing, cleanup refuses, and it sits at full
    /// strength forever. Measured, the mixture went to 0.63 on the relation while
    /// the response that had correctly reached `a1` fell to -0.03. Weighting is
    /// still superposition -- no resampling, no argmax -- and it is the same idea
    /// as committing when evidence stops rising.
    cursor_w: Vec<f32>,
    /// How many links each response has actually traversed.
    cursor_steps: Vec<u32>,
    /// The token each response has resolved to, which is what names its bank.
    cursor_tok: Vec<usize>,
    /// Tick of each response's last successful step, to break ties by recency.
    cursor_hit: Vec<u64>,
    /// The answer standing at the previous tick, so the commit rule can take the
    /// peak rather than the tick after it.
    prev_answer: Option<(Vec<f32>, usize)>,
    /// Fixed random hyperplanes whose sign bits name a bank.
    bank_planes: Vec<Vec<f32>>,
    /// The token the model itself last said. On a silent tick this is what
    /// arrives, which is what makes the self-output stream the thing that
    /// advances a chain rather than a decoration.
    last_self_token: Option<usize>,
    /// The background as it stood at the last event: the frozen half of the key.
    bands_locked: Vec<Vec<f32>>,
    /// Whether memory actually holds the key the lag blocks are a conjunction
    /// of. Recomputed when the event history changes, which is the only time it
    /// can change. See `Config::verify_gate`.
    bind_gate: f32,
    visit: Vec<f32>,
    hops: u32,
    /// The transform applied on the last tick, so the operator write is local
    /// to it and nothing has to cross a hop.
    last_step: Option<crate::graph::WalkStep>,
    /// Whether any response retrieved something that verified on this tick.
    /// Recomputed every silent tick before it is read, so it carries nothing
    /// from one tick to the next and does not belong in `Volatile`.
    retrieved: bool,
    /// Diagnostic: when set, every silent hop records how close its landing
    /// point is to the nearest codebook entry. Read by `judge::hop_cos`.
    pub log_hop_cos: bool,
    pub hop_cos_log: Vec<f32>,
    /// Diagnostic, alongside: (cosine, the token the hop lands nearest, the
    /// last token the world said). Separates a hop that falls back onto what
    /// was just said from one that arrives somewhere else.
    pub hop_tok_log: Vec<(f32, usize, usize)>,
    /// Diagnostic: silent hops attempted, and refused by the echo gate.
    pub silent_hops: u64,
    pub refused_hops: u64,

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
    /// Silent ticks that attempted an unbinding, and those whose result was
    /// close enough to a codebook entry to be accepted, with the summed best
    /// cosine. Three mechanism changes in a row moved no number at all; without
    /// this there was no way to tell a mechanism that does nothing from one
    /// whose output never reaches the charge.
    pub unbind_tries: u64,
    pub unbind_hits: u64,
    /// Retrievals that passed cleanup and failed the read-back check.
    pub verify_rejects: u64,
    pub unbind_cos: f64,
    pub mem_triples: u64,
    /// Cosine between the cursor and a named token, sampled only where it means
    /// something. Aggregates over every silent tick cannot show retrieval: most
    /// silent ticks have no triple to find and are *supposed* to be noise, so
    /// the mean is dominated by the population where failure is correct.
    pub probe_cursor_cos: f64,
    pub probe_cursor_n: u64,
    /// Cursor-to-answer cosine at each tick of a walk query's silence, indexed
    /// by how many ticks have passed. Measuring only at the answer tick cannot
    /// tell "never retrieved" from "retrieved and then stepped past", and the
    /// first link lands on the relation's own tick, so a depth-d walk wants
    /// exactly d-1 ticks of silence.
    pub cursor_trace: Vec<(f64, u64)>,
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
        let gain0 = cfg.episodic_codebook;
        let emb = Embeddings::new(&cfg);
        let ladder = Ladder::new(&cfg);
        let graph = Graph::new(&cfg);
        let store = Store::new(&cfg);
        let d = cfg.d;
        let blocks = cfg.bind_blocks();
        let cfg_rungs = cfg.rungs;
        let cfg_banks = cfg.mem_banks.max(1);
        let cfg_traj = cfg.traj.max(1);
        let cfg_seed = cfg.seed;
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
            mem: vec![vec![0.0; d]; cfg_banks],
            prev2: None,
            ep_trace: vec![0.0; d],
            ep_hist: Vec::new(),
            naming_gain: gain0,
            naming_gate: vec![0.0; d],
            cursors: vec![vec![0.0; d]; cfg_traj],
            cursor_age: vec![0; cfg_traj],
            cursor_w: vec![0.0; cfg_traj],
            cursor_steps: vec![0; cfg_traj],
            cursor_tok: vec![usize::MAX; cfg_traj],
            cursor_hit: vec![0; cfg_traj],
            prev_answer: None,
            bank_planes: (0..24)
                .map(|j| crate::num::unit_vector(cfg_seed ^ 0xBA_11C5, j as u64, d))
                .collect(),
            last_self_token: None,
            bands_locked: vec![vec![0.0; d]; cfg_rungs],
            bind_gate: 1.0,
            gnode: 0,
            visit: vec![0.0; n],
            hops: 0,
            last_step: None,
            retrieved: false,
            log_hop_cos: false,
            hop_cos_log: Vec::new(),
            hop_tok_log: Vec::new(),
            silent_hops: 0,
            refused_hops: 0,
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
            unbind_tries: 0,
            unbind_hits: 0,
            verify_rejects: 0,
            unbind_cos: 0.0,
            mem_triples: 0,
            probe_cursor_cos: 0.0,
            probe_cursor_n: 0,
            cursor_trace: vec![(0.0, 0); 10],
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
            cursors: self.cursors.clone(),
            cursor_age: self.cursor_age.clone(),
            cursor_w: self.cursor_w.clone(),
            cursor_steps: self.cursor_steps.clone(),
            cursor_tok: self.cursor_tok.clone(),
            cursor_hit: self.cursor_hit.clone(),
            prev_answer: self.prev_answer.clone(),
            last_self_token: self.last_self_token,
            prev2: self.prev2,
            ep_trace: self.ep_trace.clone(),
            ep_hist: self.ep_hist.clone(),
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
        self.cursors = v.cursors;
        self.cursor_age = v.cursor_age;
        self.cursor_w = v.cursor_w;
        self.cursor_steps = v.cursor_steps;
        self.cursor_tok = v.cursor_tok;
        self.cursor_hit = v.cursor_hit;
        self.prev_answer = v.prev_answer;
        self.last_self_token = v.last_self_token;
        self.prev2 = v.prev2;
        self.ep_trace = v.ep_trace;
        self.ep_hist = v.ep_hist;
        self.covert_log.clear();
        self.overt_log.clear();
    }

    /// One retention probe under frozen memory: the codelength it was charged
    /// and whether the answer was right. The displaced state is put back here,
    /// so a probe cannot leak into the stream that follows it.
    /// The i-th bound trace, so a test can check that what the readout is handed
    /// at an answer tick is the conjunction it is supposed to be.
    /// How close the cursor now stands to a given token.
    /// Per-response state, so a diagnostic can see which one is advancing
    /// instead of only their sum.
    pub fn cursor_report(&self, toks: &[usize]) -> Vec<(f32, u32, Vec<f32>)> {
        (0..self.cursors.len())
            .map(|i| {
                let cs = toks
                    .iter()
                    .map(|&t| {
                        let e = self.emb.row(t);
                        crate::num::dot(&self.cursors[i], e) / crate::num::norm(e).max(1e-9)
                    })
                    .collect();
                (self.cursor_w[i], self.cursor_age[i], cs)
            })
            .collect()
    }

    pub fn cursor_cos(&self, tok: usize) -> f32 {
        let e = self.emb.row(tok);
        crate::num::dot(&self.p, e) / crate::num::norm(e).max(1e-9)
    }

    /// Record that cosine for a token the caller knows to be the right answer.
    /// Record the cursor's distance to a known answer at silence offset `off`.
    pub fn note_cursor_at(&mut self, tok: usize, off: usize) {
        let c = self.cursor_cos(tok) as f64;
        let i = off.min(self.cursor_trace.len() - 1);
        self.cursor_trace[i].0 += c;
        self.cursor_trace[i].1 += 1;
    }

    pub fn note_cursor(&mut self, tok: usize) {
        self.probe_cursor_cos += self.cursor_cos(tok) as f64;
        self.probe_cursor_n += 1;
    }

    /// Everything a probe must leave exactly as it found it, flattened so a
    /// test can compare it in one equality.
    pub fn snapshot_debug(&self) -> (Vec<u32>, Vec<u32>, Option<usize>, Option<usize>, u64) {
        let cur: Vec<u32> = self
            .cursors
            .iter()
            .flat_map(|c| c.iter().map(|v| v.to_bits()))
            .collect();
        (cur, self.cursor_age.clone(), self.last_self_token, self.prev2, self.tick_index)
    }

    pub fn bound_block(&self, i: usize) -> &[f32] {
        &self.binds[i]
    }

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
    /// The token the episode trace's recall block names now, and its cosine;
    /// None when it names nothing. A diagnostic with a known answer: it reads
    /// the recall without the readout in between.
    pub fn episodic_recall(&self) -> Option<(usize, f32)> {
        if !self.cfg.episodic || self.binds.is_empty() {
            return None;
        }
        let r = self.binds.last().unwrap();
        let mut clean = Vec::new();
        let (ok, cos, tok) = self.cleanup(r, &mut clean);
        if ok { Some((tok, cos)) } else { None }
    }

    /// What the codebook term names from: the state, plus the episode trace's
    /// recall when there is one. The recall is a token, so it is named the way
    /// the state is -- through the codebook, which needs no row -- and a value
    /// heard once in this context can be said back without having been learned.
    fn naming(&self, p: &[f32], phi: &[f32]) -> Vec<f32> {
        let mut out = p.to_vec();
        let w = self.naming_weight(p);
        if self.cfg.episodic && w != 0.0 {
            let d = self.cfg.d;
            let at = self.cfg.bind_blocks() * d;
            if phi.len() >= at + d {
                for i in 0..d.min(out.len()) {
                    out[i] += w * phi[at + i];
                }
            }
        }
        out
    }

    /// Only active with `Config::episodic`, which is off by default. Note what
    /// kind of learner this is before extending it: a gain and a state gate
    /// fitted by the gradient of the charge. The design sanctions one such
    /// learner, the readout rows; this is a second.
    ///
    /// The gain on the recall's naming term, fitted like the readout: the
    /// exact gradient of the charge on the token the world said. It rises when
    /// what the trace recalls is what gets said, and falls when it is not, so
    /// whether recall is worth naming -- and how loudly -- is learned rather
    /// than set. A fixed gain of 1 left a correct recall (0.91-1.00) unsaid
    /// against learned rows, and the answer was mostly `<none>`.
    fn learn_naming_gain(&mut self, sc: &code::Scored, phi: &[f32], target: usize, eta: f32) {
        if !self.cfg.episodic || self.frozen {
            return;
        }
        let d = self.cfg.d;
        let at = self.cfg.bind_blocks() * d;
        if phi.len() < at + d {
            return;
        }
        let rec = &phi[at..at + d];
        if crate::num::norm(rec) < 1e-6 {
            return;
        }
        let mut expect = 0.0f32;
        for &(tok, e) in sc.rows.iter() {
            expect += (e / sc.z) * crate::num::dot(self.emb.row(tok as usize), rec);
        }
        let grad = crate::num::dot(self.emb.row(target), rec) - expect;
        if self.naming_weight(&phi[..d]) <= 0.0 && grad < 0.0 {
            return;
        }
        self.naming_gain += eta * grad;
        for i in 0..d {
            self.naming_gate[i] += eta * grad * phi[i];
        }
    }

    /// How loudly to name the recall here: a gain plus a linear gate on the
    /// state, floored at zero. A single gain learned from every event settled
    /// near 2, because whenever a slot is set anew the recall at that moment is
    /// the old value -- while at a question the same recall is exactly right.
    /// Whether recall is worth saying depends on where one is, and the state
    /// is where one is.
    fn naming_weight(&self, p: &[f32]) -> f32 {
        (self.naming_gain + crate::num::dot(&self.naming_gate, &p[..self.naming_gate.len().min(p.len())])).max(0.0)
    }

    /// The gain on the recall's naming term as it stands.
    pub fn naming_gain(&self) -> f32 {
        self.naming_gain
    }

    /// The episode trace's recall block as it stands (diagnostic).
    pub fn episodic_block(&self) -> Option<&[f32]> {
        if self.cfg.episodic { self.binds.last().map(|v| v.as_slice()) } else { None }
    }

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
    ///
    /// Scored exactly the way `emit` scores it, codebook term included. It used
    /// to call `code::score`, which omits that term, so the distribution this
    /// reported -- and the one the normalisation assertion checked -- was not
    /// the distribution the model emits.
    pub fn spread_now(&self) -> code::Scored {
        let p = self.p.clone();
        let phi = self.features(&p);
        code::score_with(
            &self.store,
            &phi,
            !self.cfg.no_readout,
            Some(&self.naming(&p, &phi)),
            self.cfg.readout_codebook,
            Some(&self.emb),
        )
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
        // The lag blocks are conjunctions of a key. When memory reports that
        // the key was never written, what the unbinding of that address returns
        // is a neighbour's content, and presenting it to the readout as though
        // it were stored content is the readout leaning on something that is
        // not there. The self block and the band blocks are unaffected: neither
        // stands on a key.
        let lag0 = if self.cfg.bind_self { 1 } else { 0 };
        let lag_hi = lag0
            + if matches!(self.cfg.bind_mode, BindMode::EventLag | BindMode::Both) {
                self.cfg.bind_lags
            } else {
                0
            };
        for (i, b) in self.binds.iter().enumerate() {
            if self.cfg.verify_gate && i >= lag0 && i < lag_hi && self.bind_gate == 0.0 {
                f.extend(std::iter::repeat(0.0f32).take(b.len()));
            } else {
                f.extend_from_slice(b);
            }
        }
        for k in 0..self.cfg.rungs {
            if self.cfg.event_locked_key && !self.bands_locked.is_empty() {
                f.extend_from_slice(&self.bands_locked[k]);
            } else {
                f.extend_from_slice(self.ladder.delta(k));
            }
        }
        f
    }

    /// Snapshot the background. Called on event ticks only, so the retrieval key
    /// stops moving the moment the world does.
    /// Which superposition a bound pair belongs to.
    ///
    /// The same pair must reach the same bank whether it is being written or
    /// asked about, so the bank is a function of the pair's content and of
    /// nothing else -- no state, no trajectory, no elapsed time. Fixed random
    /// keys, never trained, as the reference mechanism has it.
    /// Which superposition a link belongs to, computed from the two token
    /// identities rather than from a vector.
    ///
    /// A vector-derived address has to choose between two bad options. Sign bits
    /// against random hyperplanes are locality sensitive, so an unwritten key
    /// lands among *neighbours* whose content passes both cleanup and a read-back
    /// check -- near misses become undetectable, which is this family's central
    /// weakness. Scrambling the code destroys that locality and immediately makes
    /// the address brittle: write and read compute the same convolution by
    /// different floating-point paths, a projection near zero flips, and one
    /// flipped bit now means a completely unrelated bank rather than a nearby
    /// one.
    ///
    /// There is no need to choose. Cleanup resolves the cursor to an exact
    /// codebook token, and the write already has both token ids, so the address
    /// can be a hash of the identities: exact, non-locality-sensitive, and
    /// cheaper than any projection. The vectors carry the superposition; the
    /// identities carry the address.
    pub fn bank_of_ids(&self, a: usize, b: usize) -> usize {
        let n = self.mem.len();
        if n <= 1 {
            return 0;
        }
        let mut z = (a as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z % n as u64) as usize
    }

    /// Step every response in flight one link, against the token that arrived.
    ///
    /// The token is the world's when it speaks and the last relation the world
    /// said when it does not, so the same rule runs on every tick and nothing has
    /// to know whether a challenge is pending. The relation is the operator and
    /// stays on the table; the response is where it has got to.
    fn step_cursors(&mut self, arriving: usize) {
        if !self.cfg.superpose {
            return;
        }
        let d = self.cfg.d;
        let tok = self.emb.row(arriving).to_vec();
        for i in 0..self.cursors.len() {
            if self.cursor_w[i] <= 1e-3 || self.cursor_tok[i] == usize::MAX {
                self.cursor_age[i] += 1;
                continue;
            }
            let mut q = vec![0.0f32; d];
            crate::num::circconv(&self.cursors[i], &tok, &mut q);
            normalize(&mut q);
            let bank = self.bank_of_ids(self.cursor_tok[i], arriving);
            let mut raw = vec![0.0f32; d];
            crate::num::unbind(&self.mem[bank], &q, &mut raw);
            normalize(&mut raw);
            let mut clean = Vec::new();
            let (mut ok, cos, tok) = self.cleanup(&raw, &mut clean);
            if ok && self.cfg.verify_sigma > 0.0 {
                // Read-back: bind the candidate onto the key and ask whether that
                // triple is in this bank at all. A neighbour's content does not
                // verify, because nobody ever wrote it.
                let mut back = vec![0.0f32; d];
                crate::num::circconv(&q, &clean, &mut back);
                normalize(&mut back);
                let mnorm = crate::num::norm(&self.mem[bank]).max(1e-9);
                let score = crate::num::dot(&self.mem[bank], &back) / mnorm;
                if score < self.cfg.verify_min() {
                    ok = false;
                    if !self.frozen {
                        self.verify_rejects += 1;
                    }
                }
            }
            if !self.frozen {
                self.unbind_tries += 1;
                self.unbind_cos += cos as f64;
            }
            if ok {
                if !self.frozen {
                    self.unbind_hits += 1;
                }
                self.retrieved = true;
                self.cursors[i] = clean;
                self.cursor_tok[i] = tok;
                self.cursor_w[i] = 1.0;
                self.cursor_steps[i] += 1;
                self.cursor_hit[i] = self.tick_index;
            } else if self.cursor_steps[i] == 0 {
                // Never produced anything: this is a response that was seeded on
                // something with nowhere to go, like a relation token, and it
                // should fade.
                self.cursor_w[i] *= self.cfg.cursor_fade;
            }
            // A response that has traversed at least one link and then stops has
            // *arrived*. Fading it is fading the answer: from inside, "the chain
            // ended" and "the chain never started" look identical, and the only
            // thing that separates them is whether anything was ever retrieved.
            self.cursor_age[i] += 1;
        }
        if self.cfg.emit_strongest {
            // Several responses sit at weight 1.0 whenever more than one stepped
            // successfully, and `>` then hands it to index 0 every time -- a
            // systematic bias toward a slot rather than toward the most matured
            // answer. Recency of the last successful step is the tie-break that
            // means what "most matured" is supposed to mean.
            let mut best = (0.0f32, 0u64, usize::MAX);
            for i in 0..self.cursors.len() {
                let (w, h) = (self.cursor_w[i], self.cursor_hit[i]);
                if w > best.0 + 1e-6 || ((w - best.0).abs() <= 1e-6 && h > best.1) {
                    best = (w, h, i);
                }
            }
            let best = (best.0, best.2);
            if best.1 != usize::MAX && best.0 > 1e-3 {
                self.p.copy_from_slice(&self.cursors[best.1]);
                normalize(&mut self.p);
            }
            return;
        }
        let mut mix = vec![0.0f32; d];
        for (i, c) in self.cursors.iter().enumerate() {
            let w = self.cursor_w[i];
            if w <= 1e-3 {
                continue;
            }
            for j in 0..d {
                mix[j] += w * c[j];
            }
        }
        if crate::num::norm(&mix) > 1e-6 {
            normalize(&mut mix);
            self.p = mix;
        }
    }

    /// The world spoke: retire the least live response and start a new one here.
    fn seed_cursor(&mut self, tok: usize) {
        if !self.cfg.superpose || self.cursors.is_empty() {
            return;
        }
        // The world spoke, so every answer in flight stops being the answer.
        //
        // "Arrived responses do not fade" has to mean "do not fade *through
        // silence*". Left unqualified it means never, and then the slots fill
        // with stale answers at full weight, the mixture becomes a sum of
        // unrelated tokens, and the one that matters is diluted -- which is
        // exactly what happened when the rule was added. Holding through silence
        // and yielding to speech is the same picture as holding a half-formed
        // reply while the other person is quiet and dropping it when they say
        // something new.
        for i in 0..self.cursors.len() {
            self.cursor_w[i] *= self.cfg.cursor_fade;
        }
        let mut worst = 0usize;
        for i in 1..self.cursors.len() {
            let (wi, wb) = (self.cursor_w[i], self.cursor_w[worst]);
            if wi < wb - 1e-6
                || ((wi - wb).abs() <= 1e-6 && self.cursor_age[i] > self.cursor_age[worst])
            {
                worst = i;
            }
        }
        self.cursors[worst].copy_from_slice(self.emb.row(tok));
        self.cursor_tok[worst] = tok;
        normalize(&mut self.cursors[worst]);
        self.cursor_age[worst] = 0;
        self.cursor_w[worst] = 1.0;
        self.cursor_steps[worst] = 0;
    }

    fn lock_bands(&mut self) {
        if !self.cfg.event_locked_key {
            return;
        }
        for k in 0..self.cfg.rungs {
            self.bands_locked[k].copy_from_slice(self.ladder.delta(k));
        }
    }

    /// Snap an unbound result to the nearest token the store has a row for, if
    /// anything is near enough. Returns false when nothing is, which is the
    /// signal that the chain has run out of links.
    /// Does memory hold anything under this key?
    ///
    /// Unbind the bank the pair addresses, clean the result to a codebook
    /// entry, bind it back and ask the bank whether that triple is there. The
    /// three steps are the ones `step_cursors` already takes; what is new is
    /// only that the answer reaches the features.
    ///
    /// This is a question about storage and not about probability. It needs no
    /// counts, no sample space and no baseline -- an address was written or it
    /// was not, and one convolution and one dot product settle it.
    fn key_is_written(&self, prev2: usize, prev: usize) -> bool {
        if !self.cfg.superpose || self.mem.is_empty() {
            return true;
        }
        let d = self.cfg.d;
        let mut q = vec![0.0f32; d];
        crate::num::circconv(self.emb.row(prev2), self.emb.row(prev), &mut q);
        normalize(&mut q);
        let bank = self.bank_of_ids(prev2, prev);
        let mut raw = vec![0.0f32; d];
        crate::num::unbind(&self.mem[bank], &q, &mut raw);
        normalize(&mut raw);
        let mut clean = Vec::new();
        let (ok, _, _) = self.cleanup(&raw, &mut clean);
        if !ok {
            return false;
        }
        let mut back = vec![0.0f32; d];
        crate::num::circconv(&q, &clean, &mut back);
        normalize(&mut back);
        let mn = crate::num::norm(&self.mem[bank]).max(1e-9);
        crate::num::dot(&self.mem[bank], &back) / mn >= self.cfg.verify_min()
    }

    fn cleanup(&self, v: &[f32], out: &mut Vec<f32>) -> (bool, f32, usize) {
        let mut best = (self.cfg.cleanup_min_cos(), usize::MAX);
        let mut raw_best = 0.0f32;
        // No allocation and no norm: embeddings are unitary and therefore already
        // unit length, so the division was 2000 x 256 wasted operations per call
        // on top of a fresh Vec of every known token, once per response per
        // silent tick.
        //
        // Over the whole codebook, not over the tokens that have readout rows.
        // It iterated `store.known_slice()`, which is the same mistake the
        // scoring made and had to be corrected for: naming a token cannot
        // require that the token was written before. With the readout switched
        // off no token has a row, so cleanup could never succeed and every
        // superposed retrieval in that configuration was dead -- measured as a
        // cosine of exactly 0.000 on every one of 36000 silent hops. Under the
        // default dense write every token acquires a row quickly, which is why
        // it never showed.
        for t in 0..self.cfg.vocab {
            let e = self.emb.row(t);
            let c = crate::num::dot(v, e);
            if c > raw_best {
                raw_best = c;
            }
            if c > best.0 {
                best = (c, t);
            }
        }
        if best.1 == usize::MAX {
            return (false, raw_best, usize::MAX);
        }
        out.clear();
        out.extend_from_slice(self.emb.row(best.1));
        normalize(out);
        (true, best.0, best.1)
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
        // Lag zero: the event bound with the convolution identity, which is the
        // event. See `Config::bind_self` -- the family the lag blocks form
        // started at lag one, so the parts of a conjunction were never present
        // alongside it.
        if self.cfg.bind_self {
            let mut b = ex.clone();
            normalize(&mut b);
            self.binds[slot] = b;
            slot += 1;
        }
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
        if self.cfg.episodic {
            // Keyed by the ordered pair of the two latest events. At each event: recall what followed the pair that
            // now stands -- (previous, x) -- earlier in this context; then write
            // x under the pair that preceded it, (before-previous, previous).
            //
            // A key written again is replaced rather than superposed with its
            // old value. With plain accumulation a slot set twice recalled
            // either value about equally, and the recall named the right one
            // 0.44 of the time on a value set in the same turn.
            //
            // The first factor is dilated, i -> 3 i mod d, before it is bound.
            // A cyclic shift will not do: it commutes with convolution, so a
            // shifted key cancels algebraically against the next one and every
            // recall came back, exactly and at cosine 1.0, as the token two
            // events back. A dilation by an odd factor permutes the frequencies,
            // so it keeps a unitary vector unitary and the unbinding exact, and
            // D(a) is unrelated to a, which makes the pair directional -- except
            // at the frequencies the dilation fixes, k (u - 1) = 0 mod d. The
            // factor is 3 because that fixes only k = 0 and d/2; 97 fixed 32 of
            // 256, and the token two events back still came through.
            let key = |a: usize, b: usize, emb: &crate::embed::Embeddings| -> Vec<f32> {
                let ea = emb.row(a);
                let mut da = vec![0.0f32; d];
                for i in 0..d {
                    da[(i * 3) % d] = ea[i];
                }
                let mut k = vec![0.0f32; d];
                crate::num::circconv(&da, emb.row(b), &mut k);
                normalize(&mut k);
                k
            };
            let prev = self.ep_hist.first().copied();
            let prev2 = self.ep_hist.get(1).copied();
            if let (Some(a), Some(b)) = (prev2, prev) {
                let k = key(a, b, &self.emb);
                let lam = self.cfg.episodic_decay;
                for v in self.ep_trace.iter_mut() {
                    *v *= lam;
                }
                // Replace, do not superpose: if the key already names a token,
                // take that token's binding out by the amount it is present.
                // Only the named token -- with a unitary key, unbinding is an
                // invertible rotation of the whole trace, not a projection onto
                // one key, so subtracting the raw recall (the delta rule as the
                // readout uses it) erased everything each time it wrote.
                let mut old = vec![0.0f32; d];
                crate::num::unbind(&self.ep_trace, &k, &mut old);
                let mut clean = Vec::new();
                let mut normed = old.clone();
                normalize(&mut normed);
                let (ok, _, _) = self.cleanup(&normed, &mut clean);
                let mut w = vec![0.0f32; d];
                let mut val = ex.clone();
                if ok {
                    let amount = crate::num::dot(&old, &clean);
                    for i in 0..d {
                        val[i] -= amount * clean[i];
                    }
                }
                crate::num::circconv(&val, &k, &mut w);
                for i in 0..d {
                    self.ep_trace[i] += w[i];
                }
            }
            let mut r = vec![0.0f32; d];
            if let Some(b) = prev {
                crate::num::unbind(&self.ep_trace, &key(b, x, &self.emb), &mut r);
                normalize(&mut r);
            }
            // Name it if it is nameable; otherwise leave it as the mixture it is.
            let mut clean = Vec::new();
            let (ok, _, _) = self.cleanup(&r, &mut clean);
            // A recall that names nothing is left out rather than passed on as
            // noise: the block is also what names the recalled token below.
            self.binds[slot] = if ok { clean } else { vec![0.0f32; d] };
            self.ep_hist.insert(0, x);
            self.ep_hist.truncate(2);
        }
        self.event_hist.insert(0, x);
        self.event_hist.truncate(self.cfg.bind_lags.max(1));
        // The key the lag blocks stand on has just changed, and it cannot
        // change again until the world speaks again.
        self.bind_gate = if !self.cfg.verify_gate {
            1.0
        } else {
            match (self.event_hist.first().copied(), self.prev2) {
                (Some(p1), Some(p2)) if self.key_is_written(p2, p1) => 1.0,
                (Some(_), Some(_)) => 0.0,
                _ => 1.0,
            }
        };
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
        if !self.frozen {
            self.silent_hops += 1;
        }
        let cur = self.p.clone();
        let st = self.graph.hop(a, &cur);
        let mut cand = st.p_out.clone();
        if self.cfg.anchor > 0.0 {
            let b = self.cfg.anchor;
            for k in 0..self.cfg.rungs {
                let band = self.ladder.delta(k);
                for i in 0..self.cfg.d {
                    cand[i] += b * band[i];
                }
            }
            normalize(&mut cand);
        }
        if self.log_hop_cos && !self.frozen {
            let mut clean = Vec::new();
            let (_, cos, _) = self.cleanup(&cand, &mut clean);
            self.hop_cos_log.push(cos);
            // The nearest token regardless of the acceptance threshold.
            let mut best = (f32::MIN, 0usize);
            for t in 0..self.cfg.vocab {
                let c = crate::num::dot(&cand, self.emb.row(t));
                if c > best.0 {
                    best = (c, t);
                }
            }
            let last = self.event_hist.first().copied().unwrap_or(usize::MAX);
            self.hop_tok_log.push((best.0, best.1, last));
        }
        // OFF BY DEFAULT, and the account below is the record of a mistake.
        // Everything from here to the gate itself argues for refusing hops that
        // collapse an ungrounded state; the measurement that decided against it
        // came afterwards and is in `Config::walk_needs_retrieval`: the "drift"
        // this was built to stop was the silence making an over-confident,
        // ungrounded answer less over-confident (ECE 0.308 -> 0.184), read as
        // harm because accuracy at 0.5 was the measure.
        //
        // A silent hop that would collapse the state is not taken unless
        // something was retrieved.
        //
        // Measured where hops land, as cosine to the nearest codebook entry:
        // a walk answering a deterministic relation on its own lands at a
        // median of 0.571, the grounded judge at 0.516, and the judge with its
        // history wiped -- where the hops have nothing under them -- at 0.831.
        // The useful hops are a mixture; the drifting ones have fallen onto a
        // single token, and the codebook term in the score then reads that
        // collapse as a confident answer. That is what produced 0.74 on a first
        // choice that was right half the time.
        //
        // A state superposing k components sits near 1/sqrt(k) to each, so a
        // cosine above 1/sqrt(2) means more than half its energy lies in one
        // direction: fewer than two components, which is collapse. The
        // threshold is arithmetic, not fitted, and it falls in the gap between
        // the two distributions. Naming an answer is what the cursors are for;
        // the state's own walk does not get to name one without evidence.
        //
        // Two earlier gates were tried and are recorded because both failed.
        // Gating on the bank's retrievals alone stopped the drift and killed
        // the walk's reading of operator memory (5.78 bits to 6.03 on the
        // walk-alone relation). Gating on cleanup succeeding had the direction
        // backwards -- it would admit exactly the collapsed hops.
        //
        // And not every collapse: only a collapse back onto what the world
        // just said. Refusing every collapse stopped the drift and cost
        // composition 2.3 times over (0.096 to 0.223 bits on a new pair),
        // because the walk that answers a composition converges onto the answer,
        // which is a collapse too. Where collapsing hops actually land:
        //
        //   composition, after the cue    3.0% the last token   96.8% the answer
        //   judge, grounded               5.4% the last token   82.2% the answer
        //   judge, history wiped        100.0% the last token    0.0% the answer
        //
        // Every one of 23351 ungrounded collapses fell back onto the token just
        // said. That is an echo, not a read: with nothing under it the walk can
        // only return its input. A grounded walk arrives somewhere else.
        if self.cfg.walk_needs_retrieval && !self.retrieved {
            let mut clean = Vec::new();
            let (ok, cos, tok) = self.cleanup(&cand, &mut clean);
            let echo = ok && self.event_hist.first().copied() == Some(tok);
            if echo && cos > std::f32::consts::FRAC_1_SQRT_2 {
                if !self.frozen {
                    self.refused_hops += 1;
                }
                return;
            }
        }
        self.p = cand;

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

    /// What the model would say, scored the same way the ledger charges: the
    /// learned rows plus the direct comparison of every token to the state.
    fn best_answer(&self, phi: &[f32]) -> Option<(usize, f32)> {
        let d = self.cfg.d;
        let st: Vec<f32> = phi[..d.min(phi.len())].to_vec();
        code::score_with(
            &self.store,
            phi,
            !self.cfg.no_readout,
            Some(&self.naming(&st, &phi)),
            self.cfg.readout_codebook,
            Some(&self.emb),
        )
        .top()
        .map(|(t, q)| (t as usize, q))
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
        // What the model just said is what arrives next tick if the world stays
        // quiet. Covert rather than overt: the founding requirement is that the
        // model is always outputting and that its own output is context, not
        // that it has to commit out loud before it may think against it.
        self.last_self_token = Some(t);
        out.top1 = Some(t);
        self.covert_log.push((self.ticks_since_event, t));

        // Speak when the answer has stopped improving. A calibrated-confidence
        // threshold cannot work here: it pins at its ceiling whenever the model
        // is over-confident anywhere, and then the channel never fires, never
        // generates calibration data, and stays shut.
        //
        // What this rule is, measured: a *timing* signal, not a *sufficiency*
        // signal. On the stateful judge it commits on 100% of questions with the
        // history wiped, where the answer is right half the time, exactly as
        // often as with the history present. "The evidence stopped moving" is
        // not "the evidence is enough", and this rule must not be read as an
        // accept-or-escalate decision. What does sort accepted answers from
        // unsafe ones is the probability itself, which under natural forgetting
        // is calibrated (confidence 0.789 against accuracy 0.831 in the hardest
        // bucket, 0.959 against 0.997 in the most faded); a caller that needs
        // to decline should threshold that, as a decision model's caller does.
        //
        // A counterfactual remedy was tried and rejected on measurement:
        // scoring the situation by how far it moves the answer from what memory
        // alone would say. Accepting the most-evidenced 80% kept 0.908 correct
        // against 0.987 for the most confident 80%, because a correct answer
        // that agrees with the prior carries little evidence by construction.
        let rising = q > self.prev_answer_conf + 1e-4;
        let settled = !rising && q >= self.cfg.speak_fallback;
        self.prev_answer_conf = q;

        // Commit the answer that was standing when evidence stopped rising, not
        // the one standing after it stopped. The peak is the tick before the
        // fall, so committing the current tick's answer commits a response that
        // has already stepped past itself -- measured on the real stream, the
        // cursor sits at 0.193 with the answer one tick after the cues and at
        // 0.059 the tick after that.
        let held = self.prev_answer.take();
        self.prev_answer = Some((phi.to_vec(), t));
        if settled && self.committed.is_none() {
            if let Some((pphi, ptok)) = held {
                self.committed = Some(pphi);
                self.committed_token = Some(ptok);
                self.commitments += 1;
                self.overt = Some(ptok);
                out.overt = Some(ptok);
                self.overt_emissions += 1;
                self.overt_log.push((self.ticks_since_event, ptok));
                return;
            }
        }
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

    /// The `k` tokens the model currently ranks highest, excluding the target:
    /// the ones actually competing for this answer.
    fn top_negatives(&self, sc: &code::Scored, target: u32, k: usize) -> Vec<u32> {
        let mut v: Vec<(u32, f32)> =
            sc.rows.iter().filter(|(t, _)| *t != target).copied().collect();
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });
        v.truncate(k);
        v.into_iter().map(|(t, _)| t).collect()
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

                // The world is quiet, so what arrives is what the model itself
                // last said. There is no "is a challenge pending" to answer:
                // every tick has a token, and the self-output stream is how a
                // chain advances rather than a channel bolted alongside one.
                // The relation is the operator and stays on the table; the
                // model's own output is where it has got to, which is the
                // cursor itself. Stepping with the self-token instead computed
                // `a1 (*) a1` once the cursor had already become a1, so the
                // first link worked and the chain never advanced past it.
                self.retrieved = false;
                if let Some(t) = self.event_hist.first().copied() {
                    // Step the responses, but do NOT rebind. `step_cursors` uses
                    // `cursor (*) E_arriving` and never reads `binds`, so binding
                    // the self-token here did nothing for the mechanism and
                    // overwrote the world's conjunction, which is what the Latin
                    // and product families are answered from. Distinguish and
                    // combine: the self stream drives retrieval, the world's
                    // stream owns the key.
                    self.step_cursors(t);
                }

                // The walk moves the state only on a tick that read something.
                //
                // It used to hop every silent tick whatever the cursors found.
                // Measured with the episode wiped before a question, so that
                // nothing could be retrieved: the answer the moment it was asked
                // sat at the base rate, 0.644, and six silent ticks later it had
                // drifted to 0.555, a mean movement of 0.333 in P(YES) with
                // nothing under it. Switching the self-feedback channels off one
                // at a time changed little; the walk, or all feedback together
                // (which is what drives it), took the drift to 0.011. The
                // grounded judge did not care either way.
                //
                // Silence is for reading memory -- unbind on silence is the half
                // of the tick that speech does not do. A hop that retrieved
                // nothing is not a read; it is the state being transformed with
                // no evidence, and what it moves is the answer. So the state
                // holds when nothing verified, which is also what "a finished
                // walk holds its answer through surplus silence" already asks.
                if self.cfg.walk_during_gap
                    && (!self.cfg.walk_needs_retrieval || self.retrieved)
                {
                    self.step_walk();
                }

                let p = self.p.clone();
                let phi = self.features(&p);
                if want_entropy {
                    // The same scoring `emit` is about to use, codebook term
                    // included. Reporting the entropy of a distribution the
                    // model does not emit is a diagnostic about nothing.
                    let sc = code::score_with(
                        &self.store,
                        &phi,
                        !self.cfg.no_readout,
                        Some(&self.naming(&p, &phi)),
                        self.cfg.readout_codebook,
                        Some(&self.emb),
                    );
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
                let sc = code::score_with(&self.store, &phi, !self.cfg.no_readout && !silent, Some(&self.naming(&p0, &phi)), self.cfg.readout_codebook, Some(&self.emb));
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
                        let sc_neg = code::score_with(&self.store, &phi, !self.cfg.no_readout, Some(&self.naming(&p0, &phi)), self.cfg.readout_codebook, Some(&self.emb));
                        // Zero means every token the distribution scored: the
                        // exact softmax gradient, against exactly the
                        // distribution the ledger charged. The documentation
                        // always said zero recovered the dense update; the code
                        // took it to mean no negatives at all.
                        let negs = if self.cfg.neg_samples == 0 {
                            sc_neg.rows.iter().map(|(t, _)| *t).filter(|t| *t != x as u32).collect()
                        } else if self.cfg.hard_negatives {
                            self.top_negatives(&sc_neg, x as u32, self.cfg.neg_samples)
                        } else {
                            self.sample_negatives(x as u32, self.cfg.neg_samples)
                        };
                        // The settled features, not the live ones. Under
                        // `commit_locks_charge` these differ: the charge came
                        // from the committed snapshot while the write used the
                        // live state, so the rows were fitted against a
                        // distribution the ledger never charged -- exactly what
                        // `code.rs` says must not happen. Identical when the
                        // flag is off, which is why it stayed hidden.
                        let sc_write = code::score_with(&self.store, &phi, !self.cfg.no_readout && !silent, Some(&self.naming(&p0, &phi)), self.cfg.readout_codebook, Some(&self.emb));
                        self.store.write(&sc_write, &phi, x as u32, &negs, eta);
                        self.learn_naming_gain(&sc_write, &phi, x, eta);
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
                // Speech: bind. The standing conjunction of the two previous
                // observations is combined with what just arrived and superposed
                // into memory. For a presented link `x r y` the triple formed at
                // the `y` tick is exactly (E_x (*) E_r) (*) E_y, and no episode
                // boundary was needed to know that.
                if self.cfg.superpose {
                    if let (Some(p2), Some(&p1)) = (self.prev2, self.event_hist.first()) {
                        let d = self.cfg.d;
                        let mut pair = vec![0.0f32; d];
                        crate::num::circconv(self.emb.row(p2), self.emb.row(p1), &mut pair);
                        normalize(&mut pair);
                        let mut tri = vec![0.0f32; d];
                        crate::num::circconv(&pair, self.emb.row(x), &mut tri);
                        normalize(&mut tri);
                        let bank = self.bank_of_ids(p2, p1);
                        // Accumulate. Do NOT renormalise here.
                        //
                        // `M <- nu(M + t)` with both unit and near-orthogonal
                        // gives |M + t| ~ sqrt(2), so every write shrank
                        // everything already stored by 1/sqrt(2) and the first
                        // item's weight after k writes was 2^(-k/2): an
                        // effective horizon of two or three triples across a run
                        // of ten thousand events. A superposition that forgets
                        // in three steps is a sliding window, and the point of
                        // superposing is that it is not one. Unbinding compares
                        // by cosine, which is scale-free, so the plain sum needs
                        // no normalisation at all.
                        for i in 0..d {
                            self.mem[bank][i] += tri[i];
                        }
                        self.mem_triples += 1;
                    }
                    self.prev2 = self.event_hist.first().copied();
                    // The cursor is the second-to-last observation. When the
                    // relation arrives it is therefore the entity, which is what
                    // the first unbinding has to be asked about.
                    if let Some(&prev) = self.event_hist.first() {
                        self.p.copy_from_slice(self.emb.row(prev));
                        normalize(&mut self.p);
                    }
                }
                self.rebind(x);
                self.step_cursors(x);
                self.seed_cursor(x);
                self.ladder.observe_world(&v);
                self.feedback();
                self.ladder.refresh();
                self.lock_bands();
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
