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
    last_self_token: Option<usize>,
    prev2: Option<usize>,
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
            cursors: vec![vec![0.0; d]; cfg_traj],
            cursor_age: vec![0; cfg_traj],
            cursor_w: vec![0.0; cfg_traj],
            cursor_steps: vec![0; cfg_traj],
            cursor_tok: vec![usize::MAX; cfg_traj],
            prev_answer: None,
            bank_planes: (0..24)
                .map(|j| crate::num::unit_vector(cfg_seed ^ 0xBA_11C5, j as u64, d))
                .collect(),
            last_self_token: None,
            bands_locked: vec![vec![0.0; d]; cfg_rungs],
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
            last_self_token: self.last_self_token,
            prev2: self.prev2,
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
        self.last_self_token = v.last_self_token;
        self.prev2 = v.prev2;
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
                    self.verify_rejects += 1;
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
                self.cursors[i] = clean;
                self.cursor_tok[i] = tok;
                self.cursor_w[i] = 1.0;
                self.cursor_steps[i] += 1;
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
    fn cleanup(&self, v: &[f32], out: &mut Vec<f32>) -> (bool, f32, usize) {
        let mut best = (self.cfg.cleanup_min_cos(), usize::MAX);
        let mut raw_best = 0.0f32;
        // No allocation and no norm: embeddings are unitary and therefore already
        // unit length, so the division was 2000 x 256 wasted operations per call
        // on top of a fresh Vec of every known token, once per response per
        // silent tick.
        for &t in self.store.known_slice() {
            let e = self.emb.row(t as usize);
            let c = crate::num::dot(v, e);
            if c > raw_best {
                raw_best = c;
            }
            if c > best.0 {
                best = (c, t as usize);
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

    /// What the model would say, scored the same way the ledger charges: the
    /// learned rows plus the direct comparison of every token to the state.
    fn best_answer(&self, phi: &[f32]) -> Option<(usize, f32)> {
        let d = self.cfg.d;
        let st: Vec<f32> = phi[..d.min(phi.len())].to_vec();
        code::score_with(
            &self.store,
            phi,
            !self.cfg.no_readout,
            Some(&st),
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
                let sc = code::score_with(&self.store, &phi, !self.cfg.no_readout && !silent, Some(&p0), self.cfg.readout_codebook, Some(&self.emb));
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
                        let sc_neg = code::score_with(&self.store, &phi, !self.cfg.no_readout, Some(&p0), self.cfg.readout_codebook, Some(&self.emb));
                        let negs = if self.cfg.hard_negatives {
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
                        let sc_write = code::score_with(&self.store, &phi, !self.cfg.no_readout && !silent, Some(&p0), self.cfg.readout_codebook, Some(&self.emb));
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
