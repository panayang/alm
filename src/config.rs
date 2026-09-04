//! One configuration struct for the whole system.
//!
//! Every field is tagged with its status from the parameter audit. The point of
//! keeping the tags in the source is that adding a *free* knob should feel
//! expensive: the design claim is that exactly one genuinely free parameter was
//! added relative to the reference mechanism, and this file is where that claim
//! is either honoured or quietly broken.
//!
//!   FREE      -- a real knob, swept or chosen.
//!   DERIVED   -- computed from stream statistics or from another field.
//!   CEILING   -- a reservation, not a setting; growth happens below it.
//!   INHERITED -- carried over from the reference mechanism unchanged.

/// What the bound trace binds.
///
/// The first version bound the *accumulated payload* with the arriving token.
/// By bilinearity that spreads the pair term over cross-terms with everything
/// else in the payload -- including tokens from previous episodes -- and the
/// payload has been through the operator's rank-one rotations besides, so it is
/// not even a clean sum of embeddings. Roughly half the mass was the pair and
/// the rest was noise, which is what a 1.7x effect on the Latin square looks
/// like.
///
/// The two clean alternatives answer different questions, so both are here:
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BindMode {
    Off,
    /// `E_prev_event (*) E_now`, at event lags 1..bind_lags. Exact: at the
    /// moment the second cue arrives this is precisely the cue pair, with no
    /// cross-terms at all.
    ///
    /// It is invariant to how many baseline ticks separated the cues, so it
    /// makes the second-order window flat in separation *by construction*. That
    /// is not a defect -- it is the hypothesis that the conjunction is
    /// event-structured rather than timescale-structured, and a flat window
    /// under this mode alongside a good Latin accuracy would settle it.
    EventLag,
    /// `Delta^k (*) E_now`, one per ladder band. This is the cross-band
    /// conjunction the design originally predicted: it carries tick-scale, so
    /// it is the mode under which a plateau in separation could appear at all.
    /// Noisier, because a band is a smoothed average rather than one embedding.
    Band,
    /// Both, so the readout can use whichever carries the signal.
    Both,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RouteQuery {
    State,
    Bound,
    BoundState,
}

#[derive(Clone, Debug)]
pub struct Config {
    // ---- widths -------------------------------------------------------
    /// INHERITED. Payload / code width.
    pub d: usize,
    /// INHERITED. Vocabulary size.
    pub vocab: usize,

    // ---- the one free knob --------------------------------------------

    // ---- ladder --------------------------------------------------------
    /// CEILING. Number of ladder rungs, which is also the ceiling on tree
    /// depth. Realised depth is emergent and reported, not set.
    pub rungs: usize,
    /// FREE-ish. Rate of the fastest rung. The remaining rungs are derived.
    pub rho0: f32,
    /// DERIVED. Geometric ratio between rung rates, chosen so the rungs cover
    /// `horizon` ticks log-uniformly: beta = (rho0 * horizon)^(-1/(rungs-1)).
    pub beta: f32,
    /// DERIVED. Slowest timescale the ladder must reach, taken from the
    /// generator's inter-event interval.
    pub horizon: f32,

    // ---- memory graph ---------------------------------------------------
    /// INHERITED. Ring nodes.
    pub nodes: usize,
    /// INHERITED. Random shortcuts per node, on top of the two ring neighbours.
    pub shortcuts: usize,
    /// INHERITED. Hops on the write walk.
    /// Steps on the write walk. It has to match what a read does in one tick.
    ///
    /// A read takes one hop per tick, so the states reads ever present to an
    /// edge are "one hop from entry". At `hops = 2` the write walk's second step
    /// was fed the output of its own first hop -- a region of state space no
    /// read produces -- and only that second edge learned, so the operator was
    /// taught a mapping on an input distribution disjoint from the one it is
    /// evaluated on.
    pub hops: usize,

    // ---- allocation ------------------------------------------------------

    // ---- learning ---------------------------------------------------------
    /// INHERITED. Learning rate.
    pub eta: f32,
    /// INHERITED. Sampled negatives per write, drawn from the leaf's own
    /// emitted targets. Zero recovers the dense update.
    pub neg_samples: usize,
    /// DERIVED. Eligibility decay, matched to the mean inter-event interval.
    pub trace_lambda: f32,
    /// DERIVED. Decay of the visit accumulator, the same time constant: a
    /// response's prior is what it touched during *this* response, not what it
    /// touched two challenges ago.
    pub visit_decay: f32,

    // ---- descent ----------------------------------------------------------
    /// INHERITED. Confidence bins for the reliability curve. Tagged DERIVED
    /// once, which was wrong -- nothing computes it -- and the point of these
    /// tags is that they are checkable claims rather than decoration.
    pub calib_bins: usize,
    /// Charge what was said, not what was being thought.
    ///
    /// Without this the overt channel serves no objective at all: the ledger
    /// scores the emitted *distribution*, so speaking costs nothing and buys
    /// nothing, and any threshold on it is arbitrary -- which is the real reason
    /// the channel fired four times in a run, not the threshold rule.
    ///
    /// With it, the distribution standing when the system first spoke is frozen
    /// and that is what the settlement scores; a response that never spoke is
    /// charged its background prior. Speaking early risks locking a worse
    /// distribution, speaking late risks the world resolving first. The
    /// speed-accuracy tradeoff stops being imposed and starts being derived,
    /// and the optimal-stopping question becomes answerable because a payoff
    /// structure finally exists.
    ///
    /// Costs comparability: bits under this rule mean something different from
    /// bits without it, and no number across the switch is comparable.
    pub commit_locks_charge: bool,
    /// Floor on the threshold for saying something out loud. Lower than the
    /// branch floor because it gates a different quantity: a readout maximum
    /// over a few dozen emitted rows lives on a different scale from a branch
    /// posterior over a handful of children, and gating one by the other is
    /// what silenced the overt channel entirely.
    pub speak_fallback: f32,

    // ---- channels ----------------------------------------------------------
    /// A5. The highest ladder rung that self-generated content may write to.
    /// Fixed at 0 by the axiom; exposed only so the ablation can break it.
    pub self_max_rung: usize,
    /// Feed what was said out loud back into the context.
    pub feedback_overt: bool,
    /// Feed inner speech -- what was considered and not said -- back into the
    /// context. Separate from `feedback_overt` because a control that ablates
    /// both at once cannot say which of the two streams does the poisoning.
    pub feedback_covert: bool,
    /// Feed the write/activity channel back into the context at all.
    pub feedback_write: bool,

    /// Weight of the background anchor in the state update.
    ///
    /// Without it the operator iteration is autonomous and settles into a fixed
    /// point or a short cycle -- the same failure the gap walk had. With it the
    /// system is *driven*: the challenge keeps pulling the state back toward
    /// itself while the operators compute, so the state converges to an
    /// attractor that depends on the challenge. Convergence is what makes the
    /// emitted distribution sharpen rather than wander.
    pub anchor: f32,

    /// How many ranks below the best edge to route to.
    ///
    /// Zero is the argmax. One takes the runner-up on every hop, two the third
    /// best, and so on. This is the near-miss instrument, and it tests the claim
    /// the whole design rests on: in a table, a neighbouring address holds an
    /// unrelated candidate set and a near miss is a cliff; in an operator set,
    /// neighbouring keys are applied to similar states and so were written by
    /// similar data, and a near miss should be a small perturbation that the
    /// next tick can correct. A cliff here falsifies the design.
    pub route_perturb: usize,

    /// Write the operator toward the observed token's fixed embedding (a pure
    /// association) rather than toward its readout row.
    ///
    /// The founding requirement is that a node performs a memory *write*, not an
    /// autoregressive fit. Writing toward the embedding involves no prediction
    /// error at all and is unambiguously a write; writing toward the row is more
    /// directed but lets readout information flow back into the operator, which
    /// is a step toward fitting. Kept switchable because the difference is a
    /// design question, not a tuning one.
    pub write_toward_embedding: bool,

    /// Whether the response keeps walking during the gap.
    ///
    /// Off, the state freezes after the event's own hop and every later gap tick
    /// changes nothing. This is the control that separates "the response unfolds"
    /// from "the prior is a broad mixture and the readout does the rest" -- two
    /// explanations that produce the same aggregate numbers.
    /// FREE. Remove the graph entirely: no routing, no hop, no operator write.
    ///
    /// This is the monolith. What is left is the background ladder, the bound
    /// traces, the shared delta-rule readout and the three streams -- a single
    /// body whose capacity is the linear separability of a fixed-width phi.
    ///
    /// It exists because `walk_during_gap = false` was never this arm: the event
    /// tick walked unconditionally, so every arm in the suite carried the graph
    /// and the suite had no single-body end to compare against at all. A
    /// multi-body extension buys capacity with addressing error, so comparing it
    /// to a monolith that is not yet capacity-bound charges it the whole price
    /// of scale and credits it none of the benefit.
    /// FREE. Route reads to a uniformly random out-edge instead of the argmax.
    ///
    /// The sharp form of the near-miss control. `route_perturb` takes the
    /// runner-up edge, which on a small-world graph is a *neighbouring* address;
    /// this takes any edge at all. If the walk's value were "some learned
    /// operator sits on the state", both would be free. If the value is
    /// addressed content, the near miss should stay cheap -- neighbours hold
    /// related content -- while random routing collapses. The two controls only
    /// mean something as a pair.
    /// FREE. Leave every edge transform at its random initialisation.
    ///
    /// The suite has never had this arm, and without it "the graph is an
    /// addressed operator memory" is not distinguishable from "the graph is a
    /// bank of fixed random transforms the state gets routed through". The
    /// arithmetic makes the second live: 10741 charged events write one edge
    /// each, so at nodes=256 a 64x64 matrix receives about eleven rank-one
    /// updates for its 4096 parameters -- and that is the best-scoring arm. The
    /// contrast that was supposed to rule this out, nodes=1 against no graph at
    /// all, cannot: nodes=1 has two edges, so it varies the number of random
    /// transforms rather than holding it fixed.
    /// Where the routing query comes from.
    ///
    /// `State` is what has been running: `q = nu(sum_k Delta_k + p)`. It contains
    /// no statement of *what was just observed* -- `p` is the trajectory and
    /// `Delta` is the background -- and it moves on every gap tick, so a six-tick
    /// gap routes to six different nodes. Measured: one fact reaches 12.1 distinct
    /// nodes and its dominant node holds 14.6% of its visits. An argmax over an
    /// address that unstable is a hash of noise, which is why random routing
    /// matched it exactly.
    ///
    /// `Bound` routes on the bound traces instead. Those are content -- the
    /// conjunction of what was recently observed -- and `rebind` only runs on
    /// event ticks, so the address holds still through the gap. It needs no
    /// challenge boundary and no prefix: a bound trace is a decaying binding of
    /// recent observations, not a window over them.
    ///
    /// Address consistency is the instrument. If it does not rise well above
    /// 0.146, the repair failed and nothing downstream of it is worth running.
    /// Enter the read walk at the content-determined node, as the write walk
    /// already does, instead of continuing from wherever the last walk stopped.
    ///
    /// `gnode` is initialised to 0 and thereafter only ever follows edges, so a
    /// read is a wander from its predecessor that content nudges among at most
    /// four local out-edges. The write walk calls `entry(q)` and is content
    /// addressed; the read walk never has been. No query, however clean, can
    /// produce a content address when the starting node is set by history --
    /// which is why routing on the bound traces changed nothing.
    ///
    /// With a query that holds still through the gap, this also pins the address
    /// while the state keeps evolving under one operator, which is nearer to
    /// "iterate with memory" than stepping to a new node every tick.
    pub read_entry_by_content: bool,
    pub route_query: RouteQuery,
    pub freeze_operator: bool,
    pub route_random: bool,
    pub bypass_graph: bool,
    pub walk_during_gap: bool,

    /// What the bound trace binds. See `BindMode`.
    pub bind_mode: BindMode,
    /// Event lags carried under `EventLag`.
    pub bind_lags: usize,
    /// Bind consecutive cues by circular convolution and give the readout the
    /// bound trace alongside the payload.
    ///
    /// This is the "combine" half of keeping the streams apart but combinable,
    /// and it has a falsifiable job rather than a decorative one: a Latin square
    /// is linear in the tensor features of the two cues and not in their sum, so
    /// a linear readout can represent it over a bound trace and cannot over a
    /// superposition. If the Latin window does not move when this is switched
    /// on, the binding is not what was missing.
    pub use_binding: bool,
    /// Per-tick decay of the binding trace. Measured: it must be 1.0.
    ///
    /// Declared at 0.5 and referenced nowhere for the life of the project. Wired
    /// up, 0.5 applies per tick, so across an answer gap of six the trace
    /// retains 1.6% and the conjunction is annihilated before anything reads it.
    /// The sweep is monotone and there is no interior optimum:
    ///
    /// ```text
    ///   decay   Latin    product  retention  answer bits
    ///   1.00    0.1448   0.6884   0.2163     6.249
    ///   0.99    0.1326   0.6690   0.2123     6.507
    ///   0.95    0.0682   0.6349   0.1429     7.362
    ///   0.85    0.0192   0.4612   0.1091     8.877
    ///   0.50    0.0100   0.2845   0.0536     9.773
    /// ```
    ///
    /// The audit wanted this wired because a trace that never decays leaks across
    /// episodes. It does leak, but decay is not the cure: `rebind` overwrites
    /// every slot wholesale on each event, so what leaks is the interleaved token
    /// `event_hist` carries into the lag-1 block, and only a change there fixes
    /// it. Kept as a field rather than deleted so the measurement stays attached
    /// to the number.
    pub bind_decay: f32,

    /// Initialisation scale of the edge transforms. Small values leave the tanh
    /// in its linear regime and the residual hop close to the identity, which
    /// makes the learned payload chain -- one of the two things claimed over a
    /// suffix model -- do nothing at all.
    pub w_init: f32,

    // ---- operator tokens ------------------------------------------------
    /// Gain of the rank-one operator A_x. The baseline token has A = 0, which
    /// is what makes the write gate vanish at baseline.
    pub op_gain: f32,
    /// How much of the incoming token's identity is injected into the payload.
    /// Zero leaves only the rank-one operator, which on its own moves the
    /// payload by O(gain / sqrt(d)) and so carries almost nothing.
    pub op_mix: f32,

    // ---- ablations ---------------------------------------------------------
    /// Disable the learned leaf readout, leaving pure count-based backoff. The
    /// difference between this and the full model is what the readout buys.
    pub no_readout: bool,
    /// Disable eligibility credit to gap-time particle activity.
    pub no_eligibility: bool,

    // ---- bookkeeping --------------------------------------------------------
    pub seed: u64,
    /// Ticks between full-distribution entropy evaluations, which cost O(V).
    pub entropy_every: u64,
}

impl Config {
    pub fn local() -> Self {
        // Three, because the rung sweep measured it: with coverage held at one,
        // more bands cost accuracy on both conjunctions monotonically. Six was
        // the original guess and it is the worst arm.
        let rungs = 3;
        let rho0 = 0.5f32;
        let horizon = 256.0f32;
        let mut c = Config {
            d: 64,
            vocab: 4096,
            rungs,
            rho0,
            beta: 1.0,
            horizon,
            nodes: 32,
            shortcuts: 2,
            hops: 1,
            eta: 0.5,
            neg_samples: 16,
            trace_lambda: 0.9,
            visit_decay: 0.9,
            calib_bins: 10,
            commit_locks_charge: false,
            speak_fallback: 0.25,
            self_max_rung: 0,
            feedback_overt: true,
            feedback_covert: true,
            feedback_write: true,
            read_entry_by_content: false,
            route_query: RouteQuery::State,
            freeze_operator: false,
            route_random: false,
            bypass_graph: false,
            walk_during_gap: true,
            anchor: 0.35,
            route_perturb: 0,
            write_toward_embedding: true,
            use_binding: true,
            bind_mode: BindMode::Both,
            bind_lags: 2,
            bind_decay: 1.0,
            // 1.5, because the sweep measured it: the payload chain needs to
            // be out of the tanh's linear regime before it transforms
            // anything, and both conjunctions peak here.
            // Back to a scale that leaves the residual hop a residual. At 1.5
            // the operator term is six times the norm of the state and a
            // quarter of the units saturate, so `nu(p + tanh(Wp))` is very
            // nearly `nu(tanh(Wp))`: the state is overwritten every hop by a
            // saturated near-random map, all edges destroy it about equally,
            // and the saturation also blocks the operator write through its
            // own (1 - tanh^2) factor. 1.5 came from a sweep that was later
            // shown to be an artefact of a gradient defect, and the setting was
            // left in place after the evidence for it was withdrawn.
            w_init: 0.1,
            op_gain: 1.0,
            op_mix: 0.5,
            no_readout: false,
            no_eligibility: true,
            seed: 0x5EED_1234,
            entropy_every: 16,
        };
        c.derive();
        c
    }

    /// Recompute every DERIVED field. Call after changing a FREE or CEILING
    /// field; the experiment drivers do this for every sweep point.
    pub fn derive(&mut self) {
        // bind_blocks() branches on use_binding; rebind() branches on
        // bind_mode. If they disagree, rebind writes past the end of the
        // bind array. Catch it here rather than as an out-of-bounds panic
        // a thousand ticks in.
        let inconsistent = !self.use_binding && !matches!(self.bind_mode, BindMode::Off);
        assert!(!inconsistent, "use_binding=false requires bind_mode=Off");
        // Rungs cover [1/rho0, horizon] log-uniformly.
        self.beta = if self.rungs > 1 {
            let span = (self.rho0 * self.horizon).max(1.001);
            (span.ln() / (self.rungs - 1) as f32).exp().recip()
        } else {
            1.0
        };
        // A trace should still be alive across a typical inter-event gap.
        let mean_gap = (self.horizon / 8.0).max(2.0);
        self.trace_lambda = (-1.0f32 / mean_gap).exp();
        self.visit_decay = self.trace_lambda;
    }

    /// Number of `d`-wide blocks in a readout row: the payload, plus one per
    /// bound trace the mode carries.
    /// Blocks of width `d` in a readout row: the state, the bound traces, and
    /// the background bands.
    ///
    /// The bands are new here and the omission mattered: the ladder previously
    /// fed only the routing query and never the features, so the self channels
    /// -- which write into the ladder -- had no path to the prediction at all.
    /// That is why every three-stream ablation read as no effect.
    pub fn feature_blocks(&self) -> usize {
        1 + self.bind_blocks() + self.rungs
    }

    /// Bound traces only -- not counting the state block, and exactly the number
    /// `Model::rebind` actually fills.
    ///
    /// This used to include the state block *and* be added to `rungs` again in
    /// `feature_blocks`, so the row width came out one block short of what
    /// `features()` emitted. The readout truncated to the row width and the
    /// background bands, which sit last, were never read at all -- which is why
    /// every self-feedback ablation measured as no effect. Three of the
    /// allocated bind slots were also never written and stayed zero.
    pub fn bind_blocks(&self) -> usize {
        if !self.use_binding {
            return 0;
        }
        match self.bind_mode {
            BindMode::Off => 0,
            BindMode::EventLag => self.bind_lags,
            BindMode::Band => self.rungs,
            BindMode::Both => self.bind_lags + self.rungs,
        }
    }

    /// Rate of rung k.
    pub fn rho(&self, k: usize) -> f32 {
        self.rho0 * self.beta.powi(k as i32)
    }
}
