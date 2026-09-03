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

    /// Whether the response keeps walking during the gap.
    ///
    /// Off, the state freezes after the event's own hop and every later gap tick
    /// changes nothing. This is the control that separates "the response unfolds"
    /// from "the prior is a broad mixture and the readout does the rest" -- two
    /// explanations that produce the same aggregate numbers.
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
    /// Decay of the binding trace across a response.
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
            hops: 2,
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
            walk_during_gap: true,
            use_binding: true,
            bind_mode: BindMode::Both,
            bind_lags: 2,
            bind_decay: 0.5,
            // 1.5, because the sweep measured it: the payload chain needs to
            // be out of the tanh's linear regime before it transforms
            // anything, and both conjunctions peak here.
            w_init: 1.5,
            op_gain: 1.0,
            op_mix: 0.5,
            no_readout: false,
            no_eligibility: false,
            seed: 0x5EED_1234,
            entropy_every: 16,
        };
        c.derive();
        c
    }

    /// Recompute every DERIVED field. Call after changing a FREE or CEILING
    /// field; the experiment drivers do this for every sweep point.
    pub fn derive(&mut self) {
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
    pub fn feature_blocks(&self) -> usize {
        if !self.use_binding {
            return 1;
        }
        1 + match self.bind_mode {
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
