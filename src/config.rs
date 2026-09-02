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

/// How a node decides it should be split.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitRule {
    /// The reference mechanism's rule: split when the winning similarity is too
    /// dispersed. It detects *regimes* -- and it detects them well -- but it is
    /// a statement about the address, not about what the node holds, so nothing
    /// in it has any reason to carve out a cell for a cue pair inside a regime.
    Dispersion,
    /// Split when the node is still surprised by its own content: its mean
    /// per-write surprise, in bits, stays above a threshold after it has seen
    /// enough. This is a statement about prediction rather than about
    /// similarity, so it carves where the content is still unresolved. The
    /// threshold has units -- two bits means "still four-way confused" -- which
    /// a dimensionless dispersion ratio does not.
    Surprise,
    /// Dispersion at the coarse level, surprise below it.
    ///
    /// The two criteria are not competitors so much as answers to different
    /// questions. Dispersion asks whether the address is stretched, which is the
    /// right question for "is this one regime or two". Surprise asks whether the
    /// node still fails to predict what it holds, which is the right question
    /// for "does this regime need carving inside". Replacing one with the other
    /// cost the regime separation outright; running each where it belongs is the
    /// obvious thing to try next.
    Hybrid,
}

#[derive(Clone, Debug)]
pub struct Config {
    // ---- widths -------------------------------------------------------
    /// INHERITED. Payload / code width.
    pub d: usize,
    /// INHERITED. Vocabulary size.
    pub vocab: usize,

    // ---- the one free knob --------------------------------------------
    /// FREE. Per-tick compute budget, expressed as the number of read
    /// particles. Everything else that could have been a search knob (beam
    /// width, top-k, scheduler slots) is folded into this.
    pub particles: usize,

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
    /// INHERITED in form, rescaled in value. The criterion is the reference
    /// mechanism's sigma/|mu| on the winning similarity, but here the query and
    /// the prototypes are both unit vectors, so the similarity lives in [-1,1]
    /// with a relative dispersion around 0.3-0.5 even for a well-matched class.
    /// The reference value of 0.05 splits everything in sight.
    pub grow_theta: f64,
    /// Which criterion decides a split. The comparison between them is the
    /// experiment, not a preference.
    pub split_rule: SplitRule,
    /// Under `Hybrid`, the deepest level still judged by dispersion. Fixed at
    /// one by principle rather than swept: level 1 reads the slowest rung and is
    /// the regime level, and everything below it is content.
    pub hybrid_coarse_levels: usize,
    /// Mean per-write surprise, in bits, above which a node is still unresolved
    /// and should be split. Replaces `grow_theta` under `SplitRule::Surprise`,
    /// so the parameter count does not change.
    pub split_bits: f64,
    /// INHERITED. Consecutive steps the criterion must hold.
    pub grow_hold: u32,
    /// INHERITED. Minimum observations before a node may split.
    pub grow_min_obs: u64,
    /// CEILING. Maximum children per node.
    pub max_children: usize,
    /// CEILING. Maximum nodes in the tree.
    pub max_nodes: usize,
    /// INHERITED. Observations before a node may deepen (gain its first child).
    ///
    /// This has to be low enough that the tree reaches its depth cap. Level l
    /// reads rung `rungs - l`, so a tree that stops at depth 3 under a
    /// six-rung ladder never consults the three fastest bands at all: they are
    /// configured and unread, and a sweep over the rung count then varies which
    /// bands are used rather than how many. `rung_visits` is the instrument for
    /// that and is reported on every run.
    pub deepen_min_obs: u64,

    // ---- learning ---------------------------------------------------------
    /// INHERITED. Learning rate.
    pub eta: f32,
    /// INHERITED. Sampled negatives per write, drawn from the leaf's own
    /// emitted targets. Zero recovers the dense update.
    pub neg_samples: usize,
    /// DERIVED. Eligibility decay, matched to the mean inter-event interval.
    pub trace_lambda: f32,

    // ---- descent ----------------------------------------------------------
    /// CEILING. Maximum ticks a particle may spend accumulating evidence at one
    /// level before it is forced to commit. It has to leave room for the whole
    /// descent inside a typical response window: with a gap of g ticks and a
    /// tree of depth L, a level cannot afford more than about g/L of them.
    pub max_ticks_per_level: u32,
    /// DERIVED. Confidence bins for the per-level calibration counters. The
    /// commit threshold is read out of these, not set.
    pub calib_bins: usize,
    /// INHERITED. Minimum observations in a bin before calibration is trusted;
    /// below it the fallback threshold is used.
    pub calib_min_obs: u64,
    /// Floor on the *branch* commit threshold. Calibration may raise it and
    /// never lower it. Gap ticks cost nothing, so waiting is close to free and
    /// the bar should be high.
    pub commit_fallback_slack: f32,
    /// Floor on the threshold for saying something out loud. Lower than the
    /// branch floor because it gates a different quantity: a readout maximum
    /// over a few dozen emitted rows lives on a different scale from a branch
    /// posterior over a handful of children, and gating one by the other is
    /// what silenced the overt channel entirely.
    pub speak_fallback: f32,
    /// INHERITED. Inverse temperature on branch scores.
    pub branch_temp: f32,

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
    /// Cap the tree at depth 1, recovering the reference mechanism's flat class
    /// scheme. The sanity check, not an experiment.
    pub flatten: bool,
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
            particles: 8,
            rungs,
            rho0,
            beta: 1.0,
            horizon,
            nodes: 32,
            shortcuts: 2,
            hops: 2,
            grow_theta: 0.45,
            // Dispersion, because it measurably beat surprise on every axis --
            // bits, regime purity, pair purity and both conjunctions -- when the
            // two were compared. Leaving a known-worse default in place would
            // make every later run quietly wrong.
            split_rule: SplitRule::Dispersion,
            split_bits: 2.0,
            hybrid_coarse_levels: 1,
            grow_hold: 8,
            grow_min_obs: 200,
            max_children: 8,
            max_nodes: 512,
            deepen_min_obs: 80,
            eta: 0.5,
            neg_samples: 16,
            trace_lambda: 0.9,
            max_ticks_per_level: 2,
            calib_bins: 10,
            calib_min_obs: 32,
            commit_fallback_slack: 0.7,
            speak_fallback: 0.25,
            branch_temp: 4.0,
            self_max_rung: 0,
            feedback_overt: true,
            feedback_covert: true,
            feedback_write: true,
            use_binding: true,
            bind_decay: 0.5,
            // 1.5, because the sweep measured it: the payload chain needs to
            // be out of the tanh's linear regime before it transforms
            // anything, and both conjunctions peak here.
            w_init: 1.5,
            op_gain: 1.0,
            op_mix: 0.5,
            flatten: false,
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
    }

    pub fn effective_depth_cap(&self) -> usize {
        if self.flatten {
            1
        } else {
            self.rungs
        }
    }

    /// Rate of rung k.
    pub fn rho(&self, k: usize) -> f32 {
        self.rho0 * self.beta.powi(k as i32)
    }
}
