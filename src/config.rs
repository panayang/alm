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
/// How much a row still moves, and what decides it.
///
/// The delta rule already stops *pulling* when it is right -- `err` goes to
/// zero -- but the step never shrinks, so a settled row keeps being shaken by
/// the noise in every other event it appears in as a negative. Something has to
/// say when a row is settled. What says it is the question.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepRule {
    /// One step size for every row, forever. Converges to a hover whose width
    /// is set by `eta`.
    Fixed,
    /// `eta / (1 + corrections)`. This is not consolidation, it is frequency:
    /// 1/n is exactly the weight that makes a running estimate the empirical
    /// mean, so it relocates the counted prior from the emission into the step
    /// size. It is also irreversible -- n only grows -- so a row that settles
    /// on a world that then changes can never move again. Here to be measured,
    /// not to be adopted.
    InverseCount,
    /// `eta * (running mean of |err| for this row)`. Consolidation measured by
    /// whether the row is still wrong rather than by how often it has been
    /// touched, which is what `store.rs` says a row is for. A row that keeps
    /// being corrected stays plastic however many times it has been written; a
    /// row whose error has gone silent takes small steps. The running mean is
    /// updated from the raw error, not the scaled one, so when the world
    /// changes the error returns, the mean rises, and the step grows back.
    ErrorDriven,
}

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
    /// Learning rate, for the readout row and the operator write alike.
    ///
    /// Not inherited any more: measured, together with `row_norm_cap`, because
    /// the two were doing each other's job. At 0.5 the delta rule overshoots on
    /// a stream it cannot predict, the row accumulates a random walk, and the
    /// readout becomes confidently wrong -- 6.78 bits against a uniform of
    /// 6.00, which is fabrication. Capping the norm hides that, and destroys
    /// convergence on a stream it *can* predict: the charge on a deterministic
    /// successor bottoms out around the second decile and then climbs for the
    /// rest of the run.
    ///
    /// The step size is the actual knob. Bits per event by decile, 120k events,
    /// V = 64; deterministic successor (true answer 0) over i.i.d. Zipf (true
    /// marginal 4.864, uniform 6.000), reading the last decile:
    ///
    /// ```text
    ///    eta   cap     deterministic          i.i.d.
    ///    0.5    16     0.278  (rising)        5.751     <- was here
    ///    0.3    64     0.055                  5.849
    ///    0.2    64     0.049  (falling)       5.737
    ///    0.2     0     0.049  (falling)       5.557     <- here
    ///    0.1     0     0.067  (falling)       5.571
    ///    0.5     0     0.047  (falling)       6.776     fabricates
    /// ```
    ///
    /// 0.2 with no cap is better on both columns at once than 0.5 with a cap on
    /// either -- five times cheaper on the structure it can learn and still a
    /// fifth of a bit better on the noise it cannot. And at 0.2 the cap makes
    /// no difference at all, which is the whole finding: the bound was never
    /// buying anything except protection from a step that was too large.
    ///
    /// The i.i.d. column rises slowly in every row and is 0.69 bits above the
    /// marginal. That is the price of having no counted prior, named in
    /// `store.rs`, and it is a decision rather than a defect.
    pub eta: f32,
    /// Hold a superposed memory that speech writes into and silence reads out of.
    ///
    /// The two operations are opposites and the tick already says which one is
    /// called for: the world speaking means there is something to combine, the
    /// identity input means there is not, and therefore that what is called for
    /// is the inverse. Nothing is scheduled and no boundary is needed.
    ///
    ///   speech   M += nu( (E_prev2 (*) E_prev) (*) E_now )
    ///   silence  p <- nu( M (o) (p (*) E_last) )
    ///
    /// Under silence the state is unbound out of memory using itself bound with
    /// the most recent observation, so each quiet tick advances one link and the
    /// world decides how far the chain gets by deciding how long to stay quiet.
    /// That is "depth comes from time" as a mechanism rather than a hope, and it
    /// is why gap ticks have measured as dead weight so far: iterating a state
    /// through a transform is not retrieval, and there was nothing to retrieve.
    /// Key the readout on the background as it stood at the last event, not as
    /// it stands now.
    ///
    /// The bands carry two things at once: which regime this is, which should
    /// hold across a silence, and how long the world has been quiet, which
    /// changes every tick. Both were in the retrieval key, so how long ago the
    /// world last spoke decided whether a memory fired at all -- a fact learned
    /// at six ticks of silence did not answer at one. That is a clock in the
    /// key. Locking the bands at the last event freezes the challenge the moment
    /// the world stops speaking and leaves only the state free to evolve, which
    /// is what challenge-response means. The live fast rung stays available to
    /// the commit rule, which is where elapsed time belongs.
    pub event_locked_key: bool,
    /// Minimum cosine to the nearest codebook entry for an unbound result to be
    /// accepted.
    ///
    /// Without it a cursor given more silence than the walk has links keeps
    /// stepping and walks past the answer. At the end of a chain there is no
    /// outgoing link, so the unbinding returns noise, and refusing noise leaves
    /// the cursor where it is -- stopping for free, and making surplus thinking
    /// time harmless instead of harmful.
    /// Multiple of the codebook's noise floor an unbound result must clear.
    ///
    /// The floor is what a *random* vector scores against its best match among V
    /// codebook entries, sqrt(2 ln V / d): 0.452 at d = 64 with 685 tokens. It
    /// was a fixed 0.25 -- below the floor -- so cleanup accepted noise on 97.6%
    /// of silent ticks and the cursor was randomised every tick. Measured mean
    /// best cosine was 0.385, which is the floor and not a signal.
    ///
    /// The operating region follows, and it is the reference paper's trade
    /// written quantitatively for the first time here:
    ///
    /// ```text
    ///   signal  ~ 1/sqrt(k)            k = triples per bank  (interference)
    ///   floor   ~ sqrt(2 ln V / d)                           (retrieval difficulty)
    ///   usable  <=>  d > 2 k ln V  ~  13 k
    /// ```
    ///
    /// At d = 64 that allows five triples per bank, so sixteen thousand of them
    /// need three thousand banks. This design has never been inside its own
    /// operating region.
    pub cleanup_floor_mult: f32,
    /// How many superpositions the memory is split into.
    ///
    /// Retrieval from a superposition degrades as 1/sqrt(k) in the number of
    /// triples it holds -- measured 0.043 against a predicted 0.050 at k = 401 --
    /// so one memory cannot carry a continual stream of ten thousand events. The
    /// address chooses which bank to write and which to unbind, each stays under
    /// capacity, and the read stays one unbinding however much has been stored.
    ///
    /// This is the first job content addressing has had in this project that is
    /// not about accuracy. It measured inert on every accuracy axis because a few
    /// hundred facts fit in one linear table with room to spare; against a
    /// superposition the capacity is hard and the address is what buys past it.
    /// Weight of the direct codebook comparison in the emitted score.
    ///
    /// `s(o) = <R_o, phi> + codebook * <E_o, p>`. The second term is not
    /// learned; it is the cleanup memory that lets a retrieved state be *said*.
    ///
    /// Without it the model could retrieve and could not answer. The unbound
    /// cursor is one of nine blocks handed to a learned linear readout, and that
    /// readout is trained on presented facts where the bound block does all the
    /// work, so it has no reason to put weight on the state block. Six memory
    /// configurations -- superposition off, and one through a thousand banks --
    /// produced a bit-identical walk surface, which is what it looks like when
    /// the retrieval never reaches the charge.
    pub readout_codebook: f32,
    /// How many responses are forming at once.
    ///
    /// "It is not just one trajectory." A single cursor has to be aimed, and
    /// aiming needs to know which challenge is pending -- which is
    /// one-challenge-one-response, the autoregressive shape. Several cursors of
    /// different ages step in parallel and the emission is their superposition:
    /// half-formed answers maturing together, one of which is the one that comes
    /// out. A cursor `k` ticks old has advanced `k` links, so depth is read off
    /// which cursor matures rather than off a schedule.
    ///
    /// Superposition, not competition. Every mechanism this project killed was a
    /// competition and both survivors were superpositions, so a resampling
    /// particle filter is the wrong shape here and a parallel mixture is not.
    /// How fast a response that retrieved nothing this tick fades out.
    pub cursor_fade: f32,
    /// Sigmas above chance that a retrieval must verify at to be believed.
    ///
    /// A key that was never written still hashes into an occupied bank, and its
    /// occupants answer -- so a chain that has run out does not retrieve nothing,
    /// it retrieves a neighbour. No threshold on the retrieval separates the two,
    /// because both are real stored content at comparable strength. That is
    /// address crowding in its purest form and it is the weakness this whole
    /// family is built on.
    ///
    /// Challenge-response has a move an autoregressive forward pass does not:
    /// the answer need not be given at a fixed instant, so a retrieval can be
    /// checked. Bind the candidate back onto the key and ask memory whether that
    /// triple is actually there:
    ///
    /// ```text
    ///   <M / |M| , nu(q (*) y)>   ~  1/sqrt(k)   if the triple was written
    ///                             ~  1/sqrt(d)   if it came from a neighbour
    /// ```
    ///
    /// Eight to one at k=4, d=256, for one convolution and one dot product.
    /// Binding writes, unbinding reads, and re-binding checks -- and a near miss
    /// becomes detectable, which it has never been in this project.
    pub verify_sigma: f32,
    /// Read the most matured response rather than the average of them.
    ///
    /// Averaging unit cursors and renormalising divides each by
    /// sqrt(sum w^2), so the response holding the answer arrives at the readout
    /// at 0.24-0.28 against a codebook floor of 0.255 -- present and
    /// indistinguishable. "Several half-formed answers mature together and one of
    /// them is said" is not "say their average"; taking the strongest is the
    /// natural readout of parallel responses, not a competition between
    /// mechanisms.
    pub emit_strongest: bool,
    pub traj: usize,
    pub mem_banks: usize,
    pub superpose: bool,
    /// Weight of the unbound result against the standing state.
    pub unbind_mix: f32,
    /// Give the readout a per-token additive term.
    ///
    /// FREE, default false. The store's own rationale says counts were dropped
    /// because a frequency estimate converges on the corpus unigram and so
    /// stops discriminating as data grows. A bias is that estimate. It exists
    /// as a switch because the readout was measured to carry no frequency
    /// information whatsoever, which is either the design holding its line or a
    /// broken emission path, and those two have to be told apart rather than
    /// assumed.
    /// Largest L2 norm a readout row may reach. Zero disables the bound.
    ///
    /// The delta rule is unregularised by design -- it is a write, not a fit,
    /// and nothing upstream of a row may touch it. That was survivable only
    /// while the softmax clamped, because saturation hid the growth; with the
    /// clamp gone, rows reaching a norm of 350 against a feature norm of 3 make
    /// the readout confidently wrong on anything it cannot predict, which
    /// `store.rs` names as the price of dropping the count prior.
    ///
    /// A cap rather than weight decay, deliberately. Decay pulls every row
    /// toward zero on every step, which is a forgetting rule and this
    /// architecture does not have one; a cap does nothing at all until a row is
    /// too large to be expressing anything but its own divergence.
    ///
    /// Swept on two streams at once, because one cannot choose it: a loose cap
    /// learns structure and fabricates on noise, a tight one does the reverse.
    /// The first sweep held `eta` at 0.5 and read a single endpoint, and on
    /// that evidence 16.0 looked like the best available trade.
    ///
    /// Both halves of that were wrong. Reading the trajectory instead of the
    /// endpoint shows the cap does not plateau, it *turns*: on a deterministic
    /// successor the charge falls to 0.221 bits by the second decile and then
    /// climbs back to 0.278 by the tenth. Once a row is pinned to the ball,
    /// each update plus its projection is a rotation toward the current
    /// features rather than an accumulation, so the row tracks the most recent
    /// state and forgets the rest -- the readout un-learns.
    ///
    /// And sweeping `eta` alongside it shows the cap was only ever standing in
    /// for a step size. See `eta` for the table: at 0.2 the capped and uncapped
    /// runs are the same to three decimals, because the rows never reach the
    /// bound. Zero, therefore, and the divergence it was added to prevent is
    /// prevented where it comes from.
    pub row_norm_cap: f32,

    /// Shrink a row toward zero each time it is touched, before the update.
    ///
    /// Added while the norm bound was carrying the whole job, on the diagnosis
    /// that the bound cannot tell a row that is confidently right from one that
    /// is confidently wrong -- it caps both. That much was true. The conclusion
    /// drawn from it was not: the fix is not a second shrinkage term, it is not
    /// having a step size that needs a bound. See `eta`.
    ///
    /// Decay separates them. A row driven by a consistent gradient settles
    /// where that gradient balances the shrinkage; a row accumulating a random
    /// walk -- which is what an unpredictable stream produces, and what grew
    /// the norms to 350 against a feature norm of 3 -- has no persistent
    /// direction to hold it up and collapses toward nothing.
    ///
    /// Applied only to the rows a write touches, so rows outside that set stay
    /// exactly untouched and occupancy still measures stored content.
    pub row_decay: f32,
    pub readout_bias: bool,
    /// Draw the write's negatives from the top of the current distribution
    /// instead of uniformly from every known token.
    ///
    /// Uniform sampling takes 16 of 685 rows, so the five in-domain rivals that
    /// share a Latin square's six targets are each drawn about 2.3% of the time.
    /// A target appears roughly eighteen times in a regime's life, so a rival
    /// receives a negative gradient on it about 0.4 times: the readout is never
    /// taught to separate the six candidates and settles at 1/6, which is
    /// exactly where the Latin family sits. The product code is unaffected
    /// because it is answerable from marginals, where ranking on positives alone
    /// suffices.
    ///
    /// It also breaks the rule `code.rs` states: the ledger charges an exact
    /// softmax over every row while the write was fitted against a random
    /// subset, so the two saw different distributions. Correcting what the model
    /// would actually have said is both error-driven and the same distribution
    /// the charge came from.
    ///
    /// Irrelevant when `neg_samples` is zero, which is now the default.
    pub hard_negatives: bool,
    /// How many negatives a write pushes down. Zero means every token: the
    /// exact softmax gradient.
    ///
    /// Sixteen was inherited, and its cost scales with the vocabulary rather
    /// than with anything in the stream. Truncating the negative update to the
    /// k highest-scoring tokens is a biased estimate of the gradient, and on an
    /// i.i.d. Zipf stream -- where the right charge for a token of probability p
    /// is exactly -log2 p -- the bias lands entirely on rare tokens. V = 296,
    /// 120k events, excess over -log2 p:
    ///
    /// ```text
    ///   negatives    overall    p in [2^-10, 2^-8)    p < 2^-10
    ///   top 16        +6.015          +8.065            +41.747
    ///   top 32        +1.949          +0.504            +15.530
    ///   top 64        +0.671          +0.151             +2.312
    ///   top 128       +0.537          +0.443             +0.093
    ///   all (0)       +0.537          +0.441             +0.348    <- here
    ///   16 sampled   +63.264         +90.667            +89.230
    /// ```
    ///
    /// Tokens rarer than 1/1024 were being charged forty bits over their true
    /// cost -- given a probability near 2^-50, written off. At V = 64, where
    /// every earlier diagnostic ran, sixteen is a quarter of the vocabulary and
    /// the defect does not show. On PhysioNet it was the whole of our loss:
    /// targets seen 1-99 times before, charged three bits more than a counter.
    ///
    /// The exact gradient is also simply the rule `code.rs` states -- fit the
    /// write against the distribution the ledger charged -- which a truncated
    /// write never did. The scoring pass is already O(V), so this is at most
    /// twice the cost.
    ///
    /// The overall +0.54 that remains is flat across rarity and is a different
    /// question.
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
    /// Skip the operator graph: no routing, no hop, no operator write.
    ///
    /// **On by default since 2026-09-29 (v7).** The paper measured its
    /// closeout with the graph bypassed, having found it behaves as a random
    /// feature expansion whose learning does nothing (each edge ten to forty
    /// rank-one updates against its parameters) and that removing it improved
    /// every metric. The default had nonetheless kept it on, and every
    /// experiment from 2026-09-16 to 09-28 ran with it. Re-measured at v6:
    /// the facts matrix maintains the same facts either way, the walk alone
    /// answers a relation better without it (5.32 -> 4.87 bits, F10), and
    /// PhysioNet next-token codelength falls from 3.385 to 3.295 bits per
    /// event -- more than the paper's full closeout configuration (3.308),
    /// which also gave up the margin on F9. So the graph goes, and the other
    /// closeout switches stay at their defaults.
    pub bypass_graph: bool,
    pub walk_during_gap: bool,
    /// Refuse a silent hop that would collapse the state back onto the token
    /// the world just said, on a tick where nothing was retrieved. Off, with
    /// the evidence on both sides below, and the decision open.
    ///
    /// Against it, on synthetic streams:
    /// It was built to stop "drift": with the history wiped before a question,
    /// accuracy fell from 0.644 at the question to 0.563 after the silence.
    /// But an answer with nothing under it carries no information, and
    /// accuracy at a 0.5 threshold on such an answer is noise. Under a proper
    /// score the silence does the opposite of harm:
    ///
    /// ```text
    ///   wiped, presence   Brier 0.327 -> 0.285   ECE 0.308 -> 0.184
    ///   wiped, recency    Brier 0.464 -> 0.313   ECE 0.460 -> 0.219
    /// ```
    ///
    /// At the question an ungrounded answer is badly over-confident; the walk
    /// through the silence brings it down. The gate froze that answer instead,
    /// changed neither its accuracy nor its calibration after the silence
    /// (Brier 0.2891 on against 0.2894 off), and cost composition 2.3 times
    /// over (0.096 to 0.220 bits on a new pair) because it refused 22% of all
    /// silent hops on that stream -- the ones where the state settles onto the
    /// cue it was just given and lets go of what preceded it, which is exactly
    /// what a cue-only answer needs.
    ///
    /// For it, on PhysioNet, full stream, the collapse version of the gate
    /// against no gate with everything else identical:
    ///
    /// ```text
    ///                  gate off   gate on
    ///   seen context    +0.726    +0.770
    ///   novel context   +0.233    +0.357
    ///   overall         +0.718    +0.764    bits/event below PPM-C at its best
    /// ```
    ///
    /// 1.73M events, so 0.046 bits is far outside noise, and the largest gain
    /// is in the novel column -- composition, on real data. An unverified
    /// reading that reconciles the two: a collapse back onto the last input is
    /// the state forgetting everything but that input. On the composition
    /// stream the gaps are two ticks and the answer depends on the cue alone,
    /// so forgetting helps; PhysioNet's silences run to eleven ticks and the
    /// next event depends on more than the last one, so forgetting hurts and
    /// the gate keeps the history. If so, whether to refuse the echo depends
    /// on whether the future needs more than the last input, which the
    /// mechanism cannot know in advance.
    ///
    /// That reading was then tested and is not supported. Items all from one
    /// set of eight, the first one queried after four more, each followed by
    /// a silence of the length shown (`judge::history`):
    ///
    /// ```text
    ///   silence each     0      2      5      11
    ///   gate off       0.898  0.824  0.792  0.729
    ///   gate on        0.898  0.823  0.795  0.743
    /// ```
    ///
    /// Order information about an early item does erode with elapsed silence
    /// -- the bands decay through it -- but the gate barely touches that (0.014
    /// at eleven ticks, about one standard error). So the PhysioNet gain is an
    /// empirical fact without a verified mechanism, and it is in codelength,
    /// which is an instrument and not an objective. That is not enough to turn
    /// this on, and it stays off with the question open.
    pub walk_needs_retrieval: bool,

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
    /// Give the features the event itself, not only its conjunctions.
    ///
    /// `rebind` fills one block per lag with `E_prev_j (*) E_x` and one per band
    /// with `band_k (*) E_x`. Every one of them is a conjunction; not one of
    /// them is the token. So when a pair is new, the lag blocks are
    /// near-orthogonal to everything the readout was fitted on and contribute
    /// nothing, the band blocks carry the cue only through a vector that drifts,
    /// and the readout has to fall back on whatever weight the state block
    /// happened to accumulate -- which is little, because while the pair was
    /// familiar the conjunction predicted perfectly and the delta rule stops at
    /// zero error.
    ///
    /// Measured on a stream where the answer depends on the cue alone and half
    /// the (prev, cue) pairs are held out (48 cues, charge on first occurrence,
    /// true conditional 0 bits):
    ///
    /// ```text
    ///   arm                 seen     new    new/seen
    ///   default           0.0339  0.4082      12.0
    ///   gap 6             0.0353  0.3405       9.7
    ///   bind_decay 0.9    0.0411  0.4692      11.4
    ///   bind_decay 0.5    0.0724  0.6281       8.7
    ///   no binding        0.0731  0.1721       2.4
    /// ```
    ///
    /// A counter pays 0.0023 seen and 0.0032 new: it escapes the unseen pair
    /// onto an order-1 conditional it has counted. Removing binding fixes the
    /// ratio and costs the seen column, which is the trade of throwing away the
    /// conjunction rather than completing it.
    ///
    /// Completing it is what this is. The convolution identity is delta, so
    /// `E_x (*) delta = E_x`: the event on its own is the lag-zero member of the
    /// family the lag blocks already form, and the implementation simply started
    /// at lag one. The whole and its parts are then both present and the delta
    /// rule can put weight wherever it pays.
    pub bind_self: bool,
    /// Let the features say when a conjunction stands on nothing.
    ///
    /// A lag block is `E_prev_j (*) E_x`: a conjunction whose pair also names
    /// the memory bank that pair's triples were written into. When that pair
    /// was never written, unbinding the address does not return nothing -- it
    /// returns a neighbour's content, at a strength the emission cannot
    /// distinguish from stored content. That is address crowding, it is this
    /// family's named weakness, and the design already carries the answer:
    /// bind the retrieved candidate back onto the key and ask the bank whether
    /// that triple is there.
    ///
    /// The check was wired to the walk -- `step_cursors` refuses a cursor that
    /// fails it -- and never to the emission. Measured on the composition
    /// stream, the read-back score for the pair the model is standing on is
    /// 0.9311 when that pair was written and 0.0160 when it was not, against a
    /// chance floor of 0.0884 and an acceptance threshold of 0.3536. The
    /// separation is total, and the emission was throwing it away.
    ///
    /// So the lag blocks are zeroed when the key does not verify. The self
    /// block and the band blocks are untouched, because neither stands on a
    /// key.
    ///
    /// This is a claim about storage, not about frequency: the question is
    /// whether an address was written, which one convolution settles, and it
    /// would be the right thing to do even if it cost codelength. The
    /// assertion that guards it is written in those terms and mentions no bits.
    ///
    /// **Off by default since 2026-09-28.** It was on from 433a194 (09-21) to
    /// then, and every experiment in that window ran with it on. The paper
    /// had reported this check as inert and credited it nothing. Since then it
    /// maintains none of the facts in the facts matrix, and its only support
    /// is a snapshot probe on in-hospital death (max-pooled AUROC 0.7687 ->
    /// 0.7835, about one standard error) -- an open-loop prediction benchmark
    /// of the kind this design is not for. The argument above stands; the
    /// evidence for making it the default does not. Turn it on for an
    /// experiment that asks what it maintains, not to raise a score.
    pub verify_gate: bool,
    /// Bind what was said into the key of what the world did next.
    ///
    /// The read already treats the model's own output as a relation: in a
    /// silence the arriving token is the last thing said, and a response
    /// unbinds the bank named by (where it stands, what was said). The write
    /// never produced such keys -- triples were formed from world events only
    /// -- so nothing the model said could ever be found to have been followed
    /// by anything. In a world that reacts, that is the whole of learning what
    /// one's own acts do.
    ///
    /// With this on, when the world speaks `x` after the model said `a`, a
    /// second triple is written: key (the world's last token before `a`, `a`),
    /// value `x`. The value is always the world's; the model's output appears
    /// only in the key, so nothing it said is stored as having happened, which
    /// is the spirit of A5. The world's own triple is written as before.
    ///
    /// Off by default (added 2026-09-28). On a static stream the world ignores
    /// what was said, so this can only add load and noise there; it exists for
    /// worlds that react, and its static-data cost is measured before it is
    /// used anywhere.
    pub bind_overt: bool,
    /// Bind each stored triple to the situation it happened in.
    ///
    /// The long-term banks are keyed by the two previous tokens and nothing
    /// else, so what they return for a key is the mixture of everything that
    /// ever followed it, in every situation: semantic memory, not episodic.
    /// The situation is already carried, as a drifting multi-timescale
    /// average of what the world said (the ladder's world cascade, which the
    /// model's own output never reaches). This follows the temporal context
    /// model of episodic memory: an event is stored bound to the slowly
    /// drifting context it occurred in, and recall reinstates context.
    ///
    /// With this on, each write adds `u(c) (*) triple` to the same bank as
    /// the plain triple, where `c` is the sum of the slow levels (rung 1 up)
    /// and `u` makes it unitary so that unbinding is exact and returns more
    /// the more alike the two situations were. On each event the model reads
    /// "what followed this pair, in a situation like this" by unbinding with
    /// `u(c) (*) key`, and the result is one more feature block. The situation
    /// is gist, not transcript: it is a fuzzy average, which is what the
    /// paper asks the background to be.
    ///
    /// Off by default (added 2026-09-28); it doubles what each bank holds.
    ///
    /// Measured with `contexts` (2026-09-29; d = 512, rung 1 only, the
    /// situated triples in their own superposition `mem_ctx`): with the
    /// situation recurring, recall of its own successor is 0.81 against 0.25
    /// for the plain store at 4 situations and 0.52 against 0.12 at 8, and it
    /// barely falls with the number of other visits in between (0.88 at 0-1,
    /// 0.76 at 10+). One write in a visit already shifts recall within that
    /// visit (situation similarity 0.75). Across visits a single write is
    /// weak (0.22 at 4 situations): recall is about similarity / sqrt(load),
    /// and the load on a key is every other situation's writes of it -- the
    /// fan effect. Slower rungs made every reading worse.
    pub ep_context: bool,
    /// Compute the plain recall diagnostic (`Model::plain_recall`) on every
    /// event even with `ep_context` off, so an off arm can be read the same
    /// way. Diagnostic only; changes nothing the model does.
    pub diag_recall: bool,
    /// Which levels of the world cascade make the situation for `ep_context`
    /// (inclusive range), and whether it is made unitary before binding.
    /// Unitary makes unbinding exact but measures similarity by phase alone;
    /// raw makes the delta component of the unbinding exactly the dot product
    /// of the two situations.
    pub ep_context_levels: (usize, usize),
    pub ep_context_unitary: bool,
    /// With `ep_context_unitary`, raise each Fourier magnitude of the
    /// situation to this power instead of setting it to one (0 = unitary).
    pub ep_context_gamma: f32,
    /// An episode trace: what this context has bound, recallable by content.
    ///
    /// The superposed banks are long-term memory. They are keyed by the pair of
    /// the two previous tokens and never reset, so `(user, slot) -> value` holds
    /// every value that slot has had in every episode, and reading it returns
    /// the prior. Nothing in the situation held a binding for longer than the
    /// lag blocks' reach: `rebind` overwrites every block on each event. So
    /// "what is slot s now" -- the last value that followed s in this context
    /// -- had nowhere to be read from. Measured with a known answer (`slots`):
    /// with 64 values to choose among, a value set in the same turn was named
    /// 0.10 of the time, and wrong answers were mostly some value of the episode
    /// unrelated to the key.
    ///
    /// The trace is situation, not memory: a vector restored by probes and
    /// cleared with the context. Each event adds `E_prev (*) E_x` after
    /// decaying what is there by `episodic_decay`; each event then reads the
    /// trace back by its own token, `E_x` unbinding it to what followed `x`
    /// earlier in this context, most recent strongest. That recall is one more
    /// feature block.
    ///
    /// Measured on `slots` (64 values, d = 512, decay 1.0, 2000 episodes,
    /// second half read), accuracy on "what is slot s now":
    ///
    /// ```text
    ///                                    off     on
    ///   all questions                  0.229  0.895
    ///   user set it in this turn       0.103  0.909
    ///   user set it 6+ turns ago       0.048  0.820
    ///   clerk offered another since    0.302  0.930
    /// ```
    ///
    /// With 512 values -- each seen about ten times -- 0.787. The recall itself
    /// is read without the readout (`Model::episodic_recall`) and names the
    /// right value 0.975 of the time for one set this turn.
    ///
    /// **Off by default, and it should stay off unless the design is changed
    /// on purpose.** It was built on 2026-09-28 to answer MultiWOZ dialogue
    /// state -- a lookup the rule "the last value the user gave" answers at
    /// 0.89 and a dictionary at 0.99. That is precise context association,
    /// which the paper states as a non-goal ("a design that makes the
    /// background exact has rebuilt the prefix under another name"). What it
    /// did establish stays true and is worth keeping on record: without it
    /// nothing in the situation holds a binding past the lag blocks, so "what
    /// did this context bind to X" has nowhere to be read from; the long-term
    /// banks answer it with the prior. Whether the design wants that
    /// capability is a design question, not a tuning one.
    ///
    /// Measured with it on: facts matrix, all ten facts kept plus F11;
    /// PhysioNet codelength neutral (3.380 against 3.379 bits); MultiWOZ, 1000
    /// dialogues, d = 512, 0.525 -> 0.619 (last user mention 0.891), no gain
    /// beyond three turn pairs and none where the rule is wrong.
    pub episodic: bool,
    /// Per-event decay of the episode trace. Speech decays it; silence does not.
    /// Replacement already keeps a key current, so decay only trades reach
    /// for noise: at d = 256, recall of a value set this turn was 0.91 at 0.97
    /// and 0.75 at 1.0, of one set 6+ turns ago 0.16 and 0.39. At d = 512 the
    /// trade vanishes and 1.0 is better everywhere.
    pub episodic_decay: f32,
    /// Initial gain on the recall in the codebook term. The gain is then
    /// learned (see `Model::learn_naming_gain`) -- by the exact gradient of the
    /// charge, which is the ledger being descended by something other than the
    /// readout's delta rule. Inactive while `episodic` is off.
    pub episodic_codebook: f32,
    /// See `StepRule`. Fixed until a change-point measurement says otherwise.
    pub step_rule: StepRule,
    /// Floor on the error-driven step, so a settled row is never frozen
    /// outright and can always feel a world that has changed.
    pub step_floor: f32,
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
            eta: 0.2,
            event_locked_key: true,
            cleanup_floor_mult: 1.6,
            readout_codebook: 1.0,
            cursor_fade: 0.4,
            verify_sigma: 4.0,
            emit_strongest: true,
            traj: 4,
            mem_banks: 64,
            superpose: true,
            unbind_mix: 0.7,
            row_norm_cap: 0.0,
            row_decay: 0.0,
            readout_bias: false,
            hard_negatives: true,
            neg_samples: 0,
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
            bypass_graph: true,
            walk_during_gap: true,
            walk_needs_retrieval: false,
            anchor: 0.35,
            route_perturb: 0,
            write_toward_embedding: true,
            use_binding: true,
            bind_self: true,
            verify_gate: false,
            bind_overt: false,
            ep_context: false,
            diag_recall: false,
            ep_context_levels: (1, 1),
            ep_context_unitary: true,
            ep_context_gamma: 0.0,
            episodic: false,
            episodic_decay: 0.97,
            episodic_codebook: 1.0,
            step_rule: StepRule::Fixed,
            step_floor: 0.02,
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
    /// The cosine a random vector reaches against its best of `vocab` codebook
    /// entries. Everything about cleanup has to be measured against this.
    pub fn codebook_floor(&self) -> f32 {
        (2.0 * (self.vocab.max(2) as f32).ln() / self.d as f32).sqrt()
    }

    /// The read-back score a genuine triple has to clear.
    pub fn verify_min(&self) -> f32 {
        self.verify_sigma / (self.d as f32).sqrt()
    }

    pub fn cleanup_min_cos(&self) -> f32 {
        self.cleanup_floor_mult * self.codebook_floor()
    }

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
    /// Switches read from the environment, for experiments run from the
    /// command line: ALM_EPISODIC=1 turns the episode trace on, and
    /// ALM_EPISODIC_DECAY sets its decay (default 1.0 when turned on this way).
    pub fn apply_env(&mut self) {
        // ALM_CONFIG=paper: the configuration `experiments::closeout` measured
        // for the paper -- graph bypassed and frozen, routing by the bound
        // traces, reads entered by content, no anchor.
        if std::env::var("ALM_CONFIG").ok().as_deref() == Some("graph-bypassed") {
            self.bypass_graph = true;
            eprintln!("  graph bypassed, everything else default");
        }
        if std::env::var("ALM_CONFIG").ok().as_deref() == Some("paper") {
            self.bypass_graph = true;
            self.freeze_operator = true;
            self.route_query = RouteQuery::Bound;
            self.read_entry_by_content = true;
            self.anchor = 0.0;
            eprintln!("  paper configuration");
        }
        if let Some(r) = std::env::var("ALM_RUNGS").ok().and_then(|v| v.parse::<usize>().ok()) {
            self.rungs = r;
        }
        if let Ok(v) = std::env::var("ALM_EP_CONTEXT_LEVELS") {
            let p: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if p.len() == 2 {
                self.ep_context_levels = (p[0], p[1]);
            }
        }
        if let Some(g) = std::env::var("ALM_EP_CONTEXT_GAMMA").ok().and_then(|v| v.parse().ok()) {
            self.ep_context_gamma = g;
        }
        if std::env::var("ALM_EP_CONTEXT_RAW").is_ok() {
            self.ep_context_unitary = false;
        }
        if std::env::var("ALM_EP_CONTEXT").is_ok() {
            self.ep_context = true;
            eprintln!("  triples bound to the situation they happened in");
        }
        if std::env::var("ALM_BIND_OVERT").is_ok() {
            self.bind_overt = true;
            eprintln!("  what was said binds into the key of what followed");
        }
        if std::env::var("ALM_EPISODIC").is_ok() {
            self.episodic = true;
            self.episodic_decay =
                std::env::var("ALM_EPISODIC_DECAY").ok().and_then(|x| x.parse().ok()).unwrap_or(1.0);
            eprintln!("  episode trace on, decay {}", self.episodic_decay);
        }
    }

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
        let extra = if self.bind_self { 1 } else { 0 }
            + if self.episodic { 1 } else { 0 }
            + if self.ep_context { 1 } else { 0 };
        // diag_recall adds no block: it only reads.
        match self.bind_mode {
            // Off means no conjunctions, not no blocks: the lag-zero block is
            // the event itself and does not depend on the mode. Without this,
            // "the parts without the whole" was not expressible, and `rebind`
            // silently skipped the self block because `binds` was empty.
            BindMode::Off => extra,
            BindMode::EventLag => self.bind_lags + extra,
            BindMode::Band => self.rungs + extra,
            BindMode::Both => self.bind_lags + self.rungs + extra,
        }
    }

    /// Rate of rung k.
    pub fn rho(&self, k: usize) -> f32 {
        self.rho0 * self.beta.powi(k as i32)
    }
}
