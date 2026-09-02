# alm — one tick, one factor

A challenge–response continual memory. Not autoregressive, no reward, no labels.
The only external signal is the codelength of what the world actually said,
charged on the events where it said something, and its only job is to decide
what gets written and how deeply.

Rust, zero dependencies. The exactness assertions need bit-reproducible floating
point and a counter-based RNG whose draws do not depend on call order, and both
are easier to guarantee by writing the fifty lines than by importing them.

```bash
cargo test                                  # 12 assertions: 6 data, 6 structural
cargo run --release -- gencheck             # generator self-tests only
cargo run --release -- quick --ticks 120000
cargo run --release -- full  --ticks 400000 --out results.csv
```

A full run is about a minute on one core. There is no cluster path and nothing
here wants one at this scale.

---

## The five structural constraints

| | | where |
|---|---|---|
| **A1** | **Rate limit.** Constant compute per tick. A level may take several ticks to mature, so a hard discrimination costs *time* rather than accuracy, but the per-tick budget never moves. | `descent.rs` |
| **A2** | **Event-driven accounting.** Charged only where the world speaks. The identity operator at baseline makes "no charge" and "no content write" the same fact rather than two axioms. | `model.rs`, `embed.rs` |
| **A3** | **Read/write separation.** One deterministic write path whose routing consistency is exactly one; many exploratory read particles, for which consistency is not a requirement. | `graph.rs`, `descent.rs` |
| **A4** | **No content write at baseline.** Realised in its strongest form — exactly zero — because A2 leaves no error at baseline to project. The subspace projection the design note allowed for turned out to be unnecessary and is not implemented. | `tests/structural.rs` |
| **A5** | **Timescale reachability.** The world writes every rung; self-generated content writes only the fastest. Enforced by keeping two cascades, so the slow rungs have no storage for self content at all. | `ladder.rs` |

Everything learned is either an edge transform (`graph.rs`) or a sparse readout
row (`tree.rs`). Prototypes are *placed*, priors are *counted*, addresses are
never trained.

## The derivation the code implements

Allocation builds a tree; rung *k* of the ladder feeds level *k*, so the ladder
is not a separate structure but the tree's time axis. After committing `l`
factors the emitted distribution is

```
q_l(o) = prod_{j <= j*(o)} q(u_j | u_{j-1}) . prior(o | u_{j*})
```

where `j*(o)` is the deepest level at which `o`'s ancestor is still on the walked
path, and mass that went to branches the walk did not enter is filled by those
branches' own priors, with PPM-C escape down to a uniform. It telescopes to one
at *every* `l`, which is the point: the code can be settled at any moment, and
stopping early costs exactly the prior entropy of the subtree never descended.
`tests/structural.rs` asserts the normalisation; `code.rs` derives it.

A particle set is a bounded posterior over paths, so the emitted distribution is
the weight-mixture of the particles' own. With one particle it reduces to the
single-path formula.

---

## What the tests assert

Equalities, not statistics. Each either holds or the implementation is wrong.

* Write routing consistency is exactly 1 under learning.
* 500 baseline ticks leave every stored weight **bit-identical**.
* Self-generated content has exactly zero energy above the fastest rung.
* The emitted distribution sums to one at every depth.
* Appending a node leaves every existing prototype and row bit-identical.
* A counter-based draw does not depend on how many draws preceded it.

Plus six on the generator, which run **before** any model is built.

## Suspect the data before the mechanism

`gencheck.rs` measures every load-bearing property of the stream from the ticks
the model will actually see, and `assert_usable` panics before a model exists.
A flat curve on a source that never held the structure is a fact about the
source, and over-ablating in response to it kills mechanisms that were never
given anything to do.

This already earned its keep once. The conjunction check first read

```
I(target ; cue A) = 2.613 bits      conjunctive gain = -0.117 bits
```

which looks like a broken generator and was a broken *measurement*: in mode A a
cue also names its regime, so the marginals are dominated by log2(domains) and
the co-information subtracts that redundancy twice. The claim is a within-regime
claim and had to be measured as one. Conditioned on the regime:

```
I(T;A|D) = 0.046   I(T;B|D) = 0.107   I(T;A,B|D) = 2.571   gain = 2.418 bits
```

against log2(6) = 2.585. The report prints both, and the unconditional numbers
are kept precisely because reading them as the conjunction test is the mistake.

---

## Results, 400k ticks, seed 0x5EED1234

### What works

**The coarse address separates regimes.**

```
leaf purity by regime   0.928     (chance 0.167)
```

Writes from one regime land in leaves that serve almost only that regime, with
no task identifier, no boundary and no replay. Allocation ran to 128 nodes,
depth 3, 103 leaves, entirely from the stream.

**The mechanism beats the like-for-like control on second-order items.**

```
second-order items:  ours 7.172 bits   PPM-C order 4 (same stream) 9.939 bits
```

Both are charged on the same events under the same protocol. PPM with the
silence *removed* reaches 0.380 bits on those items, but that variant is handed
the segmentation the mechanism refuses to assume, so it is reported as an upper
bound and not as a control.

### What does not work

**The fine address does not isolate the conjunction.**

```
leaf purity by cue pair  0.123    (chance 0.049; 20.3 distinct pairs per leaf)
second-order accuracy    0.021 - 0.026    (within-regime chance 0.167)
```

This is the decisive number and it splits the failure cleanly. The square is
additive modulo *m*, so the target is not a linear function of the two cues'
superposed embeddings — a linear readout on the payload cannot represent it, in
the same way a linear model cannot represent parity. The only remaining route is
for the *address* to give each pair its own leaf, and the pair purity says it
does not, by a wide margin.

That is consistent with what the allocator is: its criterion is dispersion of
the winning similarity, which is a *regime* detector. Nothing in it has any
reason to carve out thirty-six cue-pair cells inside a regime. The mechanism is
doing the job it was built for and not a job it was never given a criterion for.

**The sharpening curve is flat.**

```
H(t+0) 8.096  ->  H(t+23) 8.105     drop -0.009 bits
```

This is the first falsification test in the design note and it currently fails.
The committed factors are not sharper than the priors they replace, so the
precondition the "one tick, one factor" accounting rests on is not met on this
source at this configuration. Given that the fine address is at chance, there is
nothing for the deeper factors to sharpen *with*, so this reading is not
independent of the one above — but it is not excused by it either.

**The rung sweep is uninformative.** Plateau width is 7 of 7 at rungs 2, 4 and 6,
i.e. the second-order window is flat in separation. With pair purity at chance
there is no signal for the window to have a shape, so this measures nothing yet
and should not be read as evidence about the band structure either way.

### Ablations

```
full                                7.955 bits/event
flatten to depth 1 (sanity)         8.545
no readout, counts only             7.914
no write channel, no eligibility    7.704
one particle                        7.903
```

Two of these are worth stating plainly rather than smoothing over. The learned
readout is currently *costing* 0.04 bits against pure count-based backoff, and
removing the gap-time credit path *improves* things by 0.25 bits. Both are
consistent with the diagnosis: with the address not isolating pairs, the readout
is being trained on a problem it cannot represent, and the eligibility credit is
distributing that same signal onto edges.

---

## What would have to change

In the order I would try them.

1. **Give the allocator a criterion that can isolate conjunctions.** Dispersion
   of similarity detects regimes. Splitting on *predictive* disagreement — a node
   whose own emitted targets stay high-entropy after it has enough observations —
   would carve where the content actually needs carving. This is the one change
   that addresses the measured quantity directly.
2. **Let the payload chain be nonlinear enough to matter.** The residual hop is
   `nu(p + tanh(Wp))` with `W` initialised small, so it is close to the identity
   and the payload arrives at the readout barely transformed. Whether a
   composition can be learned at all is currently untested rather than answered.
3. **A source with an easier conjunction alongside the hard one.** The Latin
   square is the maximally non-linear choice. A table with zero marginals but a
   linearly separable structure would say whether the failure is the address or
   the representation, instead of confounding them.

Not tried, deliberately: more ablations. Three is already more than the evidence
supports, and each of the two surprising ones above has an explanation that the
diagnosis predicts rather than a component that needs deleting.

---

## Layout

```
src/num.rs          counter-based RNG, fixed-order reductions, small dense linalg
src/config.rs       one Config; every field tagged FREE / DERIVED / CEILING / INHERITED
src/embed.rs        fixed embeddings, channel rotations, rank-one token operators
src/ladder.rs       cascaded band-pass background; two cascades, world and self
src/graph.rs        small-world ring, fixed keys, learned transforms, eligibility
src/tree.rs         allocation tree, occupancy counts, per-level calibration
src/code.rs         q_l, its normalisation, the charge, the entropy
src/descent.rs      particles, evidence accumulation, resampling, backtracking
src/model.rs        the tick loop
src/gen.rs          the stream
src/gencheck.rs     data self-tests
src/metrics.rs      three curves, one control, the purity diagnostics
src/baseline.rs     interpolated PPM-C, both variants
src/experiments.rs  drivers
```

### Parameter audit

The design claim is that exactly one genuinely free knob was added relative to
the reference mechanism. `config.rs` carries the tags in the source so that
breaking the claim is visible in a diff.

`particles` is the free one — the per-tick compute budget, with beam width, top-k
and scheduler slots all folded into it. `beta` and `trace_lambda` are derived
from the stream's timescales; the commit threshold is read out of the per-level
calibration counters; the resampling threshold is ESS < B/2. `rungs`,
`max_children` and `max_nodes` are ceilings, not settings: realised depth and
branching are emergent and reported.

Three values inherited *in form* were rescaled, and the reason is recorded at
each: `grow_theta` (the reference's 0.05 assumes an unnormalised context; here
both query and prototype are unit vectors and 0.05 splits everything in sight),
`max_ticks_per_level` (a descent has to fit inside a response window), and
`commit_fallback_slack` (calibration may raise the commit bar and never lower
it, or a badly calibrated level commits instantly on no evidence).

### Notes for anyone extending this

* Reads never allocate. Growth belongs to the write path, where it follows
  content that was actually stored.
* The memory walk happens once per drive, to the same depth and by the same
  routing as the write walk. Spreading hops across the tree descent couples two
  resources that are not the same one, and the readout then gets evaluated at a
  point it was never trained at.
* Backtracking scans the path from the root *down* and truncates at the first
  level that no longer holds. Scanning upward and stopping at the first level
  that still agrees looks equivalent and is not: deepening appends single-child
  nodes where the recorded branch is trivially the argmax, so an upward scan
  halts immediately and a stale coarse commitment is never revisited.
* Do not iterate a `HashMap` anywhere a float result depends on the order.
