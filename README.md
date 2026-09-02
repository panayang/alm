# alm — one tick, one factor

A challenge–response continual memory. Not autoregressive, no reward, no labels.
The only external signal is the codelength of what the world actually said,
charged on the events where it said something, and its only job is to decide
what gets written and how deeply.

Rust, zero dependencies. The exactness assertions need bit-reproducible floating
point and a counter-based RNG whose draws do not depend on call order, and both
are easier to guarantee by writing the fifty lines than by importing them.

```bash
cargo test                                  # 14 assertions: 6 data, 8 structural
cargo run --release -- gencheck             # generator self-tests only
cargo run --release -- quick --ticks 120000
cargo run --release -- full  --ticks 400000 --out results.csv
```

---

## The five structural constraints

| | | where |
|---|---|---|
| **A1** | **Rate limit.** Constant compute per tick. A level may take several ticks to mature and the memory walk advances one hop per tick, so depth — in the tree and in the memory alike — is a temporal resource: a response reaches as far as the world gave it time for. | `descent.rs` |
| **A2** | **Event-driven accounting.** Charged only where the world speaks. The identity operator at baseline makes "no charge" and "no content write" the same fact rather than two axioms. | `model.rs`, `embed.rs` |
| **A3** | **Read/write separation.** One deterministic write path whose routing consistency is exactly one; many exploratory read particles, for which consistency is not a requirement. | `graph.rs`, `descent.rs` |
| **A4** | **No content write at baseline.** Realised in its strongest form — exactly zero — because A2 leaves no error at baseline to project. | `tests/structural.rs` |
| **A5** | **Timescale reachability.** The world writes every rung; self-generated content writes only the fastest. Enforced by keeping two cascades, so the slow rungs have no storage for self content at all. | `ladder.rs` |

Everything learned is either an edge transform (`graph.rs`) or a sparse readout
row (`tree.rs`). Prototypes are *placed*, priors are *counted*, addresses are
never trained.

## Three streams, kept apart and combinable

Four channels — what the world said, what was said out loud, what was thought
and not said, and which memory is being touched. They are kept apart twice: by
fixed signed-permutation rotations (exactly orthogonal, norm-preserving, O(d),
never trained), and by rung reachability, which is a property of the layout
because the world and the self drive two separate cascades.

Combining them is a different operation and needs a different mechanism. A sum
superposes and loses which cue went with which; circular convolution binds, and
`a ⊛ b` is nearly orthogonal to both factors and distinct for every pair. That
distinction has a falsifiable job, not a decorative one, and it is measured
below.

## The derivation the code implements

Allocation builds a tree; rung *k* of the ladder feeds level *k*, so the ladder
is not a separate structure but the tree's time axis. After committing `l`
factors the emitted distribution is

```text
q_l(o) = prod_{j <= j*(o)} q(u_j | u_{j-1}) . prior(o | u_{j*})
```

with PPM-C escape down to a uniform, and it telescopes to one at *every* `l`, so
the code can be settled at any moment and stopping early costs exactly the prior
entropy of the subtree never descended.

---

## Suspect the data before the mechanism

`gencheck.rs` measures every load-bearing property of the stream from the ticks
the model will actually see, and `assert_usable` panics before a model exists.
It has caught two things so far, both recorded here because a measurement that
was quietly wrong is worth more as a warning than as a deleted commit.

**The conjunction check first read** `I(target;cueA) = 2.613 bits`,
`conjunctive gain = −0.117` — which looks like a broken generator and was a
broken *measurement*: in mode A a cue also names its regime, so the marginals
are dominated by log2(domains) and the co-information subtracts that redundancy
twice. Conditioned on the regime: `I(T;A|D) = 0.049`, `I(T;B|D) = 0.105`,
`I(T;A,B|D) = 2.575` against log2(6) = 2.585.

**Zero marginals cannot be linearly separable.** For each class the winning
region of a zero-marginal table is a permutation pattern — one cell per row and
column — while an additive score `w_c[a] + v_c[b]` carves the grid into
intersections of staircase half-spaces, which contain whole blocks and cannot
isolate m scattered cells for m ≥ 3. So the originally planned "separable
control with zero marginals" is an empty requirement, and the stream carries two
conjunctions instead:

* **Latin square** — zero marginals, *not* representable by a linear readout on
  the superposed cues. Only the address, or a term that stores the pair, can
  answer it.
* **Product code** — cue A names one attribute, cue B the other. Marginals are
  non-zero (forced, by the above) but the pair is needed, and it *is*
  representable. Measured: `I(T;A|D) = 1.557`, `I(T;B|D) = 0.999`,
  `I(T;A,B|D) = 2.553`, cell consistency `1.0000`.

Read together they split an address failure from a representation failure.

---

## Results — 400k ticks, seed 0x5EED1234, single run

Within-regime chance on both conjunctions is 1/6 ≈ 0.167.

### The one clean positive

**The readout learns exactly what it can represent and not what it cannot.**

```
product code (representable)      acc 0.436 – 0.533     ~3x chance
Latin square (not representable)  acc 0.059 – 0.136     below chance
```

That is the predicted signature, with a wide margin, and it is what the whole
two-conjunction design was built to detect. The `no readout` ablation collapses
it (product 0.133, Latin 0.010), so the effect is the learned rows and not the
counts.

**Binding does its predicted job, on the predicted item, four runs running.**

```
binding on:  Latin acc 0.120
binding off: Latin acc 0.072
```

A Latin square is linear in the tensor features of the two cues and not in their
sum, so this is precisely where a bound trace should show and nowhere else. It
is the most reproducible positive in the build, though the base is small.

**The coarse address separates regimes**: leaf purity 0.630 against chance
0.167, with no task identifier, boundary or replay. Cue-pair purity is 0.251
against chance 0.080 — the fine address is doing something, roughly 3× chance.

### What is not supported

**The plateau prediction.** Plateau width is 7 of 7 in every arm: the
second-order window is flat in cue separation. The predicted "plateau whose
width grows with the band count" is not there for either conjunction.

**Depth paying for itself.** At the best settings the flat control and the
shallowest tree are a tie:

```
L = 1 (flat)   Latin 0.124   product 0.491
rungs = 2      Latin 0.136   product 0.533
rungs = 3      Latin 0.120   product 0.436
rungs = 4      Latin 0.108   product 0.365
rungs = 6      Latin 0.059   product 0.234
```

**The sharpening curve.** `H(t+0) 8.439 → H(t+23) 8.416`, a drop of 0.023 bits.
This is the first falsification test in the design and it still fails: the
committed factors are not sharper than the priors they replace.

### Three findings that are about method, not mechanism

**Conclusions reversed twice under more data and under one other knob.** At 120k
ticks, more bands hurt monotonically and the flat control beat every tree; at
400k with the same settings, rungs = 3 was an interior optimum that clearly beat
flat; at 400k with `w_init` raised to its measured optimum, the ordering flipped
back. Nothing here was a coding error — the effects are simply smaller than the
variation between configurations, and single-run single-configuration readings
of them are not safe.

**`rungs` and `w_init` interact, so neither sweep alone identifies an optimum.**
With a near-linear payload chain the tree wants depth 3; with the chain out of
its linear regime it wants depth 2. A 2-D sweep has not been run, and the
default is left at `rungs = 3` deliberately rather than chased to the last
measured best — chasing it is what produced the reversals above.

**The split criterion does not have one winner.** Dispersion wins the address
metrics; surprise, which fragments far less, wins the accuracy that matters:

```
                bits/ev  regime  pair   Latin  product  leaves
dispersion       8.143    0.630  0.251  0.120  0.436     394
surprise @ 8b    8.327    0.251  0.065  0.105  0.539       8
hybrid @ 8b      8.063    0.617  0.160  0.123  0.485      61
```

Address quality and readout data density pull in opposite directions: more
leaves means a better address and less data behind each set of rows. An earlier
note in this file claimed dispersion won "on every axis"; at these settings it
does not, and the tension is the actual finding.

### The baselines

```
ours                                          8.143 bits/event
PPM-C order 4, same stream                    1.416   (second-order 10.864)
PPM-C order 4, silence removed  UPPER BOUND   3.919   (second-order  0.628)
evidence vs the like-for-like PPM: ahead by 114527 bits, anytime-valid p ≤ 2.2e-308
```

The e-process is a test martingale on the per-event likelihood ratio, valid at
any stopping time — which is what a single non-stationary stream run once needs,
and what an average with a standard error cannot give. It is evidence against
the *like-for-like* control only. The variant with the silence removed is handed
the segmentation this design refuses to assume, and it is far ahead of us.

---

## What the tests assert

Equalities, not statistics. Each either holds or the implementation is wrong.

* Write routing consistency is exactly 1 under learning.
* 500 baseline ticks leave every stored weight **bit-identical**.
* Self-generated content has exactly zero energy above the fastest rung.
* The emitted distribution sums to one at every depth.
* Appending a node leaves every existing prototype and row bit-identical.
* A counter-based draw does not depend on how many draws preceded it.
* Two identical runs agree bit for bit, including accumulated codelength.
* A read that has walked as far as the write arrives at the same payload.

Plus six on the generator, which run **before** any model is built.

## Bugs this build has had, and the instrument that now catches each

Recorded because most of them were invisible in the aggregate numbers and were
found only by an instrument that had to be added first.

| bug | how it hid | instrument now |
|---|---|---|
| The write descended the tree on the unwalked payload while reads navigated on the walked one | calibration silently scored read commits against a path computed from a different query | `read_and_write_walks_agree_once_the_read_has_finished` |
| Self channels wrote rung 0; rung 0 fed tree level 6; the tree reached depth 3 — **the self channels were inert** | the poisoning control's three arms were bit-identical, which reads as "no effect" | `rung_visits` histogram + the control prints whether the channel was read |
| The rung sweep therefore varied the band *range*, not the band *count* | every arm produced plausible numbers | `rung_coverage`, reported per arm |
| `ticks_since_event` was reset after emitting, so every response's log began with the previous response's offset | inflated the idea-onset from t+0.5 to t+8.4 | onset means are reported with their event counts |
| `split_bits` was swept over 1–5 while leaf surprise was ~8.2 bits | all four arms were "always split" and came out bit-identical | `leaf_surprise` mean is printed beside the sweep |
| Speech was gated by the *branch* calibration under a may-only-raise rule | deadlock: over-confident anywhere → bar at 0.95 → never speaks → never calibrates | a separate answer calibration, gated by its top populated bin |
| Float sums over `HashMap` iteration in `gencheck` | numbers differed run to run because Rust seeds each process's hasher | sorted before summing; `two_identical_runs_agree_bit_for_bit` |

Known and not yet fixed: `descent.rs` uses `i == 0` for the exploit particle and
for the commitment that feeds calibration, but `leader()` is the argmax by
weight, and resampling breaks the correspondence. Speech still fires on only 28
events, so the speech reaction time is not yet a measurement.

---

## What would have to change

1. **Stop fragmenting the readout.** The split table above says the address and
   the data density fight each other. Sharing readout rows across siblings
   within a regime would let depth refine the address without splitting the
   evidence — which is the reference mechanism's own split (share the transform,
   privatise the readout) applied one level down.
2. **A 2-D sweep over `rungs` × `w_init`**, with more than one seed. Three
   conclusions in this file reversed under a change of one other setting; none
   of the single-knob sweeps here should be trusted to name an optimum.
3. **The sharpening precondition.** Until entropy falls across a response, the
   "one tick, one factor" accounting is not standing on its own feet, and every
   anytime property derived from it is formal rather than earned.

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
src/gen.rs          the stream: first order, two conjunctions, composition
src/gencheck.rs     data self-tests
src/metrics.rs      the curves, the e-process, the purity diagnostics
src/baseline.rs     interpolated PPM-C, both variants
src/experiments.rs  drivers
```

### Notes for anyone extending this

* Reads never allocate. Growth belongs to the write path, where it follows
  content that was actually stored.
* The write walks the memory first and descends the tree on the *walked*
  payload, because that is what the read particles navigate with. Addressing the
  two with different vectors is silent and it poisons the calibration.
* Backtracking scans the path from the root *down* and truncates at the first
  level that no longer holds. Scanning upward and stopping at the first level
  that still agrees looks equivalent and is not: deepening appends single-child
  nodes where the recorded branch is trivially the argmax, so an upward scan
  halts immediately and a stale coarse commitment is never revisited.
* A ladder rung that no write descent reads is not a band the model has,
  whatever the config says. Check `rung_visits` before believing any result that
  varies the rung count.
* Do not iterate a `HashMap` anywhere a float result depends on the order.
