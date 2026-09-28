//! One tick, one factor.
//!
//! A challenge-response continual memory. Not autoregressive, no reward, no
//! labels: the only external signal is the codelength of what the world actually
//! said, charged on the events where it said something, and its only job is to
//! decide what gets written and how deeply.
//!
//! The five structural constraints the implementation is built to honour:
//!
//! * **A1 rate limit.** Constant compute per tick. A level may take several
//!   ticks to mature, so a hard discrimination costs time rather than accuracy,
//!   but the per-tick budget never moves. (`descent.rs`)
//! * **A2 event-driven accounting.** Charged only where the world speaks. The
//!   identity operator at baseline makes "no charge" and "no content write" the
//!   same fact rather than two axioms. (`model.rs`, `embed.rs`)
//! * **A3 read/write separation.** One deterministic write path whose
//!   consistency is exactly one; many exploratory read particles for which
//!   consistency is not a requirement. (`graph.rs`, `descent.rs`)
//! * **A4 no content write at baseline.** Realised in its strongest form --
//!   exactly zero -- because A2 leaves no error at baseline to project. The
//!   subspace projection the design note allowed for turned out to be
//!   unnecessary and is therefore not implemented. (`tests/structural.rs`)
//! * **A5 timescale reachability.** The world writes every rung; self-generated
//!   content writes only the fastest. Enforced by keeping two cascades, so the
//!   slow rungs have no storage for self content at all. (`ladder.rs`)
//!
//! Everything learned is a readout row, written by an error-driven delta rule.
//! Addresses are never trained: banks are named by a hash of token identities,
//! and the codebook, the channel rotations and the ladder are fixed.
//!
//! # Before changing anything: what this design is, and how it has drifted
//!
//! Read `paper.tex` ("One Tick, One Link") section 2 and the process layer of
//! the foundations preprint (`pzfc-preprint.tex`, section 5) first. The
//! commitments that constrain every change, in the paper's own terms:
//!
//! 1. **Organisation, not optimisation.** The ledger (prequential codelength)
//!    may decide *what gets written and how deeply*; it may not be descended.
//!    No gradient crosses a memory write. The readout's delta rule is the one
//!    sanctioned learner -- error-driven, so a row holds what needed correcting
//!    here, not what was common here -- and anything else fitted to the charge
//!    is a departure that has to argue for itself.
//! 2. **An imprecise background is the point.** Precise context association is
//!    a stated non-goal: a design that makes the context exact has rebuilt the
//!    prefix under another name.
//! 3. **The world is exogenous and influenced, not generated.** Behaviour
//!    should influence what is experienced without generating it (the paper's
//!    dark-room discussion and its `beta`). This has never been implemented:
//!    in every experiment so far the world's next event is fixed in advance
//!    and ignores what the model said. So the *response* half of
//!    challenge-response has never been tested. That is the open problem, not
//!    a benchmark to add.
//! 4. **Time is a resource.** A replayed file cannot test it.
//! 5. **No single number.** Learners are compared by the partial order of the
//!    facts they maintain (`facts.rs`), not by a score.
//!
//! How it drifted, so it is not repeated (2026-09-16 to 09-28): open-loop
//! benchmarks with fixed answers -- next-token codelength, in-hospital death,
//! process-log questions, dialogue state -- were used to judge the design, and
//! mechanisms were then added to raise their numbers. Those mechanisms are kept
//! in the code, *off by default*, each with its evidence and the reason it is
//! off in its `Config` documentation:
//!
//! * `verify_gate` -- read-back check on the lag blocks. Maintains none of the
//!   facts; its only support was a snapshot-prediction benchmark.
//! * `episodic` (+ the learned naming gain) -- an exact per-context key/value
//!   trace built to answer MultiWOZ lookups. Conflicts with commitment 2, and
//!   its naming gain is fitted to the charge, against commitment 1.
//! * `walk_needs_retrieval` -- a silence gate, built and withdrawn.
//! * `step_rule` other than `Fixed` -- consolidation rules, measured and not
//!   adopted.
//!
//! Turning one of these on to improve a benchmark is exactly the drift above.
//! The instruments that produced those benchmarks (`acquire`, `patient`,
//! `bedside`, `judge`, `bpi`, `woz`, `slots`) stay as records; each says at
//! its top that it is open-loop.

pub mod acquire;
pub mod baseline;
pub mod bedside;
pub mod bpi;
pub mod woz;
pub mod slots;
pub mod budget;
pub mod chainmem;
pub mod clinical;
pub mod code;
pub mod compose;
pub mod config;

pub mod embed;
pub mod epochs;
pub mod experiments;
pub mod facts;
pub mod gen;
pub mod gencheck;
pub mod graph;
pub mod judge;
pub mod ladder;
pub mod metrics;
pub mod model;
pub mod num;
pub mod partial;
pub mod patient;
pub mod physio;
pub mod plastic;
pub mod rarity;
pub mod probe;
pub mod scan;
pub mod store;
pub mod stepsize;
pub mod stream;
