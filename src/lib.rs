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
//! Everything learned is either an edge transform or a sparse readout row.
//! Prototypes are placed, priors are counted, addresses are never trained.

pub mod acquire;
pub mod baseline;
pub mod budget;
pub mod chainmem;
pub mod clinical;
pub mod code;
pub mod compose;
pub mod config;

pub mod embed;
pub mod experiments;
pub mod gen;
pub mod gencheck;
pub mod graph;
pub mod ladder;
pub mod metrics;
pub mod model;
pub mod num;
pub mod partial;
pub mod patient;
pub mod physio;
pub mod plastic;
pub mod rarity;
pub mod scan;
pub mod store;
pub mod stepsize;
pub mod stream;
