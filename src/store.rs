//! What is stored: one associative row per token. Nothing else.
//!
//! # Why there are no counts
//!
//! The previous design carried a per-node count table and used it as a prior.
//! A count is a frequency estimate, and as the stream accumulates any node's
//! count distribution converges on the corpus marginal -- it becomes the global
//! unigram. A counted prior therefore *stops discriminating exactly as the data
//! grows*, which is the same shape of failure as address crowding: it degrades
//! with scale.
//!
//! More data tends toward chaos. What makes it mean something is organisation,
//! and organisation is not frequency.
//!
//! The delta rule already is that organisation. It is error-driven, so a row
//! holds what needed correcting here rather than what was common here, and a
//! token that is already predicted stops updating its row because `err` goes to
//! zero. Counts accumulate and so degrade under growth; the delta rule converges
//! and so is stable under it. So the rows stay and the counts go, and with them
//! the escape chain.
//!
//! What the prior used to provide is reassigned rather than dropped: the fuzzy
//! multi-level background now enters the *features*, and the readout learns to
//! use it. Graceful degradation becomes a property of the dynamics -- an
//! unfamiliar background leaves the state in a region memory has not shaped, and
//! an unshaped state reads out diffusely -- rather than of a smoothing rule.
//!
//! The cost is real and is named where it belongs: a guaranteed codelength floor
//! is traded for an empirical one. `metrics.rs` therefore treats the worst
//! single-event charge and the 99th percentile of the charge distribution as
//! required instruments rather than diagnostics. A heavy tail here is confident
//! wrongness, which is fabrication.

use std::collections::HashMap;

use crate::config::{Config, StepRule};
use crate::num::Running;

pub struct Store {
    pub d: usize,
    /// Width of a row: the state block, one per bound trace, one per band.
    pub fw: usize,
    pub vocab: usize,
    /// One row per token, shared by everything. The only learned readout.
    pub rows: Vec<(u32, Vec<f32>)>,
    /// A per-token additive term, off by default.
    ///
    /// This is a counted prior wearing a different hat, and this file opens by
    /// explaining why counts were removed. It is here because the readout was
    /// measured, in isolation on a structureless stream, to carry *no*
    /// frequency information at all -- 5.99 bits against a marginal of 4.86 and
    /// a uniform of 6.00 -- and every real stream is skewed. Whether that is a
    /// defect or the design refusing to be a frequency counter is exactly what
    /// the two tests around it are for, so it is a flag and not a decision.
    pub bias: Vec<f32>,
    pub use_bias: bool,
    /// See `Config::row_norm_cap`. Zero disables.
    pub row_cap: f32,
    /// See `Config::row_decay`. Zero disables.
    pub row_decay: f32,
    /// Per row: how many corrections it has had, and the running mean of the
    /// size of those corrections. Neither reaches the emitted distribution --
    /// they decide only how far a row still moves. See `StepRule`.
    pub step_rule: StepRule,
    pub step_floor: f32,
    corrections: Vec<u32>,
    err_ema: Vec<f32>,
    row_index: HashMap<u32, usize>,
    /// The same token ids as `rows`, kept as a slice so a hot loop does not have
    /// to rebuild them. `known()` allocated a fresh Vec of every known token on
    /// each call, and cleanup calls it once per response per silent tick.
    known_toks: Vec<u32>,
    pub answer_calib: Calibration,
    pub write_surprise: Running,
}

/// Per-bin reliability counters: (observations, correct).
#[derive(Clone)]
pub struct Calibration {
    pub bins: Vec<(u64, u64)>,
}

impl Calibration {
    fn new(nbins: usize) -> Self {
        Calibration { bins: vec![(0, 0); nbins] }
    }
    #[inline]
    fn bin_of(&self, conf: f32) -> usize {
        let n = self.bins.len();
        ((conf * n as f32) as usize).min(n - 1)
    }
    pub fn push(&mut self, conf: f32, correct: bool) {
        let b = self.bin_of(conf);
        self.bins[b].0 += 1;
        if correct {
            self.bins[b].1 += 1;
        }
    }
    pub fn accuracy(&self, b: usize) -> f32 {
        let (n, c) = self.bins[b];
        if n == 0 {
            0.0
        } else {
            c as f32 / n as f32
        }
    }
    pub fn observations(&self) -> u64 {
        self.bins.iter().map(|x| x.0).sum()
    }
    pub fn ece(&self) -> f32 {
        let total = self.observations();
        if total == 0 {
            return 0.0;
        }
        let n = self.bins.len();
        let mut e = 0.0f32;
        for b in 0..n {
            let (cnt, _) = self.bins[b];
            if cnt == 0 {
                continue;
            }
            let centre = (b as f32 + 0.5) / n as f32;
            e += (cnt as f32 / total as f32) * (self.accuracy(b) - centre).abs();
        }
        e
    }
}

/// How fast a row's error history forgets. Slow enough that one lucky event
/// does not unsettle a row, fast enough that a changed world is felt within a
/// few dozen corrections.
const ERR_EMA_RATE: f32 = 0.05;

impl Store {
    pub fn new(cfg: &Config) -> Self {
        Store {
            d: cfg.d,
            fw: cfg.feature_blocks() * cfg.d,
            vocab: cfg.vocab,
            rows: Vec::new(),
            bias: vec![0.0; cfg.vocab],
            use_bias: cfg.readout_bias,
            row_cap: cfg.row_norm_cap,
            row_decay: cfg.row_decay,
            step_rule: cfg.step_rule,
            step_floor: cfg.step_floor,
            corrections: Vec::new(),
            err_ema: Vec::new(),
            row_index: HashMap::new(),
            known_toks: Vec::new(),
            answer_calib: Calibration::new(cfg.calib_bins),
            write_surprise: Running::default(),
        }
    }

    pub fn occupied_rows(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    pub fn row_of(&self, tok: u32) -> Option<&[f32]> {
        self.row_index.get(&tok).map(|&i| self.rows[i].1.as_slice())
    }

    pub fn row_mut_or_insert(&mut self, tok: u32) -> &mut Vec<f32> {
        if let Some(&i) = self.row_index.get(&tok) {
            return &mut self.rows[i].1;
        }
        let fw = self.fw;
        self.row_index.insert(tok, self.rows.len());
        self.known_toks.push(tok);
        self.corrections.push(0);
        // A row starts fully plastic: it has no error history to be settled by.
        self.err_ema.push(1.0);
        self.rows.push((tok, vec![0.0; fw]));
        let i = self.rows.len() - 1;
        &mut self.rows[i].1
    }

    /// The delta-rule write, against the same distribution the ledger charges.
    ///
    /// A write, not a fit: the update is local to the row and needs nothing
    /// upstream of it. Rows outside the touched set are left exactly untouched.
    ///
    /// A row can now be created by being a *competitor* and not only by being a
    /// target. That follows from the scoring fix in `code.rs`: the codebook term
    /// ranges over the whole vocabulary, so the model can emit a token the world
    /// has never said, and a token the model can emit must be one it can be
    /// corrected about -- otherwise the one path that can fabricate is the one
    /// path with no corrective term on it. The cost is that occupancy is no
    /// longer "tokens the world has said"; it is "tokens memory has an opinion
    /// about", which is the quantity the readout actually depends on.
    pub fn write(
        &mut self,
        sc: &crate::code::Scored,
        phi: &[f32],
        target: u32,
        negatives: &[u32],
        eta: f32,
    ) {
        let d = self.fw.min(phi.len());
        // Sorted and deduplicated rather than checked one by one: a dense write
        // hands over the whole vocabulary, and a linear `contains` per entry
        // made that quadratic.
        let mut negs: Vec<u32> = negatives.iter().copied().filter(|&n| n != target).collect();
        negs.sort_unstable();
        negs.dedup();
        let mut touched: Vec<u32> = Vec::with_capacity(negs.len() + 1);
        touched.push(target);
        touched.extend(negs);
        for &t in touched.iter() {
            let q = sc.prob_of(self, t);
            let err = if t == target { 1.0 - q } else { -q };
            let cap = self.row_cap;
            // How far this row still moves. Both rules are bookkeeping about
            // the row's own history of being wrong; neither is a term in the
            // distribution the ledger charges.
            let idx = self.row_index.get(&t).copied();
            let scale = match self.step_rule {
                StepRule::Fixed => 1.0,
                StepRule::InverseCount => {
                    let n = idx.map(|i| self.corrections[i]).unwrap_or(0);
                    1.0 / (1.0 + n as f32)
                }
                StepRule::ErrorDriven => {
                    let e = idx.map(|i| self.err_ema[i]).unwrap_or(1.0);
                    e.max(self.step_floor)
                }
            };
            let eta = eta * scale;
            if self.use_bias {
                if let Some(b) = self.bias.get_mut(t as usize) {
                    *b += eta * err;
                    if cap > 0.0 {
                        *b = b.clamp(-cap, cap);
                    }
                }
            }
            let decay = self.row_decay;
            let row = self.row_mut_or_insert(t);
            if decay > 0.0 {
                let keep = 1.0 - decay;
                for x in row.iter_mut() {
                    *x *= keep;
                }
            }
            for i in 0..d {
                row[i] += eta * err * phi[i];
            }
            // Projected back onto the ball only when it has left it. A row
            // inside the bound is not touched, so this is not decay.
            if cap > 0.0 {
                let n = crate::num::norm(row);
                if n > cap {
                    let k = cap / n;
                    for x in row.iter_mut() {
                        *x *= k;
                    }
                }
            }

            // The row's own history of being wrong, updated from the raw error
            // and not the scaled one. That is what lets the error-driven rule
            // come back: a settled row whose world changes starts being wrong
            // again, its mean rises, and its step grows with it. A count
            // cannot do this, which is the whole difference between the two
            // rules and the reason both are here.
            if let Some(i) = self.row_index.get(&t).copied() {
                self.corrections[i] = self.corrections[i].saturating_add(1);
                let a = ERR_EMA_RATE;
                self.err_ema[i] = (1.0 - a) * self.err_ema[i] + a * err.abs();
            }
        }
    }

    /// Tokens that have rows: where a write samples its negatives from.
    pub fn known(&self) -> Vec<u32> {
        self.known_toks.clone()
    }

    /// The known tokens without a copy.
    #[inline]
    pub fn known_slice(&self) -> &[u32] {
        &self.known_toks
    }
}
