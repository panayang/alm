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

use crate::config::Config;
use crate::num::Running;

pub struct Store {
    pub d: usize,
    /// Width of a row: the state block, one per bound trace, one per band.
    pub fw: usize,
    pub vocab: usize,
    /// One row per token, shared by everything. The only learned readout.
    pub rows: Vec<(u32, Vec<f32>)>,
    row_index: HashMap<u32, usize>,
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

impl Store {
    pub fn new(cfg: &Config) -> Self {
        Store {
            d: cfg.d,
            fw: cfg.feature_blocks() * cfg.d,
            vocab: cfg.vocab,
            rows: Vec::new(),
            row_index: HashMap::new(),
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
        self.rows.push((tok, vec![0.0; fw]));
        let i = self.rows.len() - 1;
        &mut self.rows[i].1
    }

    /// The delta-rule write, against the same distribution the ledger charges.
    ///
    /// A write, not a fit: the update is local to the row and needs nothing
    /// upstream of it. Rows outside the touched set are left exactly untouched,
    /// which is what keeps occupancy a measure of stored content.
    pub fn write(
        &mut self,
        sc: &crate::code::Scored,
        phi: &[f32],
        target: u32,
        negatives: &[u32],
        eta: f32,
    ) {
        let d = self.fw.min(phi.len());
        let mut touched: Vec<u32> = Vec::with_capacity(negatives.len() + 1);
        touched.push(target);
        for &n in negatives {
            if n != target && !touched.contains(&n) {
                touched.push(n);
            }
        }
        for &t in touched.iter() {
            let q = sc.prob_of(self, t);
            let err = if t == target { 1.0 - q } else { -q };
            let row = self.row_mut_or_insert(t);
            for i in 0..d {
                row[i] += eta * err * phi[i];
            }
        }
    }

    /// Tokens that have rows: where a write samples its negatives from.
    pub fn known(&self) -> Vec<u32> {
        self.rows.iter().map(|(t, _)| *t).collect()
    }
}
