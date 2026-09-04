//! The multi-timescale background.
//!
//! A cascade, not a bank of parallel averages. Parallel exponential averages are
//! all low-passes of the same signal and all peak at lag zero, so any single one
//! of them carries almost nothing and a set of them is nearly redundant. In a
//! cascade, rung k takes its input from rung k-1, so the impulse responses are
//! gamma kernels peaked at increasing lags and the *differences* between rungs
//! isolate what changed at each timescale.
//!
//! Two separate cascades are kept, one driven by the world and one by the
//! system's own output. This is not tidiness: A5 says self-generated content
//! must contribute exactly zero to the slow rungs, and in a single cascade it
//! would leak upward through rung 0 no matter how the update was gated. With two
//! cascades the slow rungs have no storage for self content at all, so the
//! assertion is a property of the layout rather than of a threshold.

use crate::config::Config;
use crate::num::{dot, norm, normalize};

#[derive(Clone)]
pub struct Ladder {
    pub d: usize,
    pub rungs: usize,
    pub self_max_rung: usize,
    rho: Vec<f32>,
    world: Vec<Vec<f32>>,
    selfc: Vec<Vec<f32>>,
    /// Band-pass stack, recomputed once per tick and shared by every reader.
    deltas: Vec<Vec<f32>>,
}

impl Ladder {
    pub fn new(cfg: &Config) -> Self {
        let d = cfg.d;
        let rungs = cfg.rungs;
        let self_rungs = (cfg.self_max_rung + 1).min(rungs);
        let mut l = Ladder {
            d,
            rungs,
            self_max_rung: cfg.self_max_rung,
            rho: (0..rungs).map(|k| cfg.rho(k)).collect(),
            world: vec![vec![0.0; d]; rungs],
            selfc: vec![vec![0.0; d]; self_rungs],
            deltas: vec![vec![0.0; d]; rungs],
        };
        l.refresh();
        l
    }

    /// One world observation. Rung 0 takes the input; rung k takes rung k-1.
    pub fn observe_world(&mut self, v: &[f32]) {
        let r0 = self.rho[0];
        for i in 0..self.d {
            let cur = self.world[0][i];
            self.world[0][i] = cur + r0 * (v[i] - cur);
        }
        for k in 1..self.rungs {
            let rk = self.rho[k];
            for i in 0..self.d {
                let prev = self.world[k - 1][i];
                let cur = self.world[k][i];
                self.world[k][i] = cur + rk * (prev - cur);
            }
        }
    }

    /// One self observation (overt, covert, or the write channel), confined to
    /// the fast rungs by A5.
    pub fn observe_self(&mut self, v: &[f32]) {
        if self.selfc.is_empty() {
            return;
        }
        let r0 = self.rho[0];
        for i in 0..self.d {
            let cur = self.selfc[0][i];
            self.selfc[0][i] = cur + r0 * (v[i] - cur);
        }
        for k in 1..self.selfc.len() {
            let rk = self.rho[k];
            for i in 0..self.d {
                let prev = self.selfc[k - 1][i];
                let cur = self.selfc[k][i];
                self.selfc[k][i] = cur + rk * (prev - cur);
            }
        }
    }

    /// Advance one baseline tick. The background flows even when the world is
    /// silent -- that is how elapsed time gets represented, and it is what makes
    /// gap length recoverable without anyone labelling it.
    pub fn tick_baseline(&mut self) {
        let r0 = self.rho[0];
        for i in 0..self.d {
            self.world[0][i] *= 1.0 - r0;
        }
        for k in 1..self.rungs {
            let rk = self.rho[k];
            for i in 0..self.d {
                let prev = self.world[k - 1][i];
                let cur = self.world[k][i];
                self.world[k][i] = cur + rk * (prev - cur);
            }
        }
        // The self cascade ages the same way the world's does. Decaying only
        // rung zero left the higher self rungs stale between emissions -- an
        // asymmetry that is invisible while `self_max_rung` is zero and wrong
        // the moment the ablation raises it.
        if !self.selfc.is_empty() {
            for i in 0..self.d {
                self.selfc[0][i] *= 1.0 - r0;
            }
            for k in 1..self.selfc.len() {
                let rk = self.rho[k];
                for i in 0..self.d {
                    let prev = self.selfc[k - 1][i];
                    let cur = self.selfc[k][i];
                    self.selfc[k][i] = cur + rk * (prev - cur);
                }
            }
        }
    }

    /// Recompute the band-pass stack. Call once per tick, after all
    /// observations and before any query.
    pub fn refresh(&mut self) {
        for k in 0..self.rungs {
            for i in 0..self.d {
                let a = self.world[k][i] + if k < self.selfc.len() { self.selfc[k][i] } else { 0.0 };
                let b = if k + 1 < self.rungs {
                    self.world[k + 1][i]
                        + if k + 1 < self.selfc.len() { self.selfc[k + 1][i] } else { 0.0 }
                } else {
                    0.0
                };
                self.deltas[k][i] = a - b;
            }
            // Direction only: every band is a unit vector however much or little
        // the world has put into that rung. So elapsed time reaches the readout
        // through the *rotation* of the band directions and not through their
        // amplitude, the anchor pulls with constant strength on a fresh and an
        // exhausted background alike, and "how much background is there" is not
        // a feature. That is a design choice, not an oversight, but the rest of
        // this file reads as though the bands carried energy.
        normalize(&mut self.deltas[k]);
        }
    }

    #[inline]
    pub fn delta(&self, k: usize) -> &[f32] {
        &self.deltas[k.min(self.rungs - 1)]
    }

    /// Norm of the self cascade at rung k. Exactly zero above `self_max_rung`,
    /// by layout -- there is no storage there to be non-zero.
    pub fn self_energy(&self, k: usize) -> f32 {
        if k < self.selfc.len() {
            norm(&self.selfc[k])
        } else {
            0.0
        }
    }

    /// Cosine between two band-pass levels. If the bands are nearly collinear
    /// the cascade is not doing what it is here to do, and no single rung will
    /// carry anything -- which is the diagnosis the ladder experiment checks.
    pub fn band_correlation(&self, j: usize, k: usize) -> f32 {
        dot(self.delta(j), self.delta(k))
    }
}
