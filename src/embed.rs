//! Fixed vectors: the embedding table, the channel rotations, and the rank-one
//! operator attached to each token.
//!
//! Nothing in this file is ever updated. Every quantity that decides *where* an
//! observation goes lives here, outside the computational graph.

use crate::config::Config;
use crate::num::{normalize, unit_vector, SignedPerm};

/// Which stream an observation came from.
///
/// The distinction is enforced twice: by a fixed orthogonal rotation, so the
/// channels do not superpose into each other, and by rung reachability (A5), so
/// self-generated content cannot reshape the slow background.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Channel {
    /// What the world said. Reaches every rung.
    In,
    /// What I said out loud.
    Overt,
    /// What I considered and did not say.
    Covert,
    /// Which memory I am touching. The update reporting on itself.
    Write,
}

impl Channel {
    pub fn index(self) -> usize {
        match self {
            Channel::In => 0,
            Channel::Overt => 1,
            Channel::Covert => 2,
            Channel::Write => 3,
        }
    }
}

pub struct Embeddings {
    pub d: usize,
    pub vocab: usize,
    /// Row-major `vocab x d`, rows on the unit sphere, never updated.
    table: Vec<f32>,
    /// Rank-one operator factors: A_x = gain * u_x v_x^T.
    op_u: Vec<f32>,
    op_v: Vec<f32>,
    pub op_gain: f32,
    pub mix: f32,
    rot: [SignedPerm; 4],
}

impl Embeddings {
    pub fn new(cfg: &Config) -> Self {
        let d = cfg.d;
        let vocab = cfg.vocab;
        let key = cfg.seed ^ 0xE_1BED;
        let mut table = vec![0.0f32; vocab * d];
        for t in 0..vocab {
            let v = unit_vector(key, t as u64, d);
            table[t * d..(t + 1) * d].copy_from_slice(&v);
        }
        let mut op_u = vec![0.0f32; vocab * d];
        let mut op_v = vec![0.0f32; vocab * d];
        for t in 0..vocab {
            let u = unit_vector(key ^ KEY_OP_U, t as u64, d);
            let w = unit_vector(key ^ KEY_OP_V, t as u64, d);
            op_u[t * d..(t + 1) * d].copy_from_slice(&u);
            op_v[t * d..(t + 1) * d].copy_from_slice(&w);
        }
        // The input channel is the identity: the world's embeddings must stay in
        // the space the memory was built in. The self channels are rotated away
        // from it.
        let rot = [
            SignedPerm::identity(d),
            SignedPerm::new(key ^ KEY_ROT_OVERT, d),
            SignedPerm::new(key ^ KEY_ROT_COVERT, d),
            SignedPerm::new(key ^ KEY_ROT_WRITE, d),
        ];
        Embeddings { d, vocab, table, op_u, op_v, op_gain: cfg.op_gain, mix: cfg.op_mix, rot }
    }

    #[inline]
    pub fn row(&self, t: usize) -> &[f32] {
        &self.table[t * self.d..(t + 1) * self.d]
    }

    /// The embedding as seen on a given channel.
    pub fn rotated(&self, t: usize, ch: Channel, out: &mut [f32]) {
        self.rot[ch.index()].apply(self.row(t), out);
    }

    pub fn rotate_vec(&self, v: &[f32], ch: Channel, out: &mut [f32]) {
        self.rot[ch.index()].apply(v, out);
    }

    /// Frobenius norm of the token's operator: `gain` for a real token, exactly
    /// zero for the baseline token. This is the quantity the write gate reads,
    /// which is why A2 ("no charge at baseline") and "no content write at
    /// baseline" are the same statement rather than two.
    #[inline]
    pub fn drive_norm(&self, t: Option<usize>) -> f32 {
        match t {
            None => 0.0,
            Some(_) => self.op_gain,
        }
    }

    /// p <- nu( (1 - mix) * U_x p + mix * E_x ),  with U_x = I + gain u_x v_x^T.
    ///
    /// Two things have to happen at once. The operator term is non-commutative,
    /// so a multi-token challenge composes in an order-sensitive way; the
    /// injection term carries the token's identity, without which the payload
    /// barely moves at all. With random unit factors in d dimensions the
    /// rank-one term perturbs the payload by only O(gain / sqrt(d)) -- which is
    /// how a payload ends up frozen near wherever it started.
    ///
    /// Both terms vanish for the baseline token, so the payload is left
    /// *exactly* untouched there. That is what keeps "no input" and "no change"
    /// the same fact rather than two.
    pub fn apply_operator(&self, t: Option<usize>, p: &mut [f32]) {
        let t = match t {
            None => return,
            Some(t) => t,
        };
        let u = &self.op_u[t * self.d..(t + 1) * self.d];
        let v = &self.op_v[t * self.d..(t + 1) * self.d];
        let c = self.op_gain * crate::num::dot(v, p);
        crate::num::axpy(c, u, p);
        normalize(p);
        let e = self.row(t);
        let m = self.mix;
        for i in 0..self.d {
            p[i] = (1.0 - m) * p[i] + m * e[i];
        }
        normalize(p);
    }
}

const KEY_OP_U: u64 = 0x0000_0000_0000_0001;
const KEY_OP_V: u64 = 0x0000_0000_0000_0002;
const KEY_ROT_OVERT: u64 = 0x0000_0000_0000_0011;
const KEY_ROT_COVERT: u64 = 0x0000_0000_0000_0012;
const KEY_ROT_WRITE: u64 = 0x0000_0000_0000_0013;
