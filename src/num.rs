//! Deterministic RNG and small dense linear algebra.
//!
//! Two properties matter here and both are load-bearing for the assertions in
//! `tests/structural.rs`:
//!
//! 1. The RNG is *counter-based*. A draw is a pure function of (key, index), so
//!    the value of any fixed vector -- an embedding row, an edge key -- does not
//!    depend on how many other draws happened first, or in what order. Nothing
//!    in the model carries mutable RNG state that construction order could
//!    perturb.
//! 2. Reductions run in a fixed order on a single thread. `dot` is a plain
//!    left-to-right accumulation, never a tree or a parallel fold, so repeating
//!    an operation on the same operands reproduces the same bits.

// ---------------------------------------------------------------- RNG

#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Counter-based draw: a pure function of (key, index).
#[inline]
pub fn cbrng(key: u64, index: u64) -> u64 {
    splitmix64(splitmix64(key).wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
}

/// Uniform in [0, 1).
#[inline]
pub fn uniform(key: u64, index: u64) -> f32 {
    ((cbrng(key, index) >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// Uniform integer in [0, n).
#[inline]
pub fn uniform_below(key: u64, index: u64, n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    cbrng(key, index) % n
}

/// Two standard normals from one counter position (Box-Muller).
#[inline]
fn gaussian_pair(key: u64, index: u64) -> (f32, f32) {
    let u1 = uniform(key, 2 * index).max(1e-7);
    let u2 = uniform(key, 2 * index + 1);
    let r = (-2.0 * u1.ln()).sqrt();
    let t = 2.0 * std::f32::consts::PI * u2;
    (r * t.cos(), r * t.sin())
}

/// Fill `out` with standard normals drawn from (key, base ..).
pub fn fill_gaussian(key: u64, base: u64, out: &mut [f32]) {
    let mut i = 0usize;
    while i < out.len() {
        let (a, b) = gaussian_pair(key, base + (i as u64) / 2);
        out[i] = a;
        if i + 1 < out.len() {
            out[i + 1] = b;
        }
        i += 2;
    }
}

/// A unit vector on S^{d-1}, a pure function of (key, slot).
pub fn unit_vector(key: u64, slot: u64, d: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; d];
    fill_gaussian(key, slot.wrapping_mul(0x1000), &mut v);
    normalize(&mut v);
    v
}

/// A *signed permutation* of the coordinates: an exactly orthogonal matrix that
/// applies in O(d) and needs no QR factorisation. Used for the channel
/// rotations, which have to separate streams without distorting norms.
pub struct SignedPerm {
    perm: Vec<u32>,
    sign: Vec<f32>,
}

impl SignedPerm {
    pub fn new(key: u64, d: usize) -> Self {
        let mut perm: Vec<u32> = (0..d as u32).collect();
        // Fisher-Yates with counter-based draws, so the permutation is a pure
        // function of the key.
        for i in (1..d).rev() {
            let j = uniform_below(key, i as u64, (i + 1) as u64) as usize;
            perm.swap(i, j);
        }
        let sign = (0..d)
            .map(|i| if cbrng(key ^ 0x5EED, i as u64) & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        SignedPerm { perm, sign }
    }

    /// Identity: used for the input channel, so the world's own embeddings are
    /// not rotated away from the space the memory was built in.
    pub fn identity(d: usize) -> Self {
        SignedPerm { perm: (0..d as u32).collect(), sign: vec![1.0; d] }
    }

    pub fn apply(&self, v: &[f32], out: &mut [f32]) {
        for i in 0..v.len() {
            out[self.perm[i] as usize] = self.sign[i] * v[i];
        }
    }
}

// ---------------------------------------------------------------- vectors

/// Fixed left-to-right accumulation. Never reorder this.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

#[inline]
pub fn normalize(a: &mut [f32]) -> f32 {
    let n = norm(a);
    if n > 1e-12 {
        let inv = 1.0 / n;
        for x in a.iter_mut() {
            *x *= inv;
        }
    }
    n
}

#[inline]
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for i in 0..y.len() {
        y[i] += alpha * x[i];
    }
}

#[inline]
pub fn scale(alpha: f32, y: &mut [f32]) {
    for v in y.iter_mut() {
        *v *= alpha;
    }
}

#[inline]
pub fn sub_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = a[i] - b[i];
    }
}

// ---------------------------------------------------------------- matrices

/// Row-major `rows x cols`.
#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub a: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, a: vec![0.0; rows * cols] }
    }

    /// Small random init, a pure function of (key, slot).
    pub fn random(key: u64, slot: u64, rows: usize, cols: usize, sigma: f32) -> Self {
        let mut a = vec![0.0f32; rows * cols];
        fill_gaussian(key, slot.wrapping_mul(0x10_0000), &mut a);
        for v in a.iter_mut() {
            *v *= sigma;
        }
        Mat { rows, cols, a }
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.a[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.a[r * self.cols..(r + 1) * self.cols]
    }

    /// out = self * x
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        for r in 0..self.rows {
            out[r] = dot(self.row(r), x);
        }
    }

    /// out = self^T * x
    pub fn matvec_t(&self, x: &[f32], out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        for r in 0..self.rows {
            let xr = x[r];
            if xr != 0.0 {
                axpy(xr, self.row(r), out);
            }
        }
    }

    /// self -= alpha * (u v^T)
    pub fn sub_outer(&mut self, alpha: f32, u: &[f32], v: &[f32]) {
        for r in 0..self.rows {
            let c = alpha * u[r];
            if c != 0.0 {
                let row = &mut self.a[r * self.cols..(r + 1) * self.cols];
                for i in 0..row.len() {
                    row[i] -= c * v[i];
                }
            }
        }
    }

    pub fn frob(&self) -> f32 {
        dot(&self.a, &self.a).sqrt()
    }
}

// ---------------------------------------------------------------- misc

/// Numerically safe softmax in place; returns the log-sum-exp.
pub fn softmax(scores: &mut [f32]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut m = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s > m {
            m = s;
        }
    }
    let mut z = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        z += *s;
    }
    let inv = 1.0 / z;
    for s in scores.iter_mut() {
        *s *= inv;
    }
    m + z.ln()
}

/// Index of the maximum, ties broken by lowest index (deterministic).
pub fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

/// Circular convolution, the binding operation.
///
/// A sum superposes and loses which cue went with which; a convolution binds,
/// and `a (*) b` is a vector nearly orthogonal to both factors and distinct for
/// every pair. That distinction is the whole reason a Latin square is reachable
/// at all: the target is linear in the tensor features E_A x E_B and not in
/// their sum, so a linear readout over a bound trace can represent what a linear
/// readout over a superposition cannot.
/// The involution `a*`: `a[0]` fixed, the rest reversed.
///
/// `(a (*) b) (o) a ~= b`, which is the whole reason binding was chosen over a
/// concatenation. Binding has been in this codebase since the first version and
/// its inverse never was, so memory could be written associatively and only ever
/// read by a learned lookup -- which is why an unseen combination scored exactly
/// zero: to a table it is a fresh random key, and to an unbinding it is a
/// question with an answer.
pub fn involve(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    out[0] = a[0];
    for i in 1..n {
        out[i] = a[n - i];
    }
}

/// Circular correlation: unbind `a` out of `m`.
pub fn unbind(m: &[f32], a: &[f32], out: &mut [f32]) {
    let n = a.len();
    let mut inv = vec![0.0f32; n];
    involve(a, &mut inv);
    circconv(m, &inv, out);
}

pub fn circconv(a: &[f32], b: &[f32], out: &mut [f32]) {
    let d = a.len();
    for k in 0..d {
        let mut s = 0.0f32;
        for j in 0..d {
            s += a[j] * b[(k + d - j) % d];
        }
        out[k] = s;
    }
}

/// Running mean and variance (Welford), used for the allocation criterion.
#[derive(Clone, Default)]
pub struct Running {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
}

impl Running {
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }
    pub fn var(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }
    pub fn std(&self) -> f64 {
        self.var().max(0.0).sqrt()
    }
}
