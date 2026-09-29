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

/// In-place radix-2 FFT. `re`/`im` have power-of-two length.
fn fft(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = 2.0 * std::f64::consts::PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let (wr, wi) = (ang.cos() as f32, ang.sin() as f32);
        let mut i = 0usize;
        while i < n {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    re[i + k + len / 2] * cr - im[i + k + len / 2] * ci,
                    re[i + k + len / 2] * ci + im[i + k + len / 2] * cr,
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let inv = 1.0 / n as f32;
        for k in 0..n {
            re[k] *= inv;
            im[k] *= inv;
        }
    }
}

/// Circular convolution. O(d log d) through the FFT when `d` is a power of two,
/// which every configuration here uses; the quadratic form is kept for the rest.
///
/// This was O(d^2) and called several times a tick, which is most of why a run
/// took hours on one core.
pub fn circconv(a: &[f32], b: &[f32], out: &mut [f32]) {
    let d = a.len();
    if d < 8 || d & (d - 1) != 0 {
        for k in 0..d {
            let mut s = 0.0f32;
            for j in 0..d {
                s += a[j] * b[(k + d - j) % d];
            }
            out[k] = s;
        }
        return;
    }
    let mut ar = a.to_vec();
    let mut ai = vec![0.0f32; d];
    let mut br = b.to_vec();
    let mut bi = vec![0.0f32; d];
    fft(&mut ar, &mut ai, false);
    fft(&mut br, &mut bi, false);
    for k in 0..d {
        let (r, i) = (ar[k] * br[k] - ai[k] * bi[k], ar[k] * bi[k] + ai[k] * br[k]);
        ar[k] = r;
        ai[k] = i;
    }
    fft(&mut ar, &mut ai, true);
    out[..d].copy_from_slice(&ar);
}

/// A *unitary* vector: unit-magnitude Fourier coefficients, random phase.
///
/// Binding's inverse is correlation, and `a (o) a` equals the identity only when
/// every Fourier magnitude of `a` is one. Gaussian vectors do not satisfy that,
/// and the cost is not small: measured reconstruction cosine 0.527 for a *single*
/// stored triple with no interference at all, against 1.000 for unitary vectors.
/// The capacity law `signal ~ 1/sqrt(k)` that the whole banking argument rests on
/// is a statement about unitary vectors; with Gaussian ones a constant loss of
/// about a half multiplies it, which is why even a configuration inside the
/// computed operating region retrieved at 0.089.
pub fn unitary_vector(key: u64, slot: u64, d: usize) -> Vec<f32> {
    if d < 8 || d & (d - 1) != 0 {
        return unit_vector(key, slot, d);
    }
    let mut ph = vec![0.0f32; d / 2 - 1];
    fill_gaussian(key ^ 0xF7_1CE5, slot.wrapping_mul(0x2000), &mut ph);
    let mut re = vec![0.0f32; d];
    let mut im = vec![0.0f32; d];
    re[0] = 1.0;
    re[d / 2] = 1.0;
    for k in 1..d / 2 {
        let a = ph[k - 1] * std::f32::consts::PI;
        re[k] = a.cos();
        im[k] = a.sin();
        re[d - k] = a.cos();
        im[d - k] = -a.sin();
    }
    fft(&mut re, &mut im, true);
    normalize(&mut re);
    re
}

/// The unitary vector with the same Fourier phases as `v`: every magnitude set
/// to one. Binding by it is exactly invertible, and two such vectors bind to
/// something whose delta component is the mean cosine of their phase
/// differences -- their similarity. A zero vector returns None.
pub fn unitarize(v: &[f32]) -> Option<Vec<f32>> {
    soften(v, 0.0)
}

/// Keep the Fourier phases of `v` and raise each magnitude to `gamma`:
/// 0 is `unitarize`, 1 leaves the vector as it is. Between the two, binding
/// by the result is less exactly invertible and keeps more of how much each
/// component actually carries.
pub fn soften(v: &[f32], gamma: f32) -> Option<Vec<f32>> {
    let d = v.len();
    if d < 8 || d & (d - 1) != 0 {
        return None;
    }
    let mut re = v.to_vec();
    let mut im = vec![0.0f32; d];
    fft(&mut re, &mut im, false);
    let mut any = false;
    for k in 0..d {
        let m = (re[k] * re[k] + im[k] * im[k]).sqrt();
        if m > 1e-9 {
            let scale = m.powf(gamma) / m;
            re[k] *= scale;
            im[k] *= scale;
            any = true;
        } else {
            re[k] = 1.0;
            im[k] = 0.0;
        }
    }
    if !any {
        return None;
    }
    fft(&mut re, &mut im, true);
    normalize(&mut re);
    Some(re)
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
