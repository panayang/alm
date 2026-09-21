//! Can the readout answer from a part when the whole is new?
//!
//! On PhysioNet what is left of our loss sits in one cell: the order-2 context
//! is new, the order-1 context is not, and the target has been seen 1-99 times.
//! There we charge about a bit more than PPM-C. A counter escaping from an
//! order-2 context it has never seen lands on an order-1 conditional it has
//! counted; we have to compose the answer out of a state.
//!
//! That cell has a known answer if the stream is built to give it one. Three
//! disjoint sets of tokens and one rule:
//!
//! ```text
//!   episode:  p_i , q_j , z   where  z = f(q_j)
//! ```
//!
//! The answer depends on the cue and not on what preceded it, so the order-1
//! conditional entropy is exactly zero. Half the (p_i, q_j) pairs are held out
//! of training and each is charged the first time it ever occurs, so at that
//! moment the order-2 context is new, the order-1 context has been seen
//! hundreds of times, and the right charge is zero bits for anything that can
//! use the part it has seen.
//!
//! A counter gets this nearly free: it escapes from the unseen pair to the
//! order-1 conditional and pays only the escape.
//!
//! The structural assertions decode the present token from the features at
//! 1.000, but they read the whole feature vector, which includes the bound
//! blocks -- so they do not establish that the state block carries the cue on
//! its own, and this instrument should not be read as though they did. What it
//! measures is the readout's charge, and the arms say which block the charge
//! was leaning on.

use std::collections::HashMap;

use crate::config::Config;
use crate::model::Model;
use crate::num::cbrng;

const M: usize = 48;

/// Interpolated PPM-C, the same one the acquisition instrument uses.
struct Ppm {
    order: usize,
    vocab: usize,
    ctx: HashMap<u64, HashMap<u32, u32>>,
    hist: Vec<u32>,
}

impl Ppm {
    fn new(order: usize, vocab: usize) -> Self {
        Ppm { order, vocab, ctx: HashMap::new(), hist: Vec::new() }
    }
    fn key(&self, o: usize) -> u64 {
        let mut k = 0xcbf2_9ce4_8422_2325u64 ^ o as u64;
        for &h in self.hist[self.hist.len() - o..].iter() {
            k = (k ^ h as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        k
    }
    fn observe(&mut self, sym: u32) -> f64 {
        let maxo = self.order.min(self.hist.len());
        let (mut p, mut esc) = (0.0f64, 1.0f64);
        for o in (0..=maxo).rev() {
            if let Some(c) = self.ctx.get(&self.key(o)) {
                let tot: u32 = c.values().sum();
                if tot > 0 {
                    let e = c.len() as f64 / (c.len() as f64 + tot as f64);
                    p += esc * (1.0 - e) * (*c.get(&sym).unwrap_or(&0) as f64 / tot as f64);
                    esc *= e;
                }
            }
        }
        p += esc / self.vocab as f64;
        for o in 0..=maxo {
            *self.ctx.entry(self.key(o)).or_default().entry(sym).or_insert(0) += 1;
        }
        self.hist.push(sym);
        -p.max(f64::MIN_POSITIVE).log2()
    }
}

/// Held out of training: the pairs whose sum is in the upper half.
#[inline]
fn held_out(i: usize, j: usize) -> bool {
    (i + j) % M >= M / 2
}

#[derive(Default)]
struct Cell {
    n: u64,
    bits: f64,
}

impl Cell {
    fn push(&mut self, b: f64) {
        self.n += 1;
        self.bits += b;
    }
    fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.bits / self.n as f64
        }
    }
}

struct Arm {
    name: &'static str,
    gap: usize,
    set: fn(&mut Config),
    report_verify: bool,
}

/// Does memory know whether it holds what it is about to say?
///
/// The read-back check already exists and is already wired -- `step_cursors`
/// uses it to refuse a cursor whose retrieval came from a neighbouring address
/// rather than from something written. It is not wired to the emission. At the
/// tick the world speaks, the score is `row . phi + codebook * <E_t, p>` and
/// nothing in it has asked whether this key was ever written at all.
///
/// That question is not the counter's question. A counter backs off because it
/// lacks the samples to estimate a conditional; this asks memory for a fact
/// about storage, answerable in one convolution and one dot product, and it
/// needs no ground truth -- only a candidate, which the emission already has.
///
/// So: take the model's own top-1 at the answer tick, bind it back onto the
/// pair's key, and ask the bank whether that triple is there. If the score
/// separates a candidate that is right from one that is wrong, there is a
/// signal the emission is currently throwing away.
struct Verify {
    n: u64,
    sum: f64,
    /// How many fell below the acceptance threshold. On a pair that *was*
    /// written this is a false negative, and it is what makes the gate cost
    /// anything on familiar pairs.
    below: u64,
}

impl Verify {
    fn push(&mut self, x: f64, min: f64) {
        self.n += 1;
        self.sum += x;
        if x < min {
            self.below += 1;
        }
    }
    fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }
}

fn readback(m: &Model, prev2: usize, prev: usize, cand: usize) -> f32 {
    let d = m.cfg.d;
    let mut q = vec![0.0f32; d];
    crate::num::circconv(m.emb.row(prev2), m.emb.row(prev), &mut q);
    crate::num::normalize(&mut q);
    let mut back = vec![0.0f32; d];
    crate::num::circconv(&q, m.emb.row(cand), &mut back);
    crate::num::normalize(&mut back);
    let bank = m.bank_of_ids(prev2, prev);
    let mn = crate::num::norm(&m.mem[bank]).max(1e-9);
    crate::num::dot(&m.mem[bank], &back) / mn
}

/// (ours seen, ours novel, best counter seen, best counter novel).
fn run_arm(arm: &Arm, train_episodes: usize) -> (f64, f64, f64, f64) {
    let gap = arm.gap;
    let v = 3 * M + 1;
    let (p0, q0, z0) = (1usize, 1 + M, 1 + 2 * M);

    let mut cfg = Config::local();
    cfg.seed = 0xC0FFEE;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    (arm.set)(&mut cfg);
    cfg.derive();
    let mut model = Model::new(cfg);

    const ORDERS: usize = 4;
    let mut ppms: Vec<Ppm> = (1..=ORDERS).map(|o| Ppm::new(o, v)).collect();

    // [novel pair?] for us, and per order for the counter.
    let mut ours = [Cell::default(), Cell::default()];
    let mut theirs: Vec<[Cell; 2]> =
        (0..ORDERS).map(|_| [Cell::default(), Cell::default()]).collect();

    // The test pairs, each fired once, spread evenly through the second half.
    let mut test: Vec<(usize, usize)> = Vec::new();
    for i in 0..M {
        for j in 0..M {
            if held_out(i, j) {
                test.push((i, j));
            }
        }
    }
    let total = train_episodes + test.len();
    let first_test = train_episodes / 2;
    let mut next_test = 0usize;
    // [novel pair?][top-1 was right?]
    let z = || Verify { n: 0, sum: 0.0, below: 0 };
    let mut ver = [[z(), z()], [z(), z()]];
    let vmin = model.cfg.verify_min() as f64;

    for e in 0..total {
        // After the training half, interleave one held-out pair every few
        // episodes; everything else stays a training pair, so the stream is
        // stationary and the model keeps learning throughout.
        let fire_test = e >= first_test
            && next_test < test.len()
            && (e - first_test) % 3 == 0;
        let (i, j, novel) = if fire_test {
            let (i, j) = test[next_test];
            next_test += 1;
            (i, j, true)
        } else {
            let mut i = (cbrng(0xA11CE, e as u64) % M as u64) as usize;
            let mut j = (cbrng(0xB0B, e as u64) % M as u64) as usize;
            // Redraw into the training half.
            let mut guard = 0;
            while held_out(i, j) && guard < 64 {
                j = (j + 1) % M;
                guard += 1;
                if guard % M == 0 {
                    i = (i + 1) % M;
                }
            }
            (i, j, false)
        };
        let z = z0 + j; // the answer depends on the cue alone

        for (tok, scored) in [(p0 + i, false), (q0 + j, false), (z, true)] {
            for _ in 0..gap {
                model.tick(None, false);
            }
            if scored && e >= first_test {
                // The model's own answer, before it is told, and what memory
                // says about it under this pair's key.
                if let Some((t, _)) = model.spread_now().top() {
                    let sc = readback(&model, p0 + i, q0 + j, t as usize) as f64;
                    let right = t as usize == z;
                    ver[novel as usize][right as usize].push(sc, vmin);
                }
            }
            let out = model.tick(Some(tok), false);
            for (oi, pp) in ppms.iter_mut().enumerate() {
                let b = pp.observe(tok as u32);
                if scored && out.charged && e >= first_test {
                    theirs[oi][novel as usize].push(b);
                }
            }
            if scored && out.charged && e >= first_test {
                ours[novel as usize].push(out.bits);
            }
        }
    }

    let mut best = 0usize;
    for oi in 0..ORDERS {
        if theirs[oi][1].mean() < theirs[best][1].mean() {
            best = oi;
        }
    }
    if arm.report_verify {
        println!(
            "
-- read-back: does memory hold what the model is about to say? (floor 1/sqrt(d) = {:.4}, accept >= {:.4}) --",
            1.0 / (model.cfg.d as f64).sqrt(),
            model.cfg.verify_min()
        );
        println!(
            "{:>18} {:>10} {:>12} {:>14}",
            "pair", "n", "mean score", "below accept"
        );
        for (k, name) in [(0usize, "written"), (1, "never written")] {
            let n = ver[k][0].n + ver[k][1].n;
            let sum = ver[k][0].sum + ver[k][1].sum;
            let below = ver[k][0].below + ver[k][1].below;
            println!(
                "{:>18} {:>10} {:>12.4} {:>13.1}%",
                name,
                n,
                if n == 0 { 0.0 } else { sum / n as f64 },
                100.0 * below as f64 / n.max(1) as f64
            );
        }
        println!("  a written pair below the threshold is a false negative: the gate zeroes");
        println!("  a conjunction that memory does hold, which is what it costs on familiar");
        println!("  pairs.");
    }
    (ours[0].mean(), ours[1].mean(), theirs[best][0].mean(), theirs[best][1].mean())
}

pub fn run(train_episodes: usize, _gap: usize) {
    let arms = [
        Arm { name: "default", gap: 2, set: |_c| {}, report_verify: true },
        Arm {
            name: "verify gate",
            gap: 2,
            set: |c| c.verify_gate = true,
            report_verify: false,
        },
        Arm {
            name: "self + lag only",
            gap: 2,
            set: |c| c.bind_mode = crate::config::BindMode::EventLag,
            report_verify: false,
        },
        Arm {
            name: "self + band only",
            gap: 2,
            set: |c| c.bind_mode = crate::config::BindMode::Band,
            report_verify: false,
        },
        Arm {
            name: "self only",
            gap: 2,
            set: |c| c.bind_mode = crate::config::BindMode::Off,
            report_verify: false,
        },
        Arm {
            name: "no self block",
            gap: 2,
            set: |c| c.bind_self = false,
            report_verify: false,
        },
        Arm {
            name: "no binding at all",
            gap: 2,
            set: |c| {
                c.use_binding = false;
                c.bind_self = false;
                c.bind_mode = crate::config::BindMode::Off;
            },
            report_verify: false,
        },
    ];
    println!("compose: z = f(cue), independent of what preceded the cue.");
    println!(
        "  {} cues, {} held-out (prev, cue) pairs charged on first occurrence, {} training episodes",
        M,
        M * M / 2,
        train_episodes
    );
    println!("  true conditional entropy is 0 bits given the cue alone.\n");
    println!(
        "{:>18} {:>10} {:>10} {:>10} | {:>10} {:>10}",
        "arm", "ours seen", "ours new", "new/seen", "PPM seen", "PPM new"
    );
    let rows: Vec<(&'static str, (f64, f64, f64, f64))> = std::thread::scope(|sc| {
        let hs: Vec<_> =
            arms.iter().map(|a| sc.spawn(move || (a.name, run_arm(a, train_episodes)))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (name, (os, on, ps, pn)) in rows.iter() {
        println!(
            "{:>18} {:>10.4} {:>10.4} {:>10.1} | {:>10.4} {:>10.4}",
            name,
            os,
            on,
            if *os > 0.0 { on / os } else { 0.0 },
            ps,
            pn
        );
    }
    println!("\n  the cue has been seen hundreds of times in both columns. the only thing");
    println!("  new about a novel event is the pair, and the pair is irrelevant to the");
    println!("  answer. a readout using the part it has seen charges the same for both,");
    println!("  so the ratio is the measurement and one is the target.");
}
