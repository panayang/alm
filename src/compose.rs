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
    (ours[0].mean(), ours[1].mean(), theirs[best][0].mean(), theirs[best][1].mean())
}

pub fn run(train_episodes: usize, _gap: usize) {
    let arms = [
        Arm { name: "default (gap 2)", gap: 2, set: |_c| {} },
        Arm { name: "gap 6", gap: 6, set: |_c| {} },
        Arm { name: "bind_decay 0.9", gap: 2, set: |c| c.bind_decay = 0.9 },
        Arm { name: "bind_decay 0.5", gap: 2, set: |c| c.bind_decay = 0.5 },
        Arm { name: "bind_self", gap: 2, set: |c| c.bind_self = true },
        Arm {
            name: "no binding",
            gap: 2,
            set: |c| {
                c.use_binding = false;
                c.bind_mode = crate::config::BindMode::Off;
            },
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
