//! Does the readout converge, or does it hover?
//!
//! Two streams, because neither alone can choose a step size. One has a
//! conditional entropy of exactly zero, so any charge above zero is structure
//! the readout has not taken. The other is i.i.d., so there is nothing to take
//! and any charge above the uniform is the readout being confidently wrong --
//! fabrication, which is the failure mode `store.rs` names as the price of
//! having no counted prior.
//!
//! And by decile, not at the end. The sweep that chose `row_norm_cap = 16.0`
//! read a single endpoint, and the capped curve does not plateau: it bottoms
//! near the second decile and climbs for the rest of the run. An endpoint
//! cannot tell a converged run from a run that has turned around and is on its
//! way back up.

use crate::config::Config;
use crate::model::Model;
use crate::num::cbrng;

const DEC: usize = 10;

fn zipf(v: usize) -> Vec<f64> {
    let w: Vec<f64> = (0..v).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let t: f64 = w.iter().sum();
    w.iter().map(|x| x / t).collect()
}

fn draw(p: &[f64], key: u64, i: u64) -> usize {
    let u = (cbrng(key, i) >> 11) as f64 / (1u64 << 53) as f64;
    let mut a = 0.0;
    for (j, q) in p.iter().enumerate() {
        a += q;
        if u <= a {
            return j;
        }
    }
    p.len() - 1
}

/// `det`: alternate a Zipf cue with its deterministic successor, score the
/// successor (true conditional 0 bits).
/// `iid`: every token Zipf-drawn and independent, score everything (true
/// conditional is H(marginal); uniform is log2 V).
fn charge_by_decile(det: bool, eta: f32, cap: f32, n: usize) -> Vec<f64> {
    let v = 64usize;
    let p = zipf(v);
    let mut cfg = Config::local();
    cfg.seed = 0x51D;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.eta = eta;
    cfg.row_norm_cap = cap;
    cfg.derive();
    let mut m = Model::new(cfg);
    let mut acc = vec![(0.0f64, 0u64); DEC];
    let mut cue = 0usize;
    for i in 0..n {
        let (tok, score) = if det {
            if i % 2 == 0 {
                cue = draw(&p, 0xDA7B, i as u64);
                (cue, false)
            } else {
                ((cue * 7 + 3) % v, true)
            }
        } else {
            (draw(&p, 0xDA7B, i as u64), true)
        };
        for _ in 0..2 {
            m.tick(None, false);
        }
        let out = m.tick(Some(tok), false);
        if out.charged && score {
            let d = (i * DEC / n).min(DEC - 1);
            acc[d].0 += out.bits;
            acc[d].1 += 1;
        }
    }
    acc.iter().map(|(b, c)| if *c == 0 { 0.0 } else { b / *c as f64 }).collect()
}

pub fn run(n: usize) {
    let v = 64usize;
    let p = zipf(v);
    let hm: f64 = -p.iter().map(|q| q * q.log2()).sum::<f64>();
    println!("charge by decile.  deterministic: true 0 bits.  i.i.d.: true {:.4} bits, uniform {:.4}\n", hm, (v as f64).log2());
    for det in [true, false] {
        println!("-- {} --", if det { "deterministic successor" } else { "i.i.d. Zipf" });
        println!("{:>6} {:>6}   {}", "eta", "cap", (1..=DEC).map(|d| format!("{:>7}", d)).collect::<Vec<_>>().join(""));
        for cap in [16.0f32, 64.0, 0.0] {
            for eta in [0.5f32, 0.3, 0.2, 0.1] {
                let c = charge_by_decile(det, eta, cap, n);
                println!("{:>6} {:>6}   {}", eta, cap, c.iter().map(|x| format!("{:>7.3}", x)).collect::<Vec<_>>().join(""));
            }
        }
        println!();
    }
}
