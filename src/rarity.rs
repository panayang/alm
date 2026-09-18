//! Is the readout calibrated on rare tokens?
//!
//! On PhysioNet the loss is concentrated on targets seen 1-99 times before, in
//! any context: we charge about three bits more than a counter, i.e. we give
//! those tokens a tenth of the probability it does. The novelty of the context
//! makes almost no difference. So the question is about rarity, and it has an
//! answer that is known exactly on an i.i.d. stream: the right charge for a
//! token of probability p is -log2 p, nothing more and nothing less.
//!
//! Each arm switches one mechanism and nothing else, so an arm that closes the
//! gap names the mechanism that opens it. The arms run in parallel threads;
//! they share nothing.

use crate::config::Config;
use crate::model::Model;
use crate::num::cbrng;

const BUCKETS: usize = 6;

fn zipf(v: usize) -> Vec<f64> {
    let w: Vec<f64> = (0..v).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let t: f64 = w.iter().sum();
    w.iter().map(|x| x / t).collect()
}

fn draw(cdf: &[f64], key: u64, i: u64) -> usize {
    let u = (cbrng(key, i) >> 11) as f64 / (1u64 << 53) as f64;
    match cdf.binary_search_by(|c| c.partial_cmp(&u).unwrap()) {
        Ok(k) | Err(k) => k.min(cdf.len() - 1),
    }
}

/// Bucket by -log2 p, two bits wide, so each bucket is a factor of four in
/// rarity.
fn bucket(p: f64) -> usize {
    ((-p.log2() / 2.0) as usize).min(BUCKETS - 1)
}

struct Arm {
    name: &'static str,
    set: fn(&mut Config),
}

/// (events, sum of our bits, sum of -log2 p) per bucket, over the second half.
fn run_arm(arm: &Arm, v: usize, n: usize) -> Vec<(u64, f64, f64)> {
    let p = zipf(v);
    let mut cdf = Vec::with_capacity(v);
    let mut a = 0.0;
    for q in p.iter() {
        a += q;
        cdf.push(a);
    }
    let mut cfg = Config::local();
    cfg.seed = 0x5A1E;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    (arm.set)(&mut cfg);
    cfg.derive();
    let mut m = Model::new(cfg);
    let mut acc = vec![(0u64, 0.0f64, 0.0f64); BUCKETS];
    for i in 0..n {
        let tok = draw(&cdf, 0xDA7B, i as u64);
        for _ in 0..2 {
            m.tick(None, false);
        }
        let out = m.tick(Some(tok), false);
        if out.charged && i * 2 >= n {
            let b = bucket(p[tok]);
            acc[b].0 += 1;
            acc[b].1 += out.bits;
            acc[b].2 += -p[tok].log2();
        }
    }
    acc
}

pub fn run(n: usize) {
    let v = 296usize;
    let p = zipf(v);
    let h: f64 = -p.iter().map(|q| q * q.log2()).sum::<f64>();
    let arms: Vec<Arm> = vec![
        // First pass (V = 296, 120k events, overall excess / rarest bucket):
        //   top 16 +6.015 / +41.747   sampled 16 +63.264 / +89.230
        //   top 64 +0.671 /  +2.306   bias      +8.811 / +63.480
        //   no codebook term +6.033 / +41.772   eta 0.05 +3.567 / +20.156
        // Only the number of negatives against the vocabulary moved it, so this
        // pass sweeps that one axis out to the exact gradient.
        Arm { name: "top 16", set: |c| c.neg_samples = 16 },
        Arm { name: "top 32", set: |c| c.neg_samples = 32 },
        Arm { name: "top 64", set: |c| c.neg_samples = 64 },
        Arm { name: "top 128", set: |c| c.neg_samples = 128 },
        Arm { name: "dense (exact)", set: |c| c.neg_samples = 0 },
    ];
    println!(
        "i.i.d. Zipf, V = {}, {} events, second half scored.  true entropy {:.4}  uniform {:.4}",
        v,
        n,
        h,
        (v as f64).log2()
    );
    println!("excess = our charge minus -log2 p, bits per event, by rarity bucket\n");

    let results: Vec<(&'static str, Vec<(u64, f64, f64)>)> = std::thread::scope(|s| {
        let hs: Vec<_> = arms
            .iter()
            .map(|a| s.spawn(move || (a.name, run_arm(a, v, n))))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut head = format!("{:>20} {:>8}", "arm", "all");
    for b in 0..BUCKETS {
        let lo = 2 * b;
        let label = if b + 1 == BUCKETS { format!("{}+ bits", lo) } else { format!("{}-{} bits", lo, lo + 2) };
        head += &format!(" {:>11}", label);
    }
    println!("{}", head);
    let mut counts = format!("{:>20} {:>8}", "events", "");
    for b in 0..BUCKETS {
        counts += &format!(" {:>11}", results[0].1[b].0);
    }
    println!("{}", counts);
    for (name, acc) in results.iter() {
        let (tn, tb, tt) = acc.iter().fold((0u64, 0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2));
        let mut line = format!("{:>20} {:>+8.3}", name, (tb - tt) / tn.max(1) as f64);
        for b in 0..BUCKETS {
            let (k, ob, tr) = acc[b];
            if k == 0 {
                line += &format!(" {:>11}", "-");
            } else {
                line += &format!(" {:>+11.3}", (ob - tr) / k as f64);
            }
        }
        println!("{}", line);
    }
    println!("\n  a calibrated readout has zero excess in every bucket. excess that grows");
    println!("  with rarity is a readout that under-weights what it has seen rarely.");
}
