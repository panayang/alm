//! Read the model's state as a patient, not as a predictor.
//!
//! Four domains were measured with the same yardstick: prequential codelength
//! on the next token. That is the autoregressive question, and this framework
//! was specified as not autoregressive -- the founding note said in as many
//! words that many metrics would have to be redefined. They were not. A counter
//! built to answer exactly that question beat us four times on its own ground,
//! which is the least surprising thing that could have happened.
//!
//! So this asks the question the architecture is shaped for. It never stops
//! producing; it walks chains while the world is quiet; it holds an answer once
//! it arrives. Those are response-shaped, not prediction-shaped. The clinical
//! analogue of a challenge is not "what is measured next" -- it is **what state
//! is this patient in**, asked at an arbitrary moment.
//!
//! That is also the benchmark's own task, which means it comes with the things
//! the last four attempts kept having to invent: real baselines built for
//! exactly this data (GRU-D, SeFT, mTAND, Raindrop), a metric nobody here
//! chose, and five outcome descriptors rather than one scalar.
//!
//! # What is read, and what is not reset
//!
//! The memory is continual across every patient -- it is the population, and
//! nothing about it is reset, replayed or retrained. What is reset at each
//! patient boundary is the *volatile* state: the cursors, the bound traces, the
//! ladder. A new patient is a new context, not a new world. That split is the
//! architecture's own, and `Volatile` exists precisely because this project
//! already needed to say which half is which.
//!
//! After a patient's last tick the feature vector is taken as that patient's
//! representation, and a plain logistic regression over those features is
//! scored by cross-validated AUROC. The probe is deliberately linear and
//! deliberately weak: anything it finds is in the representation, not in the
//! classifier.
//!
//! The numbers to beat are already measured, on these same patients:
//!
//! ```text
//!   timing alone, best single feature   AUROC 0.6157
//!   SAPS-I  (clinicians, from values)         0.6370
//!   SOFA    (clinicians, from values)         0.6256
//! ```
//!
//! If the state cannot beat the timing-only number, then the mechanism has not
//! even retained its own handling of time, and that points somewhere specific
//! rather than nowhere.

use std::collections::HashMap;

use crate::clinical::{flat_streams_ids, flat_streams_valued};
use crate::config::Config;
use crate::model::Model;

/// Standardise columns, then L2-regularised logistic regression by plain
/// gradient descent. Weak on purpose.
pub fn logistic_pub(
    x: &[Vec<f32>],
    y: &[u8],
    train: &[usize],
    test: &[usize],
    l2: f64,
    iters: usize,
) -> Vec<f64> {
    logistic(x, y, train, test, l2, iters)
}

pub fn auroc_pub(score: &[f64], label: &[u8]) -> f64 {
    auroc(score, label)
}

fn logistic(x: &[Vec<f32>], y: &[u8], train: &[usize], test: &[usize], l2: f64, iters: usize)
    -> Vec<f64> {
    let d = x[0].len();
    let mut mean = vec![0.0f64; d];
    let mut sd = vec![0.0f64; d];
    for &i in train {
        for j in 0..d {
            mean[j] += x[i][j] as f64;
        }
    }
    for m in mean.iter_mut() {
        *m /= train.len() as f64;
    }
    for &i in train {
        for j in 0..d {
            sd[j] += (x[i][j] as f64 - mean[j]).powi(2);
        }
    }
    for s in sd.iter_mut() {
        *s = (*s / train.len() as f64).sqrt().max(1e-6);
    }
    let z = |i: usize, w: &[f64], b: f64| -> f64 {
        let mut s = b;
        for j in 0..d {
            s += w[j] * (x[i][j] as f64 - mean[j]) / sd[j];
        }
        s
    };

    let mut w = vec![0.0f64; d];
    let mut b = 0.0f64;
    let lr = 0.5;
    for _ in 0..iters {
        let mut gw = vec![0.0f64; d];
        let mut gb = 0.0f64;
        for &i in train {
            let p = 1.0 / (1.0 + (-z(i, &w, b)).exp());
            let e = p - y[i] as f64;
            gb += e;
            for j in 0..d {
                gw[j] += e * (x[i][j] as f64 - mean[j]) / sd[j];
            }
        }
        let n = train.len() as f64;
        // Shrinkage as the proximal step, not as an explicit decay term.
        //
        // `w -= lr * (grad + l2 * w)` multiplies the weight by `1 - lr*l2` each
        // iteration, which passes through zero at l2 = 2/lr and reverses sign
        // beyond it. At lr 0.5 that made the top of the sweep unusable: l2 = 4
        // scored 0.3999 and l2 = 10 scored 0.2972 -- an anti-predictor, not a
        // regularised one -- and l2 = 100 produced NaN and brought the
        // instrument down inside `auroc`. The sweep never selected those
        // values, so no wrong number was reported, but the usable ceiling was
        // 1.0 while the grid claimed 10, and a set needing more regularisation
        // than that could not reach it.
        //
        // The proximal form divides instead: stable for every non-negative l2,
        // and monotone toward zero weights as l2 grows, which is what "more
        // regularisation" is supposed to mean.
        let shrink = 1.0 / (1.0 + lr * l2);
        for j in 0..d {
            w[j] = (w[j] - lr * gw[j] / n) * shrink;
        }
        b -= lr * gb / n;
    }
    test.iter().map(|&i| z(i, &w, b)).collect()
}

fn auroc(score: &[f64], label: &[u8]) -> f64 {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap());
    let mut rank = vec![0.0f64; score.len()];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && score[idx[j + 1]] == score[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for &k in idx[i..=j].iter() {
            rank[k] = r;
        }
        i = j + 1;
    }
    let (mut pos, mut neg, mut sp) = (0.0f64, 0.0f64, 0.0f64);
    for (k, &l) in label.iter().enumerate() {
        if l == 1 {
            pos += 1.0;
            sp += rank[k];
        } else {
            neg += 1.0;
        }
    }
    if pos == 0.0 || neg == 0.0 {
        return 0.5;
    }
    (sp - pos * (pos + 1.0) / 2.0) / (pos * neg)
}

/// Each feature set at its own best regularisation. Sets differ in width and in
/// how sparse they are, so scoring them all at one L2 measures the probe rather
/// than the representation -- adding bigrams to counts "lost" 0.03 AUROC that
/// way, which is impossible for an information reason and was overfitting.
fn cv_auroc_swept(x: &[Vec<f32>], y: &[u8], ids: &[u32], iters: usize) -> (f64, f64) {
    let mut best = (0.0f64, 0.0f64);
    // Reaching high enough to regularise a wide sparse set, which the old grid
    // could not: its top two values diverged rather than shrank.
    for &l2 in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0].iter() {
        let a = cv_auroc(x, y, ids, l2, iters);
        if a > best.0 {
            best = (a, l2);
        }
    }
    best
}

/// Five folds, assigned by record id so the split does not depend on the order
/// the memory saw them in.
fn cv_auroc(x: &[Vec<f32>], y: &[u8], ids: &[u32], l2: f64, iters: usize) -> f64 {
    let folds = 5usize;
    let mut score = vec![0.0f64; x.len()];
    for f in 0..folds {
        let train: Vec<usize> = (0..x.len()).filter(|&i| ids[i] as usize % folds != f).collect();
        let test: Vec<usize> = (0..x.len()).filter(|&i| ids[i] as usize % folds == f).collect();
        if train.is_empty() || test.is_empty() {
            continue;
        }
        let s = logistic(x, y, &train, &test, l2, iters);
        for (k, &i) in test.iter().enumerate() {
            score[i] = s[k];
        }
    }
    auroc(&score, y)
}

#[inline]
fn silence_ticks(g: u32, cap: usize) -> usize {
    ((32 - g.max(1).leading_zeros()).saturating_sub(1) as usize).min(cap)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    dir: &str,
    outcomes: &str,
    label: &str,
    d: usize,
    patients: usize,
    seed: u64,
    cap: usize,
    bins: usize,
    valued: bool,
    dump: Option<&str>,
    // `ablate`: comma-separated switches to turn off before the run, so a
    // mechanism change can be compared against its own absence under one probe.
    // The earlier comparison could not: both of its numbers were taken under a
    // probe whose shrinkage was broken, and repairing it moved the control by
    // +0.009 while moving the state by -0.001.
    ablate: Option<&str>,
    horizon: Option<f32>,
) {
    let (mut streams, mut ids, vp) = if valued {
        flat_streams_valued(dir, bins)
    } else {
        flat_streams_ids(dir)
    };
    if patients > 0 && streams.len() > patients {
        streams.truncate(patients);
        ids.truncate(patients);
    }

    let mut death: HashMap<u32, u8> = HashMap::new();
    let mut saps: HashMap<u32, f64> = HashMap::new();
    let text = std::fs::read_to_string(outcomes).unwrap_or_else(|e| panic!("{}: {}", outcomes, e));
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if f.len() < 6 {
            continue;
        }
        let id: u32 = f[0].parse().unwrap_or(0);
        if let Ok(v) = f[5].parse::<u8>() {
            death.insert(id, v);
        }
        if let Ok(v) = f[1].parse::<f64>() {
            saps.insert(id, v);
        }
    }

    let mut cfg = Config::local();
    if let Some(a) = ablate {
        for part in a.split(',') {
            match part.trim() {
                "bind_self" => cfg.bind_self = false,
                "verify_gate" => cfg.verify_gate = false,
                "" => {}
                other => panic!("unknown ablation {}", other),
            }
        }
        println!("ablated: {}", a);
    }
    if let Some(h) = horizon {
        cfg.horizon = h;
        println!("ladder horizon overridden: {} ticks", h);
    }
    cfg.seed = seed;
    cfg.vocab = vp;
    cfg.d = d;
    cfg.cleanup_floor_mult = 1.1;
    cfg.derive();
    let mut model = Model::new(cfg.clone());
    let blank = model.volatile();

    // Three readouts of the same run, because the first attempt read the wrong
    // thing. `last` is the state at the final tick -- but that state is a
    // decaying mixture, so it says what just happened, not what the record was.
    // The timing feature it lost to was an aggregate over the whole stay, so
    // the fair comparison pools too.
    let mut f_last: Vec<Vec<f32>> = Vec::new();
    let mut f_mean: Vec<Vec<f32>> = Vec::new();
    let mut f_max: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<u8> = Vec::new();
    let mut keep_ids: Vec<u32> = Vec::new();
    let mut ticks = 0u64;
    // A line every few hundred patients.
    //
    // This loop is 2.5M ticks and used to print nothing until it finished,
    // which made "slow" and "stuck" the same observation from outside. It cost
    // three wrong calls in one session: a run that had been killed and looked
    // like it was working, an estimate off by two orders of magnitude, and an
    // hour spent inferring progress from a CPU counter.
    let total_pat = streams.len();
    let step = (total_pat / 20).max(1);
    let t0 = std::time::Instant::now();
    for (pi, (s, &id)) in streams.iter().zip(ids.iter()).enumerate() {
        if pi > 0 && pi % step == 0 {
            let el = t0.elapsed().as_secs_f64();
            let frac = pi as f64 / total_pat as f64;
            eprintln!(
                "  {}/{} patients, {} ticks, {:.0}s elapsed, ~{:.0}s left",
                pi,
                total_pat,
                ticks,
                el,
                el * (1.0 - frac) / frac.max(1e-9)
            );
        }
        let dy = match death.get(&id) {
            Some(&v) => v,
            None => continue,
        };
        model.restore(blank.clone());
        let mut sum: Vec<f32> = Vec::new();
        let mut mx: Vec<f32> = Vec::new();
        let mut n = 0.0f32;
        for &(gap, tok) in s.iter() {
            for _ in 0..silence_ticks(gap, cap) {
                model.tick(None, false);
                ticks += 1;
            }
            model.tick(Some(tok as usize), false);
            ticks += 1;
            let f = model.features_now();
            if sum.is_empty() {
                sum = vec![0.0; f.len()];
                mx = vec![f32::MIN; f.len()];
            }
            for (k, v) in f.iter().enumerate() {
                sum[k] += *v;
                if *v > mx[k] {
                    mx[k] = *v;
                }
            }
            n += 1.0;
        }
        f_last.push(model.features_now());
        f_mean.push(sum.iter().map(|v| v / n.max(1.0)).collect());
        f_max.push(mx);
        y.push(dy);
        keep_ids.push(id);
    }
    let feats = &f_last;

    let deaths: usize = y.iter().filter(|&&v| v == 1).count();
    println!(
        "==================== {}   [state as patient, {}] ====================",
        label,
        if valued { "parameter x value bin" } else { "parameter only" }
    );
    println!(
        "patients {}   deaths {} ({:.4})   d = {}   feature width {}   ticks {}",
        feats.len(),
        deaths,
        deaths as f64 / feats.len() as f64,
        d,
        feats[0].len(),
        ticks
    );

    // The control that decides whether any of this is the mechanism. For each
    // patient, count how often each token appeared and what its mean bin was:
    // plain summary statistics over exactly the same input the model saw, with
    // no memory, no ticks and no silence. If this matches the state, then what
    // the probe is reading is the data's own summary and not anything the
    // architecture did.
    let mut f_summary: Vec<Vec<f32>> = Vec::new();
    for s in streams.iter() {
        let mut cnt = vec![0.0f32; vp];
        let mut gapsum = vec![0.0f32; vp];
        for &(gap, tok) in s.iter() {
            cnt[tok as usize] += 1.0;
            gapsum[tok as usize] += gap as f32;
        }
        let mut f = cnt.clone();
        for (k, g) in gapsum.iter().enumerate() {
            f.push(g / cnt[k].max(1.0));
        }
        f.push(s.len() as f32);
        f_summary.push(f);
    }
    // Only for the patients that had an outcome, in the same order.
    let keep: std::collections::HashSet<u32> = keep_ids.iter().copied().collect();
    let f_summary: Vec<Vec<f32>> = ids
        .iter()
        .zip(f_summary.into_iter())
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, f)| f)
        .collect();

    // Gate two: does the ORDER carry anything the bag of counts misses?
    //
    // The timing gate asked whether the observation process is informative, and
    // it passed. It never asked whether the *sequence* is, and that is the one
    // that decides whether a memory can distinguish itself at all. The control
    // above is order-invariant by construction -- it counts tokens and averages
    // gaps -- so if adding transition counts to it changes nothing, the answer
    // to this task is a bag, and no mechanism that models sequence can show any
    // advantage here however good it is.
    //
    // Bigrams over 296 tokens would be 87k features on 3996 patients, so they
    // are hashed into a fixed number of buckets. Order information that survives
    // hashing is order information; what hashing loses can only understate the
    // gate, which is the safe direction.
    const BIG: usize = 512;
    let mut f_order: Vec<Vec<f32>> = Vec::new();
    for s in streams.iter() {
        let mut b = vec![0.0f32; BIG];
        for w in s.windows(2) {
            let h = (w[0].1 as u64).wrapping_mul(0x9e37_79b9).wrapping_add(w[1].1 as u64);
            b[(h % BIG as u64) as usize] += 1.0;
        }
        f_order.push(b);
    }
    let f_order: Vec<Vec<f32>> = ids
        .iter()
        .zip(f_order.into_iter())
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, f)| f)
        .collect();
    let f_both: Vec<Vec<f32>> = f_summary
        .iter()
        .zip(f_order.iter())
        .map(|(a, b)| a.iter().chain(b.iter()).copied().collect())
        .collect();

    // Writing the representation out, so probing it does not cost the tick
    // loop again.
    //
    // Every sweep of a probe hyperparameter used to mean rebuilding 2.5M ticks
    // of state -- four and a half hours to answer a question about a logistic
    // regression. That makes the rule this project runs on unaffordable in
    // practice: both sides sweep their own hyperparameters or the number is not
    // reported. Producing the representation is expensive and deterministic;
    // probing it is cheap and is where the hyperparameters live. They are now
    // separate.
    if let Some(p) = dump {
        let sets: Vec<(&str, &Vec<Vec<f32>>)> = vec![
            ("last", &f_last),
            ("mean", &f_mean),
            ("max", &f_max),
            ("summary", &f_summary),
            ("order", &f_order),
            ("both", &f_both),
        ];
        let mut buf: Vec<u8> = Vec::new();
        buf.extend((keep_ids.len() as u64).to_le_bytes());
        buf.extend((sets.len() as u64).to_le_bytes());
        for i in keep_ids.iter() {
            buf.extend(i.to_le_bytes());
        }
        for l in y.iter() {
            buf.push(*l);
        }
        for (name, m) in sets.iter() {
            let nb = name.as_bytes();
            buf.extend((nb.len() as u64).to_le_bytes());
            buf.extend(nb);
            let d = m.first().map(|r| r.len()).unwrap_or(0);
            buf.extend((d as u64).to_le_bytes());
            for row in m.iter() {
                for v in row.iter() {
                    buf.extend(v.to_le_bytes());
                }
            }
        }
        std::fs::write(p, &buf).unwrap_or_else(|e| panic!("{}: {}", p, e));
        println!("representation written to {} ({:.1} MB)", p, buf.len() as f64 / 1e6);
    }

    let (a_last, l2_last) = cv_auroc_swept(&f_last, &y, &keep_ids, 300);
    let (a_sum, l2_sum) = cv_auroc_swept(&f_summary, &y, &keep_ids, 300);
    let (a_ord, l2_ord) = cv_auroc_swept(&f_order, &y, &keep_ids, 300);
    let (a_both, l2_both) = cv_auroc_swept(&f_both, &y, &keep_ids, 300);
    let (a_mean, l2_mean) = cv_auroc_swept(&f_mean, &y, &keep_ids, 300);
    let (a_max, l2_max) = cv_auroc_swept(&f_max, &y, &keep_ids, 300);

    // The same probe on the severity score, so the classifier is not the thing
    // being compared.
    let sv: Vec<Vec<f32>> = keep_ids
        .iter()
        .map(|i| vec![*saps.get(i).unwrap_or(&-1.0) as f32])
        .collect();
    let (a_saps, l2_saps) = cv_auroc_swept(&sv, &y, &keep_ids, 300);

    println!("\n{:<44} {:>9}", "representation", "AUROC");
    println!("{:<40} {:>9.4}  L2 {:>6}", "model state at the final tick", a_last, l2_last);
    println!("{:<40} {:>9.4}  L2 {:>6}", "model state, mean-pooled", a_mean, l2_mean);
    println!("{:<40} {:>9.4}  L2 {:>6}", "model state, max-pooled", a_max, l2_max);
    println!("{:<40} {:>9.4}  L2 {:>6}   <- control", "count+gap summary (order-free)", a_sum, l2_sum);
    println!("{:<40} {:>9.4}  L2 {:>6}", "hashed bigrams alone (order only)", a_ord, l2_ord);
    println!("{:<40} {:>9.4}  L2 {:>6}   <- gate 2", "counts + bigrams", a_both, l2_both);
    println!("{:<40} {:>9.4}  L2 {:>6}", "SAPS-I through the same probe", a_saps, l2_saps);
    println!("{:<44} {:>9.4}", "timing alone, best single feature", 0.6157);
    println!("{:<44} {:>9.4}", "SAPS-I, direct (measured earlier)", 0.6370);
    println!(
        "\n  the memory ran continually across all {} patients and was never reset;",
        feats.len()
    );
    println!("  only the volatile half -- cursors, bound traces, ladder -- restarts per patient.");
    println!("  the probe is linear and weak on purpose: what it finds is in the state.");
    println!();
}
