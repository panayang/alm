//! Does the *timing* of a clinical record carry the outcome, on its own?
//!
//! The concept-drift streams failed the gate this project set for itself.
//! Electricity is sampled every thirty minutes forever -- its forward gaps have
//! a coefficient of variation of exactly zero -- and airlines has a time-of-day
//! attribute, not an arrival process. Neither has any silence to spend, so the
//! one axis this architecture brings that the field does not have would have
//! been unmeasurable there.
//!
//! Irregularly sampled clinical series are the opposite case, and not by
//! accident: the entire subfield exists because the observation times are
//! informative. A patient measured every five minutes is not the same patient
//! as one measured every six hours, and the methods built for it -- GRU-D,
//! SeFT, mTAND, Raindrop -- are built around that fact, which the literature
//! calls informative missingness.
//!
//! That is our claim in somebody else's words, so it should be checked in the
//! strongest possible form, on their data, before anything is built:
//!
//! > Using **only when measurements happened**, and never what they said, how
//! > much of the outcome can be recovered?
//!
//! If the answer is nothing, then the gap is decoration here too and the
//! architecture's temporal commitment has still found no home. If the answer is
//! substantial, then this source consumes exactly what we uniquely offer, and
//! it comes with real baselines and an evaluation that was never a single
//! scalar -- PhysioNet 2012 carries in-hospital death, SAPS-I, SOFA, length of
//! stay and survival time for every record.
//!
//! Nothing here runs our model.

use std::collections::HashMap;

pub struct Patient {
    pub id: u32,
    /// Minutes from admission for every measurement event, in order.
    pub times: Vec<u32>,
    /// How many distinct parameters were recorded at each event time.
    pub width: Vec<u32>,
    pub death: Option<u8>,
    pub los: Option<i32>,
    pub saps: Option<i32>,
    pub sofa: Option<i32>,
}

fn entropy_mm(counts: &[u64]) -> f64 {
    let n: u64 = counts.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut h = 0.0;
    let mut occ = 0usize;
    for &c in counts {
        if c > 0 {
            occ += 1;
            let p = c as f64 / nf;
            h -= p * p.log2();
        }
    }
    h + (occ as f64 - 1.0) / (2.0 * nf * std::f64::consts::LN_2)
}

/// Area under the ROC curve, by rank. Ties get the average rank, so a feature
/// with few distinct values is not flattered.
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
    let (mut pos, mut neg, mut sum_pos) = (0.0f64, 0.0f64, 0.0f64);
    for (k, &l) in label.iter().enumerate() {
        if l == 1 {
            pos += 1.0;
            sum_pos += rank[k];
        } else {
            neg += 1.0;
        }
    }
    if pos == 0.0 || neg == 0.0 {
        return 0.5;
    }
    (sum_pos - pos * (pos + 1.0) / 2.0) / (pos * neg)
}

/// Mutual information between a real-valued feature, binned by quantile, and a
/// binary label.
fn mi_binned(x: &[f64], y: &[u8], bins: usize) -> (f64, f64) {
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cuts: Vec<f64> = (1..bins).map(|b| sorted[(sorted.len() - 1) * b / bins]).collect();
    let mut joint = vec![0u64; bins * 2];
    let mut mx = vec![0u64; bins];
    let mut my = vec![0u64; 2];
    for (&v, &l) in x.iter().zip(y.iter()) {
        let b = cuts.iter().filter(|&&c| v > c).count().min(bins - 1);
        joint[b * 2 + l as usize] += 1;
        mx[b] += 1;
        my[l as usize] += 1;
    }
    let h_y = entropy_mm(&my);
    let mi = entropy_mm(&mx) + h_y - entropy_mm(&joint);
    (mi.max(0.0), h_y)
}

pub fn load(dir: &str, outcomes: &str) -> Vec<Patient> {
    let mut out: HashMap<u32, (Option<u8>, Option<i32>, Option<i32>, Option<i32>)> = HashMap::new();
    let text = std::fs::read_to_string(outcomes).unwrap_or_else(|e| panic!("{}: {}", outcomes, e));
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if f.len() < 6 {
            continue;
        }
        out.insert(
            f[0].parse().unwrap_or(0),
            (
                f[5].parse::<u8>().ok(),
                f[3].parse::<i32>().ok(),
                f[1].parse::<i32>().ok(),
                f[2].parse::<i32>().ok(),
            ),
        );
    }

    let mut pats = Vec::new();
    let rd = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {}", dir, e));
    let mut files: Vec<std::path::PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    files.sort();
    for p in files {
        let body = match std::fs::read_to_string(&p) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let id: u32 = p
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        // Collapse to distinct event *times*: several parameters recorded at
        // the same minute are one observation event, which is what the arrival
        // process actually is.
        let mut by_time: Vec<(u32, u32)> = Vec::new();
        for (i, line) in body.lines().enumerate() {
            if i == 0 || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 3 {
                continue;
            }
            let (hh, mm) = match f[0].split_once(':') {
                Some((a, b)) => (a.parse::<u32>().unwrap_or(0), b.parse::<u32>().unwrap_or(0)),
                None => continue,
            };
            let t = hh * 60 + mm;
            // The zero-minute block is demographics, not a measurement.
            if t == 0 {
                continue;
            }
            match by_time.last_mut() {
                Some((lt, w)) if *lt == t => *w += 1,
                _ => by_time.push((t, 1)),
            }
        }
        by_time.sort_by_key(|&(t, _)| t);
        let o = out.get(&id).copied().unwrap_or((None, None, None, None));
        pats.push(Patient {
            id,
            times: by_time.iter().map(|&(t, _)| t).collect(),
            width: by_time.iter().map(|&(_, w)| w).collect(),
            death: o.0,
            los: o.1,
            saps: o.2,
            sofa: o.3,
        });
    }
    pats
}

pub fn run(dir: &str, outcomes: &str, label: &str, bins: usize) {
    let pats: Vec<Patient> = load(dir, outcomes)
        .into_iter()
        .filter(|p| p.death.is_some() && p.times.len() >= 2)
        .collect();
    let n = pats.len();
    println!("==================== {} ====================", label);
    let deaths: usize = pats.iter().filter(|p| p.death == Some(1)).count();
    println!(
        "patients {}   in-hospital deaths {} ({:.4})",
        n,
        deaths,
        deaths as f64 / n as f64
    );

    // ---- the arrival process --------------------------------------------
    let mut all_gaps: Vec<f64> = Vec::new();
    for p in pats.iter() {
        for w in p.times.windows(2) {
            all_gaps.push((w[1] - w[0]) as f64);
        }
    }
    all_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = all_gaps.iter().sum::<f64>() / all_gaps.len() as f64;
    let var = all_gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / all_gaps.len() as f64;
    let q = |f: f64| all_gaps[((all_gaps.len() - 1) as f64 * f) as usize];
    println!("\n-- the arrival process --");
    println!("  observation events {}   per patient mean {:.1}", all_gaps.len() + n, all_gaps.len() as f64 / n as f64 + 1.0);
    println!(
        "  inter-event gap (minutes): p10 {}  p50 {}  p90 {}  p99 {}  max {}",
        q(0.10),
        q(0.50),
        q(0.90),
        q(0.99),
        all_gaps[all_gaps.len() - 1]
    );
    println!("  mean {:.2}   cv {:.4}", mean, var.sqrt() / mean);
    println!("  (electricity's cv was 0.0000 -- regularly sampled, nothing to spend)");

    // ---- the gate: timing only, never a value ---------------------------
    let y: Vec<u8> = pats.iter().map(|p| p.death.unwrap()).collect();
    let feat = |f: &dyn Fn(&Patient) -> f64| -> Vec<f64> { pats.iter().map(f).collect() };

    let features: Vec<(&str, Vec<f64>)> = vec![
        ("event count", feat(&|p| p.times.len() as f64)),
        ("mean gap", feat(&|p| {
            let g: Vec<f64> = p.times.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
            g.iter().sum::<f64>() / g.len().max(1) as f64
        })),
        ("gap cv", feat(&|p| {
            let g: Vec<f64> = p.times.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
            let m = g.iter().sum::<f64>() / g.len().max(1) as f64;
            let v = g.iter().map(|x| (x - m).powi(2)).sum::<f64>() / g.len().max(1) as f64;
            v.sqrt() / m.max(1e-9)
        })),
        ("longest gap", feat(&|p| {
            p.times.windows(2).map(|w| (w[1] - w[0]) as f64).fold(0.0, f64::max)
        })),
        ("events in last 6h", feat(&|p| {
            let last = *p.times.last().unwrap_or(&0);
            p.times.iter().filter(|&&t| t + 360 >= last).count() as f64
        })),
        ("parameters per event", feat(&|p| {
            p.width.iter().map(|&w| w as f64).sum::<f64>() / p.width.len().max(1) as f64
        })),
        ("span of record (min)", feat(&|p| {
            (*p.times.last().unwrap_or(&0) - *p.times.first().unwrap_or(&0)) as f64
        })),
    ];

    println!("\n-- the gate: what the timing alone says about in-hospital death --");
    println!("   (no measured value is used anywhere in this table)");
    println!("{:<26} {:>9} {:>12} {:>10}", "timing-only feature", "AUROC", "I(y;x) bits", "% of H(y)");
    let mut best_auc: f64 = 0.5;
    for (nm, x) in features.iter() {
        let a = auroc(x, &y);
        let (mi, h) = mi_binned(x, &y, bins);
        best_auc = best_auc.max(a.max(1.0 - a));
        println!(
            "{:<26} {:>9.4} {:>12.4} {:>10.2}",
            nm,
            a,
            mi,
            100.0 * mi / h.max(1e-9)
        );
    }

    // For scale: the severity scores the clinicians computed from the values.
    println!("\n-- for scale, the value-derived scores that come with the data --");
    for (nm, g) in [
        ("SAPS-I", pats.iter().map(|p| p.saps.unwrap_or(-1) as f64).collect::<Vec<_>>()),
        ("SOFA", pats.iter().map(|p| p.sofa.unwrap_or(-1) as f64).collect::<Vec<_>>()),
    ] {
        let a = auroc(&g, &y);
        let (mi, h) = mi_binned(&g, &y, bins);
        println!("{:<26} {:>9.4} {:>12.4} {:>10.2}", nm, a, mi, 100.0 * mi / h.max(1e-9));
    }

    println!("\n  best timing-only AUROC {:.4}. 0.5 is a coin; the gate passes only if the", best_auc);
    println!("  observation process is informative on its own, which is what this field");
    println!("  calls informative missingness and what this architecture calls silence");
    println!("  carrying information.");
    println!();
}
