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

// ---------------------------------------------------------------------------
// What the two streams say about each other
// ---------------------------------------------------------------------------
//
// The outcome gate passed: when measurements happen carries nearly as much
// about in-hospital death as what they said. That licenses the source, but it
// does not yet say there is anything here for a *sequential* mechanism, which
// is what this architecture is. So before any model code, the label-free
// question:
//
//   * does the gap help predict what arrives next?
//   * does what arrived help predict how long the next silence will be?
//
// This project's design assumes both -- the input and output streams interleave
// as one context, and the silence is where the work happens. If the two are
// independent here, then the gaps are informative about the *patient*, which
// the outcome gate already showed, while being useless *sequentially*, and the
// tick loop would have nothing to do with the reason this source looked right.
//
// A panel is the set of parameters recorded at one observation time, hashed to
// an id: one observation event, one tick, which is the granularity the tick
// loop actually runs at.

fn gap_bucket(g: u32) -> usize {
    // Minutes, log2. p50 is 31 and p99 is 120, so this spreads the mass.
    (32 - (g.max(1)).leading_zeros()) as usize
}

/// Panel streams, rebuilt at the granularity the tick loop runs at.
pub fn panel_streams(dir: &str) -> (Vec<Vec<(u32, u32)>>, usize, usize) {
    let mut param_id: HashMap<String, u32> = HashMap::new();
    let mut panel_id: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut streams: Vec<Vec<(u32, u32)>> = Vec::new();

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {}", dir, e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    files.sort();
    for path in files.iter() {
        let body = match std::fs::read_to_string(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let mut seq: Vec<(u32, u32)> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        let mut cur_t = u32::MAX;
        let mut last_t = 0u32;
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
            if t == 0 {
                continue;
            }
            if t != cur_t && !cur.is_empty() {
                cur.sort_unstable();
                cur.dedup();
                let n = panel_id.len() as u32;
                let pid = *panel_id.entry(cur.clone()).or_insert(n);
                seq.push((cur_t.saturating_sub(last_t), pid));
                last_t = cur_t;
                cur.clear();
            }
            cur_t = t;
            let n = param_id.len() as u32;
            cur.push(*param_id.entry(f[1].to_string()).or_insert(n));
        }
        if !cur.is_empty() {
            cur.sort_unstable();
            cur.dedup();
            let n = panel_id.len() as u32;
            let pid = *panel_id.entry(cur.clone()).or_insert(n);
            seq.push((cur_t.saturating_sub(last_t), pid));
        }
        if seq.len() >= 3 {
            streams.push(seq);
        }
    }
    (streams, param_id.len(), panel_id.len())
}

/// Prequential PPM-C over a panel sequence. Split out so that any subset of
/// patients can be priced with exactly the bar it will be compared against --
/// a bar measured on a different number of patients is not a bar.
pub fn ppm_bar(
    streams: &[Vec<(u32, u32)>],
    vp: usize,
    order: usize,
    use_gap: bool,
    nb: usize,
) -> (f64, f64) {
    let mut ctx: HashMap<u64, HashMap<u32, u32>> = HashMap::new();
    let (mut bits, mut n_ev, mut hits) = (0.0f64, 0u64, 0u64);
    for s in streams.iter() {
        let mut hist: Vec<(u32, usize)> = Vec::new();
        for &(g, pan) in s.iter() {
            let gb = gap_bucket(g).min(nb - 1);
            let key = |o: usize, hist: &[(u32, usize)]| -> u64 {
                let mut k = 0xcbf2_9ce4_8422_2325u64 ^ o as u64;
                for &(h, hg) in hist[hist.len() - o..].iter() {
                    k = (k ^ h as u64).wrapping_mul(0x0000_0100_0000_01b3);
                    if use_gap {
                        k = (k ^ hg as u64).wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
                if use_gap {
                    k = (k ^ (gb as u64) << 32).wrapping_mul(0x0000_0100_0000_01b3);
                }
                k
            };
            let mut p = 0.0f64;
            let mut esc = 1.0f64;
            let mut best: Option<u32> = None;
            for o in (0..=order.min(hist.len())).rev() {
                if let Some(c) = ctx.get(&key(o, &hist)) {
                    let tot: u32 = c.values().sum();
                    if tot > 0 {
                        let e = c.len() as f64 / (c.len() as f64 + tot as f64);
                        p += esc * (1.0 - e) * (*c.get(&pan).unwrap_or(&0) as f64 / tot as f64);
                        if best.is_none() {
                            best = c.iter().max_by_key(|(_, &v)| v).map(|(&s, _)| s);
                        }
                        esc *= e;
                    }
                }
            }
            p += esc / vp as f64;
            bits += -p.max(f64::MIN_POSITIVE).log2();
            n_ev += 1;
            if best == Some(pan) {
                hits += 1;
            }
            for o in 0..=order.min(hist.len()) {
                *ctx.entry(key(o, &hist)).or_default().entry(pan).or_insert(0) += 1;
            }
            hist.push((pan, gb));
        }
    }
    (bits / n_ev as f64, hits as f64 / n_ev as f64)
}

pub fn next_event(dir: &str, label: &str, order: usize) {
    let (streams, n_param, vp) = panel_streams(dir);
    let events: usize = streams.iter().map(|s| s.len()).sum();
    println!("==================== {}   [next-event] ====================", label);
    println!(
        "patients {}   observation events {}   distinct parameters {}   distinct panels {}",
        streams.len(),
        events,
        n_param,
        vp
    );

    let nb = 16usize;
    let mut m_panel = vec![0u64; vp];
    let mut m_gap = vec![0u64; nb];
    let mut by_prev: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut by_prev_gap: HashMap<(u32, usize), Vec<u64>> = HashMap::new();
    let mut gap_by_prev: HashMap<u32, Vec<u64>> = HashMap::new();
    for s in streams.iter() {
        for w in s.windows(2) {
            let g = gap_bucket(w[1].0).min(nb - 1);
            let pan = w[1].1 as usize;
            let prev = w[0].1;
            m_panel[pan] += 1;
            m_gap[g] += 1;
            by_prev.entry(prev).or_insert_with(|| vec![0; vp])[pan] += 1;
            by_prev_gap.entry((prev, g)).or_insert_with(|| vec![0; vp])[pan] += 1;
            gap_by_prev.entry(prev).or_insert_with(|| vec![0; nb])[g] += 1;
        }
    }
    let total: u64 = m_panel.iter().sum::<u64>().max(1);
    let mut hp_prev = 0.0;
    for c in by_prev.values() {
        hp_prev += (c.iter().sum::<u64>() as f64 / total as f64) * entropy_mm(c);
    }
    let mut hp_prev_gap = 0.0;
    for c in by_prev_gap.values() {
        hp_prev_gap += (c.iter().sum::<u64>() as f64 / total as f64) * entropy_mm(c);
    }
    let mut hg_prev = 0.0;
    for c in gap_by_prev.values() {
        hg_prev += (c.iter().sum::<u64>() as f64 / total as f64) * entropy_mm(c);
    }
    let hp = entropy_mm(&m_panel);
    let hg = entropy_mm(&m_gap);

    println!("\n-- does the gap help predict what arrives next? --");
    println!("  H(panel)                      {:.4} bits", hp);
    println!("  H(panel | prev panel)         {:.4} bits", hp_prev);
    println!("  H(panel | prev panel, gap)    {:.4} bits", hp_prev_gap);
    println!(
        "  I(panel ; gap | prev panel) = {:.4} bits   ({:.2}% of H(panel|prev))",
        hp_prev - hp_prev_gap,
        100.0 * (hp_prev - hp_prev_gap) / hp_prev.max(1e-9)
    );

    println!("\n-- does what arrived help predict how long the silence will be? --");
    println!("  H(gap)                        {:.4} bits", hg);
    println!("  H(gap | prev panel)           {:.4} bits", hg_prev);
    println!(
        "  I(gap ; prev panel) = {:.4} bits   ({:.2}% of H(gap))",
        hg - hg_prev,
        100.0 * (hg - hg_prev) / hg.max(1e-9)
    );

    // --- the bar ----------------------------------------------------------
    //
    // Two of them, and the second is the one that matters. Beating a gap-blind
    // predictor would prove nothing about a gap-native mechanism -- it would
    // only prove that the gap is informative, which the table above already
    // says. So the honest bar is a PPM that gets the gap in its context. The
    // gap before an event is known when the event is predicted: in an online
    // setting you know how long you have been waiting, which is exactly what
    // the tick loop is given.
    let run_ppm = |use_gap: bool| -> (f64, f64) { ppm_bar(&streams, vp, order, use_gap, nb) };
    let _unused = |use_gap: bool| -> (f64, f64) {
        let mut ctx: HashMap<u64, HashMap<u32, u32>> = HashMap::new();
        let (mut bits, mut n_ev, mut hits) = (0.0f64, 0u64, 0u64);
        for s in streams.iter() {
            let mut hist: Vec<(u32, usize)> = Vec::new();
            for &(g, pan) in s.iter() {
                let gb = gap_bucket(g).min(nb - 1);
                let key = |o: usize, hist: &[(u32, usize)]| -> u64 {
                    let mut k = 0xcbf2_9ce4_8422_2325u64 ^ o as u64;
                    for &(h, hg) in hist[hist.len() - o..].iter() {
                        k = (k ^ h as u64).wrapping_mul(0x0000_0100_0000_01b3);
                        if use_gap {
                            k = (k ^ hg as u64).wrapping_mul(0x0000_0100_0000_01b3);
                        }
                    }
                    if use_gap {
                        k = (k ^ (gb as u64) << 32).wrapping_mul(0x0000_0100_0000_01b3);
                    }
                    k
                };
                let mut p = 0.0f64;
                let mut esc = 1.0f64;
                let mut best: Option<u32> = None;
                for o in (0..=order.min(hist.len())).rev() {
                    if let Some(c) = ctx.get(&key(o, &hist)) {
                        let tot: u32 = c.values().sum();
                        if tot > 0 {
                            let e = c.len() as f64 / (c.len() as f64 + tot as f64);
                            p += esc * (1.0 - e) * (*c.get(&pan).unwrap_or(&0) as f64 / tot as f64);
                            if best.is_none() {
                                best = c.iter().max_by_key(|(_, &v)| v).map(|(&s, _)| s);
                            }
                            esc *= e;
                        }
                    }
                }
                p += esc / vp as f64;
                bits += -p.max(f64::MIN_POSITIVE).log2();
                n_ev += 1;
                if best == Some(pan) {
                    hits += 1;
                }
                for o in 0..=order.min(hist.len()) {
                    *ctx.entry(key(o, &hist)).or_default().entry(pan).or_insert(0) += 1;
                }
                hist.push((pan, gb));
            }
        }
        (bits / n_ev as f64, hits as f64 / n_ev as f64)
    };

    let (b_blind, a_blind) = run_ppm(false);
    let (b_gap, a_gap) = run_ppm(true);
    println!("\n-- the bar: prequential PPM-C over the panel sequence, order {} --", order);
    println!("{:<28} {:>12} {:>10}", "", "bits/event", "top-1");
    println!("{:<28} {:>12.4} {:>10.4}", "gap-blind", b_blind, a_blind);
    println!("{:<28} {:>12.4} {:>10.4}", "gap in the context", b_gap, a_gap);
    println!(
        "{:<28} {:>12.4}",
        "marginal H(panel)", hp
    );
    println!(
        "
  the gap is worth {:.4} bits/event to PPM, against the {:.4} the table above",
        b_blind - b_gap,
        hp_prev - hp_prev_gap
    );
    println!("  says is there. The gap-aware row is the number a gap-native mechanism has to beat;");
    println!("  beating the gap-blind one would only re-prove that the gap is informative.");
    println!();
}

/// The same records at parameter granularity.
///
/// Panels are the honest unit of *arrival* -- one observation minute is one
/// event -- but they are a bad unit of *prediction*: 3600 distinct sets over
/// forty thousand events leaves most of them seen a handful of times, which is
/// the regime where exact counting wins and a distributed code cannot. At
/// parameter granularity the vocabulary is 37.
///
/// Parameters within one minute keep their recorded order and carry a gap of
/// zero, so the arrival structure is unchanged: the silence still sits exactly
/// where the world was silent, and a panel becomes a burst of adjacent ticks.
pub fn flat_streams(dir: &str) -> (Vec<Vec<(u32, u32)>>, usize) {
    let (s, _, v) = flat_streams_ids(dir);
    (s, v)
}

/// The same, keeping each record's id so outcomes can be joined to it.
pub fn flat_streams_ids(dir: &str) -> (Vec<Vec<(u32, u32)>>, Vec<u32>, usize) {
    let mut param_id: HashMap<String, u32> = HashMap::new();
    let mut streams: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {}", dir, e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    files.sort();
    for path in files.iter() {
        let body = match std::fs::read_to_string(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let rid: u32 = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let mut seq: Vec<(u32, u32)> = Vec::new();
        let mut last_t = 0u32;
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
            if t == 0 {
                continue;
            }
            let n = param_id.len() as u32;
            let pid = *param_id.entry(f[1].to_string()).or_insert(n);
            seq.push((t.saturating_sub(last_t), pid));
            last_t = t;
        }
        if seq.len() >= 3 {
            streams.push(seq);
            ids.push(rid);
        }
    }
    let v = param_id.len();
    (streams, ids, v)
}

/// The same records, with the measured value carried in the token.
///
/// Parameter granularity told the model *that* something was measured and never
/// *what it said*. That is the observation process alone, and it reached AUROC
/// 0.6301 pooled -- above the best timing-only feature and just under SAPS-I,
/// which clinicians compute from the values. This adds the values: the token
/// becomes parameter x quantile bin, so the vocabulary is 37 * bins.
///
/// Cuts are per-parameter quantiles over the whole set. That is a fixed
/// discretisation and not a label-dependent one, but it does see every patient,
/// so it is preprocessing of the kind the baselines also do rather than
/// something the model learned. PhysioNet writes -1 for a missing value, which
/// is not a measurement and is dropped.
pub fn flat_streams_valued(dir: &str, bins: usize) -> (Vec<Vec<(u32, u32)>>, Vec<u32>, usize) {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {}", dir, e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    files.sort();

    // First pass: per-parameter value distributions, for the cuts.
    let mut param_id: HashMap<String, u32> = HashMap::new();
    let mut vals: Vec<Vec<f64>> = Vec::new();
    let mut parsed: Vec<(u32, Vec<(u32, u32, f64)>)> = Vec::new();
    for path in files.iter() {
        let body = match std::fs::read_to_string(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let rid: u32 = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let mut seq: Vec<(u32, u32, f64)> = Vec::new();
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
            if t == 0 {
                continue;
            }
            let v: f64 = match f[2].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v < 0.0 {
                continue; // -1 is PhysioNet's missing marker, not a reading.
            }
            let n = param_id.len() as u32;
            let pid = *param_id.entry(f[1].to_string()).or_insert(n);
            if vals.len() <= pid as usize {
                vals.resize(pid as usize + 1, Vec::new());
            }
            vals[pid as usize].push(v);
            seq.push((t, pid, v));
        }
        if seq.len() >= 3 {
            parsed.push((rid, seq));
        }
    }

    let cuts: Vec<Vec<f64>> = vals
        .iter()
        .map(|v| {
            let mut s = v.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if s.is_empty() {
                return Vec::new();
            }
            (1..bins).map(|b| s[(s.len() - 1) * b / bins]).collect()
        })
        .collect();

    let mut streams = Vec::with_capacity(parsed.len());
    let mut ids = Vec::with_capacity(parsed.len());
    for (rid, seq) in parsed {
        let mut out: Vec<(u32, u32)> = Vec::with_capacity(seq.len());
        let mut last_t = 0u32;
        for (t, pid, v) in seq {
            let b = cuts[pid as usize].iter().filter(|&&c| v > c).count().min(bins - 1);
            out.push((t.saturating_sub(last_t), pid * bins as u32 + b as u32));
            last_t = t;
        }
        streams.push(out);
        ids.push(rid);
    }
    let v = param_id.len() * bins;
    (streams, ids, v)
}
