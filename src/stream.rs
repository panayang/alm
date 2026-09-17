//! What a concept-drift stream is actually made of.
//!
//! Data stream mining is the one field whose standard evaluation is already
//! ours: prequential. Every instance is predicted before its label is seen, the
//! label then updates the model, and nothing is ever replayed. That is the same
//! accounting this project charges -- only when the world speaks -- and it was
//! not imported for our benefit, it is what the field does. Its metrics are
//! plural for the same reason ours are: prequential accuracy, how long recovery
//! from a drift takes, memory, time per instance.
//!
//! Before any of that is worth attempting, one property has to be checked,
//! because this architecture's distinctive claim depends on it entirely.
//!
//! # The gate
//!
//! Our depth is decided by how long the world stays quiet. If a stream is
//! regularly sampled -- one instance every thirty minutes, forever -- then
//! there is no silence to spend, every instance gets the same budget, and the
//! one axis we bring that the field does not have collapses to nothing. It
//! would not make us wrong, it would make us indistinguishable.
//!
//! So this instrument reports, first, whether the inter-arrival structure
//! exists at all: is there a time column, are the gaps constant, and does the
//! gap carry any information about what arrives. Only then does it report the
//! things that decide whether the problem is hard -- the prequential bar set by
//! trivial predictors, and where the stream actually drifts.
//!
//! Three baselines, because each fails differently and the gap between them is
//! the real bar:
//!
//!   * **majority** -- the class prior, updated online. Beating this is nothing.
//!   * **no-change** -- predict the previous label. On a stream with strong
//!     temporal autocorrelation this is embarrassingly strong, and it is the
//!     baseline the electricity dataset is famous for failing against.
//!   * **naive Bayes** -- counts over binned attributes, updated online. The
//!     honest floor a real method has to clear.
//!
//! Nothing here runs our model.

use std::collections::HashMap;

/// A parsed ARFF or CSV stream, every attribute reduced to a bin id.
pub struct Stream {
    pub names: Vec<String>,
    pub nominal: Vec<bool>,
    /// cols[a][i] is instance i's bin in attribute a.
    pub cols: Vec<Vec<u32>>,
    pub vocab: Vec<usize>,
    /// Raw numeric value, kept for the attributes that had one.
    pub raw: Vec<Option<Vec<f64>>>,
    pub labels: Vec<u32>,
    pub classes: usize,
    pub rows: usize,
}

/// Quantile binning is done over the whole stream, which a real online method
/// could not do. It is fine here because nothing downstream of it is a result:
/// the bins exist so that attributes can be counted, not so that a model can
/// be trained.
pub fn load_arff(path: &str, bins: usize) -> Stream {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let mut names: Vec<String> = Vec::new();
    let mut is_nom: Vec<bool> = Vec::new();
    let mut body: Vec<Vec<String>> = Vec::new();
    let mut in_data = false;
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('%') {
            continue;
        }
        let low = l.to_ascii_lowercase();
        if !in_data {
            if low.starts_with("@attribute") {
                let rest = l["@attribute".len()..].trim();
                let (nm, ty) = rest.split_once(char::is_whitespace).unwrap_or((rest, "numeric"));
                names.push(nm.trim_matches('\'').to_string());
                is_nom.push(!ty.trim().to_ascii_lowercase().starts_with("numeric")
                    && !ty.trim().to_ascii_lowercase().starts_with("real")
                    && !ty.trim().to_ascii_lowercase().starts_with("integer"));
            } else if low.starts_with("@data") {
                in_data = true;
            }
            continue;
        }
        let f: Vec<String> = l.split(',').map(|s| s.trim().trim_matches('\'').to_string()).collect();
        if f.len() == names.len() {
            body.push(f);
        }
    }
    assert!(!body.is_empty(), "no @data rows in {}", path);
    let rows = body.len();
    let ncol = names.len();

    let mut cols = Vec::with_capacity(ncol - 1);
    let mut vocab = Vec::with_capacity(ncol - 1);
    let mut raw: Vec<Option<Vec<f64>>> = Vec::with_capacity(ncol - 1);
    for a in 0..ncol - 1 {
        let vals: Vec<&str> = body.iter().map(|r| r[a].as_str()).collect();
        let nums: Option<Vec<f64>> = vals.iter().map(|s| s.parse::<f64>().ok()).collect();
        let distinct: std::collections::HashSet<&str> = vals.iter().copied().collect();
        match nums {
            Some(nv) if !is_nom[a] && distinct.len() > bins => {
                let mut sorted = nv.clone();
                sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let cuts: Vec<f64> =
                    (1..bins).map(|b| sorted[(sorted.len() - 1) * b / bins]).collect();
                cols.push(
                    nv.iter().map(|v| cuts.iter().filter(|&&c| *v > c).count() as u32).collect(),
                );
                vocab.push(bins);
                raw.push(Some(nv));
            }
            other => {
                let mut ids: HashMap<&str, u32> = HashMap::new();
                let c: Vec<u32> = vals
                    .iter()
                    .map(|s| {
                        let n = ids.len() as u32;
                        *ids.entry(s).or_insert(n)
                    })
                    .collect();
                vocab.push(ids.len());
                cols.push(c);
                raw.push(other);
            }
        }
    }

    let mut lab_ids: HashMap<&str, u32> = HashMap::new();
    let labels: Vec<u32> = body
        .iter()
        .map(|r| {
            let s = r[ncol - 1].as_str();
            let n = lab_ids.len() as u32;
            *lab_ids.entry(s).or_insert(n)
        })
        .collect();
    let classes = lab_ids.len();

    Stream {
        names: names[..ncol - 1].to_vec(),
        nominal: is_nom[..ncol - 1].to_vec(),
        cols,
        vocab,
        raw,
        labels,
        classes,
        rows,
    }
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

pub fn run(path: &str, label: &str, bins: usize, window: usize) {
    let s = load_arff(path, bins);
    let n = s.cols.len();
    println!("==================== {} ====================", label);
    println!("instances {}   attributes {}   classes {}", s.rows, n, s.classes);
    let mut cls = vec![0u64; s.classes];
    for &l in s.labels.iter() {
        cls[l as usize] += 1;
    }
    let prior = cls.iter().copied().max().unwrap_or(0) as f64 / s.rows as f64;
    println!(
        "class balance {:?}   majority prior {:.4}",
        cls.iter().map(|&c| c as f64 / s.rows as f64).map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        prior
    );

    // ---- the gate: is there any silence to spend? ------------------------
    println!("\n-- is there inter-arrival structure? --");
    let mut found = false;
    for a in 0..n {
        let lname = s.names[a].to_ascii_lowercase();
        let looks_temporal = lname.contains("date")
            || lname.contains("time")
            || lname.contains("period")
            || lname.contains("stamp");
        if !looks_temporal {
            continue;
        }
        found = true;
        match &s.raw[a] {
            Some(v) => {
                let mut gaps: Vec<f64> = v.windows(2).map(|w| w[1] - w[0]).collect();
                let forward: Vec<f64> = gaps.iter().copied().filter(|&g| g > 0.0).collect();
                gaps.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let distinct: std::collections::HashSet<u64> =
                    forward.iter().map(|g| (g * 1e9) as u64).collect();
                let mean = forward.iter().sum::<f64>() / forward.len().max(1) as f64;
                let var = forward.iter().map(|g| (g - mean).powi(2)).sum::<f64>()
                    / forward.len().max(1) as f64;
                println!(
                    "  {:<12} forward gaps: {} distinct value(s), mean {:.6}, cv {:.4}",
                    s.names[a],
                    distinct.len(),
                    mean,
                    var.sqrt() / mean.abs().max(1e-12)
                );
                if distinct.len() <= 2 {
                    println!("      -> REGULARLY SAMPLED. There is no silence here to compute in.");
                }
            }
            None => println!("  {:<12} nominal, no usable gap", s.names[a]),
        }
    }
    if !found {
        println!("  no time-like attribute at all -- the stream is a sequence, not a history.");
    }

    // ---- the bar --------------------------------------------------------
    // All three predict before seeing the label, then update. Nothing replays.
    let mut maj = vec![0u64; s.classes];
    let (mut maj_ok, mut nochg_ok, mut nb_ok) = (0u64, 0u64, 0u64);
    let mut prev: Option<u32> = None;
    // naive Bayes counts: [attr][bin][class]
    let mut nb: Vec<Vec<Vec<u64>>> =
        (0..n).map(|a| vec![vec![0u64; s.classes]; s.vocab[a]]).collect();
    let mut nb_cls = vec![0u64; s.classes];
    let mut curve: Vec<(usize, f64, f64)> = Vec::new();
    let (mut w_nb, mut w_nc) = (0u64, 0u64);

    for i in 0..s.rows {
        let y = s.labels[i];
        // majority
        let m = maj
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(k, _)| k as u32)
            .unwrap_or(0);
        if m == y {
            maj_ok += 1;
        }
        // no-change
        if prev == Some(y) {
            nochg_ok += 1;
            w_nc += 1;
        }
        // naive Bayes
        let total: u64 = nb_cls.iter().sum();
        let mut best = 0usize;
        let mut best_lp = f64::NEG_INFINITY;
        for c in 0..s.classes {
            let mut lp = ((nb_cls[c] + 1) as f64 / (total + s.classes as u64) as f64).ln();
            for a in 0..n {
                let b = s.cols[a][i] as usize;
                let num = nb[a][b][c] + 1;
                let den = nb_cls[c] + s.vocab[a] as u64;
                lp += (num as f64 / den as f64).ln();
            }
            if lp > best_lp {
                best_lp = lp;
                best = c;
            }
        }
        if best as u32 == y {
            nb_ok += 1;
            w_nb += 1;
        }

        maj[y as usize] += 1;
        nb_cls[y as usize] += 1;
        for a in 0..n {
            nb[a][s.cols[a][i] as usize][y as usize] += 1;
        }
        prev = Some(y);

        if (i + 1) % window == 0 {
            curve.push((i + 1, w_nb as f64 / window as f64, w_nc as f64 / window as f64));
            w_nb = 0;
            w_nc = 0;
        }
    }

    println!("\n-- the prequential bar (predict, then see the label, never replay) --");
    println!("  majority (class prior)   {:.4}", maj_ok as f64 / s.rows as f64);
    println!("  no-change (previous label) {:.4}", nochg_ok as f64 / s.rows as f64);
    println!("  naive Bayes              {:.4}", nb_ok as f64 / s.rows as f64);

    // ---- where it drifts -------------------------------------------------
    println!("\n-- windowed accuracy ({} instances per window) --", window);
    println!("{:>10} {:>12} {:>12}", "upto", "naiveBayes", "no-change");
    let step = (curve.len() / 24).max(1);
    for (k, (i, a, b)) in curve.iter().enumerate() {
        if k % step == 0 {
            println!("{:>10} {:>12.4} {:>12.4}", i, a, b);
        }
    }
    let nbw: Vec<f64> = curve.iter().map(|c| c.1).collect();
    if nbw.len() > 2 {
        let mean = nbw.iter().sum::<f64>() / nbw.len() as f64;
        let sd = (nbw.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nbw.len() as f64).sqrt();
        println!(
            "\n  windowed naive-Bayes accuracy: mean {:.4}, sd {:.4}, min {:.4}, max {:.4}",
            mean,
            sd,
            nbw.iter().cloned().fold(f64::MAX, f64::min),
            nbw.iter().cloned().fold(f64::MIN, f64::max)
        );
        println!("  a large sd is the drift; a flat curve means the stream is stationary.");
    }

    // ---- how much the label sequence alone says --------------------------
    // If the previous label predicts the next one, an architecture that reads
    // the input stream is solving an easier problem than it looks.
    let mut joint = vec![0u64; s.classes * s.classes];
    for i in 1..s.rows {
        joint[s.labels[i - 1] as usize * s.classes + s.labels[i] as usize] += 1;
    }
    let mut m1 = vec![0u64; s.classes];
    let mut m2 = vec![0u64; s.classes];
    for a in 0..s.classes {
        for b in 0..s.classes {
            m1[a] += joint[a * s.classes + b];
            m2[b] += joint[a * s.classes + b];
        }
    }
    let mi = entropy_mm(&m1) + entropy_mm(&m2) - entropy_mm(&joint);
    println!(
        "\n  I(label_t ; label_t-1) = {:.4} bits of H(label) = {:.4}   ({:.1}%)",
        mi,
        entropy_mm(&m2),
        100.0 * mi / entropy_mm(&m2).max(1e-9)
    );
    println!();
}
