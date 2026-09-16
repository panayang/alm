//! What a record table costs to answer from every direction.
//!
//! The partial match problem: store T records of n fields; a query fixes the
//! values of some subset S of the fields and asks for another field j. There
//! are exponentially many such directions, and exact structures are provably
//! bad at covering them -- Patrascu's cell-probe bound says a structure
//! answering partial match in t probes needs space 2^Omega(n/t), so in practice
//! a database builds a handful of indexes and scans for everything else.
//!
//! The escape the lower bound leaves open is approximation, and that is the
//! point on the tradeoff a superposition occupies. This file measures whether
//! a given table is worth attacking that way, and it runs no model.
//!
//! # What the superposition would do, and what it costs
//!
//! Store each record as one bound product and add them all up:
//!
//! ```text
//!   M = sum_r  E_{x_1^r} (*) E_{x_2^r} (*) ... (*) E_{x_n^r}
//! ```
//!
//! A query fixing S is `q = (*)_{i in S} E_{q_i}`, and unbinding gives
//!
//! ```text
//!   M (/) q  =  sum_{r matching S} (*)_{i not in S} E_{x_i^r}  +  noise
//! ```
//!
//! because the S factors cancel exactly for a matching record and scramble for
//! every other one. With a single unknown field j this is a superposition of
//! the answers of all matching records. Records sharing an answer add
//! coherently, so a value v held by m_v of the matching records reaches
//! `cos = m_v / sqrt(T)` against a noise floor of `sqrt(2 ln V_j / d)`:
//!
//! ```text
//!   answerable  <=>  m_v  >  sqrt( 2 T ln V_j / d )
//! ```
//!
//! Three things follow that make this worth measuring rather than assuming.
//! The store is **direction-free** -- one M answers every S, which is the whole
//! point. It returns the **modal** answer, not a record, so it is natively an
//! aggregate structure. And it can only answer a cell that is **big enough**,
//! which is a property of the data, not of us: this file computes, per
//! direction and per width, what fraction of queries clear that bar.
//!
//! # The comparison
//!
//! An exact index for one direction maps the given values to the modal answer:
//! one entry per *cell*, not per record. So it costs `8 * cells(S)` bytes and
//! answers that direction only, while the superposition costs `4 d` bytes and
//! answers all of them. A direction is only worth indexing if it carries
//! information, so directions are weighted by `I(X_j ; X_S) / H(X_j)` rather
//! than counted.
//!
//! Note what the cost is *not*. In the prefetching setting each key needed its
//! own bank, which made our side `8 T ln V` and lost by a factor of ln V. Here
//! one store holds the whole table, because records are superposed rather than
//! stored -- and what is given up for that is every cell too small to rise
//! above the noise.
//!
//! That limit has a scaling property worth stating in advance. Writing a cell's
//! share of the table as `f = m_v / T`, the condition becomes
//!
//! ```text
//!   f  >  sqrt( 2 ln V_j / (d T) )
//! ```
//!
//! so the *frequency* a cell needs falls as the table grows: the coherent part
//! of the signal grows like T while the noise grows like sqrt(T). A bigger
//! table is easier for us at fixed width, and more expensive for an exact
//! index, which grows with the cell count. If that is right the gap widens with
//! scale, and it is cheap to check on a larger table.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use crate::scan::IntHasher;

type Map<K, V> = HashMap<K, V, BuildHasherDefault<IntHasher>>;

#[inline]
fn mixn(vals: &[u32]) -> u64 {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for &v in vals {
        x ^= v as u64;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 31;
    }
    x
}

fn entropy_mm(counts: &[u64]) -> f64 {
    let n: u64 = counts.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut h = 0.0;
    let mut occupied = 0usize;
    for &c in counts {
        if c > 0 {
            occupied += 1;
            let p = c as f64 / nf;
            h -= p * p.log2();
        }
    }
    h + (occupied as f64 - 1.0) / (2.0 * nf * std::f64::consts::LN_2)
}

/// A table with every column reduced to slot ids.
pub struct Table {
    pub names: Vec<String>,
    /// cols[j][r] is record r's slot in field j.
    pub cols: Vec<Vec<u32>>,
    pub vocab: Vec<usize>,
    pub rows: usize,
}

/// Read a headerless or headed comma-separated table. Columns that parse as
/// numbers and have more distinct values than `bins` are binned by quantile:
/// partial match is a categorical question, and leaving a continuous column
/// raw would give it a vocabulary the size of the table and no cell with more
/// than one record in it.
pub fn load(path: &str, bins: usize) -> Table {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let mut raw: Vec<Vec<String>> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        if f.iter().any(|s| s == "?") {
            continue; // Adult marks missing this way; a missing value is not a category.
        }
        if let Some(first) = raw.first() {
            if f.len() != first.len() {
                continue;
            }
        }
        raw.push(f);
    }
    assert!(!raw.is_empty(), "no usable rows in {}", path);

    // A first row of non-numeric strings where the column is otherwise numeric
    // is a header.
    let ncol = raw[0].len();
    let header = (0..ncol).any(|j| {
        raw[0][j].parse::<f64>().is_err() && raw.len() > 1 && raw[1][j].parse::<f64>().is_ok()
    });
    let names: Vec<String> = if header {
        raw[0].clone()
    } else {
        (0..ncol).map(|j| format!("c{}", j)).collect()
    };
    let body = if header { &raw[1..] } else { &raw[..] };
    let rows = body.len();

    let mut cols = Vec::with_capacity(ncol);
    let mut vocab = Vec::with_capacity(ncol);
    for j in 0..ncol {
        let vals: Vec<&str> = body.iter().map(|r| r[j].as_str()).collect();
        let nums: Vec<f64> = vals.iter().filter_map(|s| s.parse::<f64>().ok()).collect();
        let distinct: std::collections::HashSet<&str> = vals.iter().copied().collect();
        let numeric = nums.len() * 10 >= vals.len() * 9 && distinct.len() > bins * 2;

        let col: Vec<u32> = if numeric {
            let mut sorted = nums.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let cuts: Vec<f64> = (1..bins)
                .map(|b| sorted[(sorted.len() - 1) * b / bins])
                .collect();
            vals.iter()
                .map(|s| {
                    let v: f64 = s.parse().unwrap_or(f64::NAN);
                    cuts.iter().filter(|&&c| v > c).count() as u32
                })
                .collect()
        } else {
            let mut ids: HashMap<&str, u32> = HashMap::new();
            vals.iter()
                .map(|s| {
                    let n = ids.len() as u32;
                    *ids.entry(s).or_insert(n)
                })
                .collect()
        };
        let v = col.iter().copied().max().unwrap_or(0) as usize + 1;
        cols.push(col);
        vocab.push(v);
    }
    Table { names, cols, vocab, rows }
}

/// One (given-set, target) direction.
struct Dir {
    given: Vec<usize>,
    target: usize,
    /// I(X_target ; X_given) / H(X_target). Zero means no index would be built.
    info: f64,
    /// Always answering the modal value of the cell: the accuracy any
    /// single-answer structure is bounded by, exact ones included.
    ceiling: f64,
    /// Modal count of each cell, one entry per cell. The width sweep asks how
    /// much of the table sits in cells whose mode clears sqrt(2 T ln V / d).
    cell_modal: Vec<u32>,
}

fn measure(tab: &Table, given: &[usize], target: usize) -> Dir {
    let t = tab.rows;
    // cell -> counts of the target value
    let mut cells: Map<u64, Map<u32, u32>> = Map::default();
    let mut key = vec![0u32; given.len()];
    for r in 0..t {
        for (i, &g) in given.iter().enumerate() {
            key[i] = tab.cols[g][r];
        }
        *cells.entry(mixn(&key)).or_default().entry(tab.cols[target][r]).or_insert(0) += 1;
    }

    // H(target) and H(target | given), both Miller-Madow corrected.
    let mut marg: Vec<u64> = vec![0; tab.vocab[target]];
    for r in 0..t {
        marg[tab.cols[target][r] as usize] += 1;
    }
    let h_t = entropy_mm(&marg);
    let mut h_cond = 0.0;
    let mut cell_modal: Vec<u32> = Vec::with_capacity(cells.len());
    let mut ceiling_hits = 0u64;
    for m in cells.values() {
        let counts: Vec<u64> = m.values().map(|&x| x as u64).collect();
        let n: u64 = counts.iter().sum();
        h_cond += (n as f64 / t as f64) * entropy_mm(&counts);
        let top = m.values().copied().max().unwrap_or(0);
        cell_modal.push(top);
        ceiling_hits += top as u64;
    }
    let info = if h_t > 1e-9 { (h_t - h_cond) / h_t } else { 0.0 };

    Dir {
        given: given.to_vec(),
        target,
        info: info.max(0.0),
        ceiling: ceiling_hits as f64 / t as f64,
        cell_modal,
    }
}

fn subsets(n: usize, target: usize, max_given: usize) -> Vec<Vec<usize>> {
    let fields: Vec<usize> = (0..n).filter(|&f| f != target).collect();
    let mut out = Vec::new();
    for size in 1..=max_given.min(fields.len()) {
        let mut idx: Vec<usize> = (0..size).collect();
        loop {
            out.push(idx.iter().map(|&i| fields[i]).collect());
            // Advance to the next combination, or stop at the last one.
            let mut i = size;
            while i > 0 {
                i -= 1;
                if idx[i] != i + fields.len() - size {
                    idx[i] += 1;
                    for k in i + 1..size {
                        idx[k] = idx[k - 1] + 1;
                    }
                    break;
                }
                if i == 0 {
                    break;
                }
            }
            if idx[0] > fields.len() - size {
                break;
            }
            if idx.iter().enumerate().all(|(k, &v)| v == k + fields.len() - size) {
                out.push(idx.iter().map(|&i| fields[i]).collect());
                break;
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

pub fn run(path: &str, label: &str, bins: usize, max_given: usize, ds: &[usize], min_info: f64) {
    let tab = load(path, bins);
    let n = tab.cols.len();
    let t = tab.rows;

    println!("==================== {} ====================", label);
    println!("records T = {}   fields n = {}   binned at {} quantiles\n", t, n, bins);
    println!("{:>4} {:<20} {:>8} {:>9}", "col", "name", "V", "ln V");
    for j in 0..n {
        println!(
            "{:>4} {:<20} {:>8} {:>9.2}",
            j,
            tab.names[j],
            tab.vocab[j],
            (tab.vocab[j].max(2) as f64).ln()
        );
    }
    let ln_v_max = (0..n)
        .map(|j| (tab.vocab[j].max(2) as f64).ln())
        .fold(0.0f64, f64::max);

    // ---- every direction up to max_given ---------------------------------
    let mut dirs: Vec<Dir> = Vec::new();
    for target in 0..n {
        for s in subsets(n, target, max_given) {
            dirs.push(measure(&tab, &s, target));
        }
    }
    let informative: Vec<&Dir> = dirs.iter().filter(|d| d.info >= min_info).collect();

    println!(
        "\ndirections measured (|S| <= {}): {}   of which carry >= {:.0}% of H(target): {}",
        max_given,
        dirs.len(),
        min_info * 100.0,
        informative.len()
    );
    // An exact index for one direction is a map from the given values to the
    // modal answer: one entry per *cell*, not one per record. Charging it 8T
    // was wrong and inflated our side by more than an order of magnitude.
    const EXACT_CELL_BYTES: usize = 8;
    let exact_cells: usize = informative.iter().map(|d| d.cell_modal.len()).sum();
    let exact_bytes = exact_cells * EXACT_CELL_BYTES;

    // And ours is not 8*T*lnV either. That came from the prefetch setting,
    // where every key needed its own bank. Here one bank holds the whole
    // table -- records are superposed, not stored -- so the cost is d floats
    // and nothing else. What it gives up is every cell too small to rise above
    // the noise, which is exactly what the sweep below measures.
    println!("\n-- the crossover --");
    println!(
        "  exact: one index per informative direction    {:>11.3} MB   ({} directions, {} cells)",
        exact_bytes as f64 / 1048576.0,
        informative.len(),
        exact_cells
    );
    println!("  superposition: one store, every direction");
    for &d in ds {
        println!("      d = {:<8}                             {:>11.6} MB", d, (d * 4) as f64 / 1048576.0);
    }
    let _ = ln_v_max;

    // ---- but can it answer? ----------------------------------------------
    // A cell is retrievable when its modal count clears sqrt(2 T ln V_j / d).
    println!("\n-- what fraction of queries the superposition could actually answer --");
    println!("   (ceiling = always answer the cell's modal value; any single-answer structure is bounded by it)");
    print!("{:>8}", "|S|");
    for &d in ds {
        print!("{:>12}", format!("d={}", d));
    }
    println!("{:>12}{:>10}", "ceiling", "dirs");
    for size in 1..=max_given {
        let group: Vec<&&Dir> = informative.iter().filter(|d| d.given.len() == size).collect();
        if group.is_empty() {
            continue;
        }
        print!("{:>8}", size);
        for &d in ds {
            let mut num = 0.0;
            for dir in group.iter() {
                let thr = (2.0 * t as f64 * (tab.vocab[dir.target].max(2) as f64).ln() / d as f64)
                    .sqrt();
                // Records that would be answered correctly: those holding
                // their cell's modal value, in cells the width can resolve.
                let hits: u64 = dir
                    .cell_modal
                    .iter()
                    .filter(|&&m| (m as f64) > thr)
                    .map(|&m| m as u64)
                    .sum();
                num += hits as f64 / t as f64;
            }
            print!("{:>12.4}", num / group.len() as f64);
        }
        let ceil: f64 = group.iter().map(|d| d.ceiling).sum::<f64>() / group.len() as f64;
        println!("{:>12.4}{:>10}", ceil, group.len());
    }

    // ---- the directions worth having, so the table is not just a number --
    let mut top: Vec<&Dir> = informative.clone();
    top.sort_by(|a, b| b.info.partial_cmp(&a.info).unwrap());
    println!("\n-- the ten most informative directions --");
    for d in top.iter().take(10) {
        let g: Vec<&str> = d.given.iter().map(|&i| tab.names[i].as_str()).collect();
        println!(
            "  {:<44} -> {:<18} I/H = {:.3}   modal ceiling {:.3}",
            g.join(","),
            tab.names[d.target],
            d.info,
            d.ceiling
        );
    }
    println!();
}
