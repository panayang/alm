//! Does *our* memory hold the chain the scanner found?
//!
//! The scanner walked the per-PC chain with an exact hash table: every key
//! keeps its own bucket, nothing interferes with anything. That is not what
//! this architecture does. Ours is a banked superposition -- triples are bound
//! into a fixed number of fixed-width vectors and added on top of each other,
//! and retrieval has to pull one back out of the pile. So every number the
//! scanner reported is an *upper bound* for us, and the only question that
//! matters is how much survives.
//!
//! We already have a prediction, from our own capacity law:
//!
//! ```text
//!   signal  ~ 1/sqrt(k)          k = triples per bank        (interference)
//!   floor   ~ sqrt(2 ln V / d)                       (retrieval difficulty)
//!   usable  <=>  d > 2 k ln V
//! ```
//!
//! The write is the one from the tick loop, with the program counter in the
//! role the relation token plays on the synthetic source:
//!
//! ```text
//!   M[hash(pc, prev)] += nu( (E_pc (*) E_prev) (*) E_now )
//! ```
//!
//! and the read is its inverse, cleaned up against the codebook. Ground truth
//! is what the exact table would have said for the same key, so this measures
//! the superposition alone and nothing else.
//!
//! # Which acceptance rule
//!
//! The rule this project has been using is absolute: take the candidate if
//! `cos(best) > 1.6 * sqrt(2 ln V / d)`. That asks whether the retrieval beat
//! the noise, and it is the wrong question under load, because the two sides
//! depend on k differently. After normalisation the interference projects onto
//! any codebook entry as ~N(0, 1/d), so the floor does not move with k at all;
//! the signal falls as 1/sqrt(k). At V = 37227 and d = 512 the floor is 0.2027
//! and the threshold 0.3244, while a bank holding k = 12 triples returns its
//! answer at 1/sqrt(12) = 0.287 -- correct, the clear argmax, and refused. The
//! argmax is already right whenever the signal clears the *maximum of the
//! nulls*; the 1.6 demands a further sixty percent on top of that, and the
//! measured price is about thirty points of recall for a tenth of a point of
//! precision.
//!
//! So the alternative tested here compares the winner against the runner-up
//! instead of against a constant. Interference scales the whole retrieval, so a
//! gap between two of its own order statistics is scale-free; and it uses the
//! *realised* second place rather than the *expected* null maximum, which
//! absorbs a fluctuation that a mean prediction cannot. The null spacing of the
//! top two of V samples has a closed form:
//!
//! ```text
//!   null_gap ~ sigma / sqrt(2 ln V) = floor / (2 ln V)
//! ```
//!
//! One more thing follows from the margin rule that the absolute rule cannot
//! express. When a key genuinely has two successors, both cosines are signal
//! and the margin is small. That is not a retrieval failure to be thrown away
//! -- it is the point at which the chain branches, and this architecture
//! already carries parallel weighted cursors for exactly that. The two rules
//! answer two different questions: "is anything there" and "can it be told
//! apart". Folding them into one threshold is what discards the weak but
//! separable case along with the empty one.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::io::{BufRead, BufReader, Read};

use crate::num::{axpy, circconv, dot, normalize, unbind, unitary_vector};
use crate::scan::{parse_line, IntHasher};

type Map<K, V> = HashMap<K, V, BuildHasherDefault<IntHasher>>;

#[inline]
fn mix2(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    x.wrapping_mul(0x94d0_49bb_1331_11eb)
}

struct Cell {
    /// Fraction of queries whose argmax is what the exact table would say.
    /// Rule-free, so it is the ceiling every rule is measured against.
    agree: f64,
    /// Per query: winner cosine, runner-up cosine, whether the winner was
    /// right, and whether the key had more than one stored successor. Rules
    /// are a post-processing step over this, so any number of them can be
    /// compared on identical retrievals instead of on separate runs.
    obs: Vec<(f32, f32, bool, bool)>,
    floor: f32,
    null_gap: f32,
}

/// Best recall a family reaches without dropping below `min_prec`, and the
/// parameter that got there. Comparing rules at their own favourite operating
/// points is not a comparison; this fixes the precision and reads off recall.
fn best_at(obs: &[(f32, f32, bool, bool)], floor: f32, gap: f32, margin: bool, min_prec: f64)
    -> (f64, f64, f64, f64) {
    let n = obs.len().max(1) as f64;
    let mut best = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for step in 0..=160 {
        let p = if margin { 0.25 + step as f32 * 0.125 } else { 0.6 + step as f32 * 0.0125 };
        let (mut acc, mut ok, mut ok_unamb, mut n_unamb) = (0u64, 0u64, 0u64, 0u64);
        for &(c1, c2, hit, amb) in obs {
            if !amb {
                n_unamb += 1;
            }
            let take = if margin { c1 - c2 > p * gap } else { c1 > p * floor };
            if take {
                acc += 1;
                if hit {
                    ok += 1;
                    if !amb {
                        ok_unamb += 1;
                    }
                }
            }
        }
        if acc == 0 {
            continue;
        }
        let prec = ok as f64 / acc as f64;
        let rec = ok as f64 / n;
        if prec >= min_prec && rec > best.0 {
            best = (rec, prec, p as f64, ok_unamb as f64 / n_unamb.max(1) as f64);
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    path: &str,
    label: &str,
    gran: &str,
    ds: &[usize],
    bankses: &[usize],
    limit: usize,
    queries: usize,
    seed: u64,
) {
    let unit_shift: u32 = match gran {
        "line" => 6,
        "page" => 12,
        other => panic!("--gran wants line or page, got {}", other),
    };

    // ---- one pass over the trace, kept as slot ids ----------------------
    // The sweep runs the same stream at several widths; re-reading a gigabyte
    // of text per width would dominate everything.
    let src: Box<dyn Read> = if path == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e)))
    };
    let mut reader = BufReader::with_capacity(1 << 22, src);
    let mut raw: Vec<u8> = Vec::with_capacity(128);

    let mut slot_of: Map<u64, u32> = Map::default();
    let mut pcslot_of: Map<u64, u32> = Map::default();
    let mut events: Vec<(u32, u32, u32)> = Vec::new();
    let mut pc_last: Map<u64, u32> = Map::default();
    let mut n = 0usize;

    loop {
        raw.clear();
        if reader.read_until(b'\n', &mut raw).expect("read trace") == 0 {
            break;
        }
        if limit > 0 && n >= limit {
            break;
        }
        let r = match parse_line(&raw, unit_shift) {
            Some(r) => r,
            None => continue,
        };
        n += 1;
        let nslots = slot_of.len() as u32;
        let now = *slot_of.entry(r.line).or_insert(nslots);
        let npc = pcslot_of.len() as u32;
        let pcs = *pcslot_of.entry(r.pc).or_insert(npc);
        if let Some(&prev) = pc_last.get(&r.pc) {
            events.push((pcs, prev, now));
        }
        pc_last.insert(r.pc, now);
    }

    let vocab = slot_of.len();
    let npcs = pcslot_of.len();

    // ---- ground truth: what an exact table would answer -----------------
    let mut exact: Map<u64, Map<u32, u32>> = Map::default();
    for &(pc, prev, now) in events.iter() {
        *exact
            .entry(mix2(pc as u64, prev as u64))
            .or_default()
            .entry(now)
            .or_insert(0) += 1;
    }
    let triples: u64 = exact.values().map(|m| m.len() as u64).sum();

    // Carry each key's fan-out along: a key with two stored successors is a
    // different question from one with a single answer, and the margin rule
    // treats them differently on purpose.
    let mut keys: Vec<(u64, u32, usize)> = exact
        .iter()
        .filter_map(|(&k, m)| {
            let mut best: Option<(u32, u32)> = None;
            for (&s, &c) in m.iter() {
                if best.map_or(true, |(bs, bc)| c > bc || (c == bc && s < bs)) {
                    best = Some((s, c));
                }
            }
            best.map(|(s, _)| (k, s, m.len()))
        })
        .collect();
    keys.sort_unstable();
    let stride = (keys.len() / queries.max(1)).max(1);
    let asked: Vec<(u64, u32, usize)> =
        keys.iter().copied().step_by(stride).take(queries).collect();
    let ambiguous = asked.iter().filter(|a| a.2 > 1).count();

    let mut key_parts: Map<u64, (u32, u32)> = Map::default();
    for &(pc, prev, _) in events.iter() {
        key_parts.entry(mix2(pc as u64, prev as u64)).or_insert((pc, prev));
    }

    println!("==================== {}   [gran = {}] ====================", label, gran);
    println!(
        "events {}   vocab V = {}   PCs {}   distinct keys {}   triples T = {}",
        events.len(),
        vocab,
        npcs,
        exact.len(),
        triples
    );
    let lnv = (vocab.max(2) as f64).ln();
    println!(
        "asking {} keys, identical at every cell; {} of them have more than one stored successor",
        asked.len(),
        ambiguous
    );
    println!("\nthe capacity law's call, made before the run:");
    println!("{:>10} {:>9} {:>22}", "banks", "k = T/B", "usable above d =");
    for &b in bankses {
        let k = triples as f64 / b as f64;
        println!("{:>10} {:>9.2} {:>22.0}", b, k, 2.0 * k * lnv);
    }

    let mut cells: Vec<Vec<Cell>> = Vec::new();
    for &d in ds {
        cells.push(one_width(d, bankses, vocab, npcs, &events, &asked, &key_parts, seed));
    }

    let inside = |d: usize, b: usize| (d as f64) > 2.0 * (triples as f64 / b as f64) * lnv;

    println!("\n-- ceiling: argmax is what the exact table would say (no rule applied) --");
    print!("{:>7}", "d");
    for &b in bankses {
        print!("{:>12}", b);
    }
    println!("{:>10}{:>10}", "floor", "null gap");
    for (i, &d) in ds.iter().enumerate() {
        print!("{:>7}", d);
        for (j, &b) in bankses.iter().enumerate() {
            print!("{:>11.4}{}", cells[i][j].agree, if inside(d, b) { "*" } else { " " });
        }
        let floor = (2.0 * lnv / d as f64).sqrt();
        println!("{:>10.4}{:>10.4}", floor, floor / (2.0 * lnv));
    }

    // Every rule held to the same precision, per cell, with its own best
    // parameter found by sweep. This is the only comparison that means
    // anything: any rule can buy recall with precision.
    const MIN_PREC: f64 = 0.99;
    for margin in [false, true] {
        let fam = if margin { "margin, c1-c2 vs null gap" } else { "absolute, c1 vs floor" };
        println!("\n-- best recall at precision >= {:.2}: {} --", MIN_PREC, fam);
        print!("{:>7}", "d");
        for &b in bankses {
            print!("{:>14}", b);
        }
        println!();
        for (i, &d) in ds.iter().enumerate() {
            print!("{:>7}", d);
            for (j, &b) in bankses.iter().enumerate() {
                let c = &cells[i][j];
                let (rec, _, par, _) = best_at(&c.obs, c.floor, c.null_gap, margin, MIN_PREC);
                print!("{:>8.4}@{:<4.2}{}", rec, par, if inside(d, b) { "*" } else { " " });
            }
            println!();
        }
    }

    println!("\n-- the same, split by whether the key had one stored successor or several --");
    println!(
        "{:>6} {:>8} | {:>9} {:>11} | {:>9} {:>11}",
        "d", "banks", "abs all", "abs unamb", "margin all", "margin unamb"
    );
    for (i, &d) in ds.iter().enumerate() {
        for (j, &b) in bankses.iter().enumerate() {
            let c = &cells[i][j];
            let (ra, _, _, ua) = best_at(&c.obs, c.floor, c.null_gap, false, MIN_PREC);
            let (rm, _, _, um) = best_at(&c.obs, c.floor, c.null_gap, true, MIN_PREC);
            if ra < 0.05 && rm < 0.05 {
                continue;
            }
            println!(
                "{:>6} {:>8} | {:>9.4} {:>11.4} | {:>9.4} {:>11.4}",
                d, b, ra, ua, rm, um
            );
        }
    }

    println!("\n  * = the capacity law said this cell is inside the operating region.");
    println!("  the ceiling is rule-free; every rule's recall is bounded by it.");
    println!("  @x.xx is the parameter that reached that recall without breaking the precision floor.");
}

#[allow(clippy::too_many_arguments)]
fn one_width(
    d: usize,
    bankses: &[usize],
    vocab: usize,
    npcs: usize,
    events: &[(u32, u32, u32)],
    asked: &[(u64, u32, usize)],
    key_parts: &Map<u64, (u32, u32)>,
    seed: u64,
) -> Vec<Cell> {
    // Unitary, because binding only inverts exactly for unitary vectors --
    // established earlier and not revisited here.
    let code: Vec<Vec<f32>> = (0..vocab)
        .map(|s| {
            let mut v = unitary_vector(seed ^ 0xC0DE, s as u64, d);
            normalize(&mut v);
            v
        })
        .collect();
    let pcv: Vec<Vec<f32>> = (0..npcs)
        .map(|s| {
            let mut v = unitary_vector(seed ^ 0x9C01, s as u64, d);
            normalize(&mut v);
            v
        })
        .collect();

    // One memory per bank count. Binding is the expensive part and does not
    // depend on how many banks there are, so it is done once per width and the
    // result added into every memory.
    let mut mems: Vec<Vec<Vec<f32>>> = bankses.iter().map(|&b| vec![vec![0.0f32; d]; b]).collect();
    let mut t1 = vec![0.0f32; d];
    let mut t2 = vec![0.0f32; d];

    for &(pc, prev, now) in events.iter() {
        circconv(&pcv[pc as usize], &code[prev as usize], &mut t1);
        circconv(&t1, &code[now as usize], &mut t2);
        normalize(&mut t2);
        let key = mix2(pc as u64, prev as u64) as usize;
        for (mi, &b) in bankses.iter().enumerate() {
            axpy(1.0, &t2, &mut mems[mi][key % b]);
        }
    }

    let lnv = (vocab.max(2) as f32).ln();
    let floor = (2.0 * lnv / d as f32).sqrt();
    let null_gap = floor / (2.0 * lnv);

    let mut q = vec![0.0f32; d];
    let mut rawv = vec![0.0f32; d];
    let mut out = Vec::new();

    for (mi, &banks) in bankses.iter().enumerate() {
        let mut agree = 0u64;
        let mut obs: Vec<(f32, f32, bool, bool)> = Vec::with_capacity(asked.len());

        for &(key, truth, fanout) in asked.iter() {
            let (pc, prev) = key_parts[&key];
            circconv(&pcv[pc as usize], &code[prev as usize], &mut q);
            normalize(&mut q);
            unbind(&mems[mi][(key as usize) % banks], &q, &mut rawv);
            normalize(&mut rawv);

            // Winner and runner-up in one pass.
            let (mut c1, mut c2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            let mut best = 0usize;
            for (s, c) in code.iter().enumerate() {
                let v = dot(&rawv, c);
                if v > c1 {
                    c2 = c1;
                    c1 = v;
                    best = s;
                } else if v > c2 {
                    c2 = v;
                }
            }
            let hit = best as u32 == truth;
            if hit {
                agree += 1;
            }
            obs.push((c1, c2, hit, fanout > 1));
        }

        let nq = asked.len().max(1) as f64;
        out.push(Cell { agree: agree as f64 / nq, obs, floor, null_gap });
    }
    out
}
