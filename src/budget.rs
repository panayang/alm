//! Same metadata budget, two ways of spending it.
//!
//! Address correlation is not our idea. Temporal prefetchers -- STMS, Domino,
//! ISB, MISB, Triage, Triangel -- have exploited exactly this structure for
//! twenty years, and they are online, one-shot and untrained just as we are.
//! What has kept them out of hardware is metadata: STMS runs a 342% traffic
//! overhead, ISB 411%, and MISB is a paper about getting that to 70%. Triage's
//! title is literally *Temporal Prefetching Without the Off-Chip Metadata*. So
//! the only question worth asking in this domain is coverage and accuracy **at
//! a fixed metadata budget in bytes**, and the only structural difference we
//! bring is how the storage fails when the budget is too small:
//!
//!   * an exact table must **evict**, losing whole keys, all or nothing;
//!   * a superposition goes **blurry**, losing every key together, by an
//!     amount its own capacity law predicts in advance.
//!
//! Which is better under pressure is an empirical question and nobody has
//! asked it, so this asks it.
//!
//! # The prediction, before the run
//!
//! Our own law says what we need. With `k = T/banks` triples per bank and
//! `d > 2k ln V` to keep them separable, the smallest superposition that can
//! hold T triples at all is
//!
//! ```text
//!   banks * d * 4 bytes  >=  banks * 2(T/banks) ln V * 4  =  8 T ln V
//! ```
//!
//! while an exact table needs one tag and one successor per triple, about
//! `8 T`. The superposition is therefore **ln V times larger for the same
//! content** -- a factor of ten on omnetpp, sixteen on mcf, and it does not
//! improve with scale. So the law predicts we lose this comparison at every
//! budget, and the run is worth making anyway for two reasons: below `8T` the
//! exact table is evicting too, and nobody has measured which failure mode
//! costs less; and a law that correctly predicts its own defeat is worth more
//! than one that has only ever been checked where it wins.
//!
//! # Protocol
//!
//! Online, in stream order, the way a prefetcher actually sees the world. At
//! each access the structure is asked for this PC's next line, the answer is
//! checked when that PC fires again, and only then is the structure told what
//! happened. Coverage is hits over sampled accesses, accuracy is hits over
//! predictions issued -- the field's own two numbers.
//!
//! # What this deliberately does not charge us for
//!
//! Cleanup compares the retrieval against every codebook entry, which is O(V)
//! work per prediction and no hardware would do it. The codebook itself is
//! free here, on the grounds that a counter-based RNG can regenerate any entry
//! from the line address. Both of those favour us, and neither is fixed by
//! this experiment.

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

/// The corrected threshold. Swept per cell at matched precision on this same
/// trace, the best multiplier runs 1.00 deep inside the operating region to
/// 1.17 at its edge; 1.6, what this project shipped, costs up to forty-seven
/// points of recall and buys four tenths of a point of precision.
const FLOOR_MULT: f32 = 1.1;

/// Bytes an exact table spends per entry: a truncated tag and a successor.
/// Generous to the baseline -- a real one also needs replacement state.
const EXACT_ENTRY_BYTES: usize = 8;
/// Ways per set. Set-associative rather than fully associative, because that
/// is what hardware builds.
const WAYS: usize = 16;

/// Set-associative LRU over (key -> successor). The baseline.
struct ExactLru {
    sets: usize,
    tag: Vec<u64>,
    val: Vec<u32>,
    age: Vec<u32>,
    clock: u32,
}

impl ExactLru {
    fn new(entries: usize) -> Self {
        let sets = (entries / WAYS).max(1);
        ExactLru {
            sets,
            tag: vec![u64::MAX; sets * WAYS],
            val: vec![0; sets * WAYS],
            age: vec![0; sets * WAYS],
            clock: 0,
        }
    }
    #[inline]
    fn base(&self, key: u64) -> usize {
        (key as usize % self.sets) * WAYS
    }
    fn get(&mut self, key: u64) -> Option<u32> {
        let b = self.base(key);
        self.clock += 1;
        for w in 0..WAYS {
            if self.tag[b + w] == key {
                self.age[b + w] = self.clock;
                return Some(self.val[b + w]);
            }
        }
        None
    }
    fn put(&mut self, key: u64, v: u32) {
        let b = self.base(key);
        self.clock += 1;
        let mut victim = 0usize;
        let mut oldest = u32::MAX;
        for w in 0..WAYS {
            if self.tag[b + w] == key {
                self.val[b + w] = v;
                self.age[b + w] = self.clock;
                return;
            }
            if self.tag[b + w] == u64::MAX {
                victim = w;
                oldest = 0;
                break;
            }
            if self.age[b + w] < oldest {
                oldest = self.age[b + w];
                victim = w;
            }
        }
        self.tag[b + victim] = key;
        self.val[b + victim] = v;
        self.age[b + victim] = self.clock;
    }
}

#[derive(Default, Clone, Copy)]
struct Tally {
    sampled: u64,
    issued: u64,
    hits: u64,
}

impl Tally {
    fn coverage(&self) -> f64 {
        self.hits as f64 / self.sampled.max(1) as f64
    }
    fn accuracy(&self) -> f64 {
        self.hits as f64 / self.issued.max(1) as f64
    }
}

fn human(b: usize) -> String {
    if b >= 1 << 20 {
        format!("{}M", b >> 20)
    } else {
        format!("{}K", b >> 10)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    path: &str,
    label: &str,
    gran: &str,
    budgets: &[usize],
    ds: &[usize],
    limit: usize,
    sample: usize,
    seed: u64,
) {
    let unit_shift: u32 = match gran {
        "line" => 6,
        "page" => 12,
        other => panic!("--gran wants line or page, got {}", other),
    };

    // ---- one pass, kept as slot ids -------------------------------------
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
        let ns = slot_of.len() as u32;
        let now = *slot_of.entry(r.line).or_insert(ns);
        let np = pcslot_of.len() as u32;
        let pcs = *pcslot_of.entry(r.pc).or_insert(np);
        if let Some(&prev) = pc_last.get(&r.pc) {
            events.push((pcs, prev, now));
        }
        pc_last.insert(r.pc, now);
    }

    let vocab = slot_of.len();
    let npcs = pcslot_of.len();
    let mut distinct: Map<u64, Map<u32, ()>> = Map::default();
    for &(pc, prev, now) in events.iter() {
        distinct.entry(mix2(pc as u64, prev as u64)).or_default().insert(now, ());
    }
    let triples: u64 = distinct.values().map(|m| m.len() as u64).sum();
    let lnv = (vocab.max(2) as f64).ln();

    println!("==================== {}   [gran = {}] ====================", label, gran);
    println!(
        "events {}   V = {}   PCs {}   triples T = {}   ln V = {:.2}",
        events.len(),
        vocab,
        npcs,
        triples,
        lnv
    );
    println!("querying every {}th access\n", sample);
    println!("what each structure needs to hold all of T, before the run:");
    println!("  exact table      8*T          = {:.2} MB", 8.0 * triples as f64 / 1048576.0);
    println!(
        "  superposition    8*T*lnV      = {:.2} MB   ({:.1}x more)",
        8.0 * triples as f64 * lnv / 1048576.0,
        lnv
    );

    // ---- the baseline, once per budget ----------------------------------
    let mut exact_res: Vec<Tally> = Vec::new();
    for &b in budgets {
        exact_res.push(run_exact(&events, b / EXACT_ENTRY_BYTES, sample, npcs));
    }

    // ---- ours: every width, and for each budget the bank count it allows -
    // Indexed [d][budget].
    let mut ours: Vec<Vec<Tally>> = Vec::new();
    for &d in ds {
        let bankses: Vec<usize> = budgets.iter().map(|&b| (b / 4 / d).max(1)).collect();
        ours.push(run_super(&events, &bankses, d, vocab, npcs, sample, seed));
    }

    println!("\n-- coverage (correct prediction outstanding, over sampled accesses) --");
    print!("{:>26}", "budget");
    for &b in budgets {
        print!("{:>12}", human(b));
    }
    println!();
    print!("{:>26}", "exact table + LRU");
    for t in exact_res.iter() {
        print!("{:>12.4}", t.coverage());
    }
    println!();
    for (i, &d) in ds.iter().enumerate() {
        print!("{:>18} d={:<6}", "superposition", d);
        for (j, &b) in budgets.iter().enumerate() {
            let banks = (b / 4 / d).max(1);
            let k = triples as f64 / banks as f64;
            let ok = (d as f64) > 2.0 * k * lnv;
            print!("{:>11.4}{}", ours[i][j].coverage(), if ok { "*" } else { " " });
        }
        println!();
    }

    println!("\n-- accuracy (correct over predictions issued) --");
    print!("{:>26}", "budget");
    for &b in budgets {
        print!("{:>12}", human(b));
    }
    println!();
    print!("{:>26}", "exact table + LRU");
    for t in exact_res.iter() {
        print!("{:>12.4}", t.accuracy());
    }
    println!();
    for (i, &d) in ds.iter().enumerate() {
        print!("{:>18} d={:<6}", "superposition", d);
        for (j, _) in budgets.iter().enumerate() {
            print!("{:>12.4}", ours[i][j].accuracy());
        }
        println!();
    }

    println!("\n  * = the capacity law says this (budget, width) split can hold the chain.");
    println!("  the exact table gets a 16-way LRU and 8 bytes an entry; we get a free codebook");
    println!("  and an O(V) cleanup no hardware would run. Both concessions favour us.");
}

fn run_exact(events: &[(u32, u32, u32)], entries: usize, sample: usize, npcs: usize) -> Tally {
    let mut tbl = ExactLru::new(entries);
    let mut pending: Vec<Option<u32>> = vec![None; npcs];
    let mut t = Tally::default();
    for (i, &(pc, prev, now)) in events.iter().enumerate() {
        if let Some(p) = pending[pc as usize].take() {
            t.sampled += 1;
            t.issued += 1;
            if p == now {
                t.hits += 1;
            }
        }
        tbl.put(mix2(pc as u64, prev as u64), now);
        if i % sample == 0 {
            match tbl.get(mix2(pc as u64, now as u64)) {
                Some(v) => pending[pc as usize] = Some(v),
                // No entry: nothing issued, but the access is still counted
                // against coverage on the next event for this PC.
                None => pending[pc as usize] = None,
            }
        }
    }
    // Sampled accesses that produced no prediction still belong in coverage.
    let asked = events.len() / sample.max(1);
    t.sampled = asked as u64;
    t
}

#[allow(clippy::too_many_arguments)]
fn run_super(
    events: &[(u32, u32, u32)],
    bankses: &[usize],
    d: usize,
    vocab: usize,
    npcs: usize,
    sample: usize,
    seed: u64,
) -> Vec<Tally> {
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

    let mut mems: Vec<Vec<Vec<f32>>> = bankses.iter().map(|&b| vec![vec![0.0f32; d]; b]).collect();
    let mut pending: Vec<Vec<Option<u32>>> = bankses.iter().map(|_| vec![None; npcs]).collect();
    let mut tallies = vec![Tally::default(); bankses.len()];

    let floor = (2.0 * (vocab.max(2) as f32).ln() / d as f32).sqrt();
    let thresh = FLOOR_MULT * floor;

    let mut t1 = vec![0.0f32; d];
    let mut t2 = vec![0.0f32; d];
    let mut q = vec![0.0f32; d];
    let mut rawv = vec![0.0f32; d];

    for (i, &(pc, prev, now)) in events.iter().enumerate() {
        for (mi, _) in bankses.iter().enumerate() {
            if let Some(p) = pending[mi][pc as usize].take() {
                tallies[mi].issued += 1;
                if p == now {
                    tallies[mi].hits += 1;
                }
            }
        }

        // Learn: one binding, shared by every bank count.
        circconv(&pcv[pc as usize], &code[prev as usize], &mut t1);
        circconv(&t1, &code[now as usize], &mut t2);
        normalize(&mut t2);
        let wkey = mix2(pc as u64, prev as u64) as usize;
        for (mi, &b) in bankses.iter().enumerate() {
            axpy(1.0, &t2, &mut mems[mi][wkey % b]);
        }

        if i % sample != 0 {
            continue;
        }
        // Predict this PC's next line from where it stands now.
        circconv(&pcv[pc as usize], &code[now as usize], &mut q);
        normalize(&mut q);
        let qkey = mix2(pc as u64, now as u64) as usize;
        for (mi, &b) in bankses.iter().enumerate() {
            unbind(&mems[mi][qkey % b], &q, &mut rawv);
            normalize(&mut rawv);
            let mut c1 = f32::NEG_INFINITY;
            let mut best = 0usize;
            for (s, c) in code.iter().enumerate() {
                let v = dot(&rawv, c);
                if v > c1 {
                    c1 = v;
                    best = s;
                }
            }
            pending[mi][pc as usize] = if c1 > thresh { Some(best as u32) } else { None };
        }
    }

    let asked = (events.len() / sample.max(1)) as u64;
    for t in tallies.iter_mut() {
        t.sampled = asked;
    }
    tallies
}
