//! The corpus scanner.
//!
//! Before writing an ingest for a real stream, decide whether the stream is
//! ours at all. Three questions, in the order that can kill the idea fastest:
//!
//!   1. Is the structure here *local* or *referential*? A stream whose next
//!      address is a short function of the last few deltas belongs to a stride
//!      prefetcher and to an autoregressive model; we would be walking onto
//!      somebody else's ground with a worse instrument. A stream whose next
//!      address is determined by what followed this address *the last time it
//!      was touched* is the other kind. Both hypotheses already have names in
//!      this field -- delta correlation and address correlation -- so the
//!      scanner codes the stream under each and reports the gap.
//!
//!   2. Does the silence carry information? Our depth is decided by how long
//!      the world stays quiet. If the inter-access gap is independent of what
//!      arrives next, that whole commitment buys nothing here, and we should
//!      know it before writing any code.
//!
//!   3. What does the addressing cost? Address discrimination decaying as
//!      addresses crowd is this architecture's known weakness, and a real
//!      stream is where it gets to bite. Heaps' curve gives V, the successor
//!      fan-out gives k, and `d > 2k ln V` turns the pair into a width.
//!
//! Nothing here runs the model. These are properties of the data.
//!
//! Input is the ML-DPC load-trace format, one access per line:
//!
//!     instr_id, cycle, load_address(hex), pc(hex), llc_hit
//!
//! Read from a path or from stdin, so a trace can stay compressed on disk:
//!
//!     xz -dc gap/bfs-3.txt.xz | alm scan --trace - --label bfs-3

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufRead, BufReader, Read};

/// Addresses are 64-byte aligned, so six bits of line offset. Twelve gives the
/// 4K page. Which one the referential hypothesis is asked about matters a great
/// deal: this is an LLC trace, filtered by L2, so the lines that reach it are
/// overwhelmingly first touches and line-level recurrence is nearly absent by
/// construction. Voyager decouples page from offset for exactly this reason.
/// The scanner therefore takes the granularity as an argument and is run at
/// both, rather than quietly picking one and reporting the answer it implies.
const LINE_SHIFT: u32 = 6;
const PAGE_SHIFT: u32 = 12;

/// Every model escapes, in the end, to a uniform over the line-address space.
/// The traces carry 48-bit physical addresses, so 42 bits of line. The
/// constant is arbitrary in exactly the same way for every model, which is all
/// the comparison needs; it is also reported separately from the like-for-like
/// number so it cannot quietly drive a conclusion.
const NOVEL_BITS: f64 = 42.0;

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/// The keys here are already well-mixed integers -- addresses, program
/// counters, packed pairs. SipHash's guarantees cost a few hundred
/// milliseconds per million records and buy nothing against them.
#[derive(Default)]
pub struct IntHasher(u64);

impl Hasher for IntHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn write_u64(&mut self, n: u64) {
        let mut x = n.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        self.0 = (self.0 ^ x).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_u64(n as u64)
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64)
    }
    fn write_i64(&mut self, n: i64) {
        self.write_u64(n as u64)
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type Map<K, V> = HashMap<K, V, BuildHasherDefault<IntHasher>>;

#[inline]
fn mix2(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    x.wrapping_mul(0x94d0_49bb_1331_11eb)
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Record {
    pub cycle: u64,
    /// The addressed unit at the granularity under test.
    pub line: u64,
    /// The 4K page, always, so the profile does not change meaning with it.
    pub page: u64,
    pub pc: u64,
    pub llc_hit: bool,
}

#[inline]
fn dec(s: &[u8], i: &mut usize) -> u64 {
    let mut v = 0u64;
    while *i < s.len() && s[*i].is_ascii_digit() {
        v = v * 10 + (s[*i] - b'0') as u64;
        *i += 1;
    }
    v
}

#[inline]
fn hex(s: &[u8], i: &mut usize) -> u64 {
    let mut v = 0u64;
    while *i < s.len() {
        let c = s[*i];
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => break,
        };
        v = (v << 4) | d as u64;
        *i += 1;
    }
    v
}

#[inline]
fn sep(s: &[u8], i: &mut usize) -> bool {
    while *i < s.len() && (s[*i] == b' ' || s[*i] == b',' || s[*i] == b'\r') {
        *i += 1;
    }
    *i < s.len()
}

/// Hand-rolled field parser. `split(',').parse()` costs more than the rest of
/// the scan put together at six million lines.
fn parse_line(s: &[u8], unit_shift: u32) -> Option<Record> {
    let mut i = 0usize;
    let _id = dec(s, &mut i);
    if !sep(s, &mut i) {
        return None;
    }
    let cycle = dec(s, &mut i);
    if !sep(s, &mut i) {
        return None;
    }
    let addr = hex(s, &mut i);
    if !sep(s, &mut i) {
        return None;
    }
    let pc = hex(s, &mut i);
    if !sep(s, &mut i) {
        return None;
    }
    let hit = dec(s, &mut i);
    Some(Record {
        cycle,
        line: addr >> unit_shift,
        page: addr >> PAGE_SHIFT,
        pc,
        llc_hit: hit != 0,
    })
}

// ---------------------------------------------------------------------------
// Prequential predictors
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Counts {
    m: Map<u64, u32>,
    total: u32,
    /// The four best symbols, carried forward. Counts only ever rise, and a
    /// symbol outside the list is checked against the running fourth place on
    /// every one of its own increments, so the incremental rule is exact --
    /// and it keeps ranking off the critical path of a map whose fan-out runs
    /// to millions of symbols.
    top: [(u64, u32); TOPK],
}

const TOPK: usize = 4;

impl Counts {
    /// Place `sym`, whose count has just become `c`, into the running top-k.
    /// Ordered by count, ties to the smaller symbol, so a scan is reproducible.
    #[inline]
    fn promote(&mut self, sym: u64, c: u32) {
        let mut at = TOPK;
        for i in 0..TOPK {
            if self.top[i].0 == sym && self.top[i].1 != 0 {
                at = i;
                break;
            }
        }
        if at == TOPK {
            let last = self.top[TOPK - 1];
            if last.1 == 0 || c > last.1 || (c == last.1 && sym < last.0) {
                self.top[TOPK - 1] = (sym, c);
                at = TOPK - 1;
            } else {
                return;
            }
        } else {
            self.top[at].1 = c;
        }
        while at > 0 {
            let (a, b) = (self.top[at - 1], self.top[at]);
            if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
                self.top.swap(at - 1, at);
                at -= 1;
            } else {
                break;
            }
        }
    }

    #[inline]
    fn rank_of(&self, sym: u64) -> Option<usize> {
        (0..TOPK).find(|&i| self.top[i].1 != 0 && self.top[i].0 == sym)
    }
}

/// Interpolated PPM-C over an arbitrary ladder of contexts. The caller decides
/// what the context keys mean; that is the *only* difference between the
/// models compared here, which is the point of the comparison.
struct Ppm {
    name: &'static str,
    levels: Vec<Map<u64, Counts>>,
    /// Bits charged on every access.
    bits: f64,
    /// Bits charged only on accesses whose target had been seen before, so the
    /// novelty floor cannot carry the comparison on its own.
    bits_seen: f64,
    seen_events: u64,
    events: u64,
    top1: u64,
    /// Top-1 hits and trials, split by how often the target had been seen.
    by_rec_hit: [u64; 5],
    by_rec_n: [u64; 5],
}

impl Ppm {
    fn new(name: &'static str, levels: usize) -> Self {
        Ppm {
            name,
            levels: (0..levels).map(|_| Map::default()).collect(),
            bits: 0.0,
            bits_seen: 0.0,
            seen_events: 0,
            events: 0,
            top1: 0,
            by_rec_hit: [0; 5],
            by_rec_n: [0; 5],
        }
    }

    /// Probability of `sym`, and the symbol the model would have named.
    /// Contexts are given longest first; each contributes its remaining escape
    /// mass, and whatever is left lands on the uniform.
    fn predict(&self, keys: &[u64], sym: u64) -> (f64, Option<u64>) {
        let mut p = 0.0f64;
        let mut esc = 1.0f64;
        let mut best: Option<u64> = None;
        for (lvl, &k) in keys.iter().enumerate() {
            let cs = match self.levels[lvl].get(&k) {
                Some(c) if c.total > 0 => c,
                _ => continue,
            };
            let distinct = cs.m.len() as f64;
            let total = cs.total as f64;
            let e = distinct / (distinct + total);
            let c = *cs.m.get(&sym).unwrap_or(&0) as f64;
            p += esc * (1.0 - e) * (c / total);
            if best.is_none() {
                best = Some(cs.top[0].0);
            }
            esc *= e;
        }
        p += esc * (-NOVEL_BITS).exp2();
        (p.max(f64::MIN_POSITIVE), best)
    }

    /// Rank of `sym` among the successors of the first non-empty context, or
    /// None if it is outside the top four. A prefetcher is allowed two guesses
    /// per access, so top-1 understates every hypothesis here -- and it has to
    /// understate them all equally or the comparison is rigged.
    fn rank(&self, keys: &[u64]) -> impl Fn(u64) -> Option<usize> + '_ {
        let mut found: Option<&Counts> = None;
        for (lvl, &key) in keys.iter().enumerate() {
            if let Some(c) = self.levels[lvl].get(&key) {
                if c.total > 0 {
                    found = Some(c);
                    break;
                }
            }
        }
        move |sym| found.and_then(|c| c.rank_of(sym))
    }

    fn update(&mut self, keys: &[u64], sym: u64) {
        for (lvl, &k) in keys.iter().enumerate() {
            let cs = self.levels[lvl].entry(k).or_default();
            let c = cs.m.entry(sym).or_insert(0);
            *c += 1;
            let c = *c;
            cs.total += 1;
            cs.promote(sym, c);
        }
    }

    /// One charged access. `rb` is how often the target line had already been
    /// seen, bucketed 0 / 1 / 2-3 / 4-8 / 9+.
    fn charge(&mut self, p: f64, predicted_hit: bool, rb: usize) {
        let b = -p.log2();
        self.bits += b;
        self.events += 1;
        if rb > 0 {
            self.bits_seen += b;
            self.seen_events += 1;
        }
        if predicted_hit {
            self.top1 += 1;
            self.by_rec_hit[rb] += 1;
        }
        self.by_rec_n[rb] += 1;
    }

    fn bits_per_access(&self) -> f64 {
        if self.events == 0 { 0.0 } else { self.bits / self.events as f64 }
    }
    fn bits_per_seen(&self) -> f64 {
        if self.seen_events == 0 { 0.0 } else { self.bits_seen / self.seen_events as f64 }
    }
    fn accuracy(&self) -> f64 {
        if self.events == 0 { 0.0 } else { self.top1 as f64 / self.events as f64 }
    }
}

// ---------------------------------------------------------------------------
// Entropy, Miller-Madow corrected
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// The scan
// ---------------------------------------------------------------------------

const REC_LABEL: [&str; 5] = ["novel", "1", "2-3", "4-8", "9+"];
const CLASS_LABEL: [&str; 4] = ["local-hit", "known-succ", "novel", "other"];
const GAP_BUCKETS: usize = 18;

#[inline]
fn rec_bucket(n: u32) -> usize {
    match n {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=8 => 3,
        _ => 4,
    }
}

#[inline]
fn gap_bucket(g: u64) -> usize {
    ((64 - g.leading_zeros()) as usize).min(GAP_BUCKETS - 1)
}

pub fn run(path: &str, label: &str, limit: usize, gran: &str) {
    let unit_shift = match gran {
        "line" => LINE_SHIFT,
        "page" => PAGE_SHIFT,
        other => panic!("--gran wants line or page, got {}", other),
    };
    // Streamed, not slurped: the larger GAP traces run to a gigabyte of text
    // and there is no reason to hold any of it.
    let src: Box<dyn Read> = if path == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(
            std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e)),
        )
    };
    let mut reader = BufReader::with_capacity(1 << 22, src);
    let mut raw: Vec<u8> = Vec::with_capacity(128);

    // --- accumulators -----------------------------------------------------
    let mut seen: Map<u64, u32> = Map::default();
    let mut pcs: Map<u64, u32> = Map::default();
    let mut pages: Map<u64, u8> = Map::default();
    let mut succ: Map<u64, Map<u64, u32>> = Map::default(); // (pc,line) -> successors

    let mut addr = Ppm::new("addr-corr  (pc,line)->line", 2);
    let mut delta = Ppm::new("delta-ppm  order 3", 4);
    let mut pcd = Ppm::new("pc-delta   (pc,dlast)->d", 2);

    let mut n = 0u64;
    let mut llc_hits = 0u64;
    let mut first_cycle = 0u64;
    let mut last_cycle = 0u64;
    let mut gaps: Vec<u32> = Vec::new();
    let mut heaps: Vec<(u64, u64)> = Vec::new();
    let mut next_mark = 100_000u64;

    // local-hit x referential-hit, in that bit order: neither / ref / local / both.
    let mut comp = [0u64; 4];
    // Coverage at prefetch degree 1..4 for the best local model and for the
    // referential one, plus the split hybrid: one guess spent on each.
    let mut loc_k = [0u64; TOPK];
    let mut ref_k = [0u64; TOPK];
    let mut hybrid = 0u64;
    let mut joint = vec![0u64; 4 * GAP_BUCKETS];
    let mut cls_marg = [0u64; 4];
    let mut gap_marg = [0u64; GAP_BUCKETS];

    let mut prev: Option<Record> = None;
    let mut dhist: [i64; 3] = [0; 3];

    loop {
        raw.clear();
        if reader.read_until(b'\n', &mut raw).expect("read trace") == 0 {
            break;
        }
        if limit > 0 && n as usize >= limit {
            break;
        }
        let r = match parse_line(&raw, unit_shift) {
            Some(r) => r,
            None => continue,
        };
        n += 1;
        if r.llc_hit {
            llc_hits += 1;
        }
        if n == 1 {
            first_cycle = r.cycle;
        }
        last_cycle = r.cycle;
        *pcs.entry(r.pc).or_insert(0) += 1;
        pages.insert(r.page, 0);

        if let Some(p) = prev {
            let gap = r.cycle.saturating_sub(p.cycle);
            gaps.push(gap.min(u32::MAX as u64) as u32);

            let prior = *seen.get(&r.line).unwrap_or(&0);
            let rb = rec_bucket(prior);
            let d = (r.line as i64).wrapping_sub(p.line as i64);
            let dsym = d as u64;

            // --- address correlation: what followed this line last time ----
            let k_pc_line = mix2(p.pc, p.line);
            // --- delta correlation: a short function of the recent deltas --
            let dkeys = [
                mix2(mix2(dhist[0] as u64, dhist[1] as u64), dhist[2] as u64),
                mix2(dhist[0] as u64, dhist[1] as u64),
                dhist[0] as u64,
                0,
            ];
            // --- the same, conditioned on the program counter --------------
            let skeys = [mix2(p.pc, dhist[0] as u64), p.pc];
            let akeys = [k_pc_line, p.line];

            // Everything is read off the models *before* any of them is told
            // the answer. Ranking after the update would hand each model the
            // very symbol it is being asked to rank.
            let (p_addr, abest) = addr.predict(&akeys, r.line);
            let (p_delta, dbest) = delta.predict(&dkeys, dsym);
            let (p_pcd, sbest) = pcd.predict(&skeys, dsym);

            // Same ranking, same depth, for every hypothesis. The local models
            // rank deltas, so their rank-r symbol becomes a line by adding it
            // to the current one; the referential model ranks lines directly.
            let ar = addr.rank(&akeys)(r.line);
            let sr = match (pcd.rank(&skeys)(dsym), delta.rank(&dkeys)(dsym)) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (x, y) => x.or(y),
            };

            let ahit = abest == Some(r.line);
            let dhit = dbest.map(|b| p.line.wrapping_add(b)) == Some(r.line);
            let shit = sbest.map(|b| p.line.wrapping_add(b)) == Some(r.line);

            addr.charge(p_addr, ahit, rb);
            delta.charge(p_delta, dhit, rb);
            pcd.charge(p_pcd, shit, rb);
            addr.update(&akeys, r.line);
            delta.update(&dkeys, dsym);
            pcd.update(&skeys, dsym);

            let local_hit = shit || dhit;
            comp[(local_hit as usize) << 1 | ahit as usize] += 1;
            if let Some(r0) = ar {
                for slot in ref_k.iter_mut().skip(r0) {
                    *slot += 1;
                }
            }
            if let Some(r0) = sr {
                for slot in loc_k.iter_mut().skip(r0) {
                    *slot += 1;
                }
            }
            if ar == Some(0) || sr == Some(0) {
                hybrid += 1;
            }

            // --- does the silence say anything about what arrives? ---------
            let known = succ.get(&k_pc_line).map_or(false, |m| m.contains_key(&r.line));
            let class = if shit || dhit {
                0
            } else if known {
                1
            } else if prior == 0 {
                2
            } else {
                3
            };
            let gb = gap_bucket(gap);
            joint[class * GAP_BUCKETS + gb] += 1;
            cls_marg[class] += 1;
            gap_marg[gb] += 1;

            *succ.entry(k_pc_line).or_default().entry(r.line).or_insert(0) += 1;

            dhist[2] = dhist[1];
            dhist[1] = dhist[0];
            dhist[0] = d;
        }

        *seen.entry(r.line).or_insert(0) += 1;
        if n >= next_mark {
            heaps.push((n, seen.len() as u64));
            next_mark *= 2;
        }
        prev = Some(r);
    }
    heaps.push((n, seen.len() as u64));

    let ctx = Report {
        label,
        gran,
        n,
        llc_hits,
        first_cycle,
        last_cycle,
        gaps: &mut gaps,
        seen: &seen,
        pcs: &pcs,
        pages: &pages,
        succ: &succ,
        heaps: &heaps,
        models: [&pcd, &delta, &addr],
        joint: &joint,
        cls_marg: &cls_marg,
        gap_marg: &gap_marg,
        comp: &comp,
        loc_k: &loc_k,
        ref_k: &ref_k,
        hybrid,
    };
    ctx.print();
}

struct Report<'a> {
    label: &'a str,
    gran: &'a str,
    n: u64,
    llc_hits: u64,
    first_cycle: u64,
    last_cycle: u64,
    gaps: &'a mut Vec<u32>,
    seen: &'a Map<u64, u32>,
    pcs: &'a Map<u64, u32>,
    pages: &'a Map<u64, u8>,
    succ: &'a Map<u64, Map<u64, u32>>,
    heaps: &'a [(u64, u64)],
    /// The two local models first, the referential one last.
    models: [&'a Ppm; 3],
    joint: &'a [u64],
    cls_marg: &'a [u64; 4],
    gap_marg: &'a [u64; GAP_BUCKETS],
    /// local-hit x referential-hit: neither / ref-only / local-only / both.
    comp: &'a [u64; 4],
    /// Coverage at prefetch degree 1..=4, best local model and referential.
    loc_k: &'a [u64; TOPK],
    ref_k: &'a [u64; TOPK],
    /// Degree 2, one guess spent on each hypothesis.
    hybrid: u64,
}

impl Report<'_> {
    fn print(self) {
        let n = self.n;
        let nf = n as f64;
        let seen = self.seen;
        let addr = self.models[2];
        println!("==================== {}   [gran = {}] ====================", self.label, self.gran);

        // --- profile ------------------------------------------------------
        println!("\n-- profile --");
        println!("accesses                  {}", n);
        println!("distinct {:<12} V = {}", self.gran.to_string() + "s", seen.len());
        println!("distinct 4K pages         {}", self.pages.len());
        println!("distinct PCs              {}", self.pcs.len());
        println!(
            "cycle span                {}  ({:.1} cycles/access)",
            self.last_cycle - self.first_cycle,
            (self.last_cycle - self.first_cycle) as f64 / nf
        );
        println!("LLC hit rate              {:.4}", self.llc_hits as f64 / nf);

        self.gaps.sort_unstable();
        let g = &self.gaps;
        let q = |f: f64| g[((g.len() - 1) as f64 * f) as usize];
        println!(
            "inter-access gap          p50 {}  p90 {}  p99 {}  max {}",
            q(0.5),
            q(0.9),
            q(0.99),
            g[g.len() - 1]
        );

        // How much traffic lands on lines the world has barely mentioned.
        // Voyager demotes anything seen fewer than twice to a delta; this is
        // the size of the hole that leaves.
        let once_only = seen.values().filter(|&&c| c == 1).count();
        let charged: u64 = addr.by_rec_n.iter().sum();
        println!(
            "\nlines seen exactly once   {} / {} distinct  ({:.3})",
            once_only,
            seen.len(),
            once_only as f64 / seen.len().max(1) as f64
        );
        print!("accesses by target recurrence   ");
        for i in 0..5 {
            print!("{} {:.3}   ", REC_LABEL[i], addr.by_rec_n[i] as f64 / charged.max(1) as f64);
        }
        println!();

        print!("Heaps V(n):  ");
        for &(m, v) in self.heaps.iter() {
            print!("{}->{}  ", m, v);
        }
        println!();
        if self.heaps.len() >= 3 {
            let (n0, v0) = self.heaps[self.heaps.len() / 2];
            let (n1, v1) = self.heaps[self.heaps.len() - 1];
            if n1 > n0 && v1 > v0 {
                let beta =
                    ((v1 as f64).ln() - (v0 as f64).ln()) / ((n1 as f64).ln() - (n0 as f64).ln());
                println!("Heaps beta (tail)         {:.3}   (1.0 = every access a new line)", beta);
            }
        }

        // --- local versus referential --------------------------------------
        println!("\n-- is the structure local or referential? --");
        println!("{:<28} {:>12} {:>14} {:>9}", "model", "bits/access", "bits/seen-tgt", "top-1");
        for m in self.models {
            println!(
                "{:<28} {:>12.3} {:>14.3} {:>9.4}",
                m.name,
                m.bits_per_access(),
                m.bits_per_seen(),
                m.accuracy()
            );
        }
        let local_best = self.models[0].bits_per_seen().min(self.models[1].bits_per_seen());
        let r = (local_best - addr.bits_per_seen()) / local_best;
        println!("\nmemory gain  R = (local - addr)/local = {:+.4}   [on seen targets]", r);
        println!("  R > 0: what followed this address last time beats any short function of recent deltas.");

        println!("\ntop-1 by how often the target had been seen before:");
        print!("{:<28}", "");
        for l in REC_LABEL.iter() {
            print!("{:>11}", l);
        }
        println!();
        for m in self.models {
            print!("{:<28}", m.name);
            for i in 0..5 {
                if m.by_rec_n[i] == 0 {
                    print!("{:>11}", "-");
                } else {
                    print!("{:>11.4}", m.by_rec_hit[i] as f64 / m.by_rec_n[i] as f64);
                }
            }
            println!();
        }

        // --- what does the referential model add? ---------------------------
        //
        // R compares the two hypotheses as rivals, but they are not rivals in
        // a real machine: a stride prefetcher is already in the hardware, and
        // anything we add is added on top of it. So the number that decides
        // this is not which hypothesis wins -- it is how much the referential
        // one gets that the local one missed. A prefetcher is also allowed two
        // guesses per access, which understates every hypothesis at top-1, so
        // the depth is swept and swept for all of them alike.
        println!("\n-- what would the referential model add on top of the local one? --");
        let ct: u64 = self.comp.iter().sum();
        let ctf = ct.max(1) as f64;
        println!(
            "at degree 1:  neither {:.4}   local only {:.4}   referential only {:.4}   both {:.4}",
            self.comp[0] as f64 / ctf,
            self.comp[2] as f64 / ctf,
            self.comp[1] as f64 / ctf,
            self.comp[3] as f64 / ctf
        );
        println!("\ncoverage of the next access, by prefetch degree:");
        println!("{:<24} {:>9} {:>9} {:>9} {:>9}", "", "deg 1", "deg 2", "deg 3", "deg 4");
        for (nm, v) in [("local (best of two)", self.loc_k), ("referential", self.ref_k)] {
            print!("{:<24}", nm);
            for j in 0..TOPK {
                print!("{:>9.4}", v[j] as f64 / ctf);
            }
            println!();
        }
        let hyb = self.hybrid as f64 / ctf;
        let loc2 = self.loc_k[1] as f64 / ctf;
        println!("\nat degree 2, both guesses local   {:.4}", loc2);
        println!(
            "at degree 2, one guess each       {:.4}   ({:+.4} against spending both locally)",
            hyb,
            hyb - loc2
        );

        println!("\n-- does the gap carry information? --");
        let hc = entropy_mm(self.cls_marg);
        let hg = entropy_mm(self.gap_marg);
        let hj = entropy_mm(self.joint);
        let mi = hc + hg - hj;
        println!(
            "H(class) {:.4}   H(log2 gap) {:.4}   I(class; gap) {:.4} bits  ({:.1}% of H(class))",
            hc,
            hg,
            mi,
            100.0 * mi / hc.max(1e-9)
        );
        let ctot: u64 = self.cls_marg.iter().sum();
        print!("class mix   ");
        for i in 0..4 {
            print!("{} {:.3}   ", CLASS_LABEL[i], self.cls_marg[i] as f64 / ctot.max(1) as f64);
        }
        println!();

        // --- what does the addressing cost? --------------------------------
        println!("\n-- what does the addressing cost? --");
        let keys = self.succ.len();
        let triples: u64 = self.succ.values().map(|m| m.len() as u64).sum();
        let mut fan: Vec<usize> = self.succ.values().map(|m| m.len()).collect();
        fan.sort_unstable();
        let fq = |f: f64| fan[((fan.len() - 1) as f64 * f) as usize];
        println!("distinct (pc,unit) keys   {}", keys);
        println!("distinct triples      T = {}", triples);
        println!(
            "successors per key        mean {:.2}  p50 {}  p90 {}  p99 {}  max {}",
            triples as f64 / keys.max(1) as f64,
            fq(0.5),
            fq(0.9),
            fq(0.99),
            fan[fan.len() - 1]
        );
        let lnv = (seen.len() as f64).ln();
        println!("\nrequired width from  d > 2k ln V   (V = {}, ln V = {:.2}):", seen.len(), lnv);
        println!("{:>14} {:>14} {:>12}", "banks", "k = T/banks", "d");
        for b in [1u64 << 14, 1 << 16, 1 << 18, 1 << 20, keys as u64] {
            if b == 0 {
                continue;
            }
            let k = triples as f64 / b as f64;
            println!("{:>14} {:>14.2} {:>12.0}", b, k, 2.0 * k * lnv);
        }
        println!("  last row is one bank per key -- the floor, if addressing were perfect.");
        println!();
    }
}
