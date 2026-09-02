//! Generator self-tests, run on the emitted stream before any experiment.
//!
//! The rule this file exists to enforce: suspect the data before the mechanism.
//! A flat curve on a source that never held the structure is a fact about the
//! source, and over-ablating in response to it kills mechanisms that were never
//! given anything to do. Every load-bearing property of the stream is therefore
//! measured here, from the ticks that the model will actually see, and the ones
//! an experiment depends on are asserted rather than printed.
//!
//! The conjunction check is the important one. If `conjunctive_gain` is not
//! close to log2(m), the second-order window measures nothing, and no reading of
//! that curve -- flat or otherwise -- is about the mechanism.

use std::collections::HashMap;

use crate::gen::{Kind, Stream};

pub struct GenReport {
    pub ticks: usize,
    pub baseline_fraction: f64,
    pub episodes: usize,
    pub second_order: usize,
    pub distinct_tokens: usize,

    /// Bits, unconditional. In mode A a cue also names its regime, so these are
    /// dominated by log2(domains) and say nothing about the conjunction. Kept
    /// because reading them as the conjunction test is the exact mistake this
    /// file exists to prevent.
    pub mi_target_cue_a: f64,
    pub mi_target_cue_b: f64,
    pub mi_target_pair: f64,

    /// Bits, conditioned on the regime. This is the quantity the second-order
    /// window depends on: *within* a regime, neither cue alone may identify the
    /// target, and the pair must.
    pub mi_within_cue_a: f64,
    pub mi_within_cue_b: f64,
    pub mi_within_pair: f64,
    /// I(T;A,B|D) - I(T;A|D) - I(T;B|D).
    pub conjunctive_gain: f64,

    /// Episodes per requested separation.
    pub sep_counts: Vec<(u32, usize)>,

    /// Fitted Zipf exponent of the target rank-frequency curve.
    pub zipf_exponent: f64,
    /// Fitted Heaps exponent. Reported, never asserted: a generator that fails
    /// to reproduce the sublinear regime is an unfavourable source for capacity
    /// questions, and that is worth knowing rather than hiding.
    pub heaps_exponent: f64,

    /// How often a baseline run of at least `answer_gap` is followed by a
    /// resolution rather than by another cue. Below one means segmentation is
    /// genuinely ambiguous from the baseline alone, which is the honest case.
    pub segmentation_purity: f64,

    pub comp_support: usize,
    pub comp_query: usize,
    /// Asserted: a composition query pair never appears as a presented fact.
    pub comp_query_leaked: usize,
}

fn entropy(counts: &HashMap<u64, u64>, total: u64) -> f64 {
    let mut h = 0.0;
    for &c in counts.values() {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total as f64;
        h -= p * p.log2();
    }
    h
}

fn conditional_entropy(joint: &HashMap<(u64, u64), u64>, marginal: &HashMap<u64, u64>, total: u64) -> f64 {
    let mut h = 0.0;
    for (&(x, _y), &c) in joint.iter() {
        let px = *marginal.get(&x).unwrap_or(&0) as f64 / total as f64;
        if px <= 0.0 || c == 0 {
            continue;
        }
        let pxy = c as f64 / total as f64;
        h -= pxy * (pxy / px).log2();
    }
    h
}

/// Least-squares slope of log y on log x.
fn loglog_slope(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for &(x, y) in pts {
        if x <= 0.0 || y <= 0.0 {
            continue;
        }
        let lx = x.ln();
        let ly = y.ln();
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
    }
    let den = n * sxx - sx * sx;
    if den.abs() < 1e-12 {
        0.0
    } else {
        (n * sxy - sx * sy) / den
    }
}

pub fn check(stream: &Stream, answer_gap: u32, separations: &[u32]) -> GenReport {
    let n = stream.len();
    let baselines = (0..n).filter(|&t| stream.observe(t).is_none()).count();

    // ---- token statistics, from the ticks the model sees ----
    let mut freq: HashMap<usize, u64> = HashMap::new();
    let mut first_seen = 0usize;
    let mut heaps: Vec<(f64, f64)> = Vec::new();
    let mut seen = 0u64;
    for t in 0..n {
        if let Some(x) = stream.observe(t) {
            seen += 1;
            let e = freq.entry(x).or_insert(0);
            if *e == 0 {
                first_seen += 1;
            }
            *e += 1;
            if seen % 512 == 0 {
                heaps.push((seen as f64, first_seen as f64));
            }
        }
    }
    let mut counts: Vec<u64> = freq.values().copied().collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    let zipf_pts: Vec<(f64, f64)> =
        counts.iter().enumerate().map(|(i, &c)| ((i + 1) as f64, c as f64)).collect();
    let zipf_exponent = -loglog_slope(&zipf_pts);
    let heaps_exponent = loglog_slope(&heaps);

    // ---- conjunction, conditioned on the regime ----
    //
    // Unconditionally, a mode-A cue names its own regime, so I(target ; cue)
    // is dominated by log2(domains) and the co-information comes out negative
    // through pure redundancy. What the second-order window actually needs is
    // that *within* a regime the conjunction is the only thing that identifies
    // the target, so the estimate is made per regime and pooled.
    let mut sep_counts: HashMap<u32, usize> = HashMap::new();
    let mut by_domain: HashMap<usize, Vec<(u64, u64, u64)>> = HashMap::new();
    let mut all: Vec<(u64, u64, u64)> = Vec::new();
    for e in stream.episodes.iter() {
        if e.kind != Kind::Second || e.cues.len() < 2 {
            continue;
        }
        let rec = (e.cues[0] as u64, e.cues[1] as u64, e.target as u64);
        by_domain.entry(e.domain).or_default().push(rec);
        all.push(rec);
        *sep_counts.entry(e.separation).or_insert(0) += 1;
    }
    let second = all.len() as u64;

    let mi_of = |rows: &[(u64, u64, u64)]| -> (f64, f64, f64) {
        let n = rows.len() as u64;
        if n == 0 {
            return (0.0, 0.0, 0.0);
        }
        let mut ja = HashMap::new();
        let mut jb = HashMap::new();
        let mut jab = HashMap::new();
        let mut ma = HashMap::new();
        let mut mb = HashMap::new();
        let mut mab = HashMap::new();
        let mut mt = HashMap::new();
        for &(a, b, t) in rows {
            let ab = a.wrapping_mul(1_000_003).wrapping_add(b);
            *ja.entry((a, t)).or_insert(0) += 1;
            *jb.entry((b, t)).or_insert(0) += 1;
            *jab.entry((ab, t)).or_insert(0) += 1;
            *ma.entry(a).or_insert(0) += 1;
            *mb.entry(b).or_insert(0) += 1;
            *mab.entry(ab).or_insert(0) += 1;
            *mt.entry(t).or_insert(0) += 1;
        }
        let h_t = entropy(&mt, n);
        (
            h_t - conditional_entropy(&ja, &ma, n),
            h_t - conditional_entropy(&jb, &mb, n),
            h_t - conditional_entropy(&jab, &mab, n),
        )
    };

    let (mi_a, mi_b, mi_ab) = mi_of(&all);

    let mut doms: Vec<&usize> = by_domain.keys().collect();
    doms.sort_unstable();
    let (mut wa, mut wb, mut wab, mut wsum) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for d in doms {
        let rows = &by_domain[d];
        let w = rows.len() as f64;
        let (a, b, ab) = mi_of(rows);
        wa += w * a;
        wb += w * b;
        wab += w * ab;
        wsum += w;
    }
    let (wa, wb, wab) = if wsum > 0.0 {
        (wa / wsum, wb / wsum, wab / wsum)
    } else {
        (0.0, 0.0, 0.0)
    };

    let mut sep_list: Vec<(u32, usize)> =
        separations.iter().map(|&s| (s, *sep_counts.get(&s).unwrap_or(&0))).collect();
    sep_list.sort_unstable();

    // ---- segmentation: how ambiguous is the baseline structure? ----
    let mut runs_ending_at_target = 0usize;
    let mut runs_total = 0usize;
    let mut run = 0u32;
    for t in 0..n {
        match stream.observe(t) {
            None => run += 1,
            Some(_) => {
                if run >= answer_gap {
                    runs_total += 1;
                    if stream.ep_at[t].is_some() {
                        runs_ending_at_target += 1;
                    }
                }
                run = 0;
            }
        }
    }
    let segmentation_purity =
        if runs_total == 0 { 0.0 } else { runs_ending_at_target as f64 / runs_total as f64 };

    // ---- composition ----
    let mut support: Vec<(usize, usize)> = Vec::new();
    let mut queries: Vec<(usize, usize)> = Vec::new();
    for e in stream.episodes.iter() {
        if e.cues.len() < 2 {
            continue;
        }
        let pair = (e.cues[0], e.cues[1]);
        match e.kind {
            Kind::CompSupport => support.push(pair),
            Kind::CompQuery => queries.push(pair),
            _ => {}
        }
    }
    let comp_query_leaked = queries.iter().filter(|q| support.contains(q)).count();

    GenReport {
        ticks: n,
        baseline_fraction: baselines as f64 / n as f64,
        episodes: stream.episodes.len(),
        second_order: second as usize,
        distinct_tokens: freq.len(),
        mi_target_cue_a: mi_a,
        mi_target_cue_b: mi_b,
        mi_target_pair: mi_ab,
        mi_within_cue_a: wa,
        mi_within_cue_b: wb,
        mi_within_pair: wab,
        conjunctive_gain: wab - wa - wb,
        sep_counts: sep_list,
        zipf_exponent,
        heaps_exponent,
        segmentation_purity,
        comp_support: support.len(),
        comp_query: queries.len(),
        comp_query_leaked,
    }
}

impl GenReport {
    pub fn print(&self) {
        println!("-- generator report ------------------------------------------");
        println!("  ticks                {}", self.ticks);
        println!("  baseline fraction    {:.3}", self.baseline_fraction);
        println!("  episodes             {}  (second order {})", self.episodes, self.second_order);
        println!("  distinct tokens      {}", self.distinct_tokens);
        println!(
            "  unconditional        I(T;A) {:.3}  I(T;B) {:.3}  I(T;A,B) {:.3}  bits",
            self.mi_target_cue_a, self.mi_target_cue_b, self.mi_target_pair
        );
        println!(
            "  within regime        I(T;A|D) {:.3}  I(T;B|D) {:.3}  I(T;A,B|D) {:.3}  bits",
            self.mi_within_cue_a, self.mi_within_cue_b, self.mi_within_pair
        );
        println!("  conjunctive gain     {:.3} bits  (within regime)", self.conjunctive_gain);
        print!("  separations          ");
        for (s, c) in self.sep_counts.iter() {
            print!("{}:{} ", s, c);
        }
        println!();
        println!("  zipf exponent        {:.3}", self.zipf_exponent);
        println!("  heaps exponent       {:.3}  (reported, not asserted)", self.heaps_exponent);
        println!("  segmentation purity  {:.3}", self.segmentation_purity);
        println!(
            "  composition          support {}  query {}  leaked {}",
            self.comp_support, self.comp_query, self.comp_query_leaked
        );
    }

    /// Fail loudly on anything an experiment depends on. Called before every
    /// run, so a defective source cannot be mistaken for a defective mechanism.
    pub fn assert_usable(&self, min_per_sep: usize) {
        assert!(
            self.conjunctive_gain > 1.0,
            "conjunctive gain is {:.3} bits: the second-order items carry no \
             information beyond their marginals, so the second-order window \
             would measure nothing. Fix the generator, not the model.",
            self.conjunctive_gain
        );
        assert!(
            self.mi_within_cue_a < 0.25 && self.mi_within_cue_b < 0.25,
            "within a regime a single cue already carries {:.3}/{:.3} bits about \
             the target: first-order lookup would pass the second-order test",
            self.mi_within_cue_a,
            self.mi_within_cue_b
        );
        for &(s, c) in self.sep_counts.iter() {
            assert!(
                c >= min_per_sep,
                "separation {} has only {} episodes (need {}); the window curve \
                 would be noise at that point",
                s,
                c,
                min_per_sep
            );
        }
        assert_eq!(
            self.comp_query_leaked, 0,
            "a composition query was also presented as a fact: the composition \
             measurement would be plain lookup"
        );
        assert!(
            self.baseline_fraction > 0.2 && self.baseline_fraction < 0.95,
            "baseline fraction {:.3} leaves no usable response window",
            self.baseline_fraction
        );
    }
}
