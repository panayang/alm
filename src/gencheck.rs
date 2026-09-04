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

    /// Distinct (cue_a, cue_b) facts actually presented, and presentations per
    /// fact, per family: [Second (Latin), Product].
    ///
    /// A load sweep raises the number of things to remember while holding the
    /// practice each one gets. Nothing asserted that. Scaling ticks with domains
    /// keeps each *domain* live for the same time, which is not the same claim,
    /// and the two conjunction families need not respond to it alike. Without
    /// these counts a sweep can compare "more facts, same practice" in one
    /// family against "same facts, more practice" in the other and read the
    /// difference as a property of the mechanism.
    /// Best top-1 accuracy obtainable within a regime from ONE cue, per family:
    /// [Second (Latin), Product], as (from cue A, from cue B).
    ///
    /// This is the number a Latin accuracy has to beat to be evidence of
    /// conjunctive retrieval, and the suite never had it. With cue A Zipf and
    /// cue B uniform it stood at 0.2745 while every reported Latin accuracy sat
    /// between 0.19 and 0.246 -- so no arm ever cleared a predictor that ignores
    /// half the challenge. Empirical rather than analytic, and therefore biased
    /// upward at small samples in the same way a model's own estimate would be,
    /// which is the comparison that matters.
    pub single_cue_ceiling: [(f64, f64); 2],
    pub distinct_facts: [usize; 2],
    pub presentations_per_fact: [f64; 2],

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

    /// The same three quantities for the product-code items. Here the marginals
    /// are *supposed* to be non-zero: a table with zero marginals cannot be
    /// linearly separable, so the separable control has to pay for it with
    /// marginal information. What has to hold is that the pair still carries
    /// strictly more than the two cues do apart.
    pub product_within_cue_a: f64,
    pub product_within_cue_b: f64,
    pub product_within_pair: f64,
    pub product_items: usize,
    /// Fraction of (a, b) cells whose target matches the additive product code
    /// the generator claims to have emitted. Asserted at one: this is the
    /// property that makes the item type representable by a linear readout, and
    /// it is checked on the emitted stream rather than assumed.
    pub product_separable_fraction: f64,
    /// Second-order items per regime. The conjunction estimate is only worth
    /// asserting on when this is comfortably above the number of cells.
    pub second_per_domain: f64,

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
    // Sorted, because a float sum over hash-map iteration order is not
    // reproducible: Rust seeds each process's hasher differently, so the same
    // stream would report slightly different numbers on every run and an
    // assertion sitting near its threshold would be flaky for no reason.
    //
    // Miller-Madow corrected. The plug-in entropy is biased *down* by roughly
    // (K-1)/(2N) nats, so a plug-in mutual information is biased *up* -- and
    // with a 6x6 table seen 78 times per regime that bias is about 0.23 bits,
    // which is the whole of what an uncorrected estimator reports as "the cue
    // already tells you the answer". Correcting it is not a nicety here: it is
    // the difference between a data check that measures the source and one that
    // measures its own sample size.
    let mut cs: Vec<u64> = counts.values().copied().collect();
    cs.sort_unstable();
    let mut h = 0.0;
    let mut support = 0u64;
    for c in cs {
        if c == 0 {
            continue;
        }
        support += 1;
        let p = c as f64 / total as f64;
        h -= p * p.log2();
    }
    if total > 0 && support > 1 {
        h += (support - 1) as f64 / (2.0 * total as f64 * std::f64::consts::LN_2);
    }
    h
}

fn conditional_entropy(
    joint: &HashMap<(u64, u64), u64>,
    marginal: &HashMap<u64, u64>,
    total: u64,
) -> f64 {
    let mut keys: Vec<(u64, u64)> = joint.keys().copied().collect();
    keys.sort_unstable();
    let mut h = 0.0;
    // Support of T within each value of X, for the same correction applied
    // conditionally.
    let mut support_per_x: HashMap<u64, u64> = HashMap::new();
    for k in keys {
        let (x, _y) = k;
        let c = joint[&k];
        *support_per_x.entry(x).or_insert(0) += 1;
        let px = *marginal.get(&x).unwrap_or(&0) as f64 / total as f64;
        if px <= 0.0 || c == 0 {
            continue;
        }
        let pxy = c as f64 / total as f64;
        h -= pxy * (pxy / px).log2();
    }
    if total > 0 {
        let mut xs: Vec<u64> = support_per_x.keys().copied().collect();
        xs.sort_unstable();
        let mut extra = 0.0;
        for x in xs {
            let k = support_per_x[&x];
            if k > 1 {
                extra += (k - 1) as f64;
            }
        }
        h += extra / (2.0 * total as f64 * std::f64::consts::LN_2);
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

    // ---- the separable control ----
    let mut prod_by_domain: HashMap<usize, Vec<(u64, u64, u64)>> = HashMap::new();
    let mut prod_all: Vec<(u64, u64, u64)> = Vec::new();
    for e in stream.episodes.iter() {
        if e.kind != Kind::Product || e.cues.len() < 2 {
            continue;
        }
        let rec = (e.cues[0] as u64, e.cues[1] as u64, e.target as u64);
        prod_by_domain.entry(e.domain).or_default().push(rec);
        prod_all.push(rec);
        *sep_counts.entry(e.separation).or_insert(0) += 1;
    }
    let mut pdoms: Vec<&usize> = prod_by_domain.keys().collect();
    pdoms.sort_unstable();
    let (mut pa, mut pb, mut pab, mut psum) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for d in pdoms {
        let rows = &prod_by_domain[d];
        let w = rows.len() as f64;
        let (a, b, ab) = mi_of(rows);
        pa += w * a;
        pb += w * b;
        pab += w * ab;
        psum += w;
    }
    let (pa, pb, pab) =
        if psum > 0.0 { (pa / psum, pb / psum, pab / psum) } else { (0.0, 0.0, 0.0) };

    // A product code is a function of the pair, and every (a, b) cell must map
    // to one target. If a cell is ever seen with two different targets the code
    // is not a product code and the separability claim is void.
    let mut cell: HashMap<(u64, u64), u64> = HashMap::new();
    let mut consistent = 0usize;
    for &(a, b, t) in prod_all.iter() {
        match cell.get(&(a, b)) {
            None => {
                cell.insert((a, b), t);
                consistent += 1;
            }
            Some(&t0) => {
                if t0 == t {
                    consistent += 1;
                }
            }
        }
    }
    let product_separable_fraction =
        if prod_all.is_empty() { 0.0 } else { consistent as f64 / prod_all.len() as f64 };

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
    // A composition query leaks if its (cue, target) has been presented as a
    // fact by *any* family, not only as one of its own supports. The old check
    // compared query pairs against support pairs, and since the query carries
    // relation r12 while supports carry r1/r2, it was structurally always zero
    // and could never fire -- while chain k = 0 shared its (a, c) with
    // `firsts[0]` outright.
    let mut first_facts: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for ep in stream.episodes.iter() {
        if matches!(ep.kind, crate::gen::Kind::First) && !ep.cues.is_empty() {
            first_facts.insert((ep.cues[0], ep.target));
        }
    }
    let mut comp_query_leaked = queries.iter().filter(|q| support.contains(q)).count();
    for ep in stream.episodes.iter() {
        if matches!(ep.kind, crate::gen::Kind::CompQuery)
            && !ep.cues.is_empty()
            && first_facts.contains(&(ep.cues[0], ep.target))
        {
            comp_query_leaked += 1;
        }
    }

    // Distinct facts and practice per fact, straight off the stream: no
    // threshold, so this cannot be biased by looking only at a tail.
    let mut fact_counts: [std::collections::HashMap<(usize, usize), u64>; 2] =
        [std::collections::HashMap::new(), std::collections::HashMap::new()];
    for ep in stream.episodes.iter() {
        let fam = match ep.kind {
            crate::gen::Kind::Second => 0,
            crate::gen::Kind::Product => 1,
            _ => continue,
        };
        if ep.cues.len() >= 2 {
            *fact_counts[fam].entry((ep.cues[0], ep.cues[1])).or_insert(0) += 1;
        }
    }
    // Best within-regime single-cue accuracy, per family and per cue side --
    // *prequential*, not in-sample.
    //
    // Predict from the counts accumulated so far, then update. An in-sample
    // ceiling is an oracle: with about eighteen items per (regime, cue) group
    // spread over six targets, its empirical mode alone reaches 0.27 by chance,
    // which no model charged before it sees the answer can reach. The mechanism
    // is scored prequentially, so its baseline has to be too.
    let mut sc: [std::collections::HashMap<(usize, usize, usize), u64>; 4] = Default::default();
    let mut sc_hit = [0u64; 4];
    let mut sc_tot = [0u64; 2];
    for ep in stream.episodes.iter() {
        let fam = match ep.kind {
            crate::gen::Kind::Second => 0usize,
            crate::gen::Kind::Product => 1,
            _ => continue,
        };
        if ep.cues.len() < 2 {
            continue;
        }
        for side in 0..2 {
            let slot = fam * 2 + side;
            let mut best: Option<(u64, usize)> = None;
            let mut keys: Vec<&(usize, usize, usize)> = sc[slot].keys().collect();
            keys.sort_unstable();
            for k in keys {
                if k.0 == ep.domain && k.1 == ep.cues[side] {
                    let c = sc[slot][k];
                    match best {
                        Some((bc, bt)) if bc > c || (bc == c && bt <= k.2) => {}
                        _ => best = Some((c, k.2)),
                    }
                }
            }
            if let Some((_, t)) = best {
                if t == ep.target {
                    sc_hit[slot] += 1;
                }
            }
            *sc[slot].entry((ep.domain, ep.cues[side], ep.target)).or_insert(0) += 1;
        }
        sc_tot[fam] += 1;
    }
    let acc = |slot: usize, tot: u64| -> f64 {
        if tot == 0 { 0.0 } else { sc_hit[slot] as f64 / tot as f64 }
    };
    let single_cue_ceiling = [
        (acc(0, sc_tot[0]), acc(1, sc_tot[0])),
        (acc(2, sc_tot[1]), acc(3, sc_tot[1])),
    ];

    let mut distinct_facts = [0usize; 2];
    let mut presentations_per_fact = [0.0f64; 2];
    for f in 0..2 {
        let n = fact_counts[f].len();
        let total: u64 = fact_counts[f].values().sum();
        distinct_facts[f] = n;
        presentations_per_fact[f] = if n == 0 { 0.0 } else { total as f64 / n as f64 };
    }

    GenReport {
        single_cue_ceiling,
        distinct_facts,
        presentations_per_fact,
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
        product_within_cue_a: pa,
        product_within_cue_b: pb,
        product_within_pair: pab,
        product_items: prod_all.len(),
        product_separable_fraction,
        second_per_domain: if by_domain.is_empty() {
            0.0
        } else {
            second as f64 / by_domain.len() as f64
        },
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
        println!(
            "  single-cue baseline (prequential, within regime, one cue only):              latin A {:.4} B {:.4} | product A {:.4} B {:.4}  <-- a conjunction              result must beat these",
            self.single_cue_ceiling[0].0,
            self.single_cue_ceiling[0].1,
            self.single_cue_ceiling[1].0,
            self.single_cue_ceiling[1].1
        );
        println!(
            "  facts presented: latin {} distinct, {:.1} presentations each |              product {} distinct, {:.1} each",
            self.distinct_facts[0],
            self.presentations_per_fact[0],
            self.distinct_facts[1],
            self.presentations_per_fact[1]
        );
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
        println!(
            "  conjunctive gain     {:.3} bits  (within regime, Miller-Madow              corrected, {:.0} items per regime)",
            self.conjunctive_gain, self.second_per_domain
        );
        println!(
            "  product code         I(T;A|D) {:.3}  I(T;B|D) {:.3}  I(T;A,B|D) {:.3}  \
             bits over {} items",
            self.product_within_cue_a,
            self.product_within_cue_b,
            self.product_within_pair,
            self.product_items
        );
        println!(
            "  product consistency  {:.4}  (cells mapping to a single target)",
            self.product_separable_fraction
        );
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
            self.second_per_domain >= 2.0 * 36.0,
            "only {:.0} second-order items per regime against 36 cells: the              conjunction estimate is sample-limited and asserting on it would              be asserting on the sample size, not on the source",
            self.second_per_domain
        );
        // The assertion that would have caught the Zipf/uniform asymmetry. Both
        // marginals being *small* was already checked and passed while cue B
        // carried 0.067 bits that cue A did not, because the target inherited
        // A's Zipf profile conditioned on B. A Latin square is zero-marginal
        // only when its two margins agree, so agreement is the thing to assert.
        assert!(
            (self.mi_within_cue_a - self.mi_within_cue_b).abs() < 0.05,
            "the two cue marginals disagree: {:.3} against {:.3} bits. A Latin              square with a skewed cue distribution on one side only is not              zero-marginal, and a single-cue predictor will beat the conjunction",
            self.mi_within_cue_a,
            self.mi_within_cue_b
        );
        assert!(
            self.mi_within_cue_a < 0.40 && self.mi_within_cue_b < 0.40,
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
            self.product_items > 0,
            "no product-code items were emitted: the separable control is the \
             only thing that can tell an address failure apart from a \
             representation failure, and without it the Latin result is not \
             interpretable"
        );
        assert!(
            self.product_separable_fraction > 0.999,
            "product cells are not single-valued ({:.4}); the item type is not a \
             product code and its separability cannot be claimed",
            self.product_separable_fraction
        );
        assert!(
            self.product_within_pair > self.product_within_cue_a + 0.4
                && self.product_within_pair > self.product_within_cue_b + 0.4,
            "the product pair carries {:.3} bits against marginals of {:.3}/{:.3}: \
             a single cue nearly answers it and the control is not testing a \
             conjunction at all",
            self.product_within_pair,
            self.product_within_cue_a,
            self.product_within_cue_b
        );
        assert!(
            self.baseline_fraction > 0.2 && self.baseline_fraction < 0.95,
            "baseline fraction {:.3} leaves no usable response window",
            self.baseline_fraction
        );
    }
}
