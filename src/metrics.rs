//! The measurements.
//!
//! Three structural assertions live in `tests/structural.rs` because they are
//! equalities, not statistics. What is left is three curves and one control, and
//! each has to be able to kill a specific claim:
//!
//! * **Sharpening** -- entropy against ticks since the world spoke. Flat means
//!   the factors being committed are no sharper than the priors they replace, in
//!   which case the whole "one tick, one factor" accounting does not stand.
//! * **Second-order window** -- accuracy against the log separation of two cues.
//!   Expected to be a plateau whose width grows with the number of bands: the
//!   lower edge is where both cues fall inside the fastest band and superpose,
//!   the upper edge is where the first cue has decayed out of the slowest.
//! * **Per-level calibration** -- one reliability plot that does three jobs: it
//!   is the commit rule's input, the sharpening argument's precondition, and the
//!   fabrication measure, since a confidently wrong coarse branch is exactly
//!   what putting mass in the wrong subtree looks like.
//! * **Self-poisoning** -- accuracy against how much of the recent background is
//!   the system's own output. A5 bounds the channel but not the damage inside
//!   the fast band.
//!
//! Everything else that was once on the list -- onset and dwell distributions,
//! binding recovery, particle spread, silence discipline -- is a diagnostic and
//! is printed alongside, not promoted.

use crate::gen::{EpisodeRec, Kind, Stream};
use crate::model::{Model, TickOutcome};

/// A test martingale on the comparison with the reference coder.
///
/// Averages with standard errors are weak inference on a single non-stationary
/// stream run once: the events are not independent, the regime changes under
/// the estimator, and there is no second sample to appeal to. The betting form
/// of the prequential comparison has none of those problems. Wealth is the
/// likelihood ratio process
///
/// ```text
/// W_n = prod_i  q_ours(o_i) / q_ref(o_i)
/// ```
///
/// which under the null "the reference is at least as good" is a non-negative
/// martingale with expectation at most one. So `W_n >= 1/alpha` at *any* stopping
/// time, chosen however you like after seeing the data, is a valid test at level
/// alpha. In bits, log2 W is exactly the cumulative codelength saved, which is
/// the quantity already being accumulated -- the point is that it may be read as
/// evidence and not only as an average.
#[derive(Default, Clone)]
pub struct EProcess {
    pub log2_wealth: f64,
    pub min_log2_wealth: f64,
    pub max_log2_wealth: f64,
    pub n: u64,
}

impl EProcess {
    pub fn push(&mut self, bits_ours: f64, bits_ref: f64) {
        self.log2_wealth += bits_ref - bits_ours;
        self.n += 1;
        if self.log2_wealth < self.min_log2_wealth {
            self.min_log2_wealth = self.log2_wealth;
        }
        if self.log2_wealth > self.max_log2_wealth {
            self.max_log2_wealth = self.log2_wealth;
        }
    }
    /// Anytime-valid bound on the p-value: 1 / wealth, clamped at one.
    pub fn p_value_bound(&self) -> f64 {
        if self.log2_wealth <= 0.0 {
            1.0
        } else {
            (2.0f64).powf(-self.log2_wealth).max(f64::MIN_POSITIVE)
        }
    }
    pub fn verdict(&self) -> String {
        if self.log2_wealth > 0.0 {
            format!(
                "ours ahead by {:.0} bits total; anytime-valid p <= {:.2e}",
                self.log2_wealth,
                self.p_value_bound()
            )
        } else {
            format!(
                "reference ahead by {:.0} bits total; no evidence for ours",
                -self.log2_wealth
            )
        }
    }
}

#[derive(Clone, Default)]
pub struct Bucket {
    pub n: u64,
    pub sum: f64,
    pub sum_sq: f64,
    pub hits: u64,
}

impl Bucket {
    pub fn push(&mut self, v: f64, hit: bool) {
        self.n += 1;
        self.sum += v;
        self.sum_sq += v * v;
        if hit {
            self.hits += 1;
        }
    }
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }
    pub fn stderr(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let m = self.mean();
        // Unbiased: dividing by n understates the error, and several buckets
        // in the sweeps are small enough for the difference to matter.
        let n = self.n as f64;
        let var = ((self.sum_sq - n * m * m) / (n - 1.0)).max(0.0);
        (var / n).sqrt()
    }
    pub fn accuracy(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.hits as f64 / self.n as f64
        }
    }
}

pub struct Metrics {
    /// Entropy against ticks since the last event.
    pub sharpening: Vec<Bucket>,
    /// Latin second-order items, bucketed by separation. Not representable by a
    /// linear readout on the superposed cues.
    pub window: Vec<(u32, Bucket)>,
    /// Product-code second-order items, same buckets. Representable. The two
    /// curves read together are what separates an address failure from a
    /// representation failure.
    pub window_product: Vec<(u32, Bucket)>,
    /// First-order items, bucketed by answer gap. The first-order special case
    /// of the same curve.
    pub gap: Vec<(u32, Bucket)>,
    /// Bits by distance since the last baseline: index 0 is "the world had just
    /// been silent", so the code had to come from memory; higher indices are
    /// local continuation inside a drive.
    pub by_distance: Vec<Bucket>,
    /// Self-poisoning: bucketed by the fraction of recent ticks that carried an
    /// overt emission.
    pub poisoning: Vec<Bucket>,
    /// Retention on departed regimes, bucketed by how many spans ago the regime
    /// left the stream. Measured under frozen memory, so it is a statement about
    /// what was stored and not about what the probe left behind.
    pub retention: Vec<Bucket>,
    pub composition_query: Bucket,
    pub composition_support: Bucket,
    pub all_events: Bucket,

    /// Which regime's writes landed in which leaf. The model never sees this;
    /// it exists so an experiment can ask whether the address separates regimes
    /// at all, which is the difference between a working coarse level and a
    /// tree that has fragmented on noise.
    pub leaf_domain: std::collections::HashMap<usize, Vec<u64>>,
    pub domains: usize,
    /// Which cue pair's writes landed in which leaf, for second-order items
    /// only. This is the diagnostic that splits a failure in two.
    ///
    /// The square is a Latin square, so the target is a function of the pair and
    /// of neither cue alone -- and, being additive modulo m, it is not a linear
    /// function of the two cues' superposed embeddings either. A linear readout
    /// on the payload therefore cannot represent it. The only route left is for
    /// the *address* to isolate the pair, giving each pair a leaf whose counts
    /// then answer directly. `pair_purity` measures whether it does. Near chance
    /// means the address is not isolating pairs and the readout was never given
    /// a separable problem; near one means the address did its part and the
    /// failure is downstream.
    pub leaf_pair: std::collections::HashMap<usize, std::collections::HashMap<(usize, usize), u64>>,

    /// Ticks after the last event at which the right answer first showed up in
    /// inner speech, and at which it was first said. Two reaction times, and the
    /// distance between them is what the commitment rule costs.
    pub idea_onset: Bucket,
    pub speech_onset: Bucket,
    /// Evidence against the reference coder, accumulated on the same events.
    pub evidence: EProcess,

    pub cumulative_bits: f64,
    pub charged_events: u64,
    /// Diagnostics, printed but not promoted.
    pub overt_ticks: u64,
    pub silent_ticks: u64,
    overt_window: std::collections::VecDeque<bool>,
}

const SHARP_MAX: usize = 24;
const DIST_MAX: usize = 8;
const POISON_BINS: usize = 5;
const RETENTION_BINS: usize = 8;
const OVERT_WINDOW: usize = 64;

impl Metrics {
    pub fn new(separations: &[u32], gaps: &[u32]) -> Self {
        Metrics {
            sharpening: vec![Bucket::default(); SHARP_MAX],
            window: separations.iter().map(|&s| (s, Bucket::default())).collect(),
            window_product: separations.iter().map(|&s| (s, Bucket::default())).collect(),
            gap: gaps.iter().map(|&g| (g, Bucket::default())).collect(),
            by_distance: vec![Bucket::default(); DIST_MAX],
            poisoning: vec![Bucket::default(); POISON_BINS],
            retention: vec![Bucket::default(); RETENTION_BINS],
            composition_query: Bucket::default(),
            composition_support: Bucket::default(),
            all_events: Bucket::default(),
            leaf_domain: std::collections::HashMap::new(),
            domains: 0,
            leaf_pair: std::collections::HashMap::new(),
            idea_onset: Bucket::default(),
            speech_onset: Bucket::default(),
            evidence: EProcess::default(),
            cumulative_bits: 0.0,
            charged_events: 0,
            overt_ticks: 0,
            silent_ticks: 0,
            overt_window: std::collections::VecDeque::with_capacity(OVERT_WINDOW),
        }
    }

    fn self_fraction(&self) -> f64 {
        if self.overt_window.is_empty() {
            return 0.0;
        }
        self.overt_window.iter().filter(|b| **b).count() as f64 / self.overt_window.len() as f64
    }

    pub fn note_write(&mut self, leaf: usize, ep: &EpisodeRec) {
        let n = self.domains.max(ep.domain + 1);
        self.domains = n;
        let e = self.leaf_domain.entry(leaf).or_insert_with(|| vec![0; n]);
        if e.len() < n {
            e.resize(n, 0);
        }
        e[ep.domain] += 1;
        if matches!(ep.kind, Kind::Second | Kind::Product) && ep.cues.len() >= 2 {
            *self
                .leaf_pair
                .entry(leaf)
                .or_default()
                .entry((ep.cues[0], ep.cues[1]))
                .or_insert(0) += 1;
        }
    }

    /// Weighted mean over leaves of the share held by that leaf's dominant cue
    /// pair. Chance is roughly 1/(pairs per leaf).
    pub fn pair_purity(&self) -> (f64, f64) {
        let mut num = 0.0;
        let mut den = 0.0;
        let mut distinct = 0.0;
        let mut leaves = 0.0;
        let mut pkeys: Vec<&usize> = self.leaf_pair.keys().collect();
        pkeys.sort_unstable();
        for m in pkeys.into_iter().map(|k| &self.leaf_pair[k]) {
            let total: u64 = m.values().sum();
            if total == 0 {
                continue;
            }
            num += *m.values().max().unwrap() as f64;
            den += total as f64;
            distinct += m.len() as f64;
            leaves += 1.0;
        }
        if den == 0.0 {
            return (0.0, 0.0);
        }
        (num / den, if leaves > 0.0 { distinct / leaves } else { 0.0 })
    }

    pub fn observe(&mut self, out: &TickOutcome, ep: Option<&EpisodeRec>) {
        // Silence discipline, kept as a diagnostic.
        if out.overt.is_some() {
            self.overt_ticks += 1;
        } else {
            self.silent_ticks += 1;
        }
        // Bucket against what preceded this tick; folding the tick's own
        // overt-ness into the figure that classifies it makes the predictor
        // partly the thing being predicted.
        let self_fraction_before = self.self_fraction();
        if self.overt_window.len() == OVERT_WINDOW {
            self.overt_window.pop_front();
        }
        self.overt_window.push_back(out.overt.is_some());

        if let Some(h) = out.entropy_bits {
            let b = (out.ticks_since_event as usize).min(SHARP_MAX - 1);
            self.sharpening[b].push(h, false);
        }

        if !out.charged {
            return;
        }
        self.cumulative_bits += out.bits;
        self.charged_events += 1;
        self.all_events.push(out.bits, out.correct);

        let d = (out.ticks_since_event as usize).min(DIST_MAX - 1);
        self.by_distance[d].push(out.bits, out.correct);

        if let Some(k) = out.idea_onset {
            self.idea_onset.push(k as f64, true);
        }
        if let Some(k) = out.speech_onset {
            self.speech_onset.push(k as f64, true);
        }

        let sf = self_fraction_before;
        let pb = ((sf * POISON_BINS as f64) as usize).min(POISON_BINS - 1);
        self.poisoning[pb].push(out.bits, out.correct);

        if let Some(e) = ep {
            match e.kind {
                Kind::Second => {
                    for (s, b) in self.window.iter_mut() {
                        if *s == e.separation {
                            b.push(out.bits, out.correct);
                        }
                    }
                }
                Kind::First => {
                    for (g, b) in self.gap.iter_mut() {
                        if *g == e.answer_gap {
                            b.push(out.bits, out.correct);
                        }
                    }
                }
                Kind::Product => {
                    for (s, b) in self.window_product.iter_mut() {
                        if *s == e.separation {
                            b.push(out.bits, out.correct);
                        }
                    }
                }
                Kind::CompQuery => self.composition_query.push(out.bits, out.correct),
                Kind::CompSupport => self.composition_support.push(out.bits, out.correct),
            }
        }
    }

    /// Mean over leaves of the share held by that leaf's dominant regime,
    /// weighted by how much each leaf holds. One means every leaf serves a
    /// single regime; 1/domains means the address carries no regime information
    /// whatsoever.
    pub fn leaf_purity(&self) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        let mut keys: Vec<&usize> = self.leaf_domain.keys().collect();
        keys.sort_unstable();
        for counts in keys.into_iter().map(|k| &self.leaf_domain[k]) {
            let total: u64 = counts.iter().sum();
            if total == 0 {
                continue;
            }
            let top = *counts.iter().max().unwrap();
            num += top as f64;
            den += total as f64;
        }
        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    }

    pub fn note_probe(&mut self, age_spans: u32, bits: f64, correct: bool) {
        let b = (age_spans as usize).min(RETENTION_BINS - 1);
        self.retention[b].push(bits, correct);
    }

    pub fn bits_per_event(&self) -> f64 {
        if self.charged_events == 0 {
            0.0
        } else {
            self.cumulative_bits / self.charged_events as f64
        }
    }

    pub fn print(&self, model: &Model, label: &str) {
        println!("== {} ==========================================", label);
        println!(
            "  charged events {}   bits/event {:.3}   top-1 {:.3}",
            self.charged_events,
            self.bits_per_event(),
            self.all_events.accuracy()
        );
        println!("  evidence vs PPM: {}", self.evidence.verdict());
        if model.cfg.commit_locks_charge {
            println!(
                "  charged what was said: {} commitments, {} settled silent \
                 ({:.1}% of events) -- bits are NOT comparable to runs without \
                 this rule",
                model.commitments,
                model.silent_settlements,
                100.0 * model.silent_settlements as f64 / self.charged_events.max(1) as f64
            );
        }
        println!(
            "  onsets: idea at t+{:.2} on {} events, speech at t+{:.2} on {}",
            self.idea_onset.mean(),
            self.idea_onset.n,
            self.speech_onset.mean(),
            self.speech_onset.n
        );
        println!(
            "  tree: {} nodes, depth {}, {} leaves, {} occupied rows  (widen {}, deepen {})",
            model.tree.nodes(),
            model.tree.realised_depth(),
            model.tree.leaves(),
            model.tree.occupied_rows(),
            model.tree.widen_events,
            model.tree.deepen_events
        );
        println!(
            "  particles: resamples {}, backtracks {}, commits {} ({:.2}/event), \
             mean depth {:.2}, root spread {:.2}",
            model.swarm.resamples,
            model.swarm.backtracks,
            model.swarm.commits,
            model.swarm.commits as f64 / self.charged_events.max(1) as f64,
            model.swarm.mean_depth(),
            model.swarm.spread_at_root()
        );

        print!("  -- ticks to mature, by level ");
        for (l, t) in model.swarm.ticks_per_level().iter().enumerate() {
            if *t > 0.0 {
                print!("L{}:{:.2} ", l, t);
            }
        }
        println!(
            " (cap {}; flat at the cap means the maturity rule is inert)",
            model.cfg.max_ticks_per_level
        );

        println!("  -- sharpening (entropy bits by ticks since event)");
        for (i, b) in self.sharpening.iter().enumerate() {
            if b.n > 0 {
                println!("     t+{:<3} n={:<7} H={:.3} +/- {:.3}", i, b.n, b.mean(), b.stderr());
            }
        }

        println!(
            "  -- second-order window (by cue separation; within-regime chance \
             is 1/square_m)"
        );
        for (s, b) in self.window.iter() {
            println!(
                "     sep={:<4} n={:<6} bits={:.3} acc={:.3}",
                s,
                b.n,
                b.mean(),
                b.accuracy()
            );
        }

        println!("  -- product-code window (separable control)");
        for (s, b) in self.window_product.iter() {
            println!(
                "     sep={:<4} n={:<6} bits={:.3} acc={:.3}",
                s,
                b.n,
                b.mean(),
                b.accuracy()
            );
        }

        println!("  -- first-order gap curve");
        for (g, b) in self.gap.iter() {
            if b.n > 0 {
                println!(
                    "     gap={:<4} n={:<6} bits={:.3} acc={:.3}",
                    g,
                    b.n,
                    b.mean(),
                    b.accuracy()
                );
            }
        }

        println!("  -- retention on departed regimes (frozen memory)");
        for (a, b) in self.retention.iter().enumerate() {
            if b.n > 0 {
                println!(
                    "     age={:<3} spans  n={:<6} bits={:.3} acc={:.3}",
                    a,
                    b.n,
                    b.mean(),
                    b.accuracy()
                );
            }
        }

        println!("  -- per-level calibration");
        for (l, c) in model.tree.calib.iter().enumerate() {
            if c.observations() == 0 {
                continue;
            }
            println!("     level {}  n={:<7} ECE={:.3}", l, c.observations(), c.ece());
        }

        println!("  -- self-poisoning (by fraction of recent ticks spoken)");
        for (i, b) in self.poisoning.iter().enumerate() {
            if b.n > 0 {
                let lo = i as f64 / POISON_BINS as f64;
                println!(
                    "     self {:.1}-{:.1}  n={:<6} bits={:.3} acc={:.3}",
                    lo,
                    lo + 1.0 / POISON_BINS as f64,
                    b.n,
                    b.mean(),
                    b.accuracy()
                );
            }
        }

        println!("  -- decomposition by distance since baseline");
        for (i, b) in self.by_distance.iter().enumerate() {
            if b.n > 0 {
                println!("     d={:<3} n={:<7} bits={:.3} acc={:.3}", i, b.n, b.mean(), b.accuracy());
            }
        }

        println!(
            "  -- composition: support n={} bits={:.3} | query n={} bits={:.3}",
            self.composition_support.n,
            self.composition_support.mean(),
            self.composition_query.n,
            self.composition_query.mean()
        );
        println!(
            "  -- diagnostics: overt {:.3} of ticks, band corr(0,1)={:.3} corr(0,{})={:.3}",
            self.overt_ticks as f64 / (self.overt_ticks + self.silent_ticks).max(1) as f64,
            model.ladder.band_correlation(0, 1),
            model.cfg.rungs - 1,
            model.ladder.band_correlation(0, model.cfg.rungs - 1)
        );
        print!("  -- rung visits ");
        for (k, v) in model.tree.rung_visits.iter().enumerate() {
            print!("r{}:{} ", k, v);
        }
        println!(
            " (coverage {:.2}; a rung with no visits is a band the model does \
             not have)",
            model.tree.rung_coverage()
        );
        println!(
            "  -- leaf surprise mean {:.2} bits over {} writes  (a split \
             threshold below this range makes every node split always)",
            model.tree.leaf_surprise.mean,
            model.tree.leaf_surprise.n
        );
        println!(
            "  -- leaf purity by regime {:.3}  (chance {:.3}, over {} leaves that took writes)",
            self.leaf_purity(),
            if self.domains > 0 { 1.0 / self.domains as f64 } else { 0.0 },
            self.leaf_domain.len()
        );
        let (pp, per_leaf) = self.pair_purity();
        println!(
            "  -- leaf purity by cue pair {:.3}  (chance about {:.3}: {:.1} distinct \
             pairs per leaf)",
            pp,
            if per_leaf > 0.0 { 1.0 / per_leaf } else { 0.0 },
            per_leaf
        );
    }

    pub fn csv_window(&self, tag: &str, out: &mut String) {
        for (s, b) in self.window.iter() {
            out.push_str(&format!(
                "{},window,{},{},{:.6},{:.6},{:.6}\n",
                tag,
                s,
                b.n,
                b.mean(),
                b.stderr(),
                b.accuracy()
            ));
        }
    }

    pub fn csv_sharpening(&self, tag: &str, out: &mut String) {
        for (i, b) in self.sharpening.iter().enumerate() {
            if b.n > 0 {
                out.push_str(&format!(
                    "{},sharpening,{},{},{:.6},{:.6},0\n",
                    tag,
                    i,
                    b.n,
                    b.mean(),
                    b.stderr()
                ));
            }
        }
    }
}

/// Drive a model over a stream and collect everything.
///
/// The reference coder runs in lockstep on the same events, so the evidence
/// process is a per-event likelihood ratio rather than a comparison of two
/// separately computed averages.
pub fn run(model: &mut Model, stream: &Stream, metrics: &mut Metrics) {
    let every = model.cfg.entropy_every;
    let mut reference = crate::baseline::Ppm::new(4, stream.vocab + 1);
    let silence = stream.vocab as u32;
    for t in 0..stream.len() {
        let want_entropy = every > 0 && (t as u64) % every == 0;
        let obs = stream.observe(t);
        for spec in stream.probe_at[t].iter() {
            let (bits, correct) = model.probe(spec, 6);
            metrics.note_probe(spec.age_spans, bits, correct);
        }
        let out = model.tick(obs, want_entropy);
        let (ref_bits, _) = reference.observe(match obs {
            Some(x) => x as u32,
            None => silence,
        });
        if out.charged {
            metrics.evidence.push(out.bits, ref_bits);
        }
        let ep = stream.ep_at[t].map(|i| &stream.episodes[i]);
        if out.wrote {
            if let Some(e) = ep {
                metrics.note_write(model.last_write_leaf, e);
            }
        }
        metrics.observe(&out, ep);
    }
}
