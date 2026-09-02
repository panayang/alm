//! Experiment drivers.
//!
//! The order is deliberate. The flatten sanity check runs first, because if
//! capping the tree at depth one does not reproduce the flat-class behaviour of
//! the reference mechanism then something already established has been broken
//! and nothing downstream means anything. The generator report runs before every
//! single one of them, and its load-bearing properties are asserted, so a flat
//! curve can never be blamed on the mechanism before the source has been
//! cleared.
//!
//! Ablations are deliberately few. Over-ablation kills mechanisms that were
//! never given anything to do on this source, and each of the three here is
//! aimed at one claim rather than at a component.

use crate::baseline;
use crate::config::{Config, SplitRule};
use crate::gen::{GenConfig, Generator, Mode, Stream};
use crate::gencheck;
use crate::metrics::{run, Metrics};
use crate::model::Model;

pub struct Outcome {
    pub label: String,
    pub model: Model,
    pub metrics: Metrics,
}

pub fn build_stream(gcfg: &GenConfig, ticks: usize, min_per_sep: usize, verbose: bool) -> Stream {
    let gen = Generator::new(gcfg.clone());
    let stream = gen.generate(ticks);
    let rep = gencheck::check(&stream, gcfg.answer_gap, &gcfg.separations);
    if verbose {
        rep.print();
    }
    // Suspect the data first: this panics before any model is built.
    rep.assert_usable(min_per_sep);
    stream
}

pub fn run_one(label: &str, cfg: Config, gcfg: &GenConfig, stream: &Stream) -> Outcome {
    let mut model = Model::new(cfg);
    let gaps = vec![gcfg.answer_gap];
    let mut metrics = Metrics::new(&gcfg.separations, &gaps);
    run(&mut model, stream, &mut metrics);
    Outcome { label: label.to_string(), model, metrics }
}

/// Mean over a window's buckets, pooled by count.
pub fn window_mean(w: &[(u32, crate::metrics::Bucket)]) -> (f64, f64) {
    let (mut n, mut s, mut hits) = (0u64, 0.0f64, 0u64);
    for (_, b) in w.iter() {
        n += b.n;
        s += b.sum;
        hits += b.hits;
    }
    if n == 0 {
        (0.0, 0.0)
    } else {
        (s / n as f64, hits as f64 / n as f64)
    }
}

/// Width of the second-order plateau: how many separations sit within half a bit
/// of the best one. The prediction under test is that this grows with the number
/// of bands, and that is the quantity to watch across the rung sweep.
pub fn plateau_width(m: &Metrics) -> usize {
    let best = m
        .window
        .iter()
        .filter(|(_, b)| b.n > 0)
        .map(|(_, b)| b.mean())
        .fold(f64::INFINITY, f64::min);
    if !best.is_finite() {
        return 0;
    }
    m.window.iter().filter(|(_, b)| b.n > 0 && b.mean() <= best + 0.5).count()
}

pub struct Suite {
    pub csv: String,
    pub summary: Vec<String>,
}

impl Suite {
    fn new() -> Self {
        Suite {
            csv: String::from("run,metric,x,n,value,stderr,accuracy\n"),
            summary: Vec::new(),
        }
    }
    fn note(&mut self, s: String) {
        println!("{}", s);
        self.summary.push(s);
    }
}

pub fn full(ticks: usize, seed: u64, quick: bool) -> Suite {
    let mut suite = Suite::new();

    let mut gcfg = GenConfig::local();
    gcfg.seed = seed ^ 0xA11CE;
    if quick {
        gcfg.span_ticks = 1500;
    }
    let min_per_sep = if quick { 10 } else { 40 };

    println!("building stream: {} ticks", ticks);
    let stream = build_stream(&gcfg, ticks, min_per_sep, true);

    let mut base = Config::local();
    base.seed = seed;
    base.vocab = gcfg.vocab;
    base.horizon = 256.0;
    base.derive();

    // ---- 0. sanity: flattening must reproduce the flat-class behaviour ----
    let mut flat = base.clone();
    flat.flatten = true;
    let o_flat = run_one("flatten-L1", flat, &gcfg, &stream);
    let (f_lat, f_lat_acc) = window_mean(&o_flat.metrics.window);
    let (f_prod, f_prod_acc) = window_mean(&o_flat.metrics.window_product);
    suite.note(format!(
        "[sanity] L=1  {:.3} bits/ev  top-1 {:.3}  Latin {:.3}/{:.3}  product \
         {:.3}/{:.3}  ({} nodes, depth {})",
        o_flat.metrics.bits_per_event(),
        o_flat.metrics.all_events.accuracy(),
        f_lat,
        f_lat_acc,
        f_prod,
        f_prod_acc,
        o_flat.model.tree.nodes(),
        o_flat.model.tree.realised_depth()
    ));
    assert_eq!(
        o_flat.model.tree.realised_depth(),
        1,
        "flatten did not actually cap the tree at depth one"
    );

    // ---- 1. the full mechanism ----
    let o_full = run_one("full", base.clone(), &gcfg, &stream);
    o_full.metrics.print(&o_full.model, "full");
    o_full.metrics.csv_window("full", &mut suite.csv);
    o_full.metrics.csv_sharpening("full", &mut suite.csv);

    // Sharpening: the first thing that can kill the accounting.
    let sharp: Vec<(usize, f64)> = o_full
        .metrics
        .sharpening
        .iter()
        .enumerate()
        .filter(|(_, b)| b.n > 32)
        .map(|(i, b)| (i, b.mean()))
        .collect();
    let sharp_verdict = if sharp.len() < 2 {
        "insufficient data".to_string()
    } else {
        let first = sharp.first().unwrap().1;
        let last = sharp.last().unwrap().1;
        format!("H(t+{}) {:.3} -> H(t+{}) {:.3}, drop {:.3} bits",
            sharp.first().unwrap().0, first, sharp.last().unwrap().0, last, first - last)
    };
    suite.note(format!("[sharpening] {}", sharp_verdict));

    // ---- 2. the rung sweep ----
    //
    // Only comparable between arms whose rung coverage is one. Where the tree
    // stops short of its depth cap the fastest bands are configured and never
    // read, and the sweep then varies which bands are used instead of how many
    // -- which is a different experiment wearing this one's label.
    for rungs in [2usize, 3, 4, 6] {
        let mut c = base.clone();
        c.rungs = rungs;
        c.derive();
        let label = format!("rungs-{}", rungs);
        let o = run_one(&label, c, &gcfg, &stream);
        o.metrics.csv_window(&label, &mut suite.csv);
        let (l, la) = window_mean(&o.metrics.window);
        let (p, pa) = window_mean(&o.metrics.window_product);
        suite.note(format!(
            "[window] rungs={} plateau {} | {:.3} bits/ev | depth {} | rung \
             coverage {:.2} | Latin {:.3}/{:.3} | product {:.3}/{:.3}",
            rungs,
            plateau_width(&o.metrics),
            o.metrics.bits_per_event(),
            o.model.tree.realised_depth(),
            o.model.tree.rung_coverage(),
            l,
            la,
            p,
            pa
        ));
        for (s, b) in o.metrics.window.iter() {
            suite.summary.push(format!(
                "    rungs={} sep={:<4} n={:<5} bits={:.3} acc={:.3}",
                rungs, s, b.n, b.mean(), b.accuracy()
            ));
        }
    }

    // ---- 3. the two conjunctions, which is what the whole build is for ----
    let (lat_bits, lat_acc) = window_mean(&o_full.metrics.window);
    let (prod_bits, prod_acc) = window_mean(&o_full.metrics.window_product);
    suite.note(format!(
        "[conjunction] Latin (not linearly representable) {:.3} bits acc {:.3} | \
         product (representable) {:.3} bits acc {:.3}",
        lat_bits, lat_acc, prod_bits, prod_acc
    ));

    // ---- 4. the split criterion ----
    //
    // Surprise is swept rather than tried at one point, because abandoning a
    // criterion on a single value of its threshold is not a measurement.
    let mut split_arms: Vec<(String, Config)> = Vec::new();
    let mut d0 = base.clone();
    d0.split_rule = SplitRule::Dispersion;
    split_arms.push(("dispersion".into(), d0));
    for bits in [4.0f64, 8.0, 12.0, 16.0] {
        let mut c = base.clone();
        c.split_rule = SplitRule::Surprise;
        c.split_bits = bits;
        split_arms.push((format!("surprise@{}b", bits), c));
    }
    for bits in [8.0f64, 12.0] {
        let mut c = base.clone();
        c.split_rule = SplitRule::Hybrid;
        c.split_bits = bits;
        split_arms.push((format!("hybrid@{}b", bits), c));
    }
    for (name, c) in split_arms {
        let o = run_one(&name, c, &gcfg, &stream);
        let (l, la) = window_mean(&o.metrics.window);
        let (p, pa) = window_mean(&o.metrics.window_product);
        let (pp2, _) = o.metrics.pair_purity();
        suite.note(format!(
            "[split] {:<14} {:.3} bits/ev | regime {:.3} | pair {:.3} | Latin \
             {:.3}/{:.3} | product {:.3}/{:.3} | {} leaves",
            name,
            o.metrics.bits_per_event(),
            o.metrics.leaf_purity(),
            pp2,
            l,
            la,
            p,
            pa,
            o.model.tree.leaves()
        ));
    }

    // ---- 5. binding: the falsification test for the "combine" half ----
    let mut nb = base.clone();
    nb.use_binding = false;
    let o_nb = run_one("no-binding", nb, &gcfg, &stream);
    let (nb_lat, nb_lat_acc) = window_mean(&o_nb.metrics.window);
    suite.note(format!(
        "[binding] on: Latin {:.3} bits acc {:.3} | off: Latin {:.3} bits acc \
         {:.3}  (a Latin square is linear in the tensor features of the two \
         cues and not in their sum, so this is where it should show)",
        lat_bits, lat_acc, nb_lat, nb_lat_acc
    ));

    // ---- 6. the two self channels, separately ----
    let mut no_cov = base.clone();
    no_cov.feedback_covert = false;
    let o_no_cov = run_one("no-covert", no_cov, &gcfg, &stream);
    let mut no_ov = base.clone();
    no_ov.feedback_overt = false;
    let o_no_ov = run_one("no-overt", no_ov, &gcfg, &stream);
    let fastest_read = o_full
        .model
        .tree
        .rung_visits
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0)
        .map(|(k, _)| k)
        .min()
        .unwrap_or(usize::MAX);
    let live = fastest_read <= base.self_max_rung;
    suite.note(format!(
        "[poisoning] both on {:.3} | inner speech off {:.3} | spoken off {:.3} \
         bits/ev; self writes rung {}, descent reaches rung {} -- channel {}",
        o_full.metrics.bits_per_event(),
        o_no_cov.metrics.bits_per_event(),
        o_no_ov.metrics.bits_per_event(),
        base.self_max_rung,
        fastest_read,
        if live { "live" } else { "NEVER READ, control is vacuous" }
    ));

    // ---- 7. the payload chain, which was silently in its linear regime ----
    for w_init in [0.05f32, 0.5, 1.5, 3.0] {
        let mut c = base.clone();
        c.w_init = w_init;
        let label = format!("w-init-{}", w_init);
        let o = run_one(&label, c, &gcfg, &stream);
        let (l, la) = window_mean(&o.metrics.window);
        let (p, pa) = window_mean(&o.metrics.window_product);
        suite.note(format!(
            "[chain] w_init {:<5} {:.3} bits/event, Latin {:.3}/{:.3}, product \
             {:.3}/{:.3}",
            w_init,
            o.metrics.bits_per_event(),
            l,
            la,
            p,
            pa
        ));
    }

    // ---- 8. ablations, still few and each aimed at one claim ----
    let mut a1 = base.clone();
    a1.no_readout = true;
    let o_a1 = run_one("no-readout", a1, &gcfg, &stream);

    let mut a2 = base.clone();
    a2.feedback_write = false;
    a2.no_eligibility = true;
    let o_a2 = run_one("no-gap-credit", a2, &gcfg, &stream);

    let mut a3 = base.clone();
    a3.particles = 1;
    let o_a3 = run_one("one-particle", a3, &gcfg, &stream);
    for (name, o) in
        [("no readout", &o_a1), ("no gap credit", &o_a2), ("one particle", &o_a3)]
    {
        let (l, la) = window_mean(&o.metrics.window);
        let (p, pa) = window_mean(&o.metrics.window_product);
        suite.note(format!(
            "[ablation] {:<14} {:.3} bits/ev | Latin {:.3}/{:.3} | product \
             {:.3}/{:.3} | pair purity {:.3}",
            name,
            o.metrics.bits_per_event(),
            l,
            la,
            p,
            pa,
            o.metrics.pair_purity().0
        ));
    }

    // ---- 5. the baseline that matters ----
    let like = baseline::run(&stream, 4, false);
    let upper = baseline::run(&stream, 4, true);
    suite.note(format!(
        "[baseline] PPM-C order 4, same stream (silence is a symbol): \
         {:.3} bits/event, top-1 {:.3}, second-order {:.3} bits",
        like.ppm.bits_per_event(),
        like.ppm.accuracy(),
        like.second.mean()
    ));
    suite.note(format!(
        "[baseline] PPM-C order 4, silence removed (UPPER BOUND, it is handed \
         the segmentation): {:.3} bits/event, second-order {:.3} bits",
        upper.ppm.bits_per_event(),
        upper.second.mean()
    ));
    let ppm_second = like.second;
    let ours_second: f64 = {
        let mut n = 0u64;
        let mut s = 0.0;
        for (_, b) in o_full.metrics.window.iter() {
            n += b.n;
            s += b.sum;
        }
        if n == 0 {
            0.0
        } else {
            s / n as f64
        }
    };
    suite.note(format!(
        "[verdict]  Latin items: ours {:.3} bits vs like-for-like PPM {:.3} bits  \
         ({})",
        ours_second,
        ppm_second.mean(),
        if ours_second < ppm_second.mean() { "ahead" } else { "behind" }
    ));
    suite.note(format!("[evidence] {}", o_full.metrics.evidence.verdict()));
    suite.note(format!(
        "[onsets]   idea at t+{:.2} ({} events), speech at t+{:.2} ({})",
        o_full.metrics.idea_onset.mean(),
        o_full.metrics.idea_onset.n,
        o_full.metrics.speech_onset.mean(),
        o_full.metrics.speech_onset.n
    ));
    let (pp, per_leaf) = o_full.metrics.pair_purity();
    suite.note(format!(
        "[address]  regime purity {:.3} (chance {:.3}) | cue-pair purity {:.3} \
         (chance {:.3}); tree {} nodes, depth {}, {} leaves",
        o_full.metrics.leaf_purity(),
        1.0 / gcfg.domains as f64,
        pp,
        if per_leaf > 0.0 { 1.0 / per_leaf } else { 0.0 },
        o_full.model.tree.nodes(),
        o_full.model.tree.realised_depth(),
        o_full.model.tree.leaves()
    ));
    suite.note(
        "[diagnosis] the square is additive modulo m, so the target is not a \
         linear function of the superposed cue embeddings; a linear readout \
         cannot represent it and the only route is for the address to isolate \
         the pair. Read the two purities above in that order."
            .to_string(),
    );

    // ---- 6. mode B, the adversarial bound ----
    let mut gb = gcfg.clone();
    gb.mode = Mode::B;
    let stream_b = build_stream(&gb, ticks, min_per_sep, false);
    let o_b = run_one("mode-b", base.clone(), &gb, &stream_b);
    suite.note(format!(
        "[mode B]   entities shared across regimes: {:.3} bits/event (mode A {:.3})",
        o_b.metrics.bits_per_event(),
        o_full.metrics.bits_per_event()
    ));

    suite
}
