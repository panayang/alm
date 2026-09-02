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
use crate::config::Config;
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
    suite.note(format!(
        "[sanity] L=1  bits/event {:.3}  top-1 {:.3}  nodes {}  depth {}",
        o_flat.metrics.bits_per_event(),
        o_flat.metrics.all_events.accuracy(),
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

    // ---- 2. the rung sweep: does the plateau widen with the number of bands? ----
    for rungs in [2usize, 4, 6] {
        let mut c = base.clone();
        c.rungs = rungs;
        c.derive();
        let label = format!("rungs-{}", rungs);
        let o = run_one(&label, c, &gcfg, &stream);
        o.metrics.csv_window(&label, &mut suite.csv);
        suite.note(format!(
            "[window] rungs={}  plateau width {}  bits/event {:.3}  depth {}",
            rungs,
            plateau_width(&o.metrics),
            o.metrics.bits_per_event(),
            o.model.tree.realised_depth()
        ));
        for (s, b) in o.metrics.window.iter() {
            suite.summary.push(format!(
                "    rungs={} sep={:<4} n={:<5} bits={:.3} acc={:.3}",
                rungs, s, b.n, b.mean(), b.accuracy()
            ));
        }
    }

    // ---- 3. self-poisoning control ----
    let mut nofb = base.clone();
    nofb.feedback_overt = false;
    let o_nofb = run_one("no-feedback", nofb, &gcfg, &stream);
    suite.note(format!(
        "[poisoning] feedback on {:.3} bits/event | off {:.3} bits/event",
        o_full.metrics.bits_per_event(),
        o_nofb.metrics.bits_per_event()
    ));

    // ---- 4. the three ablations ----
    let mut a1 = base.clone();
    a1.no_readout = true;
    let o_a1 = run_one("no-readout", a1, &gcfg, &stream);
    suite.note(format!(
        "[ablation] no readout (counts only)   {:.3} bits/event",
        o_a1.metrics.bits_per_event()
    ));

    let mut a2 = base.clone();
    a2.feedback_write = false;
    a2.no_eligibility = true;
    let o_a2 = run_one("no-gap-credit", a2, &gcfg, &stream);
    suite.note(format!(
        "[ablation] no write channel, no trace {:.3} bits/event",
        o_a2.metrics.bits_per_event()
    ));

    let mut a3 = base.clone();
    a3.particles = 1;
    let o_a3 = run_one("one-particle", a3, &gcfg, &stream);
    suite.note(format!(
        "[ablation] one particle               {:.3} bits/event",
        o_a3.metrics.bits_per_event()
    ));

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
        "[verdict]  second-order items: ours {:.3} bits vs like-for-like PPM \
         {:.3} bits  ({})",
        ours_second,
        ppm_second.mean(),
        if ours_second < ppm_second.mean() { "ahead" } else { "behind" }
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
