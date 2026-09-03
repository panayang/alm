//! Experiment drivers.
//!
//! The generator self-tests run before any model is built and their
//! load-bearing properties are asserted, so a flat curve can never be blamed on
//! the mechanism before the source has been cleared.
//!
//! Arms are few and each is aimed at one claim. Everything the old suite swept
//! -- split criteria, tree depth, rung counts, prototype thresholds -- is gone
//! because the machinery it swept is gone: depth cost more than it bought on
//! every axis, and the reason was that a placed prototype had to win a
//! competition it was in no position to win.

use crate::baseline;
use crate::config::{BindMode, Config};
use crate::gen::{GenConfig, Generator, Stream};
use crate::gencheck;
use crate::metrics::{run, Bucket, Metrics};
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
pub fn window_mean(w: &[(u32, Bucket)]) -> (f64, f64) {
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

/// Slope of accuracy against hops taken, over the buckets that have data.
///
/// This is the architecture's reason to exist: a response that keeps walking
/// through the gap should answer better than one that has just started. Flat or
/// negative means the gap is doing nothing and the model is a lookup after all.
pub fn hop_slope(m: &Metrics) -> (f64, f64, usize) {
    let pts: Vec<(f64, f64)> = m
        .by_hops
        .iter()
        .enumerate()
        .filter(|(_, b)| b.n > 50)
        .map(|(h, b)| (h as f64, b.accuracy()))
        .collect();
    if pts.len() < 3 {
        return (0.0, 0.0, pts.len());
    }
    let n = pts.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for &(x, y) in pts.iter() {
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let den = n * sxx - sx * sx;
    let slope = if den.abs() < 1e-12 { 0.0 } else { (n * sxy - sx * sy) / den };
    let first = pts.first().unwrap().1;
    let last = pts.last().unwrap().1;
    (slope, last - first, pts.len())
}

pub struct Suite {
    pub csv: String,
    pub summary: Vec<String>,
}

impl Suite {
    fn new() -> Self {
        Suite { csv: String::from("run,metric,x,n,value,stderr,accuracy\n"), summary: Vec::new() }
    }
    fn note(&mut self, s: String) {
        println!("{}", s);
        self.summary.push(s);
    }
}

/// The screening suite.
///
/// Four questions, in the order in which they can invalidate each other:
///
/// 1. Does the response unfolding pay? Accuracy against hops taken.
/// 2. Is the walk driven, or does it coast? Ablate the write channel -- the one
///    thing that changes during a gap. If nothing moves, "the update is part of
///    the context" was decoration and the trajectory is not a process.
/// 3. Does retention improve now that nothing has to win a competition?
/// 4. Does the walk diffuse? Visit entropy against hops.
pub fn screen(ticks: usize, seed: u64, wide: bool) -> Suite {
    let mut suite = Suite::new();

    let mut gcfg = if wide { GenConfig::local() } else { GenConfig::fast() };
    gcfg.seed = seed ^ 0xA11CE;
    println!("stream: {} ticks", ticks);
    let stream = build_stream(&gcfg, ticks, 20, true);

    let mut base = Config::local();
    base.seed = seed;
    base.vocab = gcfg.vocab;
    base.derive();

    let mut arms: Vec<(String, Config)> = vec![("full".into(), base.clone())];

    // The decisive one: is the trajectory driven?
    let mut wo = base.clone();
    wo.feedback_write = false;
    arms.push(("write off".into(), wo));

    // Does the walk do anything, or is the gain the broad prior plus the
    // readout? These produce the same aggregate numbers and only this separates
    // them.
    let mut wg = base.clone();
    wg.walk_during_gap = false;
    arms.push(("walk off".into(), wg));

    // Is binding still what makes the conjunction learnable at all.
    let mut nb = base.clone();
    nb.use_binding = false;
    nb.bind_mode = BindMode::Off;
    arms.push(("bind off".into(), nb));

    // Is the shared readout doing the work.
    let mut nr = base.clone();
    nr.no_readout = true;
    arms.push(("no readout".into(), nr));

    if wide {
        let mut nc = base.clone();
        nc.feedback_covert = false;
        arms.push(("covert off".into(), nc));

        let mut no = base.clone();
        no.feedback_overt = false;
        arms.push(("overt off".into(), no));

        // Memory size replaces the depth sweep: more nodes is a finer prior at
        // the same level of commitment, which is the comparison depth was meant
        // to make and could not.
        for n in [16usize, 64, 128] {
            let mut c = base.clone();
            c.nodes = n;
            arms.push((format!("nodes={}", n), c));
        }
    }

    for (name, c) in arms {
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        let (slope, delta, npts) = hop_slope(&o.metrics);
        suite.note(format!(
            "[screen] {:<12} {:.3} bits/ev | Latin {:.3} | product {:.3} | \
             retention {:.3} | hop slope {:+.4} (d {:+.3}, {} pts) | {} live nodes",
            name,
            o.metrics.bits_per_event(),
            la,
            pa,
            ret_acc,
            slope,
            delta,
            npts,
            o.model.store.live_nodes()
        ));
        if name == "full" {
            o.metrics.print(&o.model, "full");
            o.metrics.csv_window("full", &mut suite.csv);
            o.metrics.csv_sharpening("full", &mut suite.csv);
        }
    }

    // The baseline that matters.
    let like = baseline::run(&stream, 4, false);
    let upper = baseline::run(&stream, 4, true);
    suite.note(format!(
        "[baseline] PPM-C order 4, same stream: {:.3} bits/ev, second-order {:.3} \
         bits | silence removed (UPPER BOUND, handed the segmentation): {:.3} \
         bits/ev, second-order {:.3}",
        like.ppm.bits_per_event(),
        like.second.mean(),
        upper.ppm.bits_per_event(),
        upper.second.mean()
    ));

    suite
}
