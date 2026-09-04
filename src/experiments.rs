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

/// Phase A: where is the monolith's capacity knee?
///
/// A multi-body extension buys capacity with addressing error. Comparing it to a
/// monolith that is not yet capacity-bound charges it the whole price of scale
/// and credits it none of the benefit -- which is what the previous ablation
/// table did, and why every routed arm read as dead weight.
///
/// So this asks the prior question. The monolith's capacity is the linear
/// separability of a fixed-width phi, so sweeping `d` sweeps capacity directly
/// at constant load. Flat in `d` means there is spare capacity, no extension can
/// pay here, and the comparison has to move to a heavier load before it means
/// anything.
///
/// Parameter counts are printed because the comparison that follows has to be
/// budget-matched: the monolith should be allowed to spend on width whatever the
/// extension spends on nodes.
pub fn capacity(ticks: usize, seed: u64) -> Suite {
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);

    let mut arms: Vec<(String, Config)> = Vec::new();
    for d in [16usize, 32, 64, 128] {
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = gcfg.vocab;
        c.d = d;
        c.bypass_graph = true;
        c.derive();
        arms.push((format!("mono d={}", d), c));
    }
    // The extension at the default width, for reference only -- it is not yet a
    // fair comparison and is not reported as one.
    let mut g = Config::local();
    g.seed = seed;
    g.vocab = gcfg.vocab;
    g.derive();
    arms.push(("graph d=64 (ref)".into(), g));

    for (name, c) in arms {
        let readout = c.feature_blocks() * c.d;
        let graph_params = if c.bypass_graph { 0 } else { c.nodes * (2 + c.shortcuts) * c.d * c.d };
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        let rows = o.model.store.occupied_rows();
        suite.note(format!(
            "[cap] {:<16} {:.3} bits/ev | Latin {:.3} | product {:.3} |              retention {:.3} | readout {} x {} = {:.2}M | graph {:.2}M",
            name,
            o.metrics.bits_per_event(),
            la,
            pa,
            ret_acc,
            rows,
            readout,
            (rows * readout) as f64 / 1e6,
            graph_params as f64 / 1e6
        ));
    }
    suite
}

/// Phase B: is the graph an addressed memory, or one nonlinear transform?
///
/// The width sweep settled that the monolith is not width-limited -- doubling
/// the readout bought nothing -- while the graph at the same readout width won
/// on every metric. But near-miss perturbation costs almost nothing, and two
/// opposite readings survive that:
///
/// * H1: the value is that *a* learned nonlinear transform sits on the state,
///   and which one hardly matters. Then this is random-feature expansion, not
///   addressing, and `nodes = 1` should match `nodes = 64`.
/// * H2: the value is addressed capacity, and a near miss is cheap because
///   neighbouring nodes on a small-world graph hold related content -- the
///   graceful degradation the design predicted.
///
/// They disagree on the shape of this curve: H1 flat, H2 rising.
///
/// `nodes = 1` is the honest monolith for this question -- one body, one place
/// for all capacity, the learned operator kept. The earlier `bypass_graph` arm
/// removed the transform and the addressing together and so could not separate
/// them.
pub fn scale(ticks: usize, seed: u64) -> Suite {
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);

    let mut arms: Vec<(String, Config)> = Vec::new();
    for n in [1usize, 4, 16, 64, 256] {
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = gcfg.vocab;
        c.nodes = n;
        c.derive();
        arms.push((format!("nodes={}", n), c));
    }
    // Budget matching from the other side: one body, given the width the
    // extension spends on nodes.
    for d in [128usize, 256] {
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = gcfg.vocab;
        c.nodes = 1;
        c.d = d;
        c.derive();
        arms.push((format!("nodes=1 d={}", d), c));
    }

    for (name, c) in arms {
        let readout = c.feature_blocks() * c.d;
        let dd = c.d;
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        let rows = o.model.store.occupied_rows();
        let edges = o.model.graph.edges();
        suite.note(format!(
            "[scale] {:<14} {:.3} bits/ev | Latin {:.3} | product {:.3} |              retention {:.3} | {} edges | readout {:.2}M | operator {:.2}M |              total {:.2}M",
            name,
            o.metrics.bits_per_event(),
            la,
            pa,
            ret_acc,
            edges,
            (rows * readout) as f64 / 1e6,
            (edges * dd * dd) as f64 / 1e6,
            (rows * readout + edges * dd * dd) as f64 / 1e6
        ));
    }
    suite
}

/// Phase C: is the nodes curve flattening because the addresses are enough, or
/// because they have started to crowd?
///
/// The two are indistinguishable at one load and make opposite predictions
/// across loads. If the addresses are merely sufficient, raising the load moves
/// the knee right and the curve keeps its shape. If they are crowding, the
/// benefit of more addresses *shrinks* as the load grows, because each new
/// address lands closer to its neighbours -- the named weakness of this whole
/// family, and the thing that would argue for replacing hard argmax routing
/// with a soft top-k mixture that has no cliff to fall off.
///
/// Load is raised by adding domains, with the tick budget scaled to match so
/// that exposure per fact is held constant. Anything else confounds "more to
/// remember" with "less chance to learn it".
pub fn load_sweep(base_ticks: usize, seed: u64) -> Suite {
    let mut suite = Suite::new();
    for domains in [12usize, 36] {
        let mut gcfg = GenConfig::fast();
        gcfg.seed = seed ^ 0xA11CE;
        gcfg.domains = domains;
        let ticks = base_ticks * domains / 12;
        let stream = build_stream(&gcfg, ticks, 20, false);
        // Node count, and -- at the heavy load only -- readout width at a fixed
        // node count. The nodes curve saturates at the same knee under both
        // loads while the ceiling falls, so the binding constraint is not the
        // number of addresses. The readout is the half of the architecture that
        // was never extended: one global table, all rows in one phi. Widening it
        // separates "the readout is out of capacity" from "the readout is out of
        // room to keep rows apart", and only the second argues for banking it.
        let widths: &[usize] = if domains > 12 { &[64, 128] } else { &[64] };
        for &wd in widths {
        for n in [1usize, 16, 64, 256] {
            if wd != 64 && n != 64 {
                continue;
            }
            let mut c = Config::local();
            c.seed = seed;
            c.vocab = gcfg.vocab;
            c.nodes = n;
            c.d = wd;
            c.derive();
            let o = run_one("load", c, &gcfg, &stream);
            let (_, la) = window_mean(&o.metrics.window);
            let (_, pa) = window_mean(&o.metrics.window_product);
            let ret: (u64, u64) =
                o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
            let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
            suite.note(format!(
                "[load] domains={:<3} ticks={:<7} nodes={:<4} d={:<4} {:.3} bits/ev | Latin {:.3} | product {:.3} | retention {:.3} | {} rows",
                domains,
                ticks,
                n,
                wd,
                o.metrics.bits_per_event(),
                la,
                pa,
                ret_acc,
                o.model.store.occupied_rows()
            ));
        }
        }
    }
    suite
}

/// Widening `d` at heavy load recovered more than quadrupling the nodes did.
/// But `d` is not a readout knob: it widens every row *and* squares up every
/// operator matrix, so that arm cannot say which half paid.
///
/// This separates them by holding the operator budget fixed and moving only the
/// readout. `nodes=16, d=128` has 59 edges x 128^2 = 0.97M operator parameters,
/// against `nodes=64, d=64`'s 250 x 64^2 = 1.02M -- near enough the same -- while
/// the readout doubles. If the recovery survives, it was readout width; if it
/// falls back, it was operator capacity.
///
/// The low-load arm is the other missing cell: width was only ever swept on the
/// monolith, whose bottleneck was the absence of addressing, so it could not
/// show a width effect even if one existed.
pub fn width(base_ticks: usize, seed: u64) -> Suite {
    let mut suite = Suite::new();
    for (domains, arms) in [
        (12usize, vec![(64usize, 64usize)]),
        (36, vec![(64, 64)]),
    ] {
        let mut gcfg = GenConfig::fast();
        gcfg.seed = seed ^ 0xA11CE;
        gcfg.domains = domains;
        let ticks = base_ticks * domains / 12;
        let stream = build_stream(&gcfg, ticks, 20, false);
        for (n, dd) in arms {
            let mut c = Config::local();
            c.seed = seed;
            c.vocab = gcfg.vocab;
            c.nodes = n;
            c.d = dd;
            c.derive();
            let rw = c.feature_blocks() * c.d;
            let o = run_one("width", c, &gcfg, &stream);
            let (_, la) = window_mean(&o.metrics.window);
            let (_, pa) = window_mean(&o.metrics.window_product);
            let ret: (u64, u64) =
                o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
            let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
            let rows = o.model.store.occupied_rows();
            let edges = o.model.graph.edges();
            let (ls, ln, lf, lo) = o.metrics.address_consistency_full(0);
            let (ps, pn, pf, po) = o.metrics.address_consistency_full(1);
            suite.note(format!(
                "[width] domains={:<3} nodes={:<4} d={:<4} {:.3} bits/ev | Latin {:.3} |                  product {:.3} | retention {:.3} | readout {:.2}M | operator {:.2}M |                  addr-consistency latin {:.3} over {:.1} nodes ({} facts, {:.1} occ each),                  product {:.3} over {:.1} nodes ({} facts, {:.1} occ each)",
                domains, n, dd,
                o.metrics.bits_per_event(), la, pa, ret_acc,
                (rows * rw) as f64 / 1e6,
                (edges * dd * dd) as f64 / 1e6,
                ls, ln, lf, lo, ps, pn, pf, po
            ));
        }
    }
    suite
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

    // The near-miss curve: the claim the whole design rests on. Taking the
    // runner-up edge on every hop is what a crowded address space does to a
    // competition. In a table that is a cliff, because a neighbouring address
    // holds an unrelated candidate set. In an operator set it should be a small
    // perturbation the next tick can correct.
    for r in [1usize, 2] {
        let mut c = base.clone();
        c.route_perturb = r;
        arms.push((format!("near-miss {}", r), c));
    }

    // The sharp form of the near-miss control: any edge, not the neighbouring
    // one. Only meaningful as a pair with the near-miss arms above.
    let mut rr = base.clone();
    rr.route_random = true;
    arms.push(("route random".into(), rr));

    // The single body: no routing, no hop, no operator write. The suite had no
    // such arm at all, which is why the routed arms read as dead weight.
    let mut mo = base.clone();
    mo.bypass_graph = true;
    arms.push(("monolith".into(), mo));

    // Does the anchor do the converging?
    let mut na = base.clone();
    na.anchor = 0.0;
    arms.push(("anchor off".into(), na));

    // Is binding still what makes the conjunction learnable at all.
    let mut nb = base.clone();
    nb.use_binding = false;
    nb.bind_mode = BindMode::Off;
    arms.push(("bind off".into(), nb));

    // The residual scale. 1.5 came from a sweep later shown to be an artefact
    // of a gradient bug, and at 1.5 the hop overwrites the state rather than
    // transforming it. These stay inside the range where |tanh(Wp)| < |p|, so
    // what is being swept is how much of the state a hop may rewrite -- not
    // whether the walk degenerates into a random projection.
    for w in [0.3f32, 0.5] {
        let mut c = base.clone();
        c.w_init = w;
        arms.push((format!("w_init={}", w), c));
    }

    // Gap-time reads left eligibility traces that nothing ever credited. This
    // is the first run in which that channel exists, so it is its own arm.
    let mut el = base.clone();
    el.no_eligibility = false;
    arms.push(("eligibility on".into(), el));

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
             retention {:.3} | hop slope {:+.4} (d {:+.3}, {} pts) | {} rows |              clamp {:.2}",
            name,
            o.metrics.bits_per_event(),
            la,
            pa,
            ret_acc,
            slope,
            delta,
            npts,
            o.model.store.occupied_rows(),
            o.model.graph.clamp_rate()
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
