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
        // Undefined, not flat. With the gap walk off every charged event lands
        // in hop bucket 0, and returning 0.0 put a refusal to compute in the
        // same column as a measurement.
        return (f64::NAN, f64::NAN, pts.len());
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
        // Shortcuts colliding with a ring neighbour or the node itself are
        // dropped at construction, so the formula overcounts -- 4 edges against
        // 2 at nodes=1, which is exactly where the budget-matching argument was
        // being made. Ask the graph.
        let dd = c.d;
        let bypass = c.bypass_graph;
        let o = run_one(&name, c, &gcfg, &stream);
        let graph_params = if bypass { 0 } else { o.model.graph.edges() * dd * dd };
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
        (12usize, vec![(1usize, 64usize), (64, 64)]),
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
            let (pb, pa) = window_mean(&o.metrics.window_product);
            // Codelength at the answer tick only. `bits_per_event` charges every
            // observed token, cues included -- next-symbol accounting on a design
            // that never claimed to predict next symbols. On this source a cue is
            // near-uniform over the domains' entities, so that figure is mostly
            // the price of not doing something the model is not asked to do.
            let (lb, _) = window_mean(&o.metrics.window);
            let ret: (u64, u64) =
                o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
            let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
            let rows = o.model.store.occupied_rows();
            let edges = o.model.graph.edges();
            let (ls, ln, lf, lo) = o.metrics.address_consistency_full(0);
            let (ps, pn, pf, po) = o.metrics.address_consistency_full(1);
            suite.note(format!(
                "[width] domains={:<3} nodes={:<4} d={:<4} ALL-TOKEN {:.3} bits | ANSWER-ONLY latin {:.3} product {:.3} bits | Latin {:.3} |                  product {:.3} | retention {:.3} | readout {:.2}M | operator {:.2}M |                  addr-consistency latin {:.3} over {:.1} nodes ({} facts, {:.1} occ each),                  product {:.3} over {:.1} nodes ({} facts, {:.1} occ each)",
                domains, n, dd,
                o.metrics.bits_per_event(), lb, pb, la, pa, ret_acc,
                (rows * rw) as f64 / 1e6,
                (edges * dd * dd) as f64 / 1e6,
                ls, ln, lf, lo, ps, pn, pf, po
            ));
        }
    }
    suite
}

/// The two questions that gate everything downstream, in one grid.
///
/// **Is the graph a memory or a hash?** Every arm is paired with the same
/// configuration whose edge transforms are frozen at initialisation. If the
/// nodes curve survives freezing, what the walk contributes is the diversity of
/// fixed random transforms the state is routed through, and the operator write
/// -- along with the question of what it should be written toward -- is moot.
///
/// **Is a near miss cheap because neighbours hold related content?** Taking the
/// runner-up edge and taking any edge at all are compared against the same
/// argmax baseline. Cheap-near-miss with expensive-random is the graceful
/// degradation the design claims; both cheap means routing carries nothing.
///
/// Answer-tick codelength is reported alongside the all-token figure, because
/// the latter charges cues the model was never asked to predict.
pub fn mechanism(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);

    let base = |nodes: usize| {
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = gcfg.vocab;
        c.nodes = nodes;
        c
    };

    let mut arms: Vec<(String, Config)> = Vec::new();
    for n in [1usize, 16, 64, 256] {
        for frozen in [false, true] {
            let mut c = base(n);
            c.freeze_operator = frozen;
            c.derive();
            arms.push((
                format!("nodes={} {}", n, if frozen { "FROZEN" } else { "learned" }),
                c,
            ));
        }
    }
    // Routing, all at nodes=64 so the three share one baseline.
    let mut c = base(64);
    c.route_perturb = 1;
    c.derive();
    arms.push(("nodes=64 near-miss 1".into(), c));
    let mut c = base(64);
    c.route_random = true;
    c.derive();
    arms.push(("nodes=64 route RANDOM".into(), c));

    for (i, (name, c)) in arms.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let o = run_one(&name, c, &gcfg, &stream);
        let (lb, la) = window_mean(&o.metrics.window);
        let (pb, pa) = window_mean(&o.metrics.window_product);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        suite.note(format!(
            "[mech] {:<22} Latin {:.3} ({:.2} bits) | product {:.3} ({:.2} bits) |              retention {:.3} | all-token {:.3} bits",
            name, la, lb, pa, pb, ret_acc, o.metrics.bits_per_event()
        ) + &comp_tail(&o));
    }
    suite
}

/// Does routing on content instead of on the trajectory make the address an
/// address?
///
/// Two steps, and the second is only worth reading if the first passes.
///
/// 1. **Address consistency.** With the query taken from the bound traces, one
///    fact should stop scattering. The baseline is 0.146 of visits on the
///    dominant node, over 12.1 distinct nodes per fact.
/// 2. **argmax against random.** Under `State` these were indistinguishable
///    (Latin 0.209 against 0.219), which is what said the addressing carried
///    nothing. If content routing is real, argmax has to separate from random
///    here. Each query mode therefore carries its own random control; comparing
///    across modes would confound the repair with the arm.
pub fn route(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::RouteQuery;
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);

    let mut arms: Vec<(String, Config)> = Vec::new();
    for (nodes, q, qn, entry) in [
        (64usize, RouteQuery::State, "state", false),
        (64, RouteQuery::Bound, "bound", false),
        (64, RouteQuery::State, "state", true),
        (64, RouteQuery::Bound, "bound", true),
        (256, RouteQuery::Bound, "bound", false),
        (256, RouteQuery::Bound, "bound", true),
    ] {
        for rnd in [false, true] {
            let mut c = Config::local();
            c.seed = seed;
            c.vocab = gcfg.vocab;
            c.nodes = nodes;
            c.route_query = q;
            c.route_random = rnd;
            c.read_entry_by_content = entry;
            c.derive();
            arms.push((
                format!(
                    "n={} q={}{} {}",
                    nodes,
                    qn,
                    if entry { "+ENTRY" } else { "" },
                    if rnd { "RANDOM" } else { "argmax" }
                ),
                c,
            ));
        }
    }

    for (i, (name, c)) in arms.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let o = run_one(&name, c, &gcfg, &stream);
        let (lb, la) = window_mean(&o.metrics.window);
        let (pb, pa) = window_mean(&o.metrics.window_product);
        let (ls, ln, _, _) = o.metrics.address_consistency_full(0);
        let (ps, pn, _, _) = o.metrics.address_consistency_full(1);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        suite.note(format!(
            "[route] {:<24} consistency latin {:.3}/{:.1}n product {:.3}/{:.1}n |              Latin {:.3} ({:.2}b) | product {:.3} ({:.2}b) | retention {:.3} |              answer {:.3} bits over {} events",
            name, ls, ln, ps, pn, la, lb, pa, pb, ret_acc,
            o.metrics.answer.mean(), o.metrics.answer.n
        ) + &comp_tail(&o));
    }
    suite
}

/// The four decisive cells, replicated across seeds.
///
/// Content routing beat its matched random control by +0.015 Latin and +0.02
/// product at both node counts, which is exactly the size at which one seed
/// decides nothing. Stream and model seed move together, so this replicates the
/// claim rather than just the initialisation.
pub fn seeds(ticks: usize, seed0: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::RouteQuery;
    let mut suite = Suite::new();
    let mut jobs: Vec<(u64, usize, bool)> = Vec::new();
    for s in 0..4u64 {
        for nodes in [64usize, 256] {
            for rnd in [false, true] {
                jobs.push((seed0.wrapping_add(s.wrapping_mul(0x9E37_79B9)), nodes, rnd));
            }
        }
    }
    let mut last_seed: Option<u64> = None;
    let mut stream = None;
    let mut gcfg = GenConfig::fast();
    for (i, (sd, nodes, rnd)) in jobs.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        if last_seed != Some(sd) {
            gcfg = GenConfig::fast();
            gcfg.seed = sd ^ 0xA11CE;
            stream = Some(build_stream(&gcfg, ticks, 20, false));
            last_seed = Some(sd);
        }
        let st = stream.as_ref().unwrap();
        let mut c = Config::local();
        c.seed = sd;
        c.vocab = gcfg.vocab;
        c.nodes = nodes;
        c.route_query = RouteQuery::Bound;
        c.read_entry_by_content = true;
        c.route_random = rnd;
        c.derive();
        let name = format!(
            "seed={:#x} n={} {}",
            sd, nodes, if rnd { "RANDOM" } else { "content" }
        );
        let o = run_one(&name, c, &gcfg, st);
        let (lb, la) = window_mean(&o.metrics.window);
        let (pb, pa) = window_mean(&o.metrics.window_product);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        suite.note(format!(
            "[seed] {:<28} Latin {:.4} ({:.3}b) | product {:.4} ({:.3}b) |              retention {:.4} | answer {:.4} bits",
            name, la, lb, pa, pb, ret_acc, o.metrics.answer.mean()
        ) + &comp_tail(&o));
    }
    suite
}

/// (a) Does gap-time iteration pay now that the address holds still, and
/// (b) what shape has the write pushed the transforms into.
///
/// Before content entry, every gap tick jumped to a new node, so "iterating with
/// memory" was a random walk and the hop slope sat at +0.006 while switching the
/// gap walk off scored better. With the address pinned the state evolves under
/// *one* operator across the gap, which is the first time the loop has had the
/// shape the founding brief asked for. If the slope is still flat, depth-from-
/// time fails on its merits rather than on a wiring mistake.
///
/// The drift figures answer the other question in the same runs: starvation and
/// target degeneration both predict that freezing costs nothing, and they are
/// told apart by how far the matrices moved and whether they moved toward rank
/// one.
pub fn depth(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::RouteQuery;
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);

    let mut arms: Vec<(String, Config)> = Vec::new();
    for (nodes, entry, gap, label) in [
        (64usize, false, true, "old: state, wander, gap-walk"),
        (64, true, true, "fixed: bound+ENTRY, gap-walk"),
        (64, true, false, "fixed: bound+ENTRY, NO gap-walk"),
        (256, false, true, "old: state, wander, gap-walk"),
        (256, true, true, "fixed: bound+ENTRY, gap-walk"),
        (256, true, false, "fixed: bound+ENTRY, NO gap-walk"),
    ] {
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = gcfg.vocab;
        c.nodes = nodes;
        c.walk_during_gap = gap;
        if entry {
            c.route_query = RouteQuery::Bound;
            c.read_entry_by_content = true;
        }
        c.derive();
        arms.push((format!("n={} {}", nodes, label), c));
    }

    for (i, (name, c)) in arms.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let cc = c.clone();
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let (slope, delta, npts) = hop_slope(&o.metrics);
        let (drift, share) = o.model.graph.operator_drift(&cc);
        suite.note(format!(
            "[depth] {:<36} hop slope {:+.4} (d {:+.3}, {} pts) | Latin {:.3} |              product {:.3} | answer {:.3} bits | operator drift {:.4},              top-direction share {:.3}",
            name, slope, delta, npts, la, pa, o.metrics.answer.mean(), drift, share
        ) + &comp_tail(&o));
    }
    suite
}

/// Does the operator write start working once the address isolates content?
///
/// The drift figures said the write is neither starved nor degenerate: the
/// matrices move 0.31--0.65 of their initial norm while their spectra stay at
/// the random baseline, and that movement scales as the square root of the
/// updates per edge (2.10x for 3.9x the updates), which is what uncorrelated
/// rank-one accumulation looks like. So the write is large, structureless and
/// free to discard -- the signature of many unrelated facts writing the same
/// edge.
///
/// That predicts the value of learning tracks address consistency rather than
/// data volume. Content entry raised consistency, so the freeze contrast is
/// re-run on top of it, crossed with the gap walk since switching that off is
/// currently the better configuration and it changes how often each edge is
/// written.
pub fn freeze2(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::RouteQuery;
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);

    let mut arms: Vec<(String, Config)> = Vec::new();
    for nodes in [64usize, 256] {
        for gap in [true, false] {
            for frozen in [false, true] {
                let mut c = Config::local();
                c.seed = seed;
                c.vocab = gcfg.vocab;
                c.nodes = nodes;
                c.route_query = RouteQuery::Bound;
                c.read_entry_by_content = true;
                c.walk_during_gap = gap;
                c.freeze_operator = frozen;
                c.derive();
                arms.push((
                    format!(
                        "n={} {} {}",
                        nodes,
                        if gap { "gap-walk" } else { "no-gap" },
                        if frozen { "FROZEN" } else { "learned" }
                    ),
                    c,
                ));
            }
        }
    }

    for (i, (name, c)) in arms.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let cc = c.clone();
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let (ps, pn, _, _) = o.metrics.address_consistency_full(1);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        let (drift, share) = o.model.graph.operator_drift(&cc);
        suite.note(format!(
            "[freeze2] {:<26} Latin {:.4} | product {:.4} | retention {:.4} |              answer {:.4} bits | consistency {:.3}/{:.1}n | drift {:.4} share {:.3}",
            name, la, pa, ret_acc, o.metrics.answer.mean(), ps, pn, drift, share
        ) + &comp_tail(&o));
    }
    suite
}

/// The composition family, reported separately because it is the only place in
/// the source where an answer requires chaining two stored facts.
///
/// The depth question was being read off `hop_slope`, which pools every charged
/// event; composition is 14% of episodes, so the one family that needs more than
/// one hop was diluted into the 86% that do not. It has had its own metric
/// bucket all along and was never once printed.
pub fn comp_tail(o: &Outcome) -> String {
    let q = &o.metrics.composition_query;
    let g = &o.metrics.generalize;
    format!(
        " | ONE-SHOT comp-query {:.4} ({:.2}b, n={}) | held-out cell {:.4} ({:.2}b, n={})",
        q.accuracy(),
        q.mean(),
        q.n,
        g.accuracy(),
        g.mean(),
        g.n
    )
}

/// How fast may a bound trace decay?
///
/// `bind_decay` was declared at 0.5 and never referenced. Wiring it at that value
/// applies it per tick, so across an answer gap of six the trace retains 1.6% --
/// the conjunction is gone before anything reads it, and binding is the single
/// most load-bearing mechanism in the system. The knob's declared value only
/// makes sense per response, not per tick, and having never run it was never
/// calibrated.
///
/// The constraint is two-sided: the trace has to survive the gap it is read
/// across, and fade before the next episode's answer.
pub fn binddecay(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::RouteQuery;
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);
    for (i, bd) in [1.0f32, 0.99, 0.95, 0.85, 0.5].into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = gcfg.vocab;
        c.nodes = 64;
        c.route_query = RouteQuery::Bound;
        c.read_entry_by_content = true;
        c.bind_decay = bd;
        c.derive();
        let name = format!("bind_decay={}", bd);
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let (ps, pn, _, _) = o.metrics.address_consistency_full(1);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        suite.note(
            format!(
                "[bd] {:<18} Latin {:.4} | product {:.4} | retention {:.4} |                  answer {:.4} bits | consistency {:.3}/{:.1}n",
                name, la, pa, ret_acc, o.metrics.answer.mean(), ps, pn
            ) + &comp_tail(&o),
        );
    }
    suite
}

/// Is the conjunction linearly decodable from the binding at all?
///
/// The model sits at 0.145 on Latin while the prequential single-cue baseline is
/// 0.161, and 1/m is 0.167 -- so the single-cue predictor is at chance and so is
/// the model. But top-1 runs over the whole 4096-token vocabulary, and chance
/// there is 1/4096, not 1/6. Reaching 1/6 means the model reliably finds the
/// regime and its six candidate targets and then picks among them at chance:
/// the domain arrives, the conjunction does not. Meanwhile the product code,
/// also six cells, reaches 0.69.
///
/// This takes the model out of the question entirely. It builds
/// `nu(E_a (*) E_b)` straight from the embeddings, runs the same prequential
/// delta rule the readout uses, and asks what that representation alone can do.
/// If the probe succeeds where the model fails, the fault is between the bound
/// trace and the decision. If the probe fails too, `d` and the number of
/// conjunctions are the constraint and no amount of debugging will move it.
///
/// The product family is carried alongside as a positive control: same probe,
/// same code path, and it is known to be solvable.
pub fn bindprobe(ticks: usize, seed: u64) -> Suite {
    use crate::gen::Kind;
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);

    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = gcfg.vocab;
    cfg.derive();

    for &d in [64usize, 128, 256].iter() {
        let mut c = cfg.clone();
        c.d = d;
        c.derive();
        let emb = crate::embed::Embeddings::new(&c);
        // One row per token, exactly like the readout, but over the bound vector
        // alone -- no state, no bands, no graph.
        let mut rows: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
        let mut hit = [0u64; 2];
        let mut tot = [0u64; 2];
        let mut bound = vec![0.0f32; d];
        for ep in stream.episodes.iter() {
            let fam = match ep.kind {
                Kind::Second => 0usize,
                Kind::Product => 1,
                _ => continue,
            };
            if ep.cues.len() < 2 {
                continue;
            }
            crate::num::circconv(emb.row(ep.cues[0]), emb.row(ep.cues[1]), &mut bound);
            crate::num::normalize(&mut bound);

            // Predict before the answer is seen, over every row that exists.
            let mut best: Option<(f32, u32)> = None;
            let mut keys: Vec<&u32> = rows.keys().collect();
            keys.sort_unstable();
            for &t in keys {
                let sc = crate::num::dot(&rows[&t], &bound);
                match best {
                    Some((bs, bt)) if bs > sc || (bs == sc && bt < t) => {}
                    _ => best = Some((sc, t)),
                }
            }
            if let Some((_, t)) = best {
                if t as usize == ep.target {
                    hit[fam] += 1;
                }
            }
            tot[fam] += 1;

            // Delta rule against the same softmax the readout uses.
            let mut cand: Vec<u32> = rows.keys().copied().collect();
            cand.sort_unstable();
            if !cand.contains(&(ep.target as u32)) {
                cand.push(ep.target as u32);
            }
            let mut z = 0.0f32;
            let mut sc: Vec<(u32, f32)> = Vec::with_capacity(cand.len());
            for &t in cand.iter() {
                let e = rows
                    .get(&t)
                    .map(|r| crate::num::dot(r, &bound))
                    .unwrap_or(0.0)
                    .clamp(-30.0, 30.0)
                    .exp();
                z += e;
                sc.push((t, e));
            }
            z += (c.vocab - cand.len()) as f32;
            for (t, e) in sc {
                let q = e / z.max(1e-20);
                let err = if t as usize == ep.target { 1.0 - q } else { -q };
                if err.abs() < 1e-4 {
                    continue;
                }
                let r = rows.entry(t).or_insert_with(|| vec![0.0; d]);
                for i in 0..d {
                    r[i] += c.eta * err * bound[i];
                }
            }
        }
        suite.note(format!(
            "[bindprobe] d={:<4} latin {:.4} over {} items | product {:.4} over {} items",
            d,
            if tot[0] == 0 { 0.0 } else { hit[0] as f64 / tot[0] as f64 },
            tot[0],
            if tot[1] == 0 { 0.0 } else { hit[1] as f64 / tot[1] as f64 },
            tot[1]
        ));
    }
    suite
}

/// Uniform negatives against negatives drawn from the top of the distribution.
pub fn negatives(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::RouteQuery;
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);
    let mut arms: Vec<(String, Config)> = Vec::new();
    for hard in [false, true] {
        for k in [16usize, 64] {
            let mut c = Config::local();
            c.seed = seed;
            c.vocab = gcfg.vocab;
            c.nodes = 64;
            c.route_query = RouteQuery::Bound;
            c.read_entry_by_content = true;
            c.hard_negatives = hard;
            c.neg_samples = k;
            c.derive();
            arms.push((
                format!("{} k={}", if hard { "TOP" } else { "uniform" }, k),
                c,
            ));
        }
    }
    for (i, (name, c)) in arms.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        suite.note(
            format!(
                "[neg] {:<14} Latin {:.4} | product {:.4} | retention {:.4} | answer {:.4} bits",
                name, la, pa, ret_acc, o.metrics.answer.mean()
            ) + &comp_tail(&o),
        );
    }
    suite
}

/// The core grid, re-run on everything that has been repaired.
///
/// Every ablation before this one was measured through a readout that could not
/// separate the candidates inside a regime, so none of them said what they were
/// read as saying. Four axes, crossed, so the interactions are visible rather
/// than assumed:
///
/// * `freeze_operator` -- is there operator memory, now that a readout exists
///   that could show it,
/// * `route_random` -- does the address carry information, against a control
///   that randomises the entry node too rather than being pinned to a local walk,
/// * `walk_during_gap` -- does iterating through the gap pay, now that
///   composition is reachable at all,
/// * `nodes` -- what the marginal address buys.
///
/// Reported against the prequential single-cue baselines the generator prints:
/// Latin 0.161, product 0.486. Composition has no such baseline -- nothing short
/// of chaining answers it.
pub fn grid(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::RouteQuery;
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, shard == 0);

    let mut arms: Vec<(String, Config)> = Vec::new();
    for nodes in [64usize, 256] {
        for gap in [true, false] {
            for rnd in [false, true] {
                for frozen in [false, true] {
                    let mut c = Config::local();
                    c.seed = seed;
                    c.vocab = gcfg.vocab;
                    c.nodes = nodes;
                    c.route_query = RouteQuery::Bound;
                    c.read_entry_by_content = true;
                    c.route_random = rnd;
                    c.walk_during_gap = gap;
                    c.freeze_operator = frozen;
                    c.derive();
                    arms.push((
                        format!(
                            "n={} {} {} {}",
                            nodes,
                            if gap { "gap" } else { "nogap" },
                            if rnd { "RAND" } else { "content" },
                            if frozen { "FROZEN" } else { "learned" }
                        ),
                        c,
                    ));
                }
            }
        }
    }

    for (i, (name, c)) in arms.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let (ps, pn, _, _) = o.metrics.address_consistency_full(1);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        suite.note(
            format!(
                "[grid] {:<28} Latin {:.4} | product {:.4} | retention {:.4} |                  answer {:.4} bits | consist {:.3}/{:.1}n",
                name, la, pa, ret_acc, o.metrics.answer.mean(), ps, pn
            ) + &comp_tail(&o),
        );
    }
    suite
}

/// The cell the grid does not contain: is the graph worth anything at all?
///
/// Crossing freeze, routing, gap-walking and node count left sixteen arms inside
/// a 0.68-0.69 band on Latin and 0.922-0.926 on the product code -- every axis
/// inert, while the consistency instrument says content entry really does
/// concentrate a fact onto 5.9 nodes against a random control's 21.2. The
/// address works and does not matter. That makes the untested question the
/// blunt one: remove the hop entirely.
///
/// `bind off` and `no readout` ride along as the anchors that must collapse; if
/// they do not, the run says nothing about anything.
pub fn worth(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::{BindMode, RouteQuery};
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);

    let base = |seed: u64, vocab: usize| {
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = vocab;
        c.nodes = 64;
        c.route_query = RouteQuery::Bound;
        c.read_entry_by_content = true;
        c.walk_during_gap = false;
        c.freeze_operator = true;
        c
    };
    let mut arms: Vec<(String, Config)> = Vec::new();

    let mut c = base(seed, gcfg.vocab);
    c.derive();
    arms.push(("best from the grid".into(), c));

    let mut c = base(seed, gcfg.vocab);
    c.bypass_graph = true;
    c.derive();
    arms.push(("NO GRAPH at all".into(), c));

    let mut c = base(seed, gcfg.vocab);
    c.bypass_graph = true;
    c.anchor = 0.0;
    c.derive();
    arms.push(("no graph, no anchor".into(), c));

    let mut c = base(seed, gcfg.vocab);
    c.use_binding = false;
    c.bind_mode = BindMode::Off;
    c.derive();
    arms.push(("bind off (must collapse)".into(), c));

    let mut c = base(seed, gcfg.vocab);
    c.no_readout = true;
    c.derive();
    arms.push(("no readout (must collapse)".into(), c));

    let mut c = base(seed, gcfg.vocab);
    c.feedback_write = false;
    c.feedback_covert = false;
    c.feedback_overt = false;
    c.derive();
    arms.push(("all three streams off".into(), c));

    for (i, (name, c)) in arms.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        suite.note(
            format!(
                "[worth] {:<28} Latin {:.4} | product {:.4} | retention {:.4} | answer {:.4} bits",
                name, la, pa, ret_acc, o.metrics.answer.mean()
            ) + &comp_tail(&o)
                + &walk_tail(&o),
        );
    }
    suite
}

/// What is the minimal system that still works?
///
/// Removing the graph improved every metric and removing the anchor improved it
/// again; switching the three self-streams off improved composition and
/// retention. So the parts are stripped together, and then the one mechanism
/// that has never been ablated on its own -- the multi-timescale background --
/// is swept. `bind off` and `no readout` collapse, so an arm that does not is
/// saying something.
pub fn minimal(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::{BindMode, RouteQuery};
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);

    let stripped = |seed: u64, vocab: usize| {
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = vocab;
        c.nodes = 64;
        c.route_query = RouteQuery::Bound;
        c.read_entry_by_content = true;
        c.walk_during_gap = false;
        c.freeze_operator = true;
        c.bypass_graph = true;
        c.anchor = 0.0;
        c
    };
    let mut arms: Vec<(String, Config)> = Vec::new();

    let mut c = stripped(seed, gcfg.vocab);
    c.derive();
    arms.push(("no graph, no anchor".into(), c));

    let mut c = stripped(seed, gcfg.vocab);
    c.feedback_write = false;
    c.feedback_covert = false;
    c.feedback_overt = false;
    c.derive();
    arms.push(("+ no streams".into(), c));

    // The background, swept for the first time on its own.
    for r in [1usize, 2, 3, 5] {
        let mut c = stripped(seed, gcfg.vocab);
        c.feedback_write = false;
        c.feedback_covert = false;
        c.feedback_overt = false;
        c.rungs = r;
        c.derive();
        arms.push((format!("+ no streams, rungs={}", r), c));
    }

    // Binding without any background reaching the readout at all.
    let mut c = stripped(seed, gcfg.vocab);
    c.feedback_write = false;
    c.feedback_covert = false;
    c.feedback_overt = false;
    c.bind_mode = BindMode::EventLag;
    c.derive();
    arms.push(("+ no streams, event-lag binds only".into(), c));

    for (i, (name, c)) in arms.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let blocks = c.feature_blocks();
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        let ret: (u64, u64) =
            o.metrics.retention.iter().fold((0, 0), |a, b| (a.0 + b.n, a.1 + b.hits));
        let ret_acc = if ret.0 == 0 { 0.0 } else { ret.1 as f64 / ret.0 as f64 };
        suite.note(
            format!(
                "[min] {:<34} Latin {:.4} | product {:.4} | retention {:.4} |                  answer {:.4} bits | {} blocks",
                name, la, pa, ret_acc, o.metrics.answer.mean(), blocks
            ) + &comp_tail(&o)
                + &exposure_tail(&o),
        );
    }
    suite
}

/// The metrics an expansion mechanism is actually for.
///
/// Accuracy at matched parameters is the wrong test: capacity expansion does not
/// promise to be more accurate, it promises that accuracy survives accumulation
/// and that retrieval stays cheap. Those are two separate quantities and the
/// reference paper's title names both -- interference against retrieval
/// difficulty -- while a single accuracy number confounds them.
///
/// `retention` has been bucketed by the age of the departed regime all along and
/// only ever been reported pooled, so the *slope* -- how fast old content decays
/// as new content arrives -- has never been looked at. Rows scanned per answer is
/// the cost side; it is currently the whole table, for every arm, because the
/// address gates the state and not the storage.
pub fn interference_tail(o: &Outcome) -> String {
    let r = &o.metrics.retention;
    let pts: Vec<(f64, f64)> = r
        .iter()
        .enumerate()
        .filter(|(_, b)| b.n > 20)
        .map(|(i, b)| (i as f64, b.accuracy()))
        .collect();
    let slope = if pts.len() < 3 {
        f64::NAN
    } else {
        let n = pts.len() as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for &(x, y) in pts.iter() {
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let den = n * sxx - sx * sx;
        if den.abs() < 1e-12 { f64::NAN } else { (n * sxy - sx * sy) / den }
    };
    let by_age: Vec<String> = r
        .iter()
        .enumerate()
        .filter(|(_, b)| b.n > 20)
        .map(|(i, b)| format!("{}:{:.3}", i, b.accuracy()))
        .collect();
    format!(
        " | retention by age [{}] slope {:+.4} | rows scanned/answer {}",
        by_age.join(" "),
        slope,
        o.model.store.occupied_rows()
    )
}

/// Push the load, and judge on what expansion is for.
///
/// Everything measured so far reduces to "on a small simple source a simpler
/// mechanism wins", which is not news. The claim worth testing is that expansion
/// keeps working as content accumulates -- so the load goes up and the metrics
/// change: the retention *slope* against the age of a departed regime, and the
/// rows scanned per answer, alongside accuracy rather than instead of it.
///
/// Graph against no-graph at each load. If the graph is ever going to earn its
/// place it is at the top of this sweep, and if the slopes are identical there
/// too then it is not an expansion mechanism on this source at all.
pub fn loadcurve(base_ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::RouteQuery;
    let mut suite = Suite::new();
    let mut jobs: Vec<(usize, bool)> = Vec::new();
    for domains in [12usize, 36, 96] {
        for graph in [false, true] {
            jobs.push((domains, graph));
        }
    }
    let mut last: Option<usize> = None;
    let mut stream = None;
    let mut gcfg = GenConfig::fast();
    for (i, (domains, graph)) in jobs.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        if last != Some(domains) {
            gcfg = GenConfig::fast();
            gcfg.seed = seed ^ 0xA11CE;
            gcfg.domains = domains;
            // 67 tokens per regime, so the vocabulary has to grow with the load
            // or the generator's own assertion stops the run.
            gcfg.vocab = (((32 + 32 + 3) * domains) * 5 / 4).next_power_of_two().max(4096);
            stream = Some(build_stream(&gcfg, base_ticks * domains / 12, 20, false));
            last = Some(domains);
        }
        let st = stream.as_ref().unwrap();
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = gcfg.vocab;
        c.nodes = 64;
        c.route_query = RouteQuery::Bound;
        c.read_entry_by_content = true;
        c.walk_during_gap = false;
        c.freeze_operator = true;
        c.anchor = if graph { c.anchor } else { 0.0 };
        c.bypass_graph = !graph;
        c.derive();
        let name = format!("domains={} {}", domains, if graph { "GRAPH" } else { "no-graph" });
        let o = run_one(&name, c, &gcfg, st);
        let (_, la) = window_mean(&o.metrics.window);
        let (_, pa) = window_mean(&o.metrics.window_product);
        suite.note(
            format!(
                "[load2] {:<22} Latin {:.4} | product {:.4} | answer {:.4} bits",
                name, la, pa, o.metrics.answer.mean()
            ) + &comp_tail(&o)
                + &interference_tail(&o),
        );
    }
    suite
}

/// Accuracy the first time a fact is charged, against later times.
///
/// The gap between the two columns is the gap between generalising and
/// memorising, and this source cannot otherwise tell them apart.
pub fn exposure_tail(o: &Outcome) -> String {
    let names = ["latin", "product", "comp-q"];
    let mut parts = Vec::new();
    for f in 0..3 {
        let b = &o.metrics.by_exposure[f];
        parts.push(format!(
            "{} 1st {:.3}(n={}) 2nd {:.3} 3rd {:.3} 4th+ {:.3}",
            names[f],
            b[0].accuracy(),
            b[0].n,
            b[1].accuracy(),
            b[2].accuracy(),
            b[3].accuracy()
        ));
    }
    format!(" | FIRST-EXPOSURE {}", parts.join(" ; "))
}

/// The depth x silence surface, which is the measurement the design exists for.
pub fn walk_tail(o: &Outcome) -> String {
    let mut rows = Vec::new();
    for d in 1..4usize {
        let cells: Vec<String> = (0..4)
            .map(|g| {
                let b = &o.metrics.walk[d][g];
                if b.n == 0 { "  -  ".to_string() } else { format!("{:.3}", b.accuracy()) }
            })
            .collect();
        let n: u64 = (0..4).map(|g| o.metrics.walk[d][g].n).sum();
        let bits: f64 = (0..4).map(|g| o.metrics.walk[d][g].sum).sum::<f64>() / n.max(1) as f64;
        rows.push(format!("d{}[{}] n={} {:.2}b", d, cells.join(" "), n, bits));
    }
    format!(
        " | WALK by depth x gap(1,2,4,8): {} | walk-support {:.3}",
        rows.join(" ; "),
        o.metrics.walk_support.accuracy()
    )
}

/// Does bind-on-speech / unbind-on-silence actually retrieve?
///
/// Confirm the mechanism before asking anything of it. Depth 1 is a stored fact
/// and must come right in one tick of silence; if it does not, the retrieval is
/// broken and the depth axis says nothing. Then the surface: accuracy by links
/// required against ticks granted. Flat means one-pass computation; a diagonal
/// means the world's silence is buying depth.
pub fn unbindtest(ticks: usize, seed: u64, shard: usize, shards: usize) -> Suite {
    use crate::config::RouteQuery;
    let mut suite = Suite::new();
    let mut gcfg = GenConfig::fast();
    gcfg.seed = seed ^ 0xA11CE;
    let stream = build_stream(&gcfg, ticks, 20, false);
    let mut arms: Vec<(String, Config)> = Vec::new();
    // Banks are the axis now: one superposition cannot hold a continual stream,
    // and the address exists to keep each bank under its capacity.
    // Inside and outside the operating region, which is d > 2 k ln V. With
    // roughly sixteen thousand triples: d=64 needs ~3000 banks to be legal at
    // all, d=256 needs ~800, d=1024 is comfortable at 1024.
    for (sup, banks, dim, label) in [
        (false, 1usize, 256usize, "superpose OFF (control)"),
        (true, 64, 256, "d=256 x 64 banks"),
        (true, 256, 256, "d=256 x 256 banks"),
        (true, 1024, 256, "d=256 x 1024 banks"),
        (true, 256, 512, "d=512 x 256 banks"),
        (true, 1024, 512, "d=512 x 1024 banks"),
    ] {
        let mix = 1.0f32;
        let mut c = Config::local();
        c.seed = seed;
        c.vocab = gcfg.vocab;
        c.nodes = 64;
        c.route_query = RouteQuery::Bound;
        c.read_entry_by_content = true;
        c.walk_during_gap = false;
        c.freeze_operator = true;
        c.bypass_graph = true;
        c.anchor = 0.0;
        c.superpose = sup;
        c.unbind_mix = mix;
        c.mem_banks = banks;
        c.d = dim;
        c.derive();
        arms.push((label.to_string(), c));
    }
    for (i, (name, c)) in arms.into_iter().enumerate() {
        if shards > 1 && i % shards != shard {
            continue;
        }
        let o = run_one(&name, c, &gcfg, &stream);
        let (_, la) = window_mean(&o.metrics.window);
        suite.note(
            format!(
                "[unbind] {:<26} Latin {:.4} | answer {:.4} bits | unbind {}/{} accepted,                  mean best cos {:.3} | {} triples stored | CURSOR->answer cos {:.3} over {}",
                name, la, o.metrics.answer.mean(),
                o.model.unbind_hits, o.model.unbind_tries,
                o.model.unbind_cos / o.model.unbind_tries.max(1) as f64,
                o.model.mem_triples,
                o.model.probe_cursor_cos / o.model.probe_cursor_n.max(1) as f64,
                o.model.probe_cursor_n
            ) + &walk_tail(&o)
                + &comp_tail(&o),
        );
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
    // `derive()` carries the assertion that these two fields agree; every other
    // driver calls it and this one did not.
    nb.derive();
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
    // Its codelength is meaningful (uniform over the vocabulary); its accuracy
    // is not. With no readout `Scored.rows` is empty, `top()` returns None and
    // `out.correct` is false on every event by construction, so the 0.000 in
    // that row is a structural fact sitting in a column of measurements.
    let mut nr = base.clone();
    nr.no_readout = true;
    nr.derive();
    arms.push(("no readout (acc is structural)".into(), nr));

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
        ) + &comp_tail(&o));
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
        "[baseline] PPM-C order 4 on CHARGED events, same denominator as the mechanism: {:.3} bits/ev, second-order {:.3} \
         bits | silence removed (UPPER BOUND, handed the segmentation): {:.3} \
         bits/ev, second-order {:.3}",
        like.charged.mean(),
        like.second.mean(),
        upper.charged.mean(),
        upper.second.mean()
    ));

    suite
}
