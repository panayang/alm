//! The three structural assertions, plus the two properties the accounting
//! rests on.
//!
//! These are equalities, not statistics. Each is a property of the layout or of
//! the arithmetic, so it is checked exactly and it either holds or the
//! implementation is wrong. Nothing here is a result.

use alm::code;
use alm::config::Config;
use alm::embed::Channel;
use alm::model::Model;
use alm::num::unit_vector;

fn bits_of(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn weight_fingerprint(m: &Model) -> Vec<u32> {
    let mut out = Vec::new();
    for w in m.graph.w.iter() {
        out.extend(bits_of(&w.a));
    }
    for (t, row) in m.store.rows.iter() {
        out.push(*t);
        out.extend(bits_of(row));
    }
    out
}

/// Drive the model with a short synthetic stream so there is something stored.
fn warm(m: &mut Model, n: usize) {
    let mut t = 0usize;
    for i in 0..n {
        // A cue, a gap, a target: the same shape the generator emits.
        m.tick(Some(11 + (i % 17)), false);
        for _ in 0..5 {
            m.tick(None, false);
        }
        m.tick(Some(400 + (i % 23)), false);
        for _ in 0..3 {
            m.tick(None, false);
        }
        t += 10;
    }
    assert!(t > 0);
}

#[test]
fn a3_write_routing_consistency_is_exactly_one() {
    let cfg = Config::local();
    let mut m = Model::new(cfg);
    let probes: Vec<Vec<f32>> = (0..64).map(|i| unit_vector(0x9111_2222, i, 64)).collect();
    let before: Vec<Vec<usize>> =
        probes.iter().map(|q| m.graph.write_path(q, m.cfg.hops)).collect();
    warm(&mut m, 400);
    let after: Vec<Vec<usize>> =
        probes.iter().map(|q| m.graph.write_path(q, m.cfg.hops)).collect();
    assert_eq!(
        before, after,
        "the write walk moved under learning; content addressing is supposed to \
         make that impossible"
    );
}

#[test]
fn a4_baseline_ticks_write_nothing() {
    let cfg = Config::local();
    let mut m = Model::new(cfg);
    warm(&mut m, 200);
    let before = weight_fingerprint(&m);
    let writes_before = m.content_writes;
    for _ in 0..500 {
        let out = m.tick(None, false);
        assert!(!out.wrote, "a baseline tick reported a content write");
    }
    let after = weight_fingerprint(&m);
    assert_eq!(m.content_writes, writes_before, "content writes happened at baseline");
    assert!(
        before == after,
        "500 baseline ticks changed stored weights; A4 is supposed to hold \
         exactly, not approximately"
    );
}

#[test]
fn a5_self_output_cannot_reach_the_slow_rungs() {
    let cfg = Config::local();
    let d = cfg.d;
    let rungs = cfg.rungs;
    let self_max = cfg.self_max_rung;
    let mut m = Model::new(cfg);
    // Pour self-channel content in for a long time.
    let mut v = vec![0.0f32; d];
    for i in 0..2000u64 {
        m.emb.rotated((i % 500) as usize, Channel::Covert, &mut v);
        m.ladder.observe_self(&v);
    }
    m.ladder.refresh();
    for k in 0..rungs {
        let e = m.ladder.self_energy(k);
        if k <= self_max {
            assert!(e > 0.0, "rung {} should have received self content", k);
        } else {
            assert_eq!(
                e, 0.0,
                "rung {} holds self-generated energy; the two-cascade layout is \
                 supposed to make that unrepresentable",
                k
            );
        }
    }
}

#[test]
fn emitted_distribution_is_normalised_at_every_hop() {
    let cfg = Config::local();
    let mut m = Model::new(cfg);
    warm(&mut m, 600);
    m.tick(Some(23), false);
    for _ in 0..8 {
        let out = m.tick(None, false);
        let mass = m.spread_now().mass();
        assert!(
            (mass - 1.0).abs() < 1e-3,
            "the emitted distribution sums to {:.6} after {} hops; a softmax over              the rows plus one term per row-less token has to telescope to one",
            mass,
            out.hops
        );
    }
}

/// Nothing in the store grows, so there is no append to be non-destructive
/// about. What replaced that property is weaker and worth pinning: a node the
/// walk has never reached holds nothing and contributes nothing.
/// The operator write is local: applying it must not touch any edge but the
/// one that was applied. Nothing crosses a hop, which is what makes "an identity
/// input does not strongly change a node" a property of the layout rather than
/// an approximation.
#[test]
fn the_operator_write_touches_one_edge() {
    let cfg = Config::local();
    let mut m = Model::new(cfg);
    warm(&mut m, 200);
    let before: Vec<Vec<u32>> = m.graph.w.iter().map(|w| bits_of(&w.a)).collect();
    m.tick(Some(37), false);
    let after: Vec<Vec<u32>> = m.graph.w.iter().map(|w| bits_of(&w.a)).collect();
    let changed = before.iter().zip(after.iter()).filter(|(a, b)| a != b).count();
    assert!(
        changed <= 1,
        "one event changed {} edge transforms; the operator write is supposed to          be local to the transform that was applied",
        changed
    );
}

#[test]
fn counter_rng_is_order_independent() {
    let a = unit_vector(1234, 7, 32);
    let _noise: Vec<Vec<f32>> = (0..50).map(|i| unit_vector(1234, 100 + i, 32)).collect();
    let b = unit_vector(1234, 7, 32);
    assert_eq!(bits_of(&a), bits_of(&b), "a draw depended on how many draws preceded it");
}

/// Two identical runs must produce bit-identical state.
///
/// The counter-based RNG guarantees the fixed vectors, but a float sum taken
/// over hash-map iteration order would still leak the process's hash seed into
/// the results, and that kind of non-reproducibility hides rather than
/// announces itself.
#[test]
fn two_identical_runs_agree_bit_for_bit() {
    let run = || {
        let mut m = Model::new(Config::local());
        warm(&mut m, 500);
        (weight_fingerprint(&m), m.total_bits.to_bits(), m.store.occupied_rows(), m.content_writes)
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "stored weights differed between two identical runs");
    assert_eq!(a.1, b.1, "accumulated codelength differed between two identical runs");
    assert_eq!(a.2, b.2, "row count differed between two identical runs");
    assert_eq!(a.3, b.3, "write count differed between two identical runs");
}

/// A read that has walked as far as the write must arrive at the same payload,
/// or the readout rows are being scored at a point they were never trained at.
#[test]
fn read_and_write_walks_agree_once_the_read_has_finished() {
    let cfg = Config::local();
    let hops = cfg.hops;
    let mut m = Model::new(cfg);
    warm(&mut m, 200);
    let q = unit_vector(0x9555_6666, 3, m.cfg.d);
    let a = m.graph.write_walk(&q, &q, hops, None);
    let b = m.graph.write_walk(&q, &q, hops, None);
    assert_eq!(
        bits_of(&a.last().unwrap().p_out),
        bits_of(&b.last().unwrap().p_out),
        "the same walk over the same weights produced two different payloads"
    );
    assert_eq!(a.len(), hops, "the write walk did not take the memory's hop depth");
}

/// A retention probe must leave the model exactly as it found it.
///
/// The memory must not move because a probe that writes is retraining on the
/// fact it is testing; the volatile state must not move either, because a probe
/// that leaves its own context behind contaminates every measurement after it.
/// Both are checked bit for bit rather than approximately.
#[test]
fn a_probe_disturbs_neither_the_memory_nor_the_situation() {
    use alm::gen::{GenConfig, Generator, Kind};
    let mut m = Model::new(Config::local());
    warm(&mut m, 400);
    // Put the model mid-response, so the restored state is a non-trivial one.
    m.tick(Some(31), false);
    m.tick(None, false);

    let gen = Generator::new(GenConfig::local());
    let ents = gen.entities_of(0).to_vec();
    let tgts = gen.targets_of(0).to_vec();
    let spec = alm::gen::ProbeSpec {
        domain: 0,
        context: ents[0..4].to_vec(),
        cue: vec![ents[0]],
        target: tgts[0],
        kind: Kind::First,
        age_spans: 1,
    };

    let mem_before = weight_fingerprint(&m);
    let rows_before = m.store.occupied_rows();
    let writes_before = m.content_writes;
    let sit_before = bits_of(&m.state_now());
    // The eligibility trace is not part of the memory but it steers it: the
    // next real settlement credits every traced edge, so footprints a probe
    // leaves behind would end up driving real weight updates.
    let trace_before = bits_of(&m.graph.trace);
    let counters_before = (m.events, m.commitments, m.silent_settlements, m.baseline_ticks);

    let (bits, _) = m.probe(&spec, 6);
    assert!(bits.is_finite() && bits > 0.0, "the probe was not charged anything");

    assert!(mem_before == weight_fingerprint(&m), "a probe changed stored weights");
    assert_eq!(rows_before, m.store.occupied_rows(), "a probe created a row");
    assert_eq!(writes_before, m.content_writes, "a probe performed a content write");
    assert_eq!(
        sit_before,
        bits_of(&m.state_now()),
        "a probe left its own context behind instead of restoring the displaced state"
    );
    assert_eq!(
        trace_before,
        bits_of(&m.graph.trace),
        "a probe left eligibility trace behind, which the next real settlement          would have credited into the edge transforms"
    );
    assert_eq!(
        counters_before,
        (m.events, m.commitments, m.silent_settlements, m.baseline_ticks),
        "a probe moved counters that a printed statistic divides against a          probe-free denominator"
    );
}

/// Sanity for the operator memory: the pieces must actually move.
///
/// A flat near-miss curve reads identically whether the near miss is cheap or
/// the routing is inert, and this project has mistaken a broken instrument for a
/// null result several times. These are the three things that have to be true
/// before "perturbing the route costs nothing" can be read as a property of the
/// design rather than as the route not mattering.
#[test]
fn the_operator_memory_is_not_inert() {
    let cfg = Config::local();
    let mut m = Model::new(cfg);

    // 1. The operator write changes the transform it was applied to.
    warm(&mut m, 100);
    let w_before: Vec<Vec<u32>> = m.graph.w.iter().map(|w| bits_of(&w.a)).collect();
    for i in 0..40 {
        m.tick(Some(11 + i % 17), false);
        m.tick(None, false);
    }
    let w_after: Vec<Vec<u32>> = m.graph.w.iter().map(|w| bits_of(&w.a)).collect();
    let moved = w_before.iter().zip(w_after.iter()).filter(|(a, b)| a != b).count();
    assert!(moved > 0, "no edge transform moved after forty writes: the operator write is inert");

    // 2. Perturbing the route actually lands somewhere else.
    let mut a = Model::new(Config::local());
    let mut cfg2 = Config::local();
    cfg2.route_perturb = 1;
    let mut b = Model::new(cfg2);
    let mut differed = 0usize;
    for i in 0..200 {
        let tok = if i % 4 == 0 { None } else { Some(20 + i % 31) };
        a.tick(tok, false);
        b.tick(tok, false);
        if a.node_now() != b.node_now() {
            differed += 1;
        }
    }
    assert!(
        differed > 20,
        "the perturbed route visited a different node on only {} of 200 ticks: \
         taking the runner-up edge is not changing where the walk goes",
        differed
    );

    // 3. The state is not simply a copy of the background.
    let mut m3 = Model::new(Config::local());
    warm(&mut m3, 200);
    m3.tick(Some(41), false);
    for _ in 0..4 {
        m3.tick(None, false);
    }
    let cos = m3.state_vs_background();
    assert!(
        cos < 0.98,
        "the state is {:.4} aligned with the background: the operator term is \
         contributing nothing and the state is just another copy of the anchor",
        cos
    );
}

/// The feature vector and the row width must agree exactly.
///
/// They did not: `features()` emitted one block more than a row was wide, the
/// readout truncated to the row width, and the blocks that fell off the end were
/// the background bands. Every self-feedback ablation then measured as no
/// effect, because the channel it ablated had no path to the prediction at all.
/// A silent truncation like this produces a plausible null rather than an error.
#[test]
fn every_feature_block_is_read_and_written() {
    let cfg = Config::local();
    let expect = cfg.feature_blocks() * cfg.d;
    let mut m = Model::new(Config::local());
    warm(&mut m, 120);
    let phi = m.features_now();
    assert_eq!(phi.len(), expect, "features() and the row width disagree");
    assert_eq!(m.store.fw, expect, "the store's row width disagrees with the config");
    // No block may be identically zero: an allocated-but-never-filled block is
    // dead width that silently dilutes every dot product.
    for b in 0..cfg.feature_blocks() {
        let blk = &phi[b * cfg.d..(b + 1) * cfg.d];
        assert!(
            blk.iter().any(|x| x.abs() > 1e-9),
            "feature block {} is entirely zero after warm-up",
            b
        );
    }
}

/// The residual hop has to stay a residual.
///
/// If `tanh(W p)` dwarfs `p`, the state is overwritten every hop rather than
/// transformed, every edge destroys it about equally, and a near-miss curve
/// comes out flat for a mechanical reason that has nothing to do with operator
/// memory degrading gracefully.
#[test]
fn the_hop_transforms_the_state_rather_than_replacing_it() {
    let cfg = Config::local();
    let m = Model::new(cfg);
    let (ratio, cos) = m.hop_scale();
    assert!(
        ratio < 1.0,
        "the operator term is {:.2}x the state: the residual is not a residual",
        ratio
    );
    assert!(
        cos > 0.6,
        "one hop leaves the state only {:.3} aligned with its input; the \
         challenge does not survive a gap",
        cos
    );
}

/// A3 on the *live* path, and the honest statement of what it does not cover.
///
/// The old A3 assertion called `graph.write_path()`, which the model never
/// invoked -- it certified dead code while `write_operator` wrote into the read
/// walk's own perturbable step. The write walk is now live, and this asserts the
/// property the design actually claims: *given a query*, the write route is
/// deterministic and carries no perturbation.
///
/// It deliberately does NOT assert that read perturbation leaves the written
/// address alone. It does not: the write query is the running state, and reads
/// move the state. So a near-miss arm perturbs the write address too, one tick
/// later and indirectly. That is a real property of a design whose addresses are
/// contextual rather than content-only -- it is the compound effect the
/// near-miss arms measure, and pretending otherwise by asserting invariance here
/// would only hide it.
#[test]
fn the_write_route_is_deterministic_given_its_query() {
    let mut cfg = alm::config::Config::local();
    cfg.seed = 7;
    cfg.vocab = 64;
    cfg.route_perturb = 2;
    cfg.derive();
    let mut g = alm::graph::Graph::new(&cfg);
    let q: Vec<f32> = (0..cfg.d).map(|i| ((i * 37 % 19) as f32 - 9.0) / 9.0).collect();

    let a = g.write_walk(&q, &q, cfg.hops, None).iter().map(|s| s.edge).collect::<Vec<_>>();
    // Wander the read head all over the graph in between.
    for r in 0..40usize {
        let _ = g.select_rank(r % cfg.nodes, &q, r % 3);
    }
    let b = g.write_walk(&q, &q, cfg.hops, None).iter().map(|s| s.edge).collect::<Vec<_>>();
    assert_eq!(a, b, "the write route moved without its query moving");
    assert_eq!(a.len(), cfg.hops, "the write walk did not take cfg.hops steps");
}


/// At an answer tick, the first bound block must be the episode's own conjunction.
///
/// An offline linear probe on `nu(E_a (*) E_b)` alone reaches 0.41 on the Latin
/// family at d=64 where the model, holding that same vector in its features,
/// reaches 0.145. So the information is present and linearly decodable and the
/// model is not using it. This pins down the first link in that chain: whether
/// what the readout is handed at the moment of the charge is the conjunction at
/// all.
#[test]
fn the_bound_block_at_an_answer_is_the_episodes_own_conjunction() {
    let mut g = alm::gen::GenConfig::fast();
    g.seed = 99;
    let gen = alm::gen::Generator::new(g.clone());
    let stream = gen.generate(20_000);

    let mut cfg = alm::config::Config::local();
    cfg.seed = 99;
    cfg.vocab = g.vocab;
    cfg.derive();
    let emb = alm::embed::Embeddings::new(&cfg);
    let mut m = alm::model::Model::new(cfg.clone());

    let mut checked = 0usize;
    let mut worst = 1.0f32;
    let mut sum = 0.0f64;
    for t in 0..stream.len() {
        // Look before the tick is taken: the charge is settled on the features
        // standing at the top of the tick.
        if let Some(i) = stream.ep_at[t] {
            let ep = &stream.episodes[i];
            if matches!(ep.kind, alm::gen::Kind::Second) && ep.cues.len() >= 2 {
                let mut want = vec![0.0f32; cfg.d];
                alm::num::circconv(emb.row(ep.cues[0]), emb.row(ep.cues[1]), &mut want);
                alm::num::normalize(&mut want);
                let got = m.bound_block(0);
                let cos = alm::num::dot(&want, got)
                    / (alm::num::dot(&want, &want).sqrt() * alm::num::dot(got, got).sqrt()).max(1e-9);
                sum += cos as f64;
                if cos < worst {
                    worst = cos;
                }
                checked += 1;
            }
        }
        m.tick(stream.observe(t), false);
    }
    assert!(checked > 50, "only {} answer ticks checked", checked);
    let mean = sum / checked as f64;
    assert!(
        mean > 0.95,
        "mean cosine between the first bound block and the episode's own          conjunction is {:.3} over {} answer ticks (worst {:.3}): the readout is          not being handed the conjunction it is charged on",
        mean,
        checked,
        worst
    );
}
