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

/// The retrieval key must stop moving when the world does.
///
/// The bands carried both "which regime this is" and "how long since the world
/// spoke", and both were in the key, so a fact learned at six ticks of silence
/// did not answer at one -- a clock deciding whether a memory fires. Everything
/// in the key except the state block must now be bit-identical across a silence;
/// the state is the one thing that is supposed to evolve.
#[test]
fn the_retrieval_key_is_frozen_while_the_world_is_silent() {
    let mut cfg = alm::config::Config::local();
    cfg.seed = 5;
    cfg.vocab = 512;
    cfg.derive();
    let d = cfg.d;
    let mut m = alm::model::Model::new(cfg.clone());

    for t in 0..600usize {
        m.tick(if t % 7 == 0 { Some(t % 97) } else { None }, false);
    }
    // An event, then a stretch of silence.
    m.tick(Some(11), false);
    let mut keys = Vec::new();
    for _ in 0..6 {
        m.tick(None, false);
        let f = m.features_now();
        keys.push(f[d..].to_vec());
    }
    for (i, k) in keys.iter().enumerate().skip(1) {
        assert_eq!(
            &keys[0], k,
            "the retrieval key moved between silent tick 1 and silent tick {}: \
             elapsed time is still inside the key",
            i + 1
        );
    }
    assert!(keys[0].iter().any(|v| *v != 0.0), "the key is empty");
}

/// The superposition's capacity law, and why the memory has to be banked.
///
/// `M <- nu(M + t)` shrank everything already stored by 1/sqrt(2) on every
/// write, giving the first item weight 2^(-k/2): a horizon of two or three
/// triples across ten thousand events, which is a sliding window and not a
/// superposition. Summing plainly fixes that and exposes the real limit --
/// retrieval cosine falls as 1/sqrt(k), measured 0.043 against a predicted 0.050
/// at k = 401.
///
/// So one superposition cannot hold a continual stream, and that is what finally
/// gives content addressing a job: the address chooses which superposition to
/// unbind, each bank stays under capacity, and the read cost per query does not
/// grow with what is stored. Addressing has been inert on every accuracy metric
/// in this project because a few hundred facts fit in one linear table with room
/// to spare. It stops being optional the moment the store is a superposition.
#[test]
fn the_superposition_holds_at_small_k_and_decays_as_one_over_root_k() {
    let d = 256usize;
    let key = 0xBEEF_u64;
    let e = |i: u64| alm::num::unit_vector(key, i, d);
    let store_one = |mem: &mut Vec<f32>, a: &[f32], r: &[f32], y: &[f32], pair: &mut Vec<f32>| {
        alm::num::circconv(a, r, pair);
        alm::num::normalize(pair);
        let mut tri = vec![0.0f32; d];
        alm::num::circconv(pair, y, &mut tri);
        alm::num::normalize(&mut tri);
        for i in 0..d {
            mem[i] += tri[i];
        }
    };

    let mut cosines = Vec::new();
    for k in [20usize, 80, 320] {
        let mut mem = vec![0.0f32; d];
        let mut pair = vec![0.0f32; d];
        store_one(&mut mem, &e(1), &e(2), &e(3), &mut pair);
        let target_pair = pair.clone();
        for j in 0..k as u64 {
            let mut p2 = vec![0.0f32; d];
            store_one(&mut mem, &e(100 + 3 * j), &e(101 + 3 * j), &e(102 + 3 * j), &mut p2);
        }
        let mut got = vec![0.0f32; d];
        alm::num::unbind(&mem, &target_pair, &mut got);
        alm::num::normalize(&mut got);
        cosines.push(alm::num::dot(&got, &e(3)));
    }

    assert!(
        cosines[0] > 0.15,
        "a bank holding 21 triples unbinds its own content at cosine {:.3}: the          superposition is broken, not merely full",
        cosines[0]
    );
    assert!(
        cosines[0] > cosines[2] * 1.8,
        "retrieval did not degrade with load ({:.3} at k=21 against {:.3} at          k=321): the 1/sqrt(k) law is what makes banking necessary, so if it is          absent the argument for addressing is absent too",
        cosines[0],
        cosines[2]
    );
}

/// Binding must be exactly invertible, which needs unitary vectors.
#[test]
fn unbinding_a_single_pair_is_exact() {
    for d in [64usize, 256] {
        let a = alm::num::unitary_vector(7, 1, d);
        let b = alm::num::unitary_vector(7, 2, d);
        let mut m = vec![0.0f32; d];
        alm::num::circconv(&a, &b, &mut m);
        alm::num::normalize(&mut m);
        let mut got = vec![0.0f32; d];
        alm::num::unbind(&m, &a, &mut got);
        alm::num::normalize(&mut got);
        let cos = alm::num::dot(&got, &b);
        assert!(
            cos > 0.97,
            "d={}: unbinding one pair with nothing else stored reconstructs at \
             cosine {:.3}. Gaussian rows give 0.527 here; the capacity law only \
             holds for unitary ones",
            d,
            cos
        );
    }
}

/// A probe must not move the responses in flight.
///
/// `cursors`, `cursor_age`, `last_self_token` and `prev2` are situation, not
/// memory, and the volatile snapshot did not carry them -- the same omission an
/// audit had to find for `tick_index`. A probe batch is twenty-four probes of
/// fifteen ticks, so leaving them out displaces every response mid-flight.
#[test]
fn a_probe_leaves_the_responses_in_flight_untouched() {
    let mut cfg = alm::config::Config::local();
    cfg.seed = 21;
    cfg.vocab = 512;
    cfg.derive();
    let mut m = alm::model::Model::new(cfg.clone());
    for t in 0..900usize {
        m.tick(if t % 5 == 0 { Some(t % 211) } else { None }, false);
    }
    let before = m.snapshot_debug();
    let spec = alm::gen::ProbeSpec {
        domain: 0,
        context: vec![3, 5, 7],
        cue: vec![11, 13],
        target: 17,
        kind: alm::gen::Kind::Second,
        age_spans: 1,
    };
    let _ = m.probe(&spec, 6);
    let after = m.snapshot_debug();
    assert_eq!(
        before, after,
        "a probe displaced the responses in flight or the self-output token"
    );
}

/// The chain must advance on the model's own output, in isolation.
///
/// Five things sit between a stored link and a correct answer -- the write, the
/// bank, the cursor step, the cleanup and the emission -- and an end-to-end zero
/// says nothing about which one is broken. Several runs were spent that way.
/// This drives the mechanism directly: present a walk's links, then a query, then
/// silence, and watch where the cursor is after each quiet tick.
#[test]
fn a_walk_advances_one_link_per_silent_tick() {
    let mut cfg = alm::config::Config::local();
    cfg.seed = 3;
    cfg.vocab = 256;
    cfg.d = 256;
    cfg.mem_banks = 512;
    cfg.traj = 4;
    cfg.derive();
    let mut m = alm::model::Model::new(cfg.clone());

    // a0 -r-> a1 -r-> a2 -r-> a3, each link presented many times, interleaved
    // with unrelated traffic so the memory is not a toy.
    let (a0, a1, a2, a3, r) = (10usize, 11, 12, 13, 20);
    let links = [(a0, a1), (a1, a2), (a2, a3)];
    for round in 0..40usize {
        for &(x, y) in links.iter() {
            m.tick(Some(x), false);
            m.tick(None, false);
            m.tick(Some(r), false);
            m.tick(None, false);
            m.tick(Some(y), false);
            for _ in 0..3 {
                m.tick(None, false);
            }
        }
        // Unrelated content, so the banks hold more than this one walk.
        for j in 0..12usize {
            m.tick(Some(30 + (round * 7 + j) % 200), false);
            m.tick(None, false);
        }
    }

    // The query: the start, the relation, then silence.
    m.tick(Some(a0), false);
    m.tick(Some(r), false);
    // The mixture, because that is what the readout is handed. The response that
    // did the work stands at cosine 1.000 with the answer at every step; the
    // mixture reads 0.41 / 0.34 / 0.32 / 0.52 because the fading responses dilute
    // it. The floor that matters is what a random vector reaches against this
    // codebook -- sqrt(2 ln V / d) = 0.208 here -- so these clear it by half
    // again, and the assertion is set there rather than at the single cursor's
    // value, which would be asserting on a quantity nothing reads.
    let at = |m: &alm::model::Model, t: usize| m.cursor_cos(t);
    let floor = cfg.codebook_floor();
    let step1 = at(&m, a1);
    m.tick(None, false);
    let step2 = at(&m, a2);
    m.tick(None, false);
    let step3 = at(&m, a3);
    m.tick(None, false);
    let held = at(&m, a3);

    assert!(
        step1 > floor * 1.3,
        "the relation's own tick did not reach a1: mixture {:.3} against a          codebook floor of {:.3}",
        step1,
        floor
    );
    assert!(
        step2 > floor * 1.3,
        "one tick of silence did not reach a2 ({:.3}): the chain does not          advance on the model's own output",
        step2
    );
    assert!(
        step3 > floor * 1.3,
        "two ticks of silence did not reach a3 ({:.3}): depth is not coming          from time",
        step3
    );
    assert!(
        held > floor * 1.3,
        "a third tick of silence walked past the end of the walk ({:.3} from          a3): surplus thinking time has to be harmless, which is what the          cleanup threshold is for",
        held
    );
}

/// A chain that has run out must stay where it stopped.
///
/// A key nobody wrote still hashes into an occupied bank and its occupants
/// answer, so the end of a walk does not retrieve nothing -- it retrieves a
/// neighbour, and the cursor wanders off its own answer. On the real stream the
/// cursor stood at 0.193 with the answer one tick after the cues and 0.059 the
/// tick after that. Read-back is what tells the two apart: bind the candidate
/// onto the key and ask whether that triple is in the bank at all.
#[test]
fn a_finished_walk_holds_its_answer_through_surplus_silence() {
    let build = |verify: f32| {
        let mut cfg = alm::config::Config::local();
        cfg.seed = 4;
        cfg.vocab = 512;
        cfg.d = 256;
        cfg.mem_banks = 512;
        cfg.traj = 4;
        cfg.verify_sigma = verify;
        cfg.derive();
        let mut m = alm::model::Model::new(cfg.clone());
        let (a0, a1, r) = (10usize, 11, 20);
        for round in 0..40usize {
            m.tick(Some(a0), false);
            m.tick(None, false);
            m.tick(Some(r), false);
            m.tick(None, false);
            m.tick(Some(a1), false);
            for _ in 0..3 {
                m.tick(None, false);
            }
            for j in 0..14usize {
                m.tick(Some(30 + (round * 11 + j) % 400), false);
                m.tick(None, false);
                m.tick(Some(r), false);
                m.tick(None, false);
                m.tick(Some(31 + (round * 11 + j) % 400), false);
                m.tick(None, false);
            }
        }
        m.tick(Some(a0), false);
        m.tick(Some(r), false);
        let mut trace = Vec::new();
        for _ in 0..6 {
            m.tick(None, false);
            trace.push(m.cursor_cos(a1));
        }
        (trace, m.verify_rejects, cfg.codebook_floor())
    };

    let (off, _, _) = build(0.0);
    let (on, rejects, floor) = build(4.0);
    println!("verify off: {:?}", off.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>());
    println!("verify on : {:?}  rejects {}", on.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>(), rejects);

    // Against the codebook floor, not a number picked by hand -- the same
    // discipline the cleanup threshold needed after a magic 0.25 sat below its
    // own noise floor for several runs.
    assert!(
        on[0] > floor,
        "the first quiet tick did not clear the codebook floor: {:.3} against          {:.3}",
        on[0],
        floor
    );
    assert!(
        on[5] > on[0] * 0.6,
        "the cursor did not hold its answer through surplus silence: {:.3} at \
         the first quiet tick, {:.3} five ticks later (without read-back: {:.3} \
         -> {:.3})",
        on[0],
        on[5],
        off[0],
        off[5]
    );
}

/// A superposed record answers only the query that leaves *one* factor unknown.
///
/// The partial-match plan for this architecture was to store each record as a
/// single bound product and let one store answer all 2^n directions:
///
/// ```text
///   M = sum_r  E_{x_1^r} (*) ... (*) E_{x_n^r}
///   query fixing S:  M (/) (*)_{i in S} E_{q_i}
/// ```
///
/// That is wrong, and the reason is structural rather than a matter of width.
/// Unbinding the fields in S leaves the product of *every* field not in S, not
/// the target alone, and a product of several unknown factors is orthogonal to
/// any single codebook entry. So the store answers `S = everything but the
/// target` and nothing else -- n directions, not 2^n.
///
/// The table below is built so the claim cannot be blamed on the data: field 2
/// is an exact function of fields 0 and 1, so the pair determines the answer
/// and dozens of records agree on it. The true value is still not the argmax.
#[test]
fn a_superposed_record_cannot_answer_with_two_factors_unknown() {
    use alm::num::{circconv, dot, normalize, unbind, unitary_vector};

    let (d, n, v, t) = (4096usize, 8usize, 8usize, 2000usize);
    let code: Vec<Vec<Vec<f32>>> = (0..n)
        .map(|f| {
            (0..v)
                .map(|x| {
                    let mut e = unitary_vector(0x5157, (f * 64 + x) as u64, d);
                    normalize(&mut e);
                    e
                })
                .collect()
        })
        .collect();

    let mut rec = vec![vec![0usize; n]; t];
    for (r, row) in rec.iter_mut().enumerate() {
        for (f, cell) in row.iter_mut().enumerate() {
            *cell = (r * 7 + f * 13 + r / v) % v;
        }
        row[2] = (row[0] + row[1]) % v;
    }

    let mut m = vec![0.0f32; d];
    let (mut a, mut b) = (vec![0.0f32; d], vec![0.0f32; d]);
    for row in rec.iter() {
        a.copy_from_slice(&code[0][row[0]]);
        for (f, &x) in row.iter().enumerate().skip(1) {
            circconv(&a, &code[f][x], &mut b);
            a.copy_from_slice(&b);
        }
        for (mi, ai) in m.iter_mut().zip(a.iter()) {
            *mi += ai;
        }
    }

    // Read field 2 back, given a set of fields.
    let read = |given: &[usize]| -> (f32, f32) {
        let mut q = code[given[0]][rec[0][given[0]]].clone();
        for &g in given.iter().skip(1) {
            circconv(&q.clone(), &code[g][rec[0][g]], &mut q);
        }
        normalize(&mut q);
        let mut raw = vec![0.0f32; d];
        unbind(&m, &q, &mut raw);
        normalize(&mut raw);
        let mut best = f32::MIN;
        for cand in 0..v {
            best = best.max(dot(&raw, &code[2][cand]));
        }
        (dot(&raw, &code[2][rec[0][2]]), best)
    };

    let floor = (2.0 * (v as f32).ln() / d as f32).sqrt();

    // Fields 0 and 1 determine field 2 exactly, and five fields stay unknown.
    let (truth_two, best_two) = read(&[0, 1]);
    assert!(
        truth_two < floor,
        "two factors unknown should leave the answer at noise, got {} against floor {}",
        truth_two,
        floor
    );
    assert!(
        truth_two < best_two,
        "with two factors unknown the true value should not even win: {} vs {}",
        truth_two,
        best_two
    );

    // Everything but the target: exactly one unknown factor, and it is the argmax.
    let all_but: Vec<usize> = (0..n).filter(|&f| f != 2).collect();
    let (truth_one, best_one) = read(&all_but);
    assert!(
        (truth_one - best_one).abs() < 1e-6,
        "one unknown factor should be the argmax: {} vs {}",
        truth_one,
        best_one
    );
    assert!(
        truth_one > 4.0 * truth_two.abs(),
        "one unknown factor should stand far above the two-unknown case: {} vs {}",
        truth_one,
        truth_two
    );
}
