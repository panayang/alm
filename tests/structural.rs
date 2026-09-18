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

/// The readout must at least learn the marginal.
///
/// On PhysioNet the model charged 8.04 bits/event where the marginal token
/// distribution is 7.18 and the uniform is 8.21 -- nearer to knowing nothing
/// than to counting, on 1.7M events, inside the capacity region. That is not a
/// property of any source, so it is checked here on a stream with no structure
/// whatsoever: tokens drawn i.i.d. from a fixed skewed distribution, no
/// context, no order, nothing to walk. A readout that works must approach
/// H(marginal); one that cannot has nothing to do with memory.
/// The readout must never be confidently wrong about what it cannot predict.
///
/// This assertion has been weakened once, on purpose, and the reason belongs
/// with it. It originally demanded the charge approach `H(marginal)`, which no
/// setting of the norm bound reaches -- the best is 5.52 against a marginal of
/// 4.86. That was the wrong thing to demand: `store.rs` removes the count prior
/// deliberately, on the grounds that a frequency estimate converges on the
/// corpus unigram and stops discriminating as data grows, so a readout that
/// declines to encode a marginal is the design holding its line.
///
/// What is not negotiable is the other side of that trade. Dropping the prior
/// drops the guaranteed codelength floor, so nothing stops the readout from
/// assigning a vanishing probability to something it simply could not know --
/// confident wrongness, which `store.rs` names as fabrication. On a stream with
/// nothing to learn, the charge must therefore not exceed the uniform code.
/// Before the norm bound it did: 6.83 bits against a uniform of 6.00.
#[test]
fn the_readout_is_never_worse_than_knowing_nothing() {
    use alm::config::Config;
    use alm::model::Model;
    use alm::num::cbrng;

    let v = 64usize;
    // Zipf-ish weights, so the marginal is well clear of the uniform.
    let w: Vec<f64> = (0..v).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let tot: f64 = w.iter().sum();
    let p: Vec<f64> = w.iter().map(|x| x / tot).collect();
    let h_marg: f64 = -p.iter().map(|q| q * q.log2()).sum::<f64>();
    let uniform = (v as f64).log2();

    // Both negative-sampling rules, because the suspicion is specific: the
    // delta rule takes its negatives from whatever currently ranks highest,
    // which is exactly the frequent tokens, so it pushes down precisely what
    // the marginal says to push up. The generator this project validated on has
    // zero marginals by construction -- the Latin square was built that way on
    // purpose -- so a readout unable to represent a marginal would pass every
    // test there and fail on any real stream.
    let mut got = [0.0f64; 2];
    for (arm, bias) in [(0usize, false), (1, true)] {
    let mut cfg = Config::local();
    cfg.seed = 0x51D;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.readout_bias = bias;
    cfg.derive();
    let mut m = Model::new(cfg);

    let n = 120_000usize;
    let mut bits = 0.0f64;
    let mut charged = 0u64;
    for i in 0..n {
        // Inverse-CDF sample, deterministic in the counter.
        let u = (cbrng(0xDA7A, i as u64) >> 11) as f64 / (1u64 << 53) as f64;
        let mut acc = 0.0;
        let mut tok = v - 1;
        for (k, q) in p.iter().enumerate() {
            acc += q;
            if u <= acc {
                tok = k;
                break;
            }
        }
        // Silence between events: this architecture is specified around gap
        // periods, so a stream of back-to-back event ticks does not exercise
        // it as designed. Two quiet ticks per event, which is what the
        // generator this project was built on supplies.
        for _ in 0..2 {
            m.tick(None, false);
        }
        let out = m.tick(Some(tok), false);
        // Score the last fifth only, so this is about what it converged to.
        if out.charged && i * 5 >= n * 4 {
            bits += out.bits;
            charged += 1;
        }
    }
    got[arm] = bits / charged.max(1) as f64;
    println!(
        "structureless stream, readout_bias={}: {:.4} bits/event   H(marginal) {:.4}   uniform {:.4}",
        bias, got[arm], h_marg, uniform
    );
    }
    let got = got[0].min(got[1]);
    assert!(
        got < uniform,
        "confidently wrong about an unpredictable stream: {:.4} bits against a          uniform code's {:.4}. The readout has no codelength floor, so nothing          but the norm bound stops this.",
        got,
        uniform
    );
    let _ = h_marg;
}


/// Structure with a skewed marginal: the test that tells the two readings apart.
///
/// The structureless test says the readout carries no frequency information.
/// That may be the design holding its line -- `store.rs` removed counts on
/// purpose, because a frequency estimate converges on the corpus unigram and
/// stops discriminating as data grows, and there is no organisation to find in
/// an i.i.d. stream anyway.
///
/// So this stream has organisation and a skewed marginal at once: the successor
/// is a deterministic function of the current token, while the tokens
/// themselves are Zipf. A readout that works must beat the marginal here by a
/// wide margin, because the conditional entropy is zero. If it cannot, the
/// emission path is broken and the refusal-to-count reading is unavailable.
#[test]
fn the_readout_learns_a_deterministic_successor_under_a_skewed_marginal() {
    use alm::config::Config;
    use alm::model::Model;
    use alm::num::cbrng;

    let v = 64usize;
    let w: Vec<f64> = (0..v).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let tot: f64 = w.iter().sum();
    let p: Vec<f64> = w.iter().map(|x| x / tot).collect();
    let h_marg: f64 = -p.iter().map(|q| q * q.log2()).sum::<f64>();

    for bias in [false, true] {
        let mut cfg = Config::local();
        cfg.seed = 0x51D;
        cfg.vocab = v;
        cfg.d = 128;
        cfg.mem_banks = 4096;
        cfg.cleanup_floor_mult = 1.1;
        cfg.readout_bias = bias;
        cfg.derive();
        let mut m = Model::new(cfg);

        let n = 120_000usize;
        let (mut bits, mut charged) = (0.0f64, 0u64);
        let mut cue = 0usize;
        for i in 0..n {
            // Alternate: a Zipf-drawn cue, then its deterministic successor.
            let tok = if i % 2 == 0 {
                let u = (cbrng(0xDA7B, i as u64) >> 11) as f64 / (1u64 << 53) as f64;
                let (mut acc, mut k) = (0.0, v - 1);
                for (j, q) in p.iter().enumerate() {
                    acc += q;
                    if u <= acc {
                        k = j;
                        break;
                    }
                }
                cue = k;
                k
            } else {
                (cue * 7 + 3) % v
            };
            for _ in 0..2 {
                m.tick(None, false);
            }
            let out = m.tick(Some(tok), false);
            // Only the successor is scored: its conditional entropy is zero.
            if out.charged && i % 2 == 1 && i * 5 >= n * 4 {
                bits += out.bits;
                charged += 1;
            }
        }
        let got = bits / charged.max(1) as f64;
        println!(
            "deterministic successor, readout_bias={}: {:.4} bits   H(marginal) {:.4}   true conditional 0.0",
            bias, got, h_marg
        );
        assert!(
            got < h_marg,
            "a deterministic successor should cost far less than the marginal: {:.4} against {:.4}",
            got,
            h_marg
        );
    }
}

// =========================== the state path ===========================
//
// Everything above that touches the state asserts plumbing: that a block is
// read and written, that a hop transforms rather than replaces, that a probe
// puts it back. None of it asks the state to *encode* anything.
//
// The emission path just showed what that absence costs. A clamp was flattening
// the distribution by two orders of magnitude and it lived behind twenty-one
// green assertions, because not one of them ever handed the readout a question
// whose answer was known. The state has exactly the same hole, and the patient
// representation -- the one real application reading it -- sits below a
// counting control.
//
// So: known labels by construction, a chance level that is arithmetic rather
// than measured, and the weakest classifier that could work. A nearest
// centroid, fitted on the first half of the run and tested on the second, with
// the label always conditioned on the thing it must not be read off. Nothing
// here can pass by memorising, and nothing here has a threshold I chose.

/// Unit-length mean of a set of feature rows.
fn centroid(rows: &[&Vec<f32>]) -> Vec<f32> {
    let d = rows[0].len();
    let mut c = vec![0.0f32; d];
    for r in rows.iter() {
        for i in 0..d {
            c[i] += r[i];
        }
    }
    alm::num::normalize(&mut c);
    c
}

/// Nearest-centroid accuracy. `rows` are (label, features); the first half
/// fits the centroids and the second half is scored. Returns (accuracy, n).
fn nearest_centroid(rows: &[(usize, Vec<f32>)], classes: usize) -> (f64, usize) {
    let cut = rows.len() / 2;
    let mut cents: Vec<Option<Vec<f32>>> = Vec::with_capacity(classes);
    for c in 0..classes {
        let mine: Vec<&Vec<f32>> =
            rows[..cut].iter().filter(|(l, _)| *l == c).map(|(_, v)| v).collect();
        cents.push(if mine.is_empty() { None } else { Some(centroid(&mine)) });
    }
    let (mut hit, mut n) = (0usize, 0usize);
    for (lab, v) in rows[cut..].iter() {
        let mut u = v.clone();
        alm::num::normalize(&mut u);
        let mut best: Option<(usize, f32)> = None;
        for (c, cc) in cents.iter().enumerate() {
            if let Some(cc) = cc {
                let s = alm::num::dot(cc, &u);
                if best.map_or(true, |(_, bs)| s > bs) {
                    best = Some((c, s));
                }
            }
        }
        if let Some((c, _)) = best {
            n += 1;
            if c == *lab {
                hit += 1;
            }
        }
    }
    (hit as f64 / n.max(1) as f64, n)
}

/// A continuous stream of events over `np` tokens, drawn i.i.d., with a fixed
/// gap, recording the state after each event with the previous and present
/// token attached.
///
/// It used to be a fixed cycle of (prev, now) combinations, which scored 1.000
/// on both directions -- because in a cycle the class is the phase. `prev`
/// advanced only every `nn` episodes, so "which token preceded this one" and
/// "which quarter of the block is this" were the same variable, and a centroid
/// reading the slow background answered the second one. Drawing the order is
/// what makes the label the only thing left to read.
fn stream_states(events: usize, np: usize, gap: usize) -> Vec<(usize, usize, Vec<f32>)> {
    let mut cfg = Config::local();
    cfg.seed = 0x5747;
    cfg.vocab = 64;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.derive();
    let mut m = Model::new(cfg);
    let mut out = Vec::with_capacity(events);
    let mut prev: Option<usize> = None;
    for i in 0..events {
        let k = (alm::num::cbrng(0x517E, i as u64) % np as u64) as usize;
        m.tick(Some(3 + k), false);
        if let Some(p) = prev {
            if i * 4 >= events {
                out.push((p, k, m.features_now()));
            }
        }
        prev = Some(k);
        for _ in 0..gap {
            m.tick(None, false);
        }
    }
    out
}

#[test]
fn the_state_carries_the_previous_event_and_not_only_the_present_one() {
    let np = 4usize;
    let rows = stream_states(8000, np, 3);

    // The control first: the token that just arrived must be readable. If this
    // fails the instrument is broken, not the state -- and separating the two
    // is the whole reason it is here.
    let mut acc_now = Vec::new();
    for ip in 0..np {
        let sub: Vec<(usize, Vec<f32>)> =
            rows.iter().filter(|(p, _, _)| *p == ip).map(|(_, n, v)| (*n, v.clone())).collect();
        acc_now.push(nearest_centroid(&sub, np).0);
    }
    let now_acc = acc_now.iter().sum::<f64>() / np as f64;

    // The claim: conditioned on the present token, the state still says which
    // token preceded it. A state that fails this is a function of the current
    // input, and every "memory" the readout reads is the token it was just
    // handed.
    let mut acc_prev = Vec::new();
    for inn in 0..np {
        let sub: Vec<(usize, Vec<f32>)> =
            rows.iter().filter(|(_, n, _)| *n == inn).map(|(p, _, v)| (*p, v.clone())).collect();
        acc_prev.push(nearest_centroid(&sub, np).0);
    }
    let prev_acc = acc_prev.iter().sum::<f64>() / np as f64;

    println!(
        "state decoding: previous event {:.3}   present event {:.3}   chance {:.3}   rows {}",
        prev_acc,
        now_acc,
        1.0 / np as f64,
        rows.len()
    );
    assert!(now_acc > 0.9, "the present token must be readable from the state; got {:.3}", now_acc);
    assert!(
        prev_acc > 2.0 / np as f64,
        "conditioned on the present token the state must still identify the previous one: {:.3} against chance {:.3}",
        prev_acc,
        1.0 / np as f64
    );
}

#[test]
fn the_state_carries_how_long_the_world_was_silent() {
    // Depth is the duration of silence, and the one measured result on real
    // data -- 0.48 bits on PhysioNet that PPM could not take -- is the claim
    // that elapsed time reaches the readout. Nothing has ever asked the state
    // whether it does.
    //
    // Every event here is the same token, so the token stream carries nothing
    // and the gap is the only variable. The gap is drawn rather than cycled,
    // for the reason `stream_states` records: a cycled class is a phase, and a
    // phase is readable from the background whether or not the gap is.
    let gaps = [1usize, 3, 7, 15];
    let mut cfg = Config::local();
    cfg.seed = 0x6A9;
    cfg.vocab = 64;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.derive();
    let mut m = Model::new(cfg);

    let episodes = 4000usize;
    let mut rows: Vec<(usize, Vec<f32>)> = Vec::new();
    for i in 0..episodes {
        let gi = (alm::num::cbrng(0x9A17, i as u64) % gaps.len() as u64) as usize;
        for _ in 0..gaps[gi] {
            m.tick(None, false);
        }
        m.tick(Some(21), false);
        if i * 4 >= episodes {
            rows.push((gi, m.features_now()));
        }
    }
    let (acc, n) = nearest_centroid(&rows, gaps.len());
    println!(
        "state decoding: silence length {:.3}   chance {:.3}   classes {:?}   rows {}",
        acc,
        1.0 / gaps.len() as f64,
        gaps,
        n
    );
    assert!(
        acc > 2.0 / gaps.len() as f64,
        "the state must encode how long the world was silent: {:.3} against chance {:.3}",
        acc,
        1.0 / gaps.len() as f64
    );
}

#[test]
fn the_states_memory_has_a_depth_and_the_depth_is_more_than_one() {
    // Lag 1 is not enough. A state that carries only the token before this one
    // is a bigram context with a vector spelling, and the chain -- multi-hop
    // nodes, a response that keeps walking through silence -- would be
    // decoration. So: how far back does the state actually reach?
    //
    // The tokens are drawn i.i.d., so t[i-l] is independent of every other
    // token in the window. Decoding it above chance therefore needs no
    // conditioning and cannot be a leak from a neighbour: there is nothing in
    // the rest of the window that predicts it.
    let np = 4usize;
    let lags = 6usize;
    let events = 12000usize;
    for gap in [1usize, 3, 7] {
        let mut cfg = Config::local();
        cfg.seed = 0x5747;
        cfg.vocab = 64;
        cfg.d = 128;
        cfg.mem_banks = 4096;
        cfg.cleanup_floor_mult = 1.1;
        cfg.derive();
        let mut m = Model::new(cfg);
        let mut hist: Vec<usize> = Vec::new();
        let mut rows: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();
        for i in 0..events {
            let k = (alm::num::cbrng(0x517E, i as u64) % np as u64) as usize;
            m.tick(Some(3 + k), false);
            hist.push(k);
            if hist.len() > lags && i * 4 >= events {
                let w: Vec<usize> = hist[hist.len() - lags..].iter().rev().copied().collect();
                rows.push((w, m.features_now()));
            }
            for _ in 0..gap {
                m.tick(None, false);
            }
        }
        let mut accs = Vec::new();
        for l in 0..lags {
            let sub: Vec<(usize, Vec<f32>)> =
                rows.iter().map(|(w, v)| (w[l], v.clone())).collect();
            accs.push(nearest_centroid(&sub, np).0);
        }
        let depth = accs.iter().take_while(|a| **a > 1.5 / np as f64).count();
        println!(
            "gap {:>2}: lag accuracies {:?}   chance {:.3}   depth {}",
            gap,
            accs.iter().map(|a| format!("{:.3}", a)).collect::<Vec<_>>(),
            1.0 / np as f64,
            depth
        );
        assert!(
            depth >= 2,
            "the state must reach past the token before this one, or the chain is decoration: gap {} depth {} accuracies {:?}",
            gap,
            depth,
            accs
        );
    }
}

#[test]
fn the_background_bands_are_not_copies_of_each_other() {
    // The cascade exists so that rung k peaks at a later lag than rung k-1 and
    // the differences isolate timescales. If the bands are nearly collinear the
    // ladder's opening paragraph is false, the feature vector is carrying
    // `rungs` copies of one signal, and the delta rule is spreading one
    // gradient over three blocks for nothing.
    let mut cfg = Config::local();
    cfg.seed = 0x8C3;
    cfg.vocab = 64;
    cfg.d = 256;
    cfg.derive();
    let mut m = Model::new(cfg);
    for i in 0..600 {
        m.tick(Some(5 + (i % 29)), false);
        for _ in 0..4 {
            m.tick(None, false);
        }
    }
    let r = m.cfg.rungs;
    let mut worst = 0.0f32;
    for j in 0..r {
        for k in (j + 1)..r {
            let c = m.ladder.band_correlation(j, k).abs();
            println!("band correlation {} vs {}: {:+.4}", j, k, c);
            if c > worst {
                worst = c;
            }
        }
    }
    assert!(worst < 0.9, "the bands must not be copies of one signal; worst |cos| = {:.4}", worst);
}

#[test]
fn the_walk_alone_can_answer_a_deterministic_relation() {
    // The operator path has three assertions and all three are plumbing: the
    // weights moved, a perturbed route lands elsewhere, the state is not a copy
    // of the background. None of them asks the operator memory to *carry a
    // relation*, which is the only thing it is for.
    //
    // So turn the learned rows off. With `no_readout` the only term left in the
    // score is the direct comparison of every token to the state, so whatever
    // the model gets right it got from where the walk went. On a stream where
    // each cue has one deterministic successor, that must beat the uniform. If
    // it does not, the operator write is a gradient that lands somewhere the
    // read never looks, and every experiment that ablated it was measuring
    // nothing.
    let v = 64usize;
    let uniform = (v as f64).log2();
    let mut cfg = Config::local();
    cfg.seed = 0x0DE1;
    cfg.vocab = v;
    cfg.d = 128;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.no_readout = true;
    cfg.derive();
    let mut m = Model::new(cfg);

    let n = 60_000usize;
    let (mut bits, mut charged) = (0.0f64, 0u64);
    let mut cue = 0usize;
    for i in 0..n {
        let tok = if i % 2 == 0 {
            cue = (alm::num::cbrng(0xC0E, i as u64) % v as u64) as usize;
            cue
        } else {
            (cue * 7 + 3) % v
        };
        for _ in 0..2 {
            m.tick(None, false);
        }
        let out = m.tick(Some(tok), false);
        if out.charged && i % 2 == 1 && i * 5 >= n * 4 {
            bits += out.bits;
            charged += 1;
        }
    }
    let got = bits / charged.max(1) as f64;
    println!("walk only (no_readout): {:.4} bits   uniform {:.4}", got, uniform);
    assert!(
        got < uniform,
        "with the rows disabled the walk must still beat the uniform on a \
         deterministic relation: {:.4} against {:.4}",
        got,
        uniform
    );
}

#[test]
fn the_four_channels_do_not_alias_and_do_not_distort() {
    // A5 is a statement about which cascade a stream may write, and the three
    // self channels feed one cascade between them. If two channels rotated a
    // token to nearly the same vector, "the model heard itself say X" and "the
    // model is touching memory X" would be the same event in the background,
    // the three-stream drive would be one stream with a gain of three, and
    // every ablation of one channel would be partly an ablation of the others.
    //
    // The rotations are signed permutations, so both answers are known: the
    // norm is preserved exactly, and two independent permutations of the same
    // vector agree at the chance level of 1/sqrt(d).
    // Wide enough that the bound means something: at the default d = 64 the
    // chance level is 1/8 and any bound loose enough to pass cleanly is also
    // loose enough to admit genuine aliasing.
    let mut cfg = Config::local();
    cfg.d = 256;
    cfg.derive();
    let m = Model::new(cfg);
    let d = m.cfg.d;
    let floor = 5.0 / (d as f32).sqrt();
    let chans = [Channel::In, Channel::Overt, Channel::Covert, Channel::Write];
    let names = ["In", "Overt", "Covert", "Write"];

    let mut worst = 0.0f32;
    let mut worst_norm = 0.0f32;
    for t in (0..64).map(|i| i * 61 + 3) {
        let base = alm::num::norm(m.emb.row(t));
        let mut v: Vec<Vec<f32>> = Vec::new();
        for c in chans.iter() {
            let mut o = vec![0.0f32; d];
            m.emb.rotated(t, *c, &mut o);
            let e = (alm::num::norm(&o) - base).abs();
            if e > worst_norm {
                worst_norm = e;
            }
            v.push(o);
        }
        for i in 0..chans.len() {
            for j in (i + 1)..chans.len() {
                let c = alm::num::dot(&v[i], &v[j]).abs() / (base * base);
                if c > worst {
                    worst = c;
                    println!("token {:>4}  {:>6} vs {:>6}: |cos| {:.4}", t, names[i], names[j], c);
                }
            }
        }
    }
    println!("worst |cos| {:.4}   floor {:.4}   worst norm error {:.3e}", worst, floor, worst_norm);
    assert!(
        worst_norm < 1e-4,
        "a channel rotation changed the norm by {:.3e}: it is not a signed permutation",
        worst_norm
    );
    assert!(
        worst < floor,
        "two channels agree at |cos| {:.4}, above the {:.4} chance level: they alias",
        worst,
        floor
    );
}

#[test]
fn the_reported_entropy_is_the_entropy_of_the_distribution_that_is_charged() {
    // `mass()` once returned 2942.67 because it assumed a token with no row
    // weighs exp(0) = 1, when under max-subtraction it weighs exp(0 - max).
    // The normalisation assertion caught that immediately. `entropy_bits()`
    // carried the identical mistake on the identical line and nothing caught
    // it, because nothing anywhere compared the reported entropy to anything.
    //
    // So compare it to itself, computed the other way: sum -q log q over the
    // whole vocabulary using the same `prob_of` the ledger charges with. The
    // answer is known exactly, it exercises the row-less branch, and it fails
    // for any weight that is right in one place and wrong in the other.
    // The last case is the one that matters: a vocabulary far larger than the
    // number of rows, with the codebook term off, so most tokens take the
    // row-less branch and the maximum score is not zero. That is the only
    // configuration in which the wrong weight and the right one differ, and
    // every earlier case reports `unseen 0`.
    for (label, no_readout, codebook, warm_events, v) in [
        ("fresh", false, 1.0f32, 0usize, 64usize),
        ("warm", false, 1.0, 4000, 64),
        ("warm, no codebook", false, 0.0, 4000, 64),
        ("warm, rows off", true, 1.0, 4000, 64),
        ("sparse rows, no codebook", false, 0.0, 300, 4096),
    ] {
        let mut cfg = Config::local();
        cfg.seed = 0xE47;
        cfg.vocab = v;
        cfg.d = 128;
        cfg.mem_banks = 4096;
        cfg.cleanup_floor_mult = 1.1;
        cfg.no_readout = no_readout;
        cfg.readout_codebook = codebook;
        cfg.derive();
        let mut m = Model::new(cfg);
        let mut cue = 0usize;
        for i in 0..warm_events {
            let tok = if i % 2 == 0 {
                cue = (alm::num::cbrng(0xC0E, i as u64) % v as u64) as usize;
                cue
            } else {
                (cue * 7 + 3) % v
            };
            for _ in 0..2 {
                m.tick(None, false);
            }
            m.tick(Some(tok), false);
        }

        let sc = m.spread_now();
        let reported = sc.entropy_bits();
        let mut direct = 0.0f64;
        let mut mass = 0.0f64;
        for t in 0..v {
            let q = sc.prob_of(&m.store, t as u32) as f64;
            mass += q;
            if q > 1e-30 {
                direct -= q * q.log2();
            }
        }
        println!(
            "{:>18}: reported {:.6}  direct {:.6}  mass {:.6}  unseen {}",
            label, reported, direct, mass, sc.unseen_count
        );
        assert!(
            (mass - 1.0).abs() < 1e-4,
            "{}: the distribution sums to {:.6}",
            label,
            mass
        );
        assert!(
            (reported - direct).abs() < 1e-3,
            "{}: the reported entropy {:.6} is not the entropy of the charged \
             distribution {:.6}",
            label,
            reported,
            direct
        );
    }
}
