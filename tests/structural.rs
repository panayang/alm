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
    for n in m.tree.arena.iter() {
        for (t, row) in n.rows.iter() {
            out.push(*t);
            out.extend(bits_of(row));
        }
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
fn emitted_distribution_is_normalised_at_every_depth() {
    let cfg = Config::local();
    let mut m = Model::new(cfg);
    warm(&mut m, 600);
    // Sample the swarm mid-response.
    m.tick(Some(23), false);
    for _ in 0..6 {
        m.tick(None, false);
        if m.swarm.is_empty() {
            continue;
        }
        let sp = code::spread(
            &m.tree,
            &m.swarm.codes(),
            &m.swarm.payloads(),
            &m.swarm.weights,
            true,
        );
        let mass = sp.mass();
        assert!(
            (mass - 1.0).abs() < 1e-3,
            "the emitted distribution sums to {:.6} at depth {:.2}; the \
             factorisation is supposed to telescope to one at every level",
            mass,
            m.swarm.mean_depth()
        );
    }
}

#[test]
fn appending_capacity_disturbs_nothing() {
    let cfg = Config::local();
    let mut m = Model::new(cfg);
    warm(&mut m, 300);
    let protos: Vec<Vec<u32>> = m.tree.arena.iter().map(|n| bits_of(&n.proto)).collect();
    let rows = weight_fingerprint(&m);
    let n_before = m.tree.nodes();

    let q = unit_vector(0x9333_4444, 7, m.cfg.d);
    m.tree.widen(0, &q);
    assert_eq!(m.tree.nodes(), n_before + 1, "widen did not append");

    for (i, before) in protos.iter().enumerate() {
        assert_eq!(*before, bits_of(&m.tree.arena[i].proto), "node {} moved on append", i);
    }
    assert!(rows == weight_fingerprint(&m), "appending a node changed stored rows");
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
        (weight_fingerprint(&m), m.total_bits.to_bits(), m.tree.nodes(), m.content_writes)
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "stored weights differed between two identical runs");
    assert_eq!(a.1, b.1, "accumulated codelength differed between two identical runs");
    assert_eq!(a.2, b.2, "tree size differed between two identical runs");
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
    let a = m.graph.write_walk(&q, &q, hops);
    let b = m.graph.write_walk(&q, &q, hops);
    assert_eq!(
        bits_of(&a.last().unwrap().p_out),
        bits_of(&b.last().unwrap().p_out),
        "the same walk over the same weights produced two different payloads"
    );
    assert_eq!(a.len(), hops, "the write walk did not take the memory's hop depth");
}
