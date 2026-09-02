//! Generator self-tests.
//!
//! Suspect the data before the mechanism. Every one of these runs on the emitted
//! ticks, not on the construction, because a generator that is correct by
//! argument and broken in fact is exactly how a measurement gets produced by the
//! measurement.

use alm::gen::{GenConfig, Generator, Kind, Mode};
use alm::gencheck;

fn build(mode: Mode, ticks: usize) -> alm::gen::Stream {
    let mut g = GenConfig::local();
    g.mode = mode;
    g.span_ticks = 2000;
    Generator::new(g).generate(ticks)
}

#[test]
fn only_the_conjunction_identifies_the_target() {
    let s = build(Mode::A, 200_000);
    let g = GenConfig::local();
    let r = gencheck::check(&s, g.answer_gap, &g.separations);
    r.print();

    // The Latin square is supposed to make each cue's marginal information
    // exactly zero and the pair's information log2(m).
    // Unconditionally a mode-A cue names its regime, so the marginals are
    // dominated by log2(domains). The conjunction claim is a within-regime
    // claim and has to be measured as one.
    assert!(
        r.mi_within_cue_a < 0.25,
        "within a regime, cue A alone already carries {:.3} bits",
        r.mi_within_cue_a
    );
    assert!(
        r.mi_within_cue_b < 0.25,
        "within a regime, cue B alone already carries {:.3} bits",
        r.mi_within_cue_b
    );
    let expected = (g.square_m as f64).log2();
    assert!(
        r.mi_within_pair > expected - 0.5,
        "the pair carries {:.3} bits within a regime, expected about {:.3}",
        r.mi_within_pair,
        expected
    );
    assert!(r.conjunctive_gain > 1.0, "conjunctive gain only {:.3}", r.conjunctive_gain);
}

#[test]
fn every_separation_is_populated() {
    let s = build(Mode::A, 200_000);
    let g = GenConfig::local();
    let r = gencheck::check(&s, g.answer_gap, &g.separations);
    for (sep, n) in r.sep_counts.iter() {
        assert!(*n >= 40, "separation {} has only {} episodes", sep, n);
    }
}

#[test]
fn composition_queries_are_never_presented_as_facts() {
    let s = build(Mode::A, 200_000);
    let g = GenConfig::local();
    let r = gencheck::check(&s, g.answer_gap, &g.separations);
    assert_eq!(r.comp_query_leaked, 0);
    assert!(r.comp_support > 0 && r.comp_query > 0, "no composition items were emitted");

    // Both support facts must exist for a query to be answerable at all.
    let gen = Generator::new(GenConfig::local());
    let (chains, r1, r2, r12) = gen.chains_of(0);
    assert!(!chains.is_empty());
    assert!(r1 != r2 && r2 != r12);
}

#[test]
fn mode_b_removes_the_routing_signal() {
    let gen_a = Generator::new({
        let mut g = GenConfig::local();
        g.mode = Mode::A;
        g
    });
    let gen_b = Generator::new({
        let mut g = GenConfig::local();
        g.mode = Mode::B;
        g
    });
    assert_ne!(gen_a.entities_of(0), gen_a.entities_of(1), "mode A should separate entities");
    assert_eq!(
        gen_b.entities_of(0),
        gen_b.entities_of(1),
        "mode B is supposed to make the entity surface byte-identical across regimes"
    );
    // Targets stay disjoint in both modes, which is what keeps the regimes
    // distinguishable at all.
    for t in gen_b.targets_of(0) {
        assert!(!gen_b.targets_of(1).contains(t));
    }
}

#[test]
fn segmentation_comes_from_the_baseline_and_is_genuinely_ambiguous() {
    let s = build(Mode::A, 200_000);
    let g = GenConfig::local();
    let r = gencheck::check(&s, g.answer_gap, &g.separations);
    // Long baseline runs also precede the second cue of a wide-separation item,
    // so a run length rule cannot recover the resolutions exactly. That is the
    // honest case and is worth pinning: if this were 1.0 the task would be
    // handing the model its own segmentation.
    assert!(
        r.segmentation_purity < 0.98,
        "segmentation purity {:.3}: baseline runs identify resolutions almost \
         perfectly, which would make the setting easier than intended",
        r.segmentation_purity
    );
    assert!(r.segmentation_purity > 0.2, "baseline structure carries no signal at all");
}

#[test]
fn episode_geometry_matches_the_records() {
    let s = build(Mode::A, 100_000);
    for e in s.episodes.iter().take(500) {
        // The target sits exactly `answer_gap` baselines after the last cue.
        assert_eq!(
            e.target_tick,
            e.last_cue_tick + 1 + e.answer_gap as usize,
            "episode geometry disagrees with its record"
        );
        for t in (e.last_cue_tick + 1)..e.target_tick {
            assert!(s.observe(t).is_none(), "the answer gap contains a token");
        }
        assert_eq!(s.observe(e.target_tick), Some(e.target));
        if e.kind == Kind::Second {
            assert_eq!(e.cues.len(), 2);
        }
    }
}
