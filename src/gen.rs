//! The stream.
//!
//! Suspect the data before the mechanism. A flat measurement on a source that
//! never contained the structure being measured says nothing about the
//! mechanism, and the cheapest way to be wrong here is to build a generator that
//! quietly lacks what the experiment is looking for. So this generator is
//! constructed so that each claim has something to bite on, and `gencheck.rs`
//! asserts -- on the emitted stream, not on the construction -- that it really
//! does.
//!
//! Three item types carry the three claims:
//!
//! * **First order.** A cue, a gap, a target. Tests retrieval across a gap.
//! * **Second order (Latin).** Two cues separated by a controlled interval, then
//!   a gap, then a target drawn from a Latin square over the cue pair. The square
//!   makes each cue's marginal information about the target *exactly zero*: only
//!   the conjunction identifies it.
//! * **Second order (product).** The same shape, but the target is a product
//!   code: cue A names one attribute, cue B names the other, and the pair names
//!   the cell. The marginals are non-zero here and that is not a defect -- it is
//!   forced. A zero-marginal table cannot be linearly separable: for each class
//!   the winning region is a permutation pattern, one cell per row and column,
//!   while an additive score w_c[a] + v_c[b] carves the grid into intersections
//!   of staircase half-spaces, which contain whole blocks and cannot isolate m
//!   scattered cells for m >= 3. So "zero marginals and linearly separable" is
//!   an empty requirement, and the two item types have to split the difference
//!   between them:
//!
//! ```text
//!       Latin   -- zero marginals, NOT representable by a linear readout on the
//!                  superposed cue embeddings. Only the address, or a term that
//!                  stores the pair, can answer it.
//!       Product -- non-zero marginals, IS representable. If the model handles
//!                  product items and fails Latin ones, the readout works and the
//!                  address is not isolating pairs; if it fails both, something
//!                  more basic is wrong.
//! ```
//! * **Composition.** `a r1 -> b` and `b r2 -> c` appear in the stream; the
//!   query `a r12 -> c` never does. Anything above the backoff prior here has to
//!   have come from the payload chain, since the address has nothing to find.
//!
//! Segmentation is carried by the baseline and nothing else. The model receives
//! `Option<usize>` per tick and never sees the episode records, which exist only
//! for measurement.
//!
//! # The curriculum
//!
//! Regimes **arrive and depart**. A domain becomes live, runs for a few spans,
//! and is never presented again. The pool is far larger than the number live at
//! any moment, so over a run the stream introduces far more content than is ever
//! concurrently active.
//!
//! This is the difference between a continual-learning source and a stationary
//! multi-domain one, and it was missing. With every domain recurring forever
//! nothing is ever old, there is no forgetting pressure at all, and a single
//! flat readout large enough to hold the whole vocabulary is not being asked the
//! question that allocation exists to answer. A flat control winning on such a
//! source says something about the source.
//!
//! Retention is measured by **probes on departed domains**, run under frozen
//! memory: the probe restores the context that regime needs, is scored, and the
//! displaced state is put back. Writing during a probe would be retraining on
//! the thing being tested.
//!
//! `stagger` off is the control the reference mechanism's appendix demands: a
//! regime arriving late benefits from entering an already-shaped system, and a
//! cost ratio below one cannot be read as amortisation until staggering has been
//! removed and the effect has survived.

use crate::num::{uniform, uniform_below};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Each regime owns its entities, so regime identity is present in the input.
    A,
    /// Entities are byte-identical across regimes and only targets differ. The
    /// adversarial bound, not the main setting.
    B,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Kind {
    First,
    /// Latin square: zero marginals, not linearly separable.
    Second,
    /// Product code: non-zero marginals, linearly separable.
    Product,
    /// One of the two support facts for a composition query.
    CompSupport,
    /// The composition query itself, never presented as a fact.
    ///
    /// Posed exactly once per chain, late in the regime's life. It used to be
    /// posed on 20% of chain draws, so after its first appearance it had been
    /// charged and written like any other pair and every later hit was a lookup
    /// -- which is why composition read 0.75 while its first exposure read
    /// 0.000. A chaining test survives exactly one question.
    CompQuery,
    /// A Latin cell that is never presented as a fact, asked once.
    ///
    /// Every family in this source is a pair presented repeatedly and then
    /// asked, so memorising the pair scores the same as learning the structure.
    /// A quarter of each square's cells are withheld from presentation and each
    /// is asked once, late. The square is `t = (a + b) mod m`, so a mechanism
    /// that has the rule answers them and a lookup table cannot.
    Generalize,
}

#[derive(Clone, Copy)]
pub enum Tick {
    Baseline,
    Token(usize),
}

/// What the metrics need and the model must never see.
#[derive(Clone)]
pub struct EpisodeRec {
    pub kind: Kind,
    pub domain: usize,
    /// Ticks between the two cues, for second-order items. Zero otherwise.
    pub separation: u32,
    /// Ticks of baseline between the last cue and the target.
    pub answer_gap: u32,
    pub target: usize,
    /// The cue tokens, in the order they were presented.
    pub cues: Vec<usize>,
    /// Index into `ticks` where the target is presented.
    pub target_tick: usize,
    /// Index into `ticks` of the last cue.
    pub last_cue_tick: usize,
}

/// One retention test on a departed regime.
///
/// The context tokens are presented first so the background is the one that
/// regime lived in -- an address conditioned on context cannot be probed from
/// the wrong context and the result would say nothing. Everything here runs
/// with memory frozen and the displaced state restored afterwards.
#[derive(Clone)]
pub struct ProbeSpec {
    pub domain: usize,
    /// Tokens presented to restore the regime's background.
    pub context: Vec<usize>,
    /// The cue tokens of the fact being probed.
    pub cue: Vec<usize>,
    pub target: usize,
    pub kind: Kind,
    /// Spans since this domain last appeared in the stream.
    pub age_spans: u32,
}

pub struct Stream {
    pub ticks: Vec<Tick>,
    pub episodes: Vec<EpisodeRec>,
    /// `ep_at[t]` is the episode whose target sits at tick t, if any.
    pub ep_at: Vec<Option<usize>>,
    /// Probe batches to run at a given tick, under frozen memory.
    pub probe_at: Vec<Vec<ProbeSpec>>,
    pub vocab: usize,
}

impl Stream {
    pub fn len(&self) -> usize {
        self.ticks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ticks.is_empty()
    }
    pub fn probes(&self) -> usize {
        self.probe_at.iter().map(|v| v.len()).sum()
    }

    /// The only thing the model is allowed to consume.
    #[inline]
    pub fn observe(&self, t: usize) -> Option<usize> {
        match self.ticks[t] {
            Tick::Baseline => None,
            Tick::Token(x) => Some(x),
        }
    }
}

#[derive(Clone)]
pub struct GenConfig {
    pub vocab: usize,
    pub mode: Mode,
    pub domains: usize,
    pub ents_per_domain: usize,
    pub tgts_per_domain: usize,
    /// Side of the Latin square, i.e. how many distinct cueA, cueB and targets
    /// take part in the Latin second-order items of each domain.
    pub square_m: usize,
    /// Product code: cue A takes `product_na` values, cue B takes `product_nb`,
    /// and the target is the cell, so there are na*nb of them.
    pub product_na: usize,
    pub product_nb: usize,
    pub zipf_s: f64,
    /// Ticks in one span.
    pub span_ticks: usize,
    /// How many domains are live at once.
    pub concurrent: usize,
    /// How many spans a domain stays live before departing for good.
    pub lifetime_spans: usize,
    /// Whether domains arrive staggered. Off means every domain is live from
    /// the start, which is the control for arrival-order effects.
    pub stagger: bool,
    /// Spans between batches of retention probes.
    pub probe_every_spans: usize,
    /// Probes per batch, spread over the departed domains by age.
    pub probes_per_batch: usize,
    /// Baseline ticks between the last cue and the target.
    pub answer_gap: u32,
    /// Baseline ticks after the target, before the next episode.
    pub tail_gap: u32,
    /// Separations swept by the second-order items, in ticks.
    pub separations: Vec<u32>,
    /// Relative frequency of the three item types.
    pub w_first: f64,
    pub w_second: f64,
    pub w_product: f64,
    pub w_comp: f64,
    pub seed: u64,
}

impl GenConfig {
    /// A shorter curriculum for screening.
    ///
    /// The tick floor is set by the data check, not by preference: the
    /// conjunction estimate needs comfortably more second-order items per regime
    /// than the square has cells, and spreading the same stream over
    /// thirty-two regimes pushes that past two hundred thousand ticks. Fewer
    /// regimes concentrates the items, so the same assertion passes at a third
    /// of the length while the curriculum keeps its shape -- regimes still
    /// arrive, run and depart for good, which is the property that makes this a
    /// continual source at all.
    pub fn fast() -> Self {
        let mut g = GenConfig::local();
        g.domains = 12;
        g.span_ticks = 2000;
        g
    }

    pub fn local() -> Self {
        GenConfig {
            vocab: 4096,
            mode: Mode::A,
            domains: 32,
            ents_per_domain: 32,
            tgts_per_domain: 32,
            square_m: 6,
            product_na: 3,
            product_nb: 2,
            zipf_s: 1.0,
            span_ticks: 4000,
            concurrent: 3,
            lifetime_spans: 3,
            stagger: true,
            probe_every_spans: 2,
            probes_per_batch: 24,
            answer_gap: 6,
            tail_gap: 4,
            separations: vec![1, 2, 4, 8, 16, 32, 64],
            w_first: 0.30,
            w_second: 0.28,
            w_product: 0.28,
            w_comp: 0.14,
            seed: 0xA11CE,
        }
    }
}

struct Domain {
    ents: Vec<usize>,
    tgts: Vec<usize>,
    /// square[a * m + b] indexes into `sq_tgts`.
    square: Vec<usize>,
    sq_a: Vec<usize>,
    sq_b: Vec<usize>,
    sq_tgts: Vec<usize>,
    pc_a: Vec<usize>,
    pc_b: Vec<usize>,
    /// pc_tgts[a * nb + b] is the cell named by the pair.
    pc_tgts: Vec<usize>,
    /// First-order facts: (cue, target).
    firsts: Vec<(usize, usize)>,
    /// Composition chains: (a, b, c) with relation tokens r1, r2, r12.
    chains: Vec<(usize, usize, usize)>,
    /// Square cells (a, b) withheld from presentation, asked once each.
    held_out: Vec<(usize, usize)>,
    r1: usize,
    r2: usize,
    r12: usize,
}

pub struct Generator {
    pub cfg: GenConfig,
    domains: Vec<Domain>,
    zipf_cdf: Vec<f64>,
}

impl Generator {
    pub fn new(cfg: GenConfig) -> Self {
        let m = cfg.square_m;
        let block = cfg.ents_per_domain + cfg.tgts_per_domain + 3;
        assert!(
            block * cfg.domains <= cfg.vocab,
            "vocabulary too small for {} domains",
            cfg.domains
        );
        let mut domains = Vec::new();
        for d in 0..cfg.domains {
            // Mode B makes every regime share the same entity surface, so the
            // routing signal is removed and only the targets differ.
            let ent_base = match cfg.mode {
                Mode::A => d * block,
                Mode::B => 0,
            };
            let tgt_base = d * block + cfg.ents_per_domain;
            let ents: Vec<usize> = (0..cfg.ents_per_domain).map(|i| ent_base + i).collect();
            let tgts: Vec<usize> = (0..cfg.tgts_per_domain).map(|i| tgt_base + i).collect();
            let r1 = tgt_base + cfg.tgts_per_domain;
            let r2 = r1 + 1;
            let r12 = r1 + 2;

            let (na, nb) = (cfg.product_na, cfg.product_nb);
            // Token budget, laid out so no two item types share a cue or a
            // target and the measurements cannot contaminate each other.
            let ent_sq = 2 * m;
            let ent_pc = na + nb;
            let free_ent = ent_sq + ent_pc;
            let tgt_pc = na * nb;
            let free_tgt = m + tgt_pc;
            assert!(
                ents.len() > free_ent + 8 && tgts.len() > free_tgt + 8,
                "not enough tokens per domain for all item types"
            );

            let sq_a: Vec<usize> = ents[0..m].to_vec();
            let sq_b: Vec<usize> = ents[m..2 * m].to_vec();
            let sq_tgts: Vec<usize> = tgts[0..m].to_vec();
            let pc_a: Vec<usize> = ents[ent_sq..ent_sq + na].to_vec();
            let pc_b: Vec<usize> = ents[ent_sq + na..free_ent].to_vec();
            let pc_tgts: Vec<usize> = tgts[m..free_tgt].to_vec();
            // A Latin square: T[a][b] = (a + b) mod m. Every row and every
            // column is a permutation of the targets, so I(target ; cueA) and
            // I(target ; cueB) are exactly zero while I(target ; cueA, cueB) is
            // log m. Nothing but the conjunction identifies the answer.
            let mut square = vec![0usize; m * m];
            for a in 0..m {
                for b in 0..m {
                    square[a * m + b] = (a + b) % m;
                }
            }

            // Composition chains a -> b -> c, taken first so they own their
            // tokens outright. Previously chains and first-order facts were both
            // laid down from `free_ent` / `free_tgt`, so chain k = 0 had
            // (a, c) == firsts[0]: the composition query was answerable from a
            // first-order fact the stream had already presented, and the whole
            // point of the family -- that the answer exists only as a chain --
            // was lost. The leak guard could not see it, because it only
            // compared query pairs against support pairs.
            let mut chains = Vec::new();
            let n_chain = 8.min((ents.len() - free_ent) / 3).min(tgts.len() - free_tgt);
            for k in 0..n_chain {
                let a = ents[free_ent + 3 * k];
                let b = ents[free_ent + 3 * k + 1];
                let c = tgts[free_tgt + k];
                chains.push((a, b, c));
            }

            // First-order facts start past everything the chains claimed.
            let mut firsts = Vec::new();
            let mut i = free_ent + 3 * n_chain;
            let mut j = free_tgt + n_chain;
            while i < ents.len() && j < tgts.len() {
                firsts.push((ents[i], tgts[j]));
                i += 1;
                j += 1;
            }

            // The diagonal: exactly one withheld cell per row and per column.
            //
            // Any withholding perturbs the shown marginals -- a row missing one
            // of its m targets has P(T|A) uniform over m-1 rather than m, worth
            // log2(m) - log2(m-1) = 0.263 bits at m = 6. The diagonal at least
            // keeps the two sides symmetric, so cue A and cue B stay equally
            // (un)informative and gencheck's single-cue baseline measures the
            // residue instead of it hiding in an asymmetry. A scattered set
            // withholds one cell from some rows and two from others and does
            // not.
            let held_out: Vec<(usize, usize)> = (0..m).map(|a| (a, a)).collect();

            domains.push(Domain {
                ents,
                held_out,
                tgts,
                square,
                sq_a,
                sq_b,
                sq_tgts,
                pc_a,
                pc_b,
                pc_tgts,
                firsts,
                chains,
                r1,
                r2,
                r12,
            });
        }

        // Zipf over episode slots inside a domain, which is what makes hub items
        // frequent and tail items rare.
        let slots = 256usize;
        let mut cdf = Vec::with_capacity(slots);
        let mut acc = 0.0f64;
        for i in 0..slots {
            acc += 1.0 / ((i + 1) as f64).powf(cfg.zipf_s);
            cdf.push(acc);
        }
        for v in cdf.iter_mut() {
            *v /= acc;
        }

        Generator { cfg, domains, zipf_cdf: cdf }
    }

    /// The domains live during span `s`.
    ///
    /// Staggered: domain d is live for `lifetime_spans` spans starting at span
    /// d, so `concurrent` of them overlap and each departs permanently. Not
    /// staggered: every domain is live throughout, which is the control -- it
    /// removes arrival order without removing anything else.
    pub fn live_domains(&self, s: usize) -> Vec<usize> {
        let cfg = &self.cfg;
        if !cfg.stagger {
            return (0..cfg.domains).collect();
        }
        let stride = cfg.lifetime_spans.max(1) / cfg.concurrent.max(1);
        let stride = stride.max(1);
        let mut live = Vec::new();
        for d in 0..cfg.domains {
            let start = d * stride;
            if s >= start && s < start + cfg.lifetime_spans {
                live.push(d);
            }
        }
        if live.is_empty() {
            // Past the end of the curriculum the last domains stay live rather
            // than emitting nothing.
            let last = cfg.domains.saturating_sub(cfg.concurrent);
            live = (last..cfg.domains).collect();
        }
        live
    }

    /// The last span in which `d` was live, or None if it has not departed by
    /// span `s`.
    fn departed_at(&self, d: usize, s: usize) -> Option<usize> {
        if !self.cfg.stagger {
            return None;
        }
        let stride = (self.cfg.lifetime_spans.max(1) / self.cfg.concurrent.max(1)).max(1);
        let end = d * stride + self.cfg.lifetime_spans;
        if end <= s {
            Some(end)
        } else {
            None
        }
    }

    fn zipf_pick(&self, key: u64, idx: u64, n: usize) -> usize {
        let u = uniform(key, idx) as f64;
        let mut lo = 0usize;
        let mut hi = self.zipf_cdf.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.zipf_cdf[mid] < u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo % n.max(1)
    }

    pub fn generate(&self, total_ticks: usize) -> Stream {
        let cfg = &self.cfg;
        let key = cfg.seed;
        let mut ticks: Vec<Tick> = Vec::with_capacity(total_ticks + 64);
        let mut episodes: Vec<EpisodeRec> = Vec::new();
        let mut counter: u64 = 0;

        let mut probe_at: Vec<Vec<ProbeSpec>> = Vec::new();
        let mut last_span = usize::MAX;

        while ticks.len() < total_ticks {
            let span = ticks.len() / cfg.span_ticks;
            if span != last_span {
                last_span = span;

                // The final exam. When a regime is in its last span, every
                // withheld cell and every composition chain is asked exactly
                // once, and never again -- so nothing here can be answered from
                // a copy of the question. These are the only items in the source
                // whose answer was never presented as a fact, and the only ones
                // that a lookup table cannot reach.
                if span > 0 {
                    for d in 0..cfg.domains {
                        let stride =
                            (cfg.lifetime_spans.max(1) / cfg.concurrent.max(1)).max(1);
                        let last = d * stride + cfg.lifetime_spans - 1;
                        if last != span {
                            continue;
                        }
                        let dm = &self.domains[d];
                        for &(a, b) in dm.held_out.iter() {
                            let tgt = dm.sq_tgts[dm.square[a * cfg.square_m + b]];
                            Self::emit_pair(
                                &mut ticks, &mut episodes, cfg, d,
                                dm.sq_a[a], dm.sq_b[b], tgt, Kind::Generalize,
                            );
                        }
                        for &(a, _b, c) in dm.chains.iter() {
                            Self::emit_pair(
                                &mut ticks, &mut episodes, cfg, d,
                                a, dm.r12, c, Kind::CompQuery,
                            );
                        }
                    }
                }

                // At a span boundary, test what has already departed.
                if span > 0 && span % cfg.probe_every_spans == 0 {
                    let batch = self.build_probes(span, counter);
                    if !batch.is_empty() {
                        while probe_at.len() <= ticks.len() {
                            probe_at.push(Vec::new());
                        }
                        probe_at[ticks.len()] = batch;
                    }
                }
            }
            let live = self.live_domains(span);
            let domain = live[uniform_below(key ^ 0x71, counter, live.len() as u64) as usize];
            let dom = &self.domains[domain];
            counter += 1;

            let r = uniform(key ^ 0x11, counter) as f64;
            let total_w = cfg.w_first + cfg.w_second + cfg.w_product + cfg.w_comp;
            let kind = if r < cfg.w_first / total_w {
                Kind::First
            } else if r < (cfg.w_first + cfg.w_second) / total_w {
                Kind::Second
            } else if r < (cfg.w_first + cfg.w_second + cfg.w_product) / total_w {
                Kind::Product
            } else {
                Kind::CompSupport
            };

            match kind {
                Kind::First => {
                    if dom.firsts.is_empty() {
                        ticks.push(Tick::Baseline);
                        continue;
                    }
                    let i = self.zipf_pick(key ^ 0x21, counter, dom.firsts.len());
                    let (cue, tgt) = dom.firsts[i];
                    let last_cue_tick = ticks.len();
                    ticks.push(Tick::Token(cue));
                    for _ in 0..cfg.answer_gap {
                        ticks.push(Tick::Baseline);
                    }
                    let target_tick = ticks.len();
                    ticks.push(Tick::Token(tgt));
                    episodes.push(EpisodeRec {
                        kind: Kind::First,
                        domain,
                        separation: 0,
                        answer_gap: cfg.answer_gap,
                        target: tgt,
                        cues: vec![cue],
                        target_tick,
                        last_cue_tick,
                    });
                }
                Kind::Second => {
                    let m = cfg.square_m;
                    // Both cues uniform, and it has to be both.
                    //
                    // For T = (a+b) mod m, P(T|A=a) is uniform exactly when
                    // p_B is, and P(T|B=b) is uniform exactly when p_A is. So
                    // zero marginals on both sides require both cue
                    // distributions uniform -- no choice of Latin square
                    // changes that. `a` Zipf with `b` uniform gave I(T;B) =
                    // 0.067 bits and a single-cue ceiling of 0.2745, above
                    // every Latin accuracy the suite had ever reported; making
                    // both Zipf only makes the leak symmetric and leaves that
                    // ceiling exactly where it was.
                    //
                    // The frequency skew therefore lives outside the square:
                    // which regime and which item family are presented, the
                    // product code's cue A (that family is deliberately
                    // non-zero-marginal and is the control for exactly this),
                    // and which chain is drawn.
                    let mut a = uniform_below(key ^ 0x31, counter, m as u64) as usize;
                    let mut b = uniform_below(key ^ 0x32, counter, m as u64) as usize;
                    // Withheld cells are never presented. Resample rather than
                    // nudge: advancing to the next cell piles probability onto
                    // whatever follows a withheld one, which cost 0.304 bits of
                    // cue-A marginal and was caught by the generator's own guard.
                    let mut tries = 0u64;
                    while dom.held_out.contains(&(a, b)) && tries < 16 {
                        tries += 1;
                        a = uniform_below(key ^ 0x33, counter ^ (tries << 32), m as u64) as usize;
                        b = uniform_below(key ^ 0x34, counter ^ (tries << 40), m as u64) as usize;
                    }
                    let sep_i =
                        uniform_below(key ^ 0x33, counter, cfg.separations.len() as u64) as usize;
                    let sep = cfg.separations[sep_i];
                    let tgt = dom.sq_tgts[dom.square[a * m + b]];

                    ticks.push(Tick::Token(dom.sq_a[a]));
                    for _ in 0..sep {
                        ticks.push(Tick::Baseline);
                    }
                    let last_cue_tick = ticks.len();
                    ticks.push(Tick::Token(dom.sq_b[b]));
                    for _ in 0..cfg.answer_gap {
                        ticks.push(Tick::Baseline);
                    }
                    let target_tick = ticks.len();
                    ticks.push(Tick::Token(tgt));
                    episodes.push(EpisodeRec {
                        kind: Kind::Second,
                        domain,
                        separation: sep,
                        answer_gap: cfg.answer_gap,
                        target: tgt,
                        cues: vec![dom.sq_a[a], dom.sq_b[b]],
                        target_tick,
                        last_cue_tick,
                    });
                }
                Kind::Product => {
                    let (na, nb) = (cfg.product_na, cfg.product_nb);
                    let a = self.zipf_pick(key ^ 0x51, counter, na);
                    let b = uniform_below(key ^ 0x52, counter, nb as u64) as usize;
                    let sep_i =
                        uniform_below(key ^ 0x53, counter, cfg.separations.len() as u64) as usize;
                    let sep = cfg.separations[sep_i];
                    let tgt = dom.pc_tgts[a * nb + b];

                    ticks.push(Tick::Token(dom.pc_a[a]));
                    for _ in 0..sep {
                        ticks.push(Tick::Baseline);
                    }
                    let last_cue_tick = ticks.len();
                    ticks.push(Tick::Token(dom.pc_b[b]));
                    for _ in 0..cfg.answer_gap {
                        ticks.push(Tick::Baseline);
                    }
                    let target_tick = ticks.len();
                    ticks.push(Tick::Token(tgt));
                    episodes.push(EpisodeRec {
                        kind: Kind::Product,
                        domain,
                        separation: sep,
                        answer_gap: cfg.answer_gap,
                        target: tgt,
                        cues: vec![dom.pc_a[a], dom.pc_b[b]],
                        target_tick,
                        last_cue_tick,
                    });
                }
                _ => {
                    if dom.chains.is_empty() {
                        ticks.push(Tick::Baseline);
                        continue;
                    }
                    let i = self.zipf_pick(key ^ 0x41, counter, dom.chains.len());
                    let (a, b, c) = dom.chains[i];
                    // Most of the time present one of the two support facts;
                    // occasionally pose the composition query, which has never
                    // been presented as a fact.
                    // Supports only. The query is posed once per chain, on the
                    // schedule below, so it can never be answered from a copy of
                    // itself.
                    let roll = uniform(key ^ 0x42, counter);
                    if roll < 0.5 {
                        Self::emit_pair(&mut ticks, &mut episodes, cfg, domain, a, dom.r1, b, Kind::CompSupport);
                    } else {
                        Self::emit_pair(&mut ticks, &mut episodes, cfg, domain, b, dom.r2, c, Kind::CompSupport);
                    }
                }
            }

            for _ in 0..cfg.tail_gap {
                ticks.push(Tick::Baseline);
            }
        }

        ticks.truncate(total_ticks);
        episodes.retain(|e| e.target_tick < total_ticks);
        let mut ep_at = vec![None; ticks.len()];
        for (i, e) in episodes.iter().enumerate() {
            ep_at[e.target_tick] = Some(i);
        }
        probe_at.resize(ticks.len(), Vec::new());
        Stream { ticks, episodes, ep_at, probe_at, vocab: cfg.vocab }
    }

    /// One batch of retention probes, spread over the departed domains so that
    /// recently departed and long departed regimes are both represented.
    fn build_probes(&self, span: usize, counter: u64) -> Vec<ProbeSpec> {
        let cfg = &self.cfg;
        let mut departed: Vec<(usize, u32)> = Vec::new();
        for d in 0..cfg.domains {
            if let Some(end) = self.departed_at(d, span) {
                departed.push((d, (span - end) as u32));
            }
        }
        if departed.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(cfg.probes_per_batch);
        for i in 0..cfg.probes_per_batch {
            let (d, age) = departed[i % departed.len()];
            let dom = &self.domains[d];
            let m = cfg.square_m;
            let (na, nb) = (cfg.product_na, cfg.product_nb);
            // Context: entities of this regime that are *not* cues of any item
            // family, so the background is the one the regime lived in without
            // the probe being run against four competing cue-A tokens sitting in
            // the ladder and in `event_hist`. `ents[..2m]` are the square's cues
            // and `ents[2m..free_ent]` the product code's, so the context is
            // drawn from past them -- an interference condition the live stream
            // never presents was otherwise being applied to every probe.
            let ctx_base = 2 * m + na + nb;
            let ctx: Vec<usize> =
                dom.ents.iter().skip(ctx_base).take(4).copied().collect();
            // Rotate through the item types so retention is not measured on one
            // kind of fact only.
            let (cue, target, kind) = match i % 3 {
                0 if !dom.firsts.is_empty() => {
                    let j = (counter as usize + i) % dom.firsts.len();
                    let (c, t) = dom.firsts[j];
                    (vec![c], t, Kind::First)
                }
                1 => {
                    let a = (counter as usize + i) % m;
                    let b = (counter as usize + 2 * i) % m;
                    (vec![dom.sq_a[a], dom.sq_b[b]], dom.sq_tgts[dom.square[a * m + b]], Kind::Second)
                }
                _ => {
                    let a = (counter as usize + i) % na;
                    let b = (counter as usize + 2 * i) % nb;
                    (vec![dom.pc_a[a], dom.pc_b[b]], dom.pc_tgts[a * nb + b], Kind::Product)
                }
            };
            out.push(ProbeSpec { domain: d, context: ctx, cue, target, kind, age_spans: age });
        }
        out
    }

    fn emit_pair(
        ticks: &mut Vec<Tick>,
        episodes: &mut Vec<EpisodeRec>,
        cfg: &GenConfig,
        domain: usize,
        head: usize,
        rel: usize,
        tail: usize,
        kind: Kind,
    ) {
        ticks.push(Tick::Token(head));
        let last_cue_tick = ticks.len();
        ticks.push(Tick::Token(rel));
        for _ in 0..cfg.answer_gap {
            ticks.push(Tick::Baseline);
        }
        let target_tick = ticks.len();
        ticks.push(Tick::Token(tail));
        episodes.push(EpisodeRec {
            kind,
            domain,
            separation: 0,
            answer_gap: cfg.answer_gap,
            target: tail,
            cues: vec![head, rel],
            target_tick,
            last_cue_tick,
        });
    }

    pub fn product_of(&self, domain: usize) -> (&[usize], &[usize], &[usize]) {
        let d = &self.domains[domain];
        (&d.pc_a, &d.pc_b, &d.pc_tgts)
    }

    /// Ground-truth tables, for the generator self-tests only.
    pub fn square_of(&self, domain: usize) -> (usize, &[usize], &[usize], &[usize], &[usize]) {
        let d = &self.domains[domain];
        (self.cfg.square_m, &d.sq_a, &d.sq_b, &d.sq_tgts, &d.square)
    }

    pub fn chains_of(&self, domain: usize) -> (&[(usize, usize, usize)], usize, usize, usize) {
        let d = &self.domains[domain];
        (&d.chains, d.r1, d.r2, d.r12)
    }

    pub fn entities_of(&self, domain: usize) -> &[usize] {
        &self.domains[domain].ents
    }

    pub fn targets_of(&self, domain: usize) -> &[usize] {
        &self.domains[domain].tgts
    }
}
