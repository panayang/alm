//! Can the mechanism retrieve a value by its key at all?
//!
//! **Open-loop instrument.** The world's next event here is fixed in advance
//! and ignores what the model says, so this measures memory and prediction,
//! not response. It is kept as a record; it is not the test this design is
//! for (see the top of `lib.rs`).
//!
//! On MultiWOZ the model answers "what does the user want for slot s" at
//! 0.52 against 0.90 for the rule "the last value the user gave", and only
//! 0.53 when the value was said in the very same turn. Wrong answers there are
//! not confusions between values in the dialogue -- the same slot's value in
//! another domain accounts for none of them -- they are the prior (45%),
//! nothing (21%), or a value never mentioned (24%). The value just heard does
//! not come through. And where the clerk offered a value the user never took,
//! the model names the offer.
//!
//! Every earlier synthetic question was yes/no. Retrieving one value out of
//! many by its key, keeping who said it, was never asked with a known answer.
//! This asks it, with everything MultiWOZ varies held under control:
//!
//! ```text
//!   turn:     <user> slot value      the user sets slot (overwriting)
//!             <clerk> slot value     an offer; it sets nothing
//!             <user> filler          talk that sets nothing
//!   question: <ask> <user> slot  ...silence...  value | <none>   (spoken by the world)
//! ```
//!
//! The right answer is exactly the last value the *user* gave for that slot,
//! so the rule scores 1.000 by construction. What is varied: how many values
//! there are to choose among, how many turns ago the user set it, and whether
//! the clerk has since offered something else for it. The wiped arm erases
//! the episode before the question: what memory alone answers.

use crate::config::Config;
use crate::model::Model;
use crate::num::cbrng;

const SLOTS: usize = 8;
const FILLERS: usize = 8;
const SILENCE: usize = 6;

struct Toks {
    user: usize,
    clerk: usize,
    ask: usize,
    none: usize,
    slot0: usize,
    filler0: usize,
    value0: usize,
    vocab: usize,
}

fn toks(values: usize) -> Toks {
    let slot0 = 4;
    let filler0 = slot0 + SLOTS;
    let value0 = filler0 + FILLERS;
    Toks { user: 0, clerk: 1, ask: 2, none: 3, slot0, filler0, value0, vocab: value0 + values }
}

struct Rec {
    ok: bool,
    wiped_ok: bool,
    /// Turns since the user last set the asked slot; None if never.
    age: Option<usize>,
    /// The clerk offered a different value for this slot after the user's.
    offered_since: bool,
    conf: f64,
    /// The episode trace's recall at the question names the answer.
    recall_ok: bool,
    /// ... names something at all.
    recall_any: bool,
    /// What a wrong answer was: 0 none, 1 an offer, 2 the slot's older value,
    /// 3 another slot's value in this episode, 4 anything else.
    wrong: Option<usize>,
}

fn episode_run(values: usize, episodes: usize, seed: u64, set: fn(&mut Config)) -> Vec<Rec> {
    let t = toks(values);
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = t.vocab;
    cfg.d = 256;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    set(&mut cfg);
    cfg.derive();
    let mut m = Model::new(cfg);
    let blank = m.volatile();
    let cand: Vec<usize> = std::iter::once(t.none).chain(t.value0..t.value0 + values).collect();
    let read = |m: &Model| {
        let sc = m.spread_now();
        let ps: Vec<f64> = cand.iter().map(|&c| sc.prob_of(&m.store, c as u32) as f64).collect();
        let z: f64 = ps.iter().sum::<f64>().max(1e-30);
        let (bi, bp) = ps.iter().enumerate().fold((0, -1.0), |b, (i, &p)| if p > b.1 { (i, p) } else { b });
        (cand[bi], bp / z)
    };
    let mut out = Vec::with_capacity(episodes);
    for e in 0..episodes {
        if e > 0 && e % (episodes / 4).max(1) == 0 {
            eprintln!("  V={} values: {}/{} episodes, naming gain {:.2}", values, e, episodes, m.naming_gain());
        }
        m.restore(blank.clone());
        let r = |k: u64, i: u64| cbrng(seed ^ k, (e as u64) << 8 | i);
        let turns = 4 + (r(0x71, 0) % 9) as usize;
        // (turn, by user, slot, value)
        let mut said: Vec<(usize, bool, usize, usize)> = Vec::new();
        for ti in 0..turns {
            let user = r(0x5E, ti as u64) % 10 < 6;
            m.tick(Some(if user { t.user } else { t.clerk }), false);
            if r(0xC0, ti as u64) % 10 < 7 {
                let s = (r(0x51, ti as u64) % SLOTS as u64) as usize;
                let v = (r(0x7A, ti as u64) % values as u64) as usize;
                m.tick(Some(t.slot0 + s), false);
                m.tick(Some(t.value0 + v), false);
                said.push((ti, user, s, v));
            } else {
                m.tick(Some(t.filler0 + (r(0xF1, ti as u64) % FILLERS as u64) as usize), false);
            }
        }
        // Ask mostly about slots the user has set, so a value is usually the answer.
        let set_slots: Vec<usize> = (0..SLOTS).filter(|&s| said.iter().any(|x| x.1 && x.2 == s)).collect();
        let s = if !set_slots.is_empty() && r(0xA5, 0) % 10 < 7 {
            set_slots[(r(0xA6, 0) % set_slots.len() as u64) as usize]
        } else {
            (r(0xA7, 0) % SLOTS as u64) as usize
        };
        let last_user = said.iter().rev().find(|x| x.1 && x.2 == s).copied();
        let gold = last_user.map(|x| t.value0 + x.3).unwrap_or(t.none);
        let age = last_user.map(|x| turns - 1 - x.0);
        let offered_since =
            said.iter().any(|x| !x.1 && x.2 == s && last_user.map_or(true, |u| x.0 > u.0) && t.value0 + x.3 != gold);

        let saved = m.volatile();
        m.restore(blank.clone());
        m.frozen = true;
        m.tick(Some(t.ask), false);
        m.tick(Some(t.user), false);
        m.tick(Some(t.slot0 + s), false);
        for _ in 0..SILENCE {
            m.tick(None, false);
        }
        let wiped_ok = read(&m).0 == gold;
        m.frozen = false;
        m.restore(saved);

        m.tick(Some(t.ask), false);
        m.tick(Some(t.user), false);
        m.tick(Some(t.slot0 + s), false);
        let rec = m.episodic_recall();
        if std::env::var("ALM_SLOTS_DEBUG").is_ok() && e >= episodes - 6 {
            eprintln!("episode {}: asked slot {} gold {} recall {:?}; said {:?}", e, s, gold, rec,
                said.iter().map(|x| (x.1, t.slot0 + x.2, t.value0 + x.3)).collect::<Vec<_>>());
        }
        for _ in 0..SILENCE {
            m.tick(None, false);
        }
        let (top, conf) = read(&m);
        m.tick(Some(gold), false);
        let wrong = if top == gold {
            None
        } else if top == t.none {
            Some(0)
        } else if said.iter().any(|x| !x.1 && x.2 == s && t.value0 + x.3 == top) {
            Some(1)
        } else if said.iter().any(|x| x.1 && x.2 == s && t.value0 + x.3 == top) {
            Some(2)
        } else if said.iter().any(|x| t.value0 + x.3 == top) {
            Some(3)
        } else {
            Some(4)
        };
        out.push(Rec {
            ok: top == gold,
            wiped_ok,
            age,
            offered_since,
            conf,
            recall_ok: rec.map_or(false, |x| x.0 == gold),
            recall_any: rec.is_some(),
            wrong,
        });
    }
    out
}

/// Accuracy on a value the user set in the very last turn, 64 values to
/// choose among, second half read: the facts matrix's F11.
pub(crate) fn this_turn(episodes: usize, seed: u64, set: fn(&mut Config)) -> f64 {
    let recs = episode_run(64, episodes, seed, set);
    let r: Vec<&Rec> = recs[recs.len() / 2..].iter().filter(|a| a.age == Some(0) && !a.offered_since).collect();
    r.iter().filter(|a| a.ok).count() as f64 / r.len().max(1) as f64
}

fn report(values: usize, recs: &[Rec]) {
    let second = &recs[recs.len() / 2..];
    let row = |name: &str, sel: &dyn Fn(&Rec) -> bool| {
        let r: Vec<&Rec> = second.iter().filter(|a| sel(a)).collect();
        if r.is_empty() {
            return;
        }
        let f = |g: &dyn Fn(&Rec) -> bool| r.iter().filter(|a| g(a)).count() as f64 / r.len() as f64;
        let conf = r.iter().map(|a| a.conf).sum::<f64>() / r.len() as f64;
        println!(
            "  {:>30} {:>6} {:>7.3} {:>7.3} {:>6.3} {:>7.3} {:>7.3}",
            name,
            r.len(),
            f(&|a| a.ok),
            f(&|a| a.wiped_ok),
            conf,
            f(&|a| a.recall_ok),
            f(&|a| a.recall_any)
        );
    };
    println!("\n{} values to choose among (chance {:.3}); second half read", values, 1.0 / (values + 1) as f64);
    println!(
        "  {:>30} {:>6} {:>7} {:>7} {:>6} {:>7} {:>7}",
        "questions", "n", "ours", "wiped", "conf", "recall", "names"
    );
    row("all", &|_| true);
    row("answer is none", &|a| a.age.is_none());
    for g in 0..6 {
        row(&format!("user set it {} turns ago", g), &|a| a.age == Some(g) && !a.offered_since);
    }
    row("user set it 6+ turns ago", &|a| a.age.map_or(false, |x| x >= 6) && !a.offered_since);
    row("clerk offered another since", &|a| a.offered_since);
    let w: Vec<usize> = second.iter().filter_map(|a| a.wrong).collect();
    print!("  wrong answers ({}):", w.len());
    for (i, lab) in ["none", "an offer", "older value", "another slot's", "other"].iter().enumerate() {
        print!("  {} {:.3}", lab, w.iter().filter(|&&x| x == i).count() as f64 / w.len().max(1) as f64);
    }
    println!();
}

/// Walk one fixed episode through the trace and print what it recalls.
fn trace_walk(seed: u64) {
    let t = toks(8);
    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = t.vocab;
    cfg.d = 256;
    cfg.mem_banks = 4096;
    cfg.cleanup_floor_mult = 1.1;
    cfg.episodic = true;
    cfg.derive();
    let mut m = Model::new(cfg);
    let seq = [t.user, t.slot0 + 1, t.value0 + 3, t.clerk, t.slot0 + 2, t.value0 + 5, t.user, t.slot0 + 1, t.value0 + 6, t.ask, t.user, t.slot0 + 1];
    for &x in &seq {
        m.tick(Some(x), false);
        let blk = m.episodic_block().unwrap().to_vec();
        let mut c: Vec<(f32, usize)> =
            (0..t.vocab).map(|k| (crate::num::dot(&blk, m.emb.row(k)), k)).collect();
        c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        eprintln!("  after {:>3}: recall {:?}  top {:?}", x, m.episodic_recall(), &c[..3]);
    }
}

pub fn run(episodes: usize, seed: u64) {
    if std::env::var("ALM_SLOTS_WALK").is_ok() {
        trace_walk(seed);
        return;
    }
    println!("what is slot s now? the last value the user gave it; the rule scores 1.000");
    println!("  {} slots, turns 4-12, 60% user / 40% clerk, {} episodes per row", SLOTS, episodes);
    // ALM_SLOTS_VALUES=8,64 picks the value-set sizes; default all three.
    let sizes: Vec<usize> = std::env::var("ALM_SLOTS_VALUES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![8, 64, 512]);
    for &values in &sizes {
        if std::env::var("ALM_SLOTS_NO_DEFAULT").is_err() {
            println!("
==== default");
            let recs = episode_run(values, episodes, seed, |_| {});
            report(values, &recs);
        }
        // ALM_SLOTS_DECAY=0.97,1.0 runs one trace arm per decay.
        let decays: Vec<f32> = std::env::var("ALM_SLOTS_DECAY")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1.0]);
        for dk in decays {
            println!("
==== with the episode trace, decay {}", dk);
            std::env::set_var("ALM_SLOTS_DECAY_NOW", dk.to_string());
            let recs = episode_run(values, episodes, seed, |c| {
                c.episodic = true;
                if let Some(x) = std::env::var("ALM_SLOTS_CB").ok().and_then(|v| v.parse().ok()) {
                    c.episodic_codebook = x;
                }
                if let Some(x) = std::env::var("ALM_SLOTS_D").ok().and_then(|v| v.parse().ok()) {
                    c.d = x;
                }
                if let Some(x) = std::env::var("ALM_SLOTS_DECAY_NOW").ok().and_then(|v| v.parse().ok()) {
                    c.episodic_decay = x;
                }
            });
            report(values, &recs);
        }
    }
}
