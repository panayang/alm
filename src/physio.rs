//! The tick loop on a real irregular stream.
//!
//! Everything up to here was an instrument. This is the first time the model
//! runs on a source it was not written for.
//!
//! The mapping follows from the architecture rather than from taste. A panel --
//! the set of parameters recorded at one observation minute -- is what the
//! world says, so it is a token tick and it is charged. The wait before it is
//! what the world does not say, so it becomes silent ticks, on which the model
//! keeps running and is charged nothing. Gaps run from one minute to twenty-five
//! hours, so they are spent logarithmically: a gap of g minutes buys
//! `floor(log2 g)` silent ticks, which is zero to eleven. Linear would give a
//! single wait fifteen hundred ticks and drown everything around it.
//!
//! # The comparison that decides it
//!
//! The bar is prequential PPM-C over the same panel sequence, fed the events
//! only and never diluted by silence, which is the stronger form of that
//! baseline:
//!
//! ```text
//!   gap-blind PPM-C, order 3      5.6502 bits/event
//!   gap in the PPM context        5.6973 bits/event   (worse: contexts fragment)
//!   marginal H(panel)             6.9218
//!   I(panel ; gap | prev panel)   0.4848 available
//! ```
//!
//! and the internal control is the same model with the silence removed. That
//! second one is the one that matters: if spending the gap as ticks changes
//! nothing, then this architecture's temporal commitment is decoration here as
//! it was everywhere else, whatever the number against PPM turns out to be.

use crate::clinical::{flat_streams, panel_streams, ppm_bar};
use crate::config::Config;
use crate::model::Model;

/// How many silent ticks a wait of `g` minutes buys.
#[inline]
fn silence_ticks(g: u32, mode: &str, cap: usize) -> usize {
    let t = match mode {
        "none" => 0,
        "log" => (32 - g.max(1).leading_zeros()).saturating_sub(1) as usize,
        "linear" => g as usize,
        other => panic!("--gapmode wants none, log or linear, got {}", other),
    };
    t.min(cap)
}

struct Arm {
    name: String,
    bits: f64,
    events: u64,
    hits: u64,
    ticks: u64,
}

pub fn run(
    dir: &str,
    label: &str,
    ds: &[usize],
    patients: usize,
    seed: u64,
    cap: usize,
    gran: &str,
) {
    let (mut streams, n_param, vp) = match gran {
        "panel" => panel_streams(dir),
        "param" => {
            let (s, v) = flat_streams(dir);
            (s, v, v)
        }
        other => panic!("--gran wants panel or param, got {}", other),
    };
    if patients > 0 && streams.len() > patients {
        streams.truncate(patients);
    }
    let events: usize = streams.iter().map(|s| s.len()).sum();
    println!("==================== {}   [tick loop, {} granularity] ====================", label, gran);
    println!(
        "patients {}   events {}   parameters {}   vocabulary {}",
        streams.len(),
        events,
        n_param,
        vp
    );

    let mut arms: Vec<Arm> = Vec::new();
    for &d in ds {
    for mode in ["log", "none"] {
        let mut cfg = Config::local();
        cfg.seed = seed;
        cfg.vocab = vp;
        cfg.d = d;
        // The threshold this project shipped was measured wrong on real data:
        // swept per cell at matched precision on omnetpp, the best multiplier
        // runs 1.00 deep inside the operating region to 1.17 at its edge, and
        // 1.6 costs up to forty-seven points of recall for four tenths of a
        // point of precision. Leaving it at 1.6 here would understate the
        // mechanism for a reason already known to be an error.
        cfg.cleanup_floor_mult = 1.1;
        cfg.derive();
        let mut model = Model::new(cfg);

        let mut a = Arm {
            name: format!("d={:<5} gap as {} silence", d, mode),
            bits: 0.0,
            events: 0,
            hits: 0,
            ticks: 0,
        };
        for s in streams.iter() {
            for &(gap, panel) in s.iter() {
                for _ in 0..silence_ticks(gap, mode, cap) {
                    model.tick(None, false);
                    a.ticks += 1;
                }
                let out = model.tick(Some(panel as usize), false);
                a.ticks += 1;
                if out.charged {
                    a.bits += out.bits;
                    a.events += 1;
                    if out.top1 == Some(panel as usize) {
                        a.hits += 1;
                    }
                }
            }
        }
        arms.push(a);
    }
    }

    println!("\n{:<36} {:>12} {:>10} {:>12}", "", "bits/event", "top-1", "ticks");
    for a in arms.iter() {
        println!(
            "{:<36} {:>12.4} {:>10.4} {:>12}",
            a.name,
            a.bits / a.events.max(1) as f64,
            a.hits as f64 / a.events.max(1) as f64,
            a.ticks
        );
    }
    // The bar, on exactly these patients -- a bar from a different subset is
    // not a bar, because PPM sees more context with more patients and its
    // number moves. And its order is swept rather than fixed at the first
    // setting that was tried, because a baseline held at one setting is not a
    // baseline either.
    let mut best_blind = (f64::MAX, 0.0, 0usize);
    let mut best_gap = (f64::MAX, 0.0, 0usize);
    for o in 1..=6 {
        let (b, a) = ppm_bar(&streams, vp, o, false, 16);
        println!("{:<36} {:>12.4} {:>10.4}", format!("PPM-C order {}, gap-blind", o), b, a);
        if b < best_blind.0 {
            best_blind = (b, a, o);
        }
        let (b, a) = ppm_bar(&streams, vp, o, true, 16);
        if b < best_gap.0 {
            best_gap = (b, a, o);
        }
    }
    println!(
        "{:<36} {:>12.4} {:>10.4}   <- the bar",
        format!("PPM-C best, order {}, gap-blind", best_blind.2),
        best_blind.0,
        best_blind.1
    );
    println!(
        "{:<36} {:>12.4} {:>10.4}",
        format!("PPM-C best, order {}, gap in context", best_gap.2),
        best_gap.0,
        best_gap.1
    );

    println!("\n  what the silence is worth to us, per width:");
    for (i, &d) in ds.iter().enumerate() {
        let with = arms[2 * i].bits / arms[2 * i].events.max(1) as f64;
        let without = arms[2 * i + 1].bits / arms[2 * i + 1].events.max(1) as f64;
        println!(
            "    d = {:<6} {:+.4} bits/event    best margin over the bar {:+.4}",
            d,
            without - with,
            best_blind.0 - with
        );
    }
    println!("  against the 0.4848 the joint entropy says is available. That internal");
    println!("  difference is the claim; the swept PPM row is the bar.");
    println!();
}
