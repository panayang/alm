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

use crate::clinical::{panel_streams, ppm_bar};
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

pub fn run(dir: &str, label: &str, ds: &[usize], patients: usize, seed: u64, cap: usize) {
    let (mut streams, n_param, vp) = panel_streams(dir);
    if patients > 0 && streams.len() > patients {
        streams.truncate(patients);
    }
    let events: usize = streams.iter().map(|s| s.len()).sum();
    println!("==================== {}   [tick loop] ====================", label);
    println!(
        "patients {}   events {}   parameters {}   panel vocabulary {}",
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
    // The bar, measured on exactly these patients. A bar from a different
    // subset is not a bar: PPM sees more context with more patients and the
    // number moves.
    let (b_blind, a_blind) = ppm_bar(&streams, vp, 3, false, 16);
    let (b_gap, a_gap) = ppm_bar(&streams, vp, 3, true, 16);
    println!("{:<36} {:>12.4} {:>10.4}", "PPM-C order 3, gap-blind  (the bar)", b_blind, a_blind);
    println!("{:<36} {:>12.4} {:>10.4}", "PPM-C order 3, gap in context", b_gap, a_gap);

    if arms.len() == 2 {
        println!(
            "\n  silence is worth {:+.4} bits/event to us, against the {:.4} the joint entropy says",
            arms[1].bits / arms[1].events.max(1) as f64
                - arms[0].bits / arms[0].events.max(1) as f64,
            0.4848
        );
        println!("  is there. That internal difference is the claim; the PPM row is the bar.");
    }
    println!(
        "\n  NOTE: the bar was measured on all {} patients. Compare only at the same subset.",
        3990
    );
    println!();
}
