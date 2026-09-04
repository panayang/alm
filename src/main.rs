//! Command line driver.
//!
//!   cargo run --release -- gencheck            data self-tests only
//!   cargo run --release -- quick               short run, everything
//!   cargo run --release -- full                the whole suite
//!
//! Options: --ticks N  --seed S  --out FILE

use std::fs;

use alm::experiments;
use alm::gen::GenConfig;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("screen");

    let mut ticks: usize = match cmd {
        "full" => 600_000,
        "screen" => 90_000,
        _ => 90_000,
    };
    let mut seed: u64 = 0x5EED_1234;
    let mut out: Option<String> = None;
    let (mut shard, mut shards) = (0usize, 1usize);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ticks" => {
                ticks = args[i + 1].parse().expect("--ticks wants a number");
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("--seed wants a number");
                i += 2;
            }
            "--out" => {
                out = Some(args[i + 1].clone());
                i += 2;
            }
            "--shard" => {
                let v = &args[i + 1];
                let (a, b) = v.split_once('/').expect("--shard wants i/n");
                shard = a.parse().expect("shard index");
                shards = b.parse().expect("shard count");
                i += 2;
            }
            other => panic!("unknown argument {}", other),
        }
    }

    match cmd {
        "gencheck" => {
            // Both rungs of the load sweep, so the manipulation each family
            // actually received is visible side by side.
            for domains in [12usize, 36] {
                let mut g = GenConfig::fast();
                g.seed = seed ^ 0xA11CE;
                g.domains = domains;
                let t = ticks * domains / 12;
                println!("== domains={} ticks={} ==", domains, t);
                let stream = experiments::build_stream(&g, t, 20, true);
                println!("stream ok: {} ticks, {} episodes", stream.len(), stream.episodes.len());
            }
        }
        // The baseline alone, so a reporting fix does not cost a full suite run.
        "baseline" => {
            // Same source the screening suite uses, or the two numbers are
            // not about the same stream.
            let mut g = GenConfig::fast();
            g.seed = seed ^ 0xA11CE;
            let stream = experiments::build_stream(&g, ticks, 20, false);
            for (name, skip) in [("like-for-like", false), ("silence removed (UPPER BOUND)", true)] {
                let r = alm::baseline::run(&stream, 4, skip);
                println!(
                    "[baseline {}] charged events: {:.3} bits/ev, acc {:.3} |                      second-order {:.3} bits, acc {:.3} | product {:.3} bits, acc {:.3}                      | per-symbol incl. silence {:.3} (different denominator)",
                    name,
                    r.charged.mean(), r.charged.accuracy(),
                    r.second.mean(), r.second.accuracy(),
                    r.product.mean(), r.product.accuracy(),
                    r.ppm.bits_per_event()
                );
            }
        }
        "seeds" => {
            let suite = experiments::seeds(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "route" => {
            let suite = experiments::route(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "mechanism" => {
            let suite = experiments::mechanism(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "width" => {
            let suite = experiments::width(ticks, seed);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "load" => {
            let suite = experiments::load_sweep(ticks, seed);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "scale" => {
            let suite = experiments::scale(ticks, seed);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "capacity" => {
            let suite = experiments::capacity(ticks, seed);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "screen" | "full" => {
            let suite = experiments::screen(ticks, seed, cmd == "full");
            println!();
            println!("== summary ====================================================");
            for line in suite.summary.iter() {
                println!("{}", line);
            }
            if let Some(path) = out {
                fs::write(&path, &suite.csv).expect("could not write csv");
                println!("
csv written to {}", path);
            }
        }
        other => panic!("unknown command {} (try gencheck, baseline, capacity, screen, full)", other),
    }
}
