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
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("quick");

    let mut ticks: usize = match cmd {
        "full" => 600_000,
        "screen" => 300_000,
        _ => 250_000,
    };
    let mut seed: u64 = 0x5EED_1234;
    let mut out: Option<String> = None;

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
            other => panic!("unknown argument {}", other),
        }
    }

    match cmd {
        "gencheck" => {
            let mut g = GenConfig::local();
            g.seed = seed ^ 0xA11CE;
            let stream = experiments::build_stream(&g, ticks, 20, true);
            println!("stream ok: {} ticks, {} episodes", stream.len(), stream.episodes.len());
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
        other => panic!("unknown command {} (try gencheck, screen, quick, full)", other),
    }
}
