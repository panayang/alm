# Raw run outputs — current mechanism

Every file here was produced by the mechanism described in `paper.tex`: unitary
codebook, FFT convolution, identity-hashed banks, weighted parallel responses,
strongest-response emission. Each is the stdout of one shard of one command;
shards were run in parallel and each prints the arms whose index it owns.

Reproduce with `cargo run --release -- <command> --ticks <n> --shard i/N`.

| prefix | command | what it backs |
|---|---|---|
| `CO_*` | `closeout --ticks 90000 --shard i/14` | REPORT §4, paper Table 8. The re-measurement of every conclusion the mechanism changes invalidated: background rungs, ladder feedback, the operator graph, live-vs-locked bands. |
| `E_*` | `unbindtest --ticks 90000 --shard i/6` | The first run with strongest-response emission, where the walk surface leaves zero. |
| `F_*` | `unbindtest --ticks 90000 --shard i/6` | Paper Table 5: response-to-answer cosine holding at 0.24–0.28 against a control at 0.015, and the single-response collapse. Same mechanism as `E_*` but averaging the responses rather than reading the strongest. |
| `R_*` | `unbindtest --ticks 90000 --shard i/6` | Bank-count arms placed by the capacity law, including one deliberately below the line. |
| `Z_*` | `unbindtest --ticks 90000 --shard i/6` | Per-tick response-to-answer trace; the instrument that separated "never retrieved" from "retrieved and stepped past". |
| `NG_*` | `negatives --ticks 90000 --shard i/4` | Paper Table 6: uniform against top-k negatives. |
| `BD_*` | `binddecay --ticks 90000 --shard i/4` | The `bind_decay` sweep. Monotone with no interior optimum, which is why the knob is pinned at 1.0. |

Single seed unless a filename says otherwise. The walk, one-shot composition and
withheld-cell families carry 60–96 events in total, so their cells hold six to
nine each; see the caution in `REPORT.md`.
