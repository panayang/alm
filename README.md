# alm — one tick, one link

A challenge–response continual memory. **Not autoregressive**, no reward, no
labels, no gradient across a memory write. The only external signal is the
codelength of what the world actually said, charged on the ticks where it said
something.

One input per tick, one output per tick. In the intervals between the world's
tokens the input is the identity and the model keeps emitting — and because
speech means *there is something to combine* while silence means *there is
not*, the two halves of the loop are inverses selected by the world:

```
speech    M[hash(id₋₂, id₋₁)] += ν( (E₋₂ ⊛ E₋₁) ⊛ E_now )      bind, write
silence   cᵢ ← cleanup( M[hash(τᵢ, τ_arr)] ⊘ (cᵢ ⊛ E_arr) )    unbind, read
```

Depth is therefore neither a layer count nor a hyperparameter: it is **how long
the world stays quiet**. Several partial responses mature in parallel, and the
one that has matured is what gets said.

Rust, zero dependencies. The exactness assertions need bit-reproducible floating
point and a counter-based RNG whose draws do not depend on call order; both are
easier to guarantee by writing the code than by importing it.

## Run it

```bash
cargo test                        # 20 structural assertions + 6 generator self-tests
cargo run --release -- gencheck   # generator self-tests, both load levels
cargo run --release -- baseline   # PPM control, charged-event denominator
cargo run --release -- screen     # screening suite
cargo run --release -- closeout   # the re-measurement suite (paper Table 8)
```

Any command takes `--ticks N`, `--seed S`, and `--shard i/N` to split an arm
list across processes. Arms are single-threaded, so shard across cores:

```bash
for k in $(seq 0 13); do
  cargo run --release -- closeout --ticks 90000 --shard $k/14 > CO_$k.txt &
done
```

## What is in here

| path | |
|---|---|
| `paper.tex`, `paper.pdf` | the write-up |
| `REPORT.md` | the closing report, in four tiers by whether the evidence can leave this data source |
| `src/` | the implementation (~9k lines) |
| `tests/` | 20 structural assertions, 6 generator self-tests |
| `data/current/` | raw run outputs backing the paper |
| `data/superseded/` | earlier runs, void, kept for the record |

## How to read the results

The report is split four ways on purpose, and the line is **whether the evidence
survives leaving this source**:

- **established** — verified in isolation by unit tests, or a structural
  argument. One link per quiet tick; the halt at a chain's end; the capacity
  region `d > 2k ln V`; that binding inverts exactly only for unitary
  codebooks; that locality-sensitive addressing makes a near miss undetectable
  by construction.
- **falsified** — source-independent negatives, including one of our own
  explanations.
- **net-negative here, but this source cannot settle it** — three components,
  one mechanism that explains all three, and **no ruling**, because the
  explanation rests on this source's answers being determined by its cues.
- **open** — with the reason attached.

Two cautions apply throughout: everything is single seed unless stated, and the
families that cannot be memorised carry 60–96 events in total.

## Why a replayed stream cannot finish the job

Our silences are drawn from `{1,2,4,8}` — authored, not observed. The model is
never late: it receives exactly the ticks the file specifies however much
computation it needed. In a file, time is an index; here it is meant to be a
resource. Deciding the questions in the third tier needs real language content
carrying real, non-authored temporal structure — timestamped transcripts,
subtitle streams, conversation logs — and a compute budget bound to wall-clock
rather than to ticks. We do not know what these mechanisms do on real data and
do not think it can be predicted from here.

## Related

The fixed-addressing position, and the interference-for-retrieval-difficulty
trade this design is organised around, are argued in
[*Trading Interference for Retrieval Difficulty*](https://doi.org/10.5281/zenodo.22161096)
(Zenodo, 2026).

## Author

Xinyu Yang · [0009-0007-2600-0948](https://orcid.org/0009-0007-2600-0948) ·
Pana.Yang@hotmail.com
