# Einsum Benchmark Results

`scripts/run_all.sh` no longer writes benchmark reports under `docs/`.

Run the benchmark and read the latest generated einsum report at:

```bash
./scripts/run_all.sh 1
less result/cpu/einsum.md
```

Raw logs, run metadata, and timestamped intermediate tables are written under `data/results/cpu/einsum/<timestamp>/`.
