#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d extern/tenferro-einsum-benchmark/data/instances ]]; then
  echo "skip: extern/tenferro-einsum-benchmark/data/instances not present" >&2
  exit 0
fi

uv run python scripts/import_einsum_dataset.py >/dev/null
bash tests/test_instance_colmajor_metadata.sh

EXPECTED=(
  bin_batched_matmul_b32_m64_n64_k64.json
  bin_elementwise_mul_2048x2048.json
  bin_matmul_256.json
  bin_outer_product_4096.json
  gm_queen5_5_3.wcsp.json
  lm_batch_likelihood_brackets_4_4d.json
  lm_batch_likelihood_sentence_3_12d.json
  lm_batch_likelihood_sentence_4_4d.json
  str_matrix_chain_multiplication_100.json
  str_mps_varying_inner_product_200.json
  str_nw_mera_closed_120.json
  str_nw_mera_open_26.json
  tensornetwork_permutation_focus_step409_316.json
  tensornetwork_permutation_light_415.json
)

for name in "${EXPECTED[@]}"; do
  test -s "data/instances/$name"
done

test -s data/instances/nary_matmul_chain_64.json

uv run python - <<'PY'
import json
from pathlib import Path

nary = json.loads(Path("data/instances/nary_matmul_chain_64.json").read_text())
assert nary["num_tensors"] == 3
assert "inner ExecProgram" in nary["intent"]
PY

grep -q 'nary_matmul_chain_64' benchmarks/cpu/einsum.yaml
