# CPU route-contract diagnostics

Mechanism diagnostics (provider spy + counting allocator), **not timing**. Verdicts are deterministic; see `scripts/route_contract.py` for the contract.

- Raw run: `data/results/amd-cpu/cpu/route_contract/20261006_215911`
- Overall: **PASS** — PASS 45, FAIL 0, INCOMPLETE 0, UNSUPPORTED 3, NOT_OBSERVABLE 0

- Metadata: `run_t1.yaml`
- Metadata: `run_t4.yaml`

| Pair | Threads | Verdict | Allocating calls (uninit/init, lanes, batches) | _into calls (uninit/init, lanes, batches) | Reason |
|---|---:|---|---|---|---|
| `bdot_c64_b1024_m4n4k4_direct_auto` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_c64_b1024_m4n4k4_direct_auto` | 4 | PASS | 4/0, lanes 4, [256, 256, 256, 256] | 0/4, lanes 4, [256, 256, 256, 256] | same lane decision; batches covered exactly once |
| `bdot_c64_b1024_m4n4k4_strided_auto` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_c64_b1024_m4n4k4_strided_auto` | 4 | PASS | 4/0, lanes 4, [256, 256, 256, 256] | 0/4, lanes 4, [256, 256, 256, 256] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_canonical_auto` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_canonical_auto` | 4 | PASS | 4/0, lanes 4, [256, 256, 256, 256] | 0/4, lanes 4, [256, 256, 256, 256] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_auto` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_auto` | 4 | PASS | 4/0, lanes 4, [256, 256, 256, 256] | 0/4, lanes 4, [256, 256, 256, 256] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_backend-outer-min-items-4096` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_backend-outer-min-items-4096` | 4 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_forced-outer-parallel` | 1 | UNSUPPORTED | unsupported | unsupported | both routes unsupported: dot_general: unsupported operation: batch strategy OuterParallel is not available: the selected CPU domain has one thread, so there are no outer lanes; use CpuBatchStrategy::Auto or another strategy |
| `bdot_f64_b1024_m4n4k4_direct_forced-outer-parallel` | 4 | PASS | 4/0, lanes 4, [256, 256, 256, 256] | 0/4, lanes 4, [256, 256, 256, 256] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_forced-provider-items` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_forced-provider-items` | 4 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_forced-sequential` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_forced-sequential` | 4 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_forced-whole-batch-vendor` | 1 | UNSUPPORTED | unsupported | unsupported | both routes unsupported: dot_general: unsupported operation: configured CPU GEMM provider reported unsupported: RuntimeUnavailable |
| `bdot_f64_b1024_m4n4k4_direct_forced-whole-batch-vendor` | 4 | UNSUPPORTED | unsupported | unsupported | both routes unsupported: dot_general: unsupported operation: configured CPU GEMM provider reported unsupported: RuntimeUnavailable |
| `bdot_f64_b1024_m4n4k4_direct_scoped-outer-min-items-4096` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_direct_scoped-outer-min-items-4096` | 4 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_strided_auto` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b1024_m4n4k4_strided_auto` | 4 | PASS | 4/0, lanes 4, [256, 256, 256, 256] | 0/4, lanes 4, [256, 256, 256, 256] | same lane decision; batches covered exactly once |
| `bdot_f64_b16_m64n64k64_direct_auto` | 1 | PASS | 1/0, lanes 0, [16] | 0/1, lanes 0, [16] | same lane decision; batches covered exactly once |
| `bdot_f64_b16_m64n64k64_direct_auto` | 4 | PASS | 4/0, lanes 4, [4, 4, 4, 4] | 0/4, lanes 4, [4, 4, 4, 4] | same lane decision; batches covered exactly once |
| `bdot_f64_b256_m4n4k4_direct_auto` | 1 | PASS | 1/0, lanes 0, [256] | 0/1, lanes 0, [256] | same lane decision; batches covered exactly once |
| `bdot_f64_b256_m4n4k4_direct_auto` | 4 | PASS | 1/0, lanes 0, [256] | 0/1, lanes 0, [256] | same lane decision; batches covered exactly once |
| `bdot_f64_b320_m4n4k4_direct_auto` | 1 | PASS | 1/0, lanes 0, [320] | 0/1, lanes 0, [320] | same lane decision; batches covered exactly once |
| `bdot_f64_b320_m4n4k4_direct_auto` | 4 | PASS | 2/0, lanes 2, [160, 160] | 0/2, lanes 2, [160, 160] | same lane decision; batches covered exactly once |
| `bdot_f64_b4096_m4n4k4_direct_auto` | 1 | PASS | 1/0, lanes 0, [4096] | 0/1, lanes 0, [4096] | same lane decision; batches covered exactly once |
| `bdot_f64_b4096_m4n4k4_direct_auto` | 4 | PASS | 4/0, lanes 4, [1024, 1024, 1024, 1024] | 0/4, lanes 4, [1024, 1024, 1024, 1024] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m16n16k16_direct_auto` | 1 | PASS | 1/0, lanes 0, [64] | 0/1, lanes 0, [64] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m16n16k16_direct_auto` | 4 | PASS | 2/0, lanes 2, [32, 32] | 0/2, lanes 2, [32, 32] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m32n32k32_direct_auto` | 1 | PASS | 1/0, lanes 0, [64] | 0/1, lanes 0, [64] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m32n32k32_direct_auto` | 4 | PASS | 4/0, lanes 4, [16, 16, 16, 16] | 0/4, lanes 4, [16, 16, 16, 16] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m4n4k4_direct_auto` | 1 | PASS | 1/0, lanes 0, [64] | 0/1, lanes 0, [64] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m4n4k4_direct_auto` | 4 | PASS | 1/0, lanes 0, [64] | 0/1, lanes 0, [64] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m4n4k4_direct_backend-lane-min-work-0` | 1 | PASS | 1/0, lanes 0, [64] | 0/1, lanes 0, [64] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m4n4k4_direct_backend-lane-min-work-0` | 4 | PASS | 4/0, lanes 4, [16, 16, 16, 16] | 0/4, lanes 4, [16, 16, 16, 16] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m4n4k4_direct_scoped-lane-min-work-0` | 1 | PASS | 1/0, lanes 0, [64] | 0/1, lanes 0, [64] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m4n4k4_direct_scoped-lane-min-work-0` | 4 | PASS | 4/0, lanes 4, [16, 16, 16, 16] | 0/4, lanes 4, [16, 16, 16, 16] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m64n1k64_direct_auto` | 1 | PASS | 1/0, lanes 0, [64] | 0/1, lanes 0, [64] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m64n1k64_direct_auto` | 4 | PASS | 2/0, lanes 2, [32, 32] | 0/2, lanes 2, [32, 32] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m8n8k8_direct_auto` | 1 | PASS | 1/0, lanes 0, [64] | 0/1, lanes 0, [64] | same lane decision; batches covered exactly once |
| `bdot_f64_b64_m8n8k8_direct_auto` | 4 | PASS | 1/0, lanes 0, [64] | 0/1, lanes 0, [64] | same lane decision; batches covered exactly once |
| `beinsum_f64_b1024_m4n4k4_direct_auto` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `beinsum_f64_b1024_m4n4k4_direct_auto` | 4 | PASS | 4/0, lanes 4, [256, 256, 256, 256] | 0/4, lanes 4, [256, 256, 256, 256] | same lane decision; batches covered exactly once |
| `beinsum_f64_b1024_m4n4k4_strided_auto` | 1 | PASS | 1/0, lanes 0, [1024] | 0/1, lanes 0, [1024] | same lane decision; batches covered exactly once |
| `beinsum_f64_b1024_m4n4k4_strided_auto` | 4 | PASS | 4/0, lanes 4, [256, 256, 256, 256] | 0/4, lanes 4, [256, 256, 256, 256] | same lane decision; batches covered exactly once |
