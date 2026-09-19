"""Exercise CLI thread setup without measuring operations or requiring a GPU."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GpuThreadConfigTests(unittest.TestCase):
    def test_cli_configures_both_torch_thread_pools(self):
        for module in ("benchmark_gpu_python", "benchmark_gpu_linalg_ad",
                       "benchmark_gpu_permutation_python"):
            with self.subTest(module=module):
                # Inter-op configuration is once per process, as in the real CLIs.
                code = f'''
import importlib, pathlib, sys, tempfile, torch
sys.path.insert(0, {str(ROOT / "scripts")!r})
m = importlib.import_module({module!r})
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    suite = root / "empty.yaml"
    suite.write_text("suite_id: gpu/dense\\nbackends: [pytorch-cuda]\\nproblems: []\\n")
    sys.argv = [{module!r}, str(root / "records.jsonl"), "0", "", "pytorch-cuda", "--", str(suite)]
    if {module!r} == "benchmark_gpu_permutation_python":
        sys.argv = [{module!r}]
        m.torch_cuda_available = lambda: True
        m.gpu_name = lambda ordinal: "configuration-test"
        m.load_pattern_suite = lambda: {{"patterns": []}}
    m.main()
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1
'''
                env = dict(os.environ, OMP_NUM_THREADS="1")
                env.pop("BENCH_OUTPUT", None)
                env.pop("PATTERN_ID", None)
                result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                        env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("intra-op=1, inter-op=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
