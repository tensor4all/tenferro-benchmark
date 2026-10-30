#!/usr/bin/env python3
"""run.yaml carries the hostinfo host block and the build toolchain."""
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect_run_metadata as meta  # noqa: E402
import validate_benchmark_suite as validate  # noqa: E402

HOST = {
    "cpu": "Fake CPU 9000",
    "logical_cpus": 16,
    "power": {
        "cpufreq_boost": "0",
        "governor": "performance",
        "platform_profile": "performance",
        "smt": {"active": "1", "control": "on"},
    },
    "numa": {"nodes": 2, "cpulists": {"0": "0-7", "1": "8-15"}},
    "smt_siblings": ["0,8", "1,9"],
    "l3_domains": ["0-3,8-11", "4-7,12-15"],
}
TOOLCHAIN = {
    "rustc": "rustc 1.99.1 (58cf8f9c3 2026-10-13)",
    "rustc_release": "1.99.1",
    "rustflags": "-C target-cpu=native",
    "target_cpu": "native",
}


class FakeHostinfo:
    """The campaign_toolkit.hostinfo surface this recorder calls."""

    def collect(self, root="/"):
        return dict(HOST)

    def toolchain(self, env=None, run=None):
        return dict(TOOLCHAIN)


class HostInfoMetadataTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        # The interpreter probes (torch, JAX, Julia) are not what this test covers.
        replacements = {
            "campaign_hostinfo": FakeHostinfo(),
            "collect_python_backends": lambda: {},
            "collect_julia_backend": lambda: {"available": False, "reason": "test"},
        }
        for name, value in replacements.items():
            saved = getattr(meta, name)
            setattr(meta, name, value)
            self.addCleanup(setattr, meta, name, saved)

    def build_args(self):
        return Namespace(
            suite_id="cpu/einsum",
            target_profile="zen5-cpu",
            suite_file=Path("benchmarks/cpu/einsum.yaml"),
            timestamp="2026-11-01T00:00:00Z",
            tenferro_dir=None,
            tenferro_commit="abcdef1",
            features=[],
            harness_dir=self.tmp,
            command=None,
            collection_json=None,
            blas=None,
            cuda_device_ordinal=None,
        )

    def test_host_and_build_blocks_validate(self):
        output = self.tmp / "run.yaml"
        meta.write_yaml(output, meta.build_metadata(self.build_args()))
        run = yaml.safe_load(output.read_text())

        self.assertEqual(run["host"], HOST)
        self.assertEqual(run["build"], {"toolchain": TOOLCHAIN})
        self.assertIn("affinity", run["environment"])
        validator = validate.validator_for(validate.RUN_SCHEMA)
        validator.validate(run)

        # The second collection profile is accepted by the schema; run_all.sh
        # names it in the profile case that guards collection.
        run["target_profile"] = "zen2-cpu"
        validator.validate(run)

    def test_old_manifest_without_host_still_validates(self):
        output = self.tmp / "run.yaml"
        meta.write_yaml(output, meta.build_metadata(self.build_args()))
        run = yaml.safe_load(output.read_text())
        del run["host"]
        del run["build"]
        validate.validator_for(validate.RUN_SCHEMA).validate(run)


if __name__ == "__main__":
    unittest.main()
