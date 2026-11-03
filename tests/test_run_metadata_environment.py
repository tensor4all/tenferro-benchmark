#!/usr/bin/env python3
"""Run metadata records the collection environment (container, CUDA toolkit,
vendor versions) and reports read those recorded values, never the live host."""
from argparse import Namespace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect_cpu_info as cpu  # noqa: E402
import collect_gpu_info as gpu  # noqa: E402
import collect_run_metadata as meta  # noqa: E402
import format_elementwise_results as elementwise  # noqa: E402
import validate_benchmark_suite as validate  # noqa: E402

RECORDED_CUDA = {
    "device_ordinal": 0,
    "device_name": "NVIDIA A100-SXM4-40GB",
    "device_uuid": "GPU-1234abcd",
    "device_memory_bytes": 40 * 1024**3,
    "driver_version": "550.54.15",
    "runtime_version": "12.4",
    "cuda_version": "12.4",
    "toolkit_version": "12.4",
    "toolkit_root": "/home/app/cuda/12.4",
    "cutensor_version": "2.7.0",
    "cusolver_version": "12.6.4",
    "cudnn_version": None,
}
NEW_CUDA_KEYS = ("toolkit_version", "toolkit_root", "cutensor_version", "cusolver_version")


class FakeHostinfo:
    def collect(self, root="/"):
        return {"cpu": "Fake CPU 9000", "logical_cpus": 16}

    def toolchain(self, env=None, run=None):
        return {"rustc": "rustc 1.99.1"}


class ContainerDetectionTest(unittest.TestCase):
    def test_dockerenv_is_detection_not_a_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / ".dockerenv"
            self.assertEqual(gpu.detect_container(lock), "none")
            lock.touch()
            self.assertEqual(gpu.detect_container(lock), "devcontainer")


class MetadataEnvironmentTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.replace(meta, "campaign_hostinfo", FakeHostinfo())
        self.replace(meta, "collect_python_backends", lambda: {})
        self.replace(meta, "collect_julia_backend", lambda: {"available": False, "reason": "test"})
        self.replace(meta, "detect_container", lambda: "none")
        self.replace(meta, "collect_cuda_metadata", lambda ordinal: dict(RECORDED_CUDA))

    def replace(self, module, name, value):
        saved = getattr(module, name)
        setattr(module, name, value)
        self.addCleanup(setattr, module, name, saved)

    def build_args(self):
        return Namespace(
            suite_id="gpu/linalg_jvp_vjp",
            target_profile="nvidia-gpu",
            suite_file=Path("benchmarks/gpu/linalg_jvp_vjp.yaml"),
            timestamp="2026-11-01T00:00:00Z",
            tenferro_dir=None,
            tenferro_commit="abcdef1",
            features=["cuda"],
            harness_dir=self.tmp,
            command=None,
            collection_json=None,
            blas=None,
            cuda_device_ordinal=0,
        )

    def build_run(self):
        output = self.tmp / "run.yaml"
        meta.write_yaml(output, meta.build_metadata(self.build_args()))
        return yaml.safe_load(output.read_text())

    def test_container_and_toolkit_are_recorded_and_validate(self):
        run = self.build_run()
        self.assertEqual(run["environment"]["container"], "none")
        for key in NEW_CUDA_KEYS:
            self.assertEqual(run["cuda"][key], RECORDED_CUDA[key])
        validate.validator_for(validate.RUN_SCHEMA).validate(run)

    def test_records_without_the_new_fields_still_validate(self):
        run = self.build_run()
        del run["environment"]["container"]
        for key in NEW_CUDA_KEYS:
            del run["cuda"][key]
        # A record from before these fields existed is still a record.
        validate.validator_for(validate.RUN_SCHEMA).validate(run)


class ToolkitSelectionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.cuda = self.tmp / "cuda-12.4"
        (self.cuda / "bin").mkdir(parents=True)
        (self.cuda / "lib64").mkdir()
        nvcc = self.cuda / "bin" / "nvcc"
        nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 12.4, V12.4.131'\n")
        nvcc.chmod(0o755)

    def test_root_order_is_cuda_home_then_cuda_path_then_nvcc(self):
        other = self.tmp / "cuda-13.0"
        other.mkdir()
        with patch.dict(os.environ, {"CUDA_HOME": str(self.cuda), "CUDA_PATH": str(other)}, clear=True):
            self.assertEqual(gpu._cuda_toolkit_root(), self.cuda)
        with patch.dict(os.environ, {"CUDA_PATH": str(other)}, clear=True):
            self.assertEqual(gpu._cuda_toolkit_root(), other)
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(gpu.shutil, "which", return_value=str(self.cuda / "bin" / "nvcc")):
                self.assertEqual(gpu._cuda_toolkit_root(), self.cuda)

    def test_toolkit_version_comes_only_from_the_selected_root(self):
        path_nvcc = self.tmp / "path-nvcc" / "nvcc"
        path_nvcc.parent.mkdir()
        path_nvcc.write_text("#!/bin/sh\necho 'release 99.9'\n")
        path_nvcc.chmod(0o755)
        with patch.dict(os.environ, {"PATH": str(path_nvcc.parent)}, clear=True):
            with patch.object(gpu.shutil, "which", return_value=str(path_nvcc)):
                root = gpu._cuda_toolkit_root()
        self.assertEqual(root, path_nvcc.parent.parent)
        self.assertEqual(gpu._nvcc_version(self.cuda), "12.4")
        # No selected root means no version, not an unrelated PATH nvcc.
        with patch.object(gpu, "_run", side_effect=AssertionError("nvcc consulted")):
            self.assertIsNone(gpu._nvcc_version(None))
        self.assertIsNone(gpu._nvcc_version(self.tmp / "no-toolkit"))

    def test_runtime_version_never_falls_back_to_a_compiler(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(gpu.shutil, "which", return_value=None), \
                patch.object(gpu, "_run", side_effect=AssertionError("nvcc consulted")), \
                patch.object(gpu.ctypes, "CDLL", side_effect=OSError("missing")):
            self.assertIsNone(gpu._cuda_runtime_version())


class FakeCutensor:
    def __init__(self, version=None):
        if version is not None:
            def cutensor_get_version():
                return version
            self.cutensorGetVersion = cutensor_get_version


class FakeCusolver:
    def __init__(self, versions=(12, 6, 4), failing=None):
        def get_property(property_index, output):
            if property_index == failing:
                return 1
            output._obj.value = versions[property_index]
            return 0
        self.cusolverGetProperty = get_property


class VendorVersionTest(unittest.TestCase):
    def loader(self, mapping):
        calls = []

        def load(path):
            calls.append(path)
            value = mapping.get(path)
            if isinstance(value, Exception):
                raise value
            if value is None:
                raise OSError(path)
            return value
        return calls, load

    def test_env_override_replaces_defaults_even_when_empty(self):
        with patch.dict(os.environ, {"TENFERRO_CUTENSOR_PATH": "a.so:b.so"}, clear=True):
            self.assertEqual(gpu._library_paths("TENFERRO_CUTENSOR_PATH", gpu.CUTENSOR_LIBRARY_PATHS), ["a.so", "b.so"])
        with patch.dict(os.environ, {"TENFERRO_CUTENSOR_PATH": ""}, clear=True):
            self.assertEqual(gpu._library_paths("TENFERRO_CUTENSOR_PATH", gpu.CUTENSOR_LIBRARY_PATHS), [])
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                gpu._library_paths("TENFERRO_CUTENSOR_PATH", gpu.CUTENSOR_LIBRARY_PATHS),
                list(gpu.CUTENSOR_LIBRARY_PATHS),
            )

    def test_cutensor_is_the_first_loadable_candidate(self):
        first, second = gpu.CUTENSOR_LIBRARY_PATHS[:2]
        calls, load = self.loader({first: OSError("missing"), second: FakeCutensor(20700)})
        with patch.dict(os.environ, {}, clear=True), patch.object(gpu.ctypes, "CDLL", side_effect=load):
            self.assertEqual(gpu._cutensor_version(), "2.7.0")
        self.assertEqual(calls, [first, second])

    def test_absent_version_symbol_is_unknown_without_trying_another(self):
        first = gpu.CUTENSOR_LIBRARY_PATHS[0]
        calls, load = self.loader({first: FakeCutensor()})
        with patch.dict(os.environ, {}, clear=True), patch.object(gpu.ctypes, "CDLL", side_effect=load):
            self.assertIsNone(gpu._cutensor_version())
        self.assertEqual(calls, [first])
        # An empty override has no candidates at all.
        with patch.dict(os.environ, {"TENFERRO_CUTENSOR_PATH": ""}, clear=True), \
                patch.object(gpu.ctypes, "CDLL", side_effect=AssertionError("loaded a library")):
            self.assertIsNone(gpu._cutensor_version())

    def test_cusolver_uses_the_public_property_api(self):
        first = gpu.CUSOLVER_LIBRARY_PATHS[0]
        calls, load = self.loader({first: FakeCusolver()})
        with patch.dict(os.environ, {}, clear=True), patch.object(gpu.ctypes, "CDLL", side_effect=load):
            self.assertEqual(gpu._cusolver_version(), "12.6.4")
        self.assertEqual(calls, [first])
        _, failing = self.loader({first: FakeCusolver(failing=1)})
        with patch.dict(os.environ, {}, clear=True), patch.object(gpu.ctypes, "CDLL", side_effect=failing):
            self.assertIsNone(gpu._cusolver_version())
        _, absent = self.loader({first: object()})
        with patch.dict(os.environ, {}, clear=True), patch.object(gpu.ctypes, "CDLL", side_effect=absent):
            self.assertIsNone(gpu._cusolver_version())


class RecordedReportTest(unittest.TestCase):
    def test_report_shows_recorded_environment_without_probing(self):
        metadata = {
            "cuda": dict(RECORDED_CUDA),
            "environment": {"container": "none"},
        }
        with patch.object(gpu, "collect_cuda_metadata", side_effect=AssertionError("probed")), \
                patch.object(gpu, "collect_gpu_info", side_effect=AssertionError("probed")), \
                patch.object(gpu.ctypes, "CDLL", side_effect=AssertionError("probed")):
            info = gpu.resolve_gpu_info([], metadata)
        rendered = gpu.markdown(info)
        for line in (
            "- Container: `none`",
            "- CUDA toolkit: `12.4`",
            "- CUDA toolkit root: `/home/app/cuda/12.4`",
            "- cuTENSOR version: `2.7.0`",
            "- cuSOLVER version: `12.6.4`",
        ):
            self.assertIn(line, rendered)
        self.assertIn("- Driver version: `550.54.15`", rendered)

    def test_older_records_do_not_gain_guessed_lines(self):
        rendered = gpu.markdown(gpu.resolve_gpu_info([], {"cuda": {"device_ordinal": 0, "device_name": "A100"}}))
        self.assertNotIn("Container", rendered)
        self.assertNotIn("CUDA toolkit", rendered)
        self.assertNotIn("cuTENSOR", rendered)

    def test_recorded_unknown_versions_are_reported_as_unknown(self):
        metadata = {
            "cuda": {"device_ordinal": 0, "device_name": "A100", "toolkit_version": None, "cutensor_version": None},
            "environment": {"container": "none"},
        }
        rendered = gpu.markdown(gpu.resolve_gpu_info([], metadata))
        self.assertIn("- CUDA toolkit: `unknown`", rendered)
        self.assertIn("- cuTENSOR version: `unknown`", rendered)


class CpuReportTest(unittest.TestCase):
    def test_cpu_info_records_its_container(self):
        with patch.object(cpu, "detect_container", return_value="none"):
            info = cpu.collect_cpu_info()
        self.assertEqual(info["Container"], "none")
        self.assertIn("- Container: `none`", cpu.markdown(info))

    def test_elementwise_report_uses_only_the_recorded_container(self):
        # The live value stands in for a renderer running somewhere else.
        with patch.object(elementwise, "collect_cpu_info", side_effect=lambda: {"Model": "m", "Container": "devcontainer"}):
            recorded = elementwise.format_markdown([], {"environment": {"container": "none"}})
            legacy = elementwise.format_markdown([], {"target_profile": "amd-cpu"})
            unstamped = elementwise.format_markdown([], None)
        self.assertIn("Container: `none`", recorded)
        # A run that recorded nothing keeps it unknown instead of borrowing this host.
        self.assertNotIn("Container", legacy)
        self.assertNotIn("Container", unstamped)


if __name__ == "__main__":
    unittest.main()
