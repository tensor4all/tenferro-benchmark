#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from small_work_provenance import (
    collect_identity,
    prepare_cargo,
    verify_preparation_receipt,
    _run_cargo,
    _cargo_identity,
    _input_identity,
    _metadata_summary,
)


class ProvenanceTests(unittest.TestCase):
    def _git(self, path: Path, files: dict[str, str]) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        subprocess.run(["git", "init", "-q"], cwd=path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
        subprocess.run(["git", "add", "."], cwd=path, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)

    def _fixture(self, root: Path, *, debug: bool = False):
        benchmark, library = root / "benchmark", root / "tenferro-rs"
        self._git(benchmark, {"Cargo.toml": "[package]\nname='fixture'\nversion='0.1.0'\n", "Cargo.lock": "# lock\n", ".gitignore": "target/\n"})
        self._git(library, {"Cargo.toml": "[package]\nname='tenferro-core'\nversion='0.1.0'\n"})
        cargo = root / "cargo-fixture.py"
        profile = repr({"opt_level": "0", "debuginfo": 2, "debug_assertions": True, "overflow_checks": True, "test": False} if debug else {"opt_level": "3", "debuginfo": 0, "debug_assertions": False, "overflow_checks": False, "test": False})
        cargo.write_text("#!/usr/bin/env python3\n" +
            "import json, pathlib, sys\n" +
            "if '--version' in sys.argv: print('cargo 1.80.0 (fixture)'); raise SystemExit(0)\n" +
            "if 'metadata' in sys.argv:\n" +
            " print(json.dumps({'packages':[{'id':'fixture 0.1.0 (path+fixture)','name':'fixture','manifest_path':str(pathlib.Path('Cargo.toml').resolve()),'features':{}},{'id':'tenferro-core 0.1.0 (path+fixture)','name':'tenferro-core','manifest_path':str((pathlib.Path(" + repr(str(library / "Cargo.toml")) + ")).resolve()),'features':{}}], 'workspace_members':['fixture 0.1.0 (path+fixture)'], 'resolve':{'root':'fixture 0.1.0 (path+fixture)','nodes':[{'id':'fixture 0.1.0 (path+fixture)','features':['cpu-faer']},{'id':'tenferro-core 0.1.0 (path+fixture)','features':[]}]}})); raise SystemExit(0)\n" +
            "binary=pathlib.Path('target/release/small_work_case'); binary.parent.mkdir(parents=True,exist_ok=True); binary.write_bytes(b'fixture-binary'); binary.chmod(0o755)\n" +
            "print(json.dumps({'reason':'compiler-artifact','package_id':'fixture 0.1.0 (path+fixture)','target':{'name':'small_work_case','kind':['bin']},'profile':" + profile + ", 'features':['cpu-faer'],'executable':str(binary.resolve()),'fresh':False}))\n" +
            "print(json.dumps({'reason':'build-finished','success':True}))\n")
        cargo.chmod(0o755)
        return benchmark, library, cargo, benchmark / "target/release/small_work_case"

    def test_preparation_receipt_requires_events_and_exact_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory))
            receipt_path = Path(directory) / "receipt.json"
            receipt = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                    receipt=receipt_path, features=["cpu-faer"], cargo=str(cargo))
            self.assertEqual(receipt["status"], "verified", receipt["errors"])
            self.assertEqual(receipt["build"]["artifact"]["target"]["kind"], ["bin"])
            self.assertEqual(verify_preparation_receipt(receipt, benchmark=benchmark,
                             library=library, binary=binary, require_timing=True, cargo=str(cargo)), [])
            for key in ("artifact_kind", "build_project", "target_name", "package"):
                with self.subTest(missing_key=key):
                    incomplete = json.loads(json.dumps(receipt))
                    del incomplete["build"][key]
                    self.assertTrue(verify_preparation_receipt(
                        incomplete, benchmark=benchmark, library=library, binary=binary,
                        require_timing=True, cargo=str(cargo)))
            binary.write_bytes(b"substituted")
            self.assertTrue(any("substituted" in error for error in verify_preparation_receipt(
                receipt, benchmark=benchmark, library=library, binary=binary,
                require_timing=True, cargo=str(cargo))))

    def test_failed_or_missing_cargo_event_is_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory))
            text = cargo.read_text()
            cargo.write_text(text.replace("print(json.dumps({'reason':'build-finished','success':True}))",
                                         "raise SystemExit(7)"))
            receipt = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                   receipt=Path(directory) / "receipt.json", features=["cpu-faer"], cargo=str(cargo))
            self.assertEqual(receipt["status"], "unverified")
            self.assertTrue(any("build" in error for error in receipt["errors"]))

    def test_debug_artifact_is_correctness_only(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory), debug=True)
            receipt = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                   receipt=Path(directory) / "receipt.json", features=["cpu-faer"], cargo=str(cargo))
            self.assertEqual(receipt["status"], "unverified")
            # A caller may inspect/run it for correctness, but timing evidence is rejected.
            receipt["status"] = "verified"
            self.assertTrue(any("debug" in error for error in verify_preparation_receipt(
                receipt, benchmark=benchmark, library=library, binary=binary,
                require_timing=True, cargo=str(cargo))))

    def test_dirty_source_after_build_invalidates_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory))
            marker = repr(str(library / "Cargo.toml"))
            text = cargo.read_text()
            cargo.write_text(text.replace(
                "binary=pathlib.Path('target/release/small_work_case');",
                f"pathlib.Path({marker}).write_text('[package]\\nname=\\\"tenferro-core\\\"\\nversion=\\\"0.1.0\\\"\\n'); binary=pathlib.Path('target/release/small_work_case');"))
            receipt = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                    receipt=Path(directory) / "dirty-after.json", features=["cpu-faer"], cargo=str(cargo))
            self.assertEqual(receipt["status"], "unverified")
            self.assertTrue(any("changed" in error or "dirty" in error or "identity" in error
                                for error in receipt["errors"]))

    def test_ls_files_failure_is_not_verified_empty_membership(self):
        from small_work_provenance import _git_state
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._git(path, {"Cargo.toml": "[package]\nname='fixture'\nversion='0.1.0'\n"})
            original = subprocess.run
            def fail_ls(command, *args, **kwargs):
                if command[:2] == ["git", "ls-files"]:
                    raise subprocess.CalledProcessError(1, command)
                return original(command, *args, **kwargs)
            with mock.patch("small_work_provenance.subprocess.run", side_effect=fail_ls):
                state = _git_state(path)
            self.assertIsNone(state["tracked_file_count"])
            self.assertIn("tracked membership", state["error"])

    def test_lock_change_stales_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory))
            receipt = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                   receipt=Path(directory) / "receipt.json", features=["cpu-faer"], cargo=str(cargo))
            self.assertEqual(receipt["status"], "verified")
            (benchmark / "Cargo.lock").write_text("# changed\n")
            errors = verify_preparation_receipt(receipt, benchmark=benchmark, library=library,
                                                binary=binary, require_timing=True, cargo=str(cargo))
            self.assertTrue(any("stale" in error or "changed" in error for error in errors))

    def test_dirty_checkout_is_not_timing_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory))
            (library / "untracked.txt").write_text("dirty")
            identity = collect_identity(benchmark, library, cargo=str(cargo))
            self.assertTrue(identity["library"]["dirty"])
            self.assertTrue(identity["library"]["untracked"])
            self.assertGreater(identity["library"]["tracked_file_count"], 0)

    def test_dirty_source_skips_cargo_build(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory))
            (library / "dirty.txt").write_text("changed")
            receipt = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                    receipt=Path(directory) / "dirty.json", cargo=str(cargo))
            self.assertEqual(receipt["status"], "unverified")
            self.assertIn("known clean", " ".join(receipt["errors"]))
            self.assertFalse(binary.exists())

    def test_cargo_timeout_retains_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "slow.py"
            script.write_text("import sys,time; print('out', flush=True); print('err', file=sys.stderr, flush=True); time.sleep(2)")
            result = _run_cargo([sys.executable, str(script)], Path(directory), timeout=0.1)
            self.assertTrue(result["timed_out"])
            self.assertIn("out", result["stdout"])
            self.assertIn("err", result["stderr"])

    def test_root_package_is_exempt_but_external_tenferro_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, _ = self._fixture(Path(directory))
            metadata = json.loads(subprocess.run([str(cargo), "metadata"], cwd=benchmark,
                                                  check=True, capture_output=True, text=True).stdout)
            from small_work_provenance import _metadata_summary
            metadata["packages"][0]["name"] = "tenferro-einsum-benchmark"
            metadata["packages"][0]["features"] = {"fabricated": []}
            metadata["resolve"]["nodes"][0]["features"] = ["resolved"]
            summary = _metadata_summary(metadata, library, benchmark)
            self.assertTrue(any(item["root"] for item in summary["packages"]))
            self.assertEqual(next(item for item in summary["packages"] if item["root"])["features"], ["resolved"])
            external = Path(directory) / "outside.toml"
            external.write_text("[package]\nname='tenferro-outside'\n")
            metadata["packages"].append({"id": "tenferro-outside 0.1.0 (path+fixture)",
                "name": "tenferro-outside", "manifest_path": str(external), "features": {}})
            metadata["resolve"]["nodes"].append({"id": "tenferro-outside 0.1.0 (path+fixture)", "features": []})
            with self.assertRaises(ValueError):
                _metadata_summary(metadata, library, benchmark)

    def test_foreign_path_dependency_identity_is_captured_and_dirty_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, _ = self._fixture(Path(directory))
            foreign = Path(directory) / "foreign"
            self._git(foreign, {"Cargo.toml": "[package]\nname='foreign'\nversion='0.1.0'\n", "src/lib.rs": "pub fn f() {}\n"})
            metadata = json.loads(subprocess.run([str(cargo), "metadata"], cwd=benchmark,
                                                  check=True, capture_output=True, text=True).stdout)
            metadata["packages"].append({"id": "foreign 0.1.0 (path+fixture)", "name": "foreign",
                "manifest_path": str(foreign / "Cargo.toml"), "source": None})
            metadata["resolve"]["nodes"].append({"id": "foreign 0.1.0 (path+fixture)", "features": ["default"]})
            summary = _metadata_summary(metadata, library, benchmark)
            package = next(item for item in summary["packages"] if item["name"] == "foreign")
            self.assertEqual(package["source_identity"]["dirty"], False)
            (foreign / "src/lib.rs").write_text("pub fn f() { panic!() }\n")
            with self.assertRaises(ValueError):
                _metadata_summary(metadata, library, benchmark)

    def test_config_hashes_and_selected_rustc_are_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, _ = self._fixture(Path(directory))
            config = benchmark.parent / ".cargo" / "config.toml"
            config.parent.mkdir()
            config.write_text("[build]\nrustflags=[]\n")
            cargo_home = Path(directory) / "cargo-home"
            cargo_home.mkdir()
            (cargo_home / "config.toml").write_text("[net]\noffline=true\n")
            old_env = dict(os.environ)
            try:
                os.environ["CARGO_HOME"] = str(cargo_home)
                first = collect_identity(benchmark, library, cargo=str(cargo))
                self.assertIn(str(config.resolve()), first["files"]["benchmark"]["configs"])
                self.assertIn(str((cargo_home / "config.toml").resolve()), first["files"]["benchmark"]["configs"])
                config.write_text("[build]\nrustflags=['-Ctarget-cpu=native']\n")
                second = collect_identity(benchmark, library, cargo=str(cargo))
                self.assertNotEqual(_input_identity(first), _input_identity(second))
                (cargo_home / "config.toml").write_text("[net]\noffline=false\n")
                third = collect_identity(benchmark, library, cargo=str(cargo))
                self.assertNotEqual(_input_identity(second), _input_identity(third))
                rustc = Path(directory) / "selected-rustc"
                rustc.write_text("#!/bin/sh\necho selected-rustc 9.9\n")
                rustc.chmod(0o755)
                os.environ["RUSTC"] = str(rustc)
                self.assertIn("selected-rustc", _cargo_identity(str(cargo), benchmark)["rustc"])
            finally:
                os.environ.clear(); os.environ.update(old_env)

    def test_resolved_graph_is_complete_and_edges_are_known(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, _ = self._fixture(Path(directory))
            metadata = json.loads(subprocess.run([str(cargo), "metadata"], cwd=benchmark,
                                                  check=True, capture_output=True, text=True).stdout)
            metadata["resolve"]["nodes"][0]["dependencies"] = ["missing-package"]
            with self.assertRaises(ValueError):
                _metadata_summary(metadata, library, benchmark)
            metadata = json.loads(subprocess.run([str(cargo), "metadata"], cwd=benchmark,
                                                  check=True, capture_output=True, text=True).stdout)
            metadata["resolve"]["nodes"].pop()
            with self.assertRaises(ValueError):
                _metadata_summary(metadata, library, benchmark)

    def test_rustflags_override_skips_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory))
            old = os.environ.get("RUSTFLAGS")
            os.environ["RUSTFLAGS"] = "-Ctarget-cpu=native"
            try:
                receipt = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                       receipt=Path(directory) / "rustflags.json", cargo=str(cargo))
            finally:
                if old is None:
                    os.environ.pop("RUSTFLAGS", None)
                else:
                    os.environ["RUSTFLAGS"] = old
            self.assertEqual(receipt["status"], "unverified")
            self.assertIn("RUSTFLAGS", " ".join(receipt["errors"]))
            self.assertNotIn("RUSTFLAGS", receipt["before"]["environment"])
            self.assertIn("RUSTFLAGS", receipt["before"]["overrides"])
            self.assertFalse(binary.exists())

    def test_real_profile_shape_has_no_legacy_debug_key(self):
        from small_work_provenance import _profile_is_debug, _profile_known
        profile = {"opt_level": "0", "debuginfo": 2, "debug_assertions": True,
                   "overflow_checks": True, "test": False}
        self.assertTrue(_profile_known(profile))
        self.assertTrue(_profile_is_debug(profile))
        self.assertFalse(_profile_known({**profile, "debug": True}))

    def test_fresh_cached_artifact_requires_previous_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory))
            receipt_path = Path(directory) / "receipt.json"
            first = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                  receipt=receipt_path, features=["cpu-faer"], cargo=str(cargo))
            self.assertEqual(first["status"], "verified")
            cargo.write_text(cargo.read_text().replace("'fresh':False", "'fresh':True"))
            without_previous = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                  receipt=Path(directory) / "fresh-no-prev.json", features=["cpu-faer"], cargo=str(cargo))
            self.assertEqual(without_previous["status"], "unverified")
            with_previous = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                  receipt=Path(directory) / "fresh-prev.json", features=["cpu-faer"], cargo=str(cargo),
                                  previous_receipt=first)
            self.assertEqual(with_previous["status"], "verified", with_previous["errors"])
            self.assertEqual(verify_preparation_receipt(with_previous, benchmark=benchmark,
                library=library, binary=binary, require_timing=True, cargo=str(cargo)), [])

    def test_fresh_cached_artifact_rejects_tampered_previous_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library, cargo, binary = self._fixture(Path(directory))
            first = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                  receipt=Path(directory) / "first.json", features=["cpu-faer"], cargo=str(cargo))
            self.assertEqual(first["status"], "verified")
            first["errors"] = ["fabricated"]
            cargo.write_text(cargo.read_text().replace("'fresh':False", "'fresh':True"))
            receipt = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                    receipt=Path(directory) / "fresh.json", features=["cpu-faer"], cargo=str(cargo),
                                    previous_receipt=first)
            self.assertEqual(receipt["status"], "unverified")
            self.assertIn("previous receipt", " ".join(receipt["errors"]))


class RealCargoPreparationTests(unittest.TestCase):
    def _git(self, path: Path, files: dict[str, str]) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        subprocess.run(["git", "init", "-q"], cwd=path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
        subprocess.run(["git", "add", "."], cwd=path, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)

    def _workspace(self, root: Path) -> tuple[Path, Path]:
        benchmark = root / "benchmark"
        self._git(benchmark, {"Cargo.toml": "[package]\nname='benchmark'\nversion='0.1.0'\n\n[[bin]]\nname='tiny_bin'\npath='src/main.rs'\n",
                              "Cargo.lock": "version = 3\n\n[[package]]\nname = \"benchmark\"\nversion = \"0.1.0\"\n", ".gitignore": "target/\n", "src/main.rs": "fn main() {}\n"})
        library = root / "tenferro-rs"
        self._git(library, {
            "Cargo.toml": "[workspace]\nmembers=['crates/tiny-lib']\nresolver='2'\n",
            "Cargo.lock": "version = 3\n\n[[package]]\nname = \"tiny-lib\"\nversion = \"0.1.0\"\n",
            ".gitignore": "target/\n",
            "crates/tiny-lib/Cargo.toml": "[package]\nname='tiny-lib'\nversion='0.1.0'\n",
            "crates/tiny-lib/src/lib.rs": "#[test]\nfn smoke() {}\n",
        })
        return benchmark, library

    def test_real_bin_debug_and_release_preparation_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library = self._workspace(Path(directory))
            for profile in ("debug", "release"):
                binary = benchmark / "target" / profile / "tiny_bin"
                receipt = prepare_cargo(benchmark=benchmark, library=library, binary=binary,
                                        receipt=Path(directory) / f"bin-{profile}.json", profile=profile)
                self.assertEqual(receipt["status"], "verified", receipt["errors"])
                self.assertEqual(verify_preparation_receipt(
                    receipt, benchmark=benchmark, library=library, binary=binary,
                    require_timing=profile == "release"), [])

    def test_real_virtual_workspace_lib_test_is_discovered_and_reverified(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library = self._workspace(Path(directory))
            receipt_path = Path(directory) / "lib-test.json"
            receipt = prepare_cargo(benchmark=benchmark, library=library, binary=None,
                                    receipt=receipt_path, package="tiny-lib", target_name="tiny_lib",
                                    artifact_kind="lib-test", profile="release", build_project=library)
            self.assertEqual(receipt["status"], "verified", receipt["errors"])
            artifact = receipt["build"]["artifact"]
            self.assertEqual(artifact["target"]["kind"], ["lib"])
            self.assertTrue(artifact["profile"]["test"])
            self.assertEqual(verify_preparation_receipt(
                receipt, benchmark=benchmark, library=library, package="tiny-lib",
                target_name="tiny_lib", artifact_kind="lib-test", require_timing=True), [])

    def test_prepare_cli_reuses_receipt_as_cached_lib_test_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library = self._workspace(Path(directory))
            scripts = Path(__file__).resolve().parents[1] / "scripts"
            (benchmark / "scripts").mkdir()
            for name in ("run_small_work.py", "small_work.py", "small_work_provenance.py"):
                shutil.copy2(scripts / name, benchmark / "scripts" / name)
            subprocess.run(["git", "add", "scripts"], cwd=benchmark, check=True)
            subprocess.run(["git", "commit", "-qm", "cli fixture"], cwd=benchmark, check=True)
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            common = [sys.executable, str(benchmark / "scripts/run_small_work.py"), "--prepare",
                      "--artifact-kind", "lib-test", "--build-project", str(library),
                      "--cargo-package", "tiny-lib", "--cargo-target-name", "tiny_lib",
                      "--tenferro-dir", str(library)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            first = subprocess.run(common + ["--output", str(first_path)], cwd=benchmark,
                                   env=env, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            reused = subprocess.run(common + ["--preparation-receipt", str(first_path),
                                              "--output", str(second_path)], cwd=benchmark,
                                    env=env, capture_output=True, text=True, check=False)
            self.assertEqual(reused.returncode, 0, reused.stderr)
            self.assertTrue(json.loads(second_path.read_text())["build"]["fresh_reused"])
            unproved_path = Path(directory) / "unproved.json"
            unproved = subprocess.run(common + ["--output", str(unproved_path)], cwd=benchmark,
                                      env=env, capture_output=True, text=True, check=False)
            self.assertNotEqual(unproved.returncode, 0)
            self.assertIn("previous receipt", unproved.stdout)

    def test_lib_test_receipt_cannot_be_verified_as_bin(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark, library = self._workspace(Path(directory))
            receipt = prepare_cargo(benchmark=benchmark, library=library, binary=None,
                                    receipt=Path(directory) / "lib-test.json", package="tiny-lib",
                                    target_name="tiny_lib", artifact_kind="lib-test", build_project=library)
            self.assertEqual(receipt["status"], "verified", receipt["errors"])
            errors = verify_preparation_receipt(receipt, benchmark=benchmark, library=library)
            self.assertTrue(any("selector" in error or "target" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
