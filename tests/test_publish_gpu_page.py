"""The GPU public-API page is published from the run it names.

``tests/fixtures/render_run/gpu_linalg_public_api_throughput/20261102_210935``
is a recorded run directory: the case selection the collector materialized
(``instances.json``), one JSONL per backend and the folded ``records.jsonl``.
Publishing it must reproduce the committed page and summary byte for byte, so
"the page is reproducible from a run id" is a test rather than a claim.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import regenerate_report_summaries as regenerate  # noqa: E402
import report_summary  # noqa: E402

FIXTURES = ROOT / "tests/fixtures/render_run"
GPU_RUN = "tests/fixtures/render_run/gpu_linalg_public_api_throughput/20261102_210935"
EXPECTED = FIXTURES / "expected"
PAGE = "result/nvidia-gpu/gpu/linalg_public_api_throughput.md"
SUMMARY = "result/nvidia-gpu/gpu/linalg_public_api_throughput.summary.json"
SUITE = "gpu/linalg_public_api_throughput"


class PublishGpuPageTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "repository"
        shutil.copytree(FIXTURES, self.root / "tests/fixtures/render_run")
        # The page spells its source run relative to the repository it belongs to.
        patcher = patch.object(report_summary, "REPO_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def copy_run(self, family: str, stamp: str, *, commit: str | None = None) -> Path:
        """Record the fixture run again under ``gpu/<family>``/``stamp``."""
        target = self.root / "data/results/nvidia-gpu/gpu" / family / stamp
        shutil.copytree(self.root / GPU_RUN, target)
        if commit is not None:
            metadata = target / "run.yaml"
            metadata.write_text(
                metadata.read_text().replace(
                    "433121a7756fdb4f109b02f0422d3714a04514c6", commit
                )
            )
        return target

    def page_naming(self, run: Path) -> Path:
        """A published page that names ``run``, the way a failed run leaves one."""
        report = self.root / PAGE
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            f"# {SUITE}: CUDA public API benchmark\n\n"
            f"Raw run: `{run.relative_to(self.root)}/`.\n"
        )
        return report

    def regenerate(self) -> dict:
        written = regenerate.regenerate(
            self.root / "result", self.root / "data/results", {"nvidia-gpu"}, {SUITE}
        )
        self.assertEqual(len(written), 1, written)
        return json.loads((self.root / SUMMARY).read_text())

    def test_published_gpu_page_matches_the_committed_expected_page(self):
        written = regenerate.publish(
            [self.root / GPU_RUN], self.root / "result", self.root
        )
        self.assertEqual(written, [self.root / PAGE, self.root / SUMMARY])
        self.assertEqual(
            (self.root / PAGE).read_bytes(),
            (EXPECTED / "gpu_linalg_public_api_throughput.page.md").read_bytes(),
        )
        self.assertEqual(
            (self.root / SUMMARY).read_bytes(),
            (
                EXPECTED / "gpu_linalg_public_api_throughput.page.summary.json"
            ).read_bytes(),
        )

    def test_regenerate_writes_the_gpu_summary_beside_its_page(self):
        run = self.copy_run("linalg_public_api_throughput", "20261102_210935")
        self.page_naming(run)
        written = regenerate.regenerate(
            self.root / "result", self.root / "data/results", {"nvidia-gpu"}, {SUITE}
        )
        self.assertEqual(written, [self.root / SUMMARY])
        self.assertTrue((self.root / SUMMARY).is_file())

    def test_the_page_states_that_its_shape_class_is_not_a_measured_classification(self):
        regenerate.publish([self.root / GPU_RUN], self.root / "result", self.root)
        page = (self.root / PAGE).read_text()
        self.assertIn("`large` is a shape class, not a measured classification", page)

    def test_declared_run_in_the_previous_suite_layout_is_the_source(self):
        declared = self.copy_run("linalg_public_api", "20261102_210935")
        newer = self.copy_run(
            "linalg_public_api_throughput",
            "20261103_000000",
            commit="9999999999999999999999999999999999999999",
        )
        self.assertGreater(newer.name, declared.name)
        report = self.page_naming(declared)
        suite_dir = self.root / "data/results/nvidia-gpu/gpu/linalg_public_api_throughput"
        self.assertEqual(
            regenerate.run_dir_from_report(report, suite_dir), declared.resolve()
        )
        # The run the page names wins over the newer local run it does not name.
        self.assertEqual(self.regenerate()["tenferro_commit"], "433121a7756fdb4f109b02f0422d3714a04514c6")

    def test_absent_declared_run_fails_naming_the_run(self):
        absent = self.root / "data/results/nvidia-gpu/gpu/linalg_public_api/20260101_000000"
        self.page_naming(absent)
        with self.assertRaises(regenerate.RegenerateError) as raised:
            self.regenerate()
        self.assertIn(str(absent), str(raised.exception))
        self.assertFalse((self.root / SUMMARY).exists())

    def test_record_missing_a_field_fails_naming_field_and_run(self):
        run = self.copy_run("linalg_public_api_throughput", "20261102_210935")
        self.page_naming(run)
        records = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]
        for record in records:
            record.pop("backend", None)
        (run / "records.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        with self.assertRaises(regenerate.RegenerateError) as raised:
            self.regenerate()
        message = str(raised.exception)
        self.assertIn("'backend'", message)
        self.assertIn(str(run), message)

    def test_combined_pre_split_run_fails_naming_its_missing_selection(self):
        """The pre-split run held both shape classes and cannot become one page."""
        run = self.copy_run("linalg_public_api", "20261102_210935")
        metadata = run / "run.yaml"
        metadata.write_text(
            metadata.read_text().replace(SUITE, "gpu/linalg_public_api")
        )
        (run / "instances.json").unlink()
        self.page_naming(run)
        with self.assertRaises(regenerate.RegenerateError) as raised:
            self.regenerate()
        message = str(raised.exception)
        self.assertIn("instances.json", message)
        self.assertIn(str(run), message)
        self.assertFalse((self.root / SUMMARY).exists())

    def test_publish_fails_naming_a_backend_row_field_that_is_gone(self):
        run = self.copy_run("linalg_public_api_throughput", "20261102_210935")
        rows = [
            json.loads(line)
            for line in (run / "tenferro-cuda-direct.jsonl").read_text().splitlines()
        ]
        for row in rows:
            (row.get("timing") or {}).pop("batch_ms", None)
        (run / "tenferro-cuda-direct.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        with self.assertRaises(regenerate.PublishError) as raised:
            regenerate.publish([run], self.root / "result", self.root)
        message = str(raised.exception)
        self.assertIn("'batch_ms'", message)
        self.assertIn(str(run), message)

    def summary_rows(self) -> list[dict]:
        return json.loads((self.root / SUMMARY).read_text())["rows"]

    def test_summary_holds_only_the_cells_the_page_declares(self):
        run = self.copy_run("linalg_public_api_throughput", "20261102_210935")
        extra = json.loads((run / "records.jsonl").read_text().splitlines()[0])
        extra["problem_id"] = "cholesky_f32_32x32"
        extra["op"] = "cholesky"
        with (run / "records.jsonl").open("a") as handle:
            handle.write(json.dumps(extra) + "\n")
        regenerate.publish([run], self.root / "result", self.root)
        cases = {row["case"] for row in self.summary_rows()}
        self.assertEqual(
            cases, {"cholesky_f32_512x512", "householder_r_default_f32_512x256"}
        )

    def test_publish_fails_on_an_ok_summary_row_without_a_median(self):
        run = self.copy_run("linalg_public_api_throughput", "20261102_210935")
        records = [
            json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()
        ]
        for record in records:
            if record["status"] == "ok":
                (record.get("timing") or {}).pop("median_ms", None)
        (run / "records.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        with self.assertRaises(regenerate.PublishError) as raised:
            regenerate.publish([run], self.root / "result", self.root)
        message = str(raised.exception)
        self.assertIn("timing.median_ms", message)
        self.assertIn(str(run), message)

    def test_publish_fails_when_the_run_kept_no_case_selection(self):
        run = self.copy_run("linalg_public_api_throughput", "20261102_210935")
        (run / "instances.json").unlink()
        with self.assertRaises(regenerate.PublishError) as raised:
            regenerate.publish([run], self.root / "result", self.root)
        self.assertIn("instances.json", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
