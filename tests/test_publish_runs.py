"""Publication of recorded runs as pages: --publish, --check, raw-only mode.

A fixture repository holds the ``tests/fixtures/render_run`` run directories in
git.  Publishing derives pages from those records; ``--check`` re-renders them
and compares bytes, so a hand edit of a page, a summary or a raw measurement all
fail, and a source run that is not tracked is an error.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import regenerate_report_summaries as regenerate  # noqa: E402
import render_run  # noqa: E402
import report_summary  # noqa: E402

# The fixture repository keeps the recorded runs under tests/fixtures/, so the
# published pages name them the same way a real run under data/results/ is named.
FIXTURES = ROOT / "tests/fixtures/render_run"
RECORD = "tests/fixtures/render_run"
EINSUM_RUNS = (
    f"{RECORD}/einsum/20261018_000001",
    f"{RECORD}/einsum/20261018_000004",
)
SINGLE_RUNS = (
    f"{RECORD}/perf_issues/20261018_030000",
    f"{RECORD}/permutation/20261018_010000",
    f"{RECORD}/overhead_smoke/20261018_020000",
)
# A CPU-ops run keeps its raw batch samples next to the CSV they were summarized into.
CPU_OPS_RUN = f"{RECORD}/einsum/20261018_000002"
CPU_OPS_SAMPLES = f"{CPU_OPS_RUN}/cpu_ops_samples_t2_20261018_000002.jsonl"


def snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


class PublishRunsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "repository"
        (self.root / "benchmarks").mkdir(parents=True)
        (self.root / "tests").mkdir()
        shutil.copytree(FIXTURES, self.root / "tests/fixtures/render_run")
        (self.root / "benchmarks/profiles.yaml").write_text(
            "maintained:\n  - mac-cpu\n  - zen5-cpu\n"
        )
        self.unverified = self.root / "benchmarks/unverified-pages.txt"
        self.unverified.write_text("# every page is checked\n")
        # The renderer and the summaries spell run directories relative to the
        # repository they belong to, so point both at the fixture repository.
        for module, attribute in ((render_run, "ROOT"), (report_summary, "REPO_ROOT")):
            patcher = patch.object(module, attribute, self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.git("init", "--initial-branch=main")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")

    def git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def commit(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def publish(self, runs=EINSUM_RUNS):
        return regenerate.publish(
            [self.root / run for run in runs], self.root / "result", self.root
        )

    def check(self) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = regenerate.check(self.root / "result", self.root)
        return status, out.getvalue(), err.getvalue()

    def publish_all(self) -> None:
        """Publish an einsum page from two thread runs and three single-run pages."""
        self.publish()
        for run in SINGLE_RUNS:
            self.publish([run])
        self.commit("Record the runs and publish their pages")

    def test_published_pages_derive_from_the_selected_runs(self):
        written = self.publish()
        self.assertEqual(
            written,
            [
                self.root / "result/mac-cpu/cpu/einsum.md",
                self.root / "result/mac-cpu/cpu/einsum.summary.json",
            ],
        )
        page = (self.root / "result/mac-cpu/cpu/einsum.md").read_text()
        for threads, run in zip((1, 4), EINSUM_RUNS):
            self.assertIn(f"## Threads: {threads}", page)
            self.assertIn(f"{run}/report.md", page)
        document = report_summary.load_run_metadata(
            self.root / "result/mac-cpu/cpu/einsum.summary.json"
        )
        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(
            [entry["run"] for entry in document["source_runs"]], list(EINSUM_RUNS)
        )
        self.assertEqual([entry["threads"] for entry in document["source_runs"]], [[1], [4]])

    def test_check_passes_for_pages_published_from_tracked_runs(self):
        self.publish_all()
        status, out, err = self.check()
        self.assertEqual(status, 0, err)
        self.assertEqual(err, "")
        self.assertIn("checked 4 maintained pages", out)

    def test_one_byte_page_edit_fails(self):
        self.publish_all()
        page = self.root / "result/mac-cpu/cpu/einsum.md"
        page.write_text(page.read_text() + "\n")
        status, _, err = self.check()
        self.assertEqual(status, 1)
        self.assertIn("result/mac-cpu/cpu/einsum.md: differs", err)

    def test_one_byte_summary_edit_fails(self):
        self.publish_all()
        summary = self.root / "result/zen5-cpu/cpu/perf_issues.summary.json"
        summary.write_text(summary.read_text() + "\n")
        status, _, err = self.check()
        self.assertEqual(status, 1)
        self.assertIn("perf_issues.summary.json: differs", err)

    def test_edited_raw_timing_fails(self):
        self.publish_all()
        log = self.root / EINSUM_RUNS[0] / "tenferro_trace_t1_20261018.log"
        log.write_text(log.read_text().replace("1.0 0.1", "1.2 0.1"))
        self.commit("Edit a raw measurement")
        status, _, err = self.check()
        self.assertEqual(status, 1)
        self.assertIn("result/mac-cpu/cpu/einsum.md: differs", err)

    def test_edited_cpu_ops_sample_fails_check(self):
        # The published CSV still matches the page, but it no longer matches the
        # raw batch timings it was summarized from.
        self.publish([CPU_OPS_RUN])
        self.commit("Record the CPU-ops run and publish its pages")
        status, _, err = self.check()
        self.assertEqual(status, 0, err)
        samples = self.root / CPU_OPS_SAMPLES
        records = [json.loads(line) for line in samples.read_text().splitlines()]
        records[0]["per_operation_ns"] = [
            value * 1.5 for value in records[0]["per_operation_ns"]
        ]
        samples.write_text("".join(json.dumps(record) + "\n" for record in records))
        self.commit("Edit a raw CPU-ops sample")
        status, _, err = self.check()
        self.assertEqual(status, 1)
        self.assertIn("per_operation_ns", err)
        self.assertIn("cpu_ops", err)

    def test_untracked_source_run_fails(self):
        self.publish_all()
        self.git("rm", "-r", "--cached", "-q", "tests/fixtures/render_run/perf_issues")
        status, _, err = self.check()
        self.assertEqual(status, 1)
        self.assertIn(f"source run `{SINGLE_RUNS[0]}` is not tracked by git", err)

    def test_page_without_summary_fails(self):
        self.publish_all()
        (self.root / "result/zen5-cpu/cpu/permutation.summary.json").unlink()
        status, _, err = self.check()
        self.assertEqual(status, 1)
        self.assertIn("no permutation.summary.json", err)

    def test_unverified_pages_are_reported_not_hidden(self):
        self.publish_all()
        self.unverified.write_text(
            "# raw records not imported yet\n"
            "result/zen5-cpu/cpu/perf_issues.md  # fixtures are not the record\n"
        )
        (self.root / "result/zen5-cpu/cpu/perf_issues.md").write_text("# edited\n")
        status, out, err = self.check()
        self.assertEqual(status, 0, err)
        self.assertIn(
            "UNVERIFIED result/zen5-cpu/cpu/perf_issues.md: fixtures are not the record",
            out,
        )

    def test_raw_only_mode_writes_no_report_summary_or_page(self):
        self.publish_all()
        before = snapshot(self.root / "result")
        run_dir = self.root / EINSUM_RUNS[0]
        record = snapshot(run_dir)
        script = "\n".join(
            [
                "set -euo pipefail",
                f'source "{ROOT / "scripts/publish_run.sh"}"',
                f'publish_cpu_runs "{self.root}" "{run_dir}"',
                "",
            ]
        )
        environment = dict(os.environ, BENCHMARK_RAW_ONLY="1")
        completed = subprocess.run(
            ["bash", "-c", script], env=environment, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("BENCHMARK_RAW_ONLY=1", completed.stderr)
        self.assertEqual(before, snapshot(self.root / "result"))
        self.assertEqual(record, snapshot(run_dir))


if __name__ == "__main__":
    unittest.main()
