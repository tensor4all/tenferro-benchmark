#!/usr/bin/env python3
"""Run with: uv run python tests/test_gpu_item_notes.py"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import format_gpu_results as formatter
import format_gpu_linalg_ad_results as ad_formatter


class ItemNotesTest(unittest.TestCase):
    def test_ad_formatter_links_the_documented_problem(self):
        problem = "linalg_ad_small_grad_sum_qr_jvp_f64_8"
        href = "../../../notes/nvidia-gpu/gpu/linalg_ad_latency.md#" + problem
        records = [{"problem_id": problem, "backend": "tenferro-cuda-trace",
                    "status": "ok", "timing": {"median_ms": 1.0, "iqr_ms": 0.1}}]
        with patch.object(ad_formatter, "collect_cpu_info", return_value={}), \
             patch.object(ad_formatter, "cpu_info_markdown", return_value=""), \
             patch.object(ad_formatter, "resolve_gpu_info", return_value={}), \
             patch.object(ad_formatter, "gpu_info_markdown", return_value=""):
            markdown = ad_formatter.format_markdown(records, problem_notes={problem: href})
        self.assertIn("[`grad_sum_qr_jvp`](" + href + ")", markdown)
        self.assertIn("without a reference_backend are execution-only", markdown)

    def test_persistent_notes_are_linked_only_for_existing_anchors(self):
        records = [
            {"problem_id": name, "backend": "tenferro-cuda-eager",
             "suite_id": "gpu/dense", "op": "batched_matmul", "status": "ok",
             "timing": {"median_ms": 2.0}}
            for name in ("documented", "undocumented")
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result/nvidia-gpu/gpu/dense.md"
            notes = root / "notes/nvidia-gpu/gpu/dense.md"
            with patch.object(formatter, "__file__", str(root / "scripts/format_gpu_results.py")):
                self.assertEqual(formatter.load_problem_notes(output, records), {})
                notes.parent.mkdir(parents=True)
                text = '<a id="documented"></a>\n## Analysis\nNot yet adopted.\n'
                notes.write_text(text)
                links = formatter.load_problem_notes(output, records)
                self.assertEqual(links, {
                    "documented": "../../../notes/nvidia-gpu/gpu/dense.md#documented"
                })
                self.assertEqual(formatter.load_problem_notes(None, records), {})
                self.assertEqual(formatter.load_problem_notes(root / "external.md", records), {})
                with patch.object(formatter, "collect_cpu_info", return_value={}), \
                     patch.object(formatter, "cpu_info_markdown", return_value=""), \
                     patch.object(formatter, "resolve_gpu_info", return_value={}), \
                     patch.object(formatter, "gpu_info_markdown", return_value=""):
                    markdown = formatter.format_markdown(records, problem_notes=links)
                    self.assertIn("| [documented](" + links["documented"] + ") | 2.000 |", markdown)
                    self.assertIn("| undocumented | 2.000 |", markdown)
                    output.parent.mkdir(parents=True)
                    output.write_text(markdown)
                    output.write_text(formatter.format_markdown(records, problem_notes=links))
                self.assertEqual(notes.read_text(), text)
                self.assertFalse(output.read_text().endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
