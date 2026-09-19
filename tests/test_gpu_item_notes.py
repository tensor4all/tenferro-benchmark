#!/usr/bin/env python3
"""Run with: uv run python tests/test_gpu_item_notes.py"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import format_gpu_results as formatter


class ItemNotesTest(unittest.TestCase):
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
