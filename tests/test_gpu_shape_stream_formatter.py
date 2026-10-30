import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import format_gpu_shape_stream_results as formatter


class ShapeStreamFormatterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.timing, self.counter = root / "timing", root / "counter"
        self.cases = []
        for instrument, path, role, index, key in (
                (False, self.timing, "benchmark", 0, "latency"),
                (True, self.counter, "debug", 1, "counters")):
            path.mkdir()
            params = {"n": 16, "instrument": instrument}
            scope = "counter_diagnostic" if instrument else "shape_stream_diagnostic"
            self.cases.append(dict(id=key, kind="shape_stream", params=params,
                                   arm="fixed-warm", measurement_scope=scope))
            metadata = dict(harness=dict(commit="abcdef"), tenferro_rs=dict(commit="same"),
                            cuda=dict(device_uuid=f"GPU-{index}", device_name="RTX 3060", driver_version="580"))
            (path / "run.yaml").write_text(yaml.safe_dump(metadata))
            provenance = dict(harness_commit="abc", device_uuid=f"GPU-{index}",
                              physical_device_index=index, gpu_role=role)
            provenance.update(dict(source_sha256="same", verification_only=True) if instrument else
                              dict(shape_stream_source_sha256="same", effort="scan", warmups=1, samples=3))
            (path / "provenance.json").write_text(json.dumps(provenance))
            row = dict(case_id=key, kind="shape_stream", dtype="f64", threads=1, status="passed",
                       correctness_status="passed", max_rel_error=0.0, params=params,
                       measurement_scope=scope, retained_output_bytes=8,
                       sequence=[dict(key=0, lhs_shape=[1], lhs_strides=[1])], operations_per_sequence=1,
                       instrumentation_only=instrument, samples=[], counter_observations=[])
            if instrument:
                def snapshot(launches, allocations, live_bytes):
                    return dict(
                        device=dict(live_allocations=1, live_bytes=live_bytes, pool_used_bytes=live_bytes,
                                    pool_reserved_bytes=live_bytes, idle_pinned_bytes=0),
                        retirement=dict(pending_bytes=live_bytes, module_bytes=0, pending_records=1,
                                        epochs=1, frontiers=0, reclaimed_epochs=0, launches=launches,
                                        allocations=allocations, zero_fills=0, events_recorded=0),
                        artifact_cache=dict(entries=1, bytes=100),
                        module_cache=dict(entries=1, bytes=100),
                        compile_attempts=1, module_loads=1, artifact_cache_hits=0)
                before = snapshot(0, 0, 8)
                after = snapshot(1, 2, 8)
                row["counter_observations"] = [dict(phase="before", stats=before),
                    dict(phase="enqueue", index=0, key=0, stats=after),
                    dict(phase="completion", stats=after), dict(phase="drop", stats=after)]
            else:
                row["samples"] = [dict(sample_index=i, elapsed_ns=1000+i*10, iterations=1) for i in range(3)]
                row["host_enqueue_samples_ns"] = [900, 910, 920]
            (path / "samples_t1.jsonl").write_text(json.dumps(row)+"\n")

    def mutate(self, path, change):
        row = json.loads((path / "samples_t1.jsonl").read_text())
        change(row)
        (path / "samples_t1.jsonl").write_text(json.dumps(row)+"\n")

    def render(self):
        return formatter.render(self.cases, self.timing, self.counter)

    def test_sequence_units_and_separate_counter_deltas(self):
        text = self.render()
        self.assertIn("µs per **sequence**", text)
        self.assertIn("| latency | 1 | 0.91 (0.01) | 1.01 (0.01)", text)
        self.assertIn("| counters | 8→8 | 2 | 1 | 0 | 1 | 1 | 8 |", text)
        self.assertIn('"order":[0]', text)

    def test_duplicate_is_rejected(self):
        path = self.timing / "samples_t1.jsonl"
        path.write_text(path.read_text()*2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.render()

    def test_counter_timings_are_rejected(self):
        self.mutate(self.counter, lambda r: r.update(samples=[dict(elapsed_ns=1000)]))
        with self.assertRaisesRegex(ValueError, "timings present"):
            self.render()

    def test_missing_case_is_rejected(self):
        (self.timing / "samples_t1.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "missing"):
            self.render()

    def test_instrumented_timing_is_rejected(self):
        self.mutate(self.timing, lambda r: r.update(instrumentation_only=True))
        with self.assertRaisesRegex(ValueError, "instrumentation"):
            self.render()

    def test_different_sequences_are_rejected(self):
        self.mutate(self.counter, lambda r: r.update(sequence=[dict(key=1)]))
        with self.assertRaisesRegex(ValueError, "sequences differ"):
            self.render()


if __name__ == "__main__":
    unittest.main()
