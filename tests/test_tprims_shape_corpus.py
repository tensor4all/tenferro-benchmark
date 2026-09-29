import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tprims_shape_corpus import build, main

MM = {"op": "dot_general", "dtype": "F64", "outcome": "executed",
      "a": {"dims": [4, 3], "strides": [1, 4]}, "b": {"dims": [3, 5], "strides": [1, 3]},
      "lc": [1], "rc": [0], "lb": [], "rb": [], "conj": [False, False]}
BMM = {"op": "dot_general", "dtype": "C64", "outcome": "executed",
       "a": {"dims": [2, 3, 7], "strides": [1, 2, 6]}, "b": {"dims": [3, 4, 7], "strides": [1, 3, 12]},
       "lc": [1], "rc": [0], "lb": [2], "rb": [2], "conj": [True, False]}
GEMM = {"op": "gemm_batched", "dtype": "F32", "outcome": "executed", "m": 2, "n": 3, "k": 4,
        "batch": 5, "a": [1, 2, 8], "b": [1, 4, 12], "c": [1, 2, 6], "conj": [False, False]}


class ShapeCorpus(unittest.TestCase):
    def test_groups_ranks_and_derives_outputs(self):
        recs = [dict(MM, ns=10)] * 3 + [dict(BMM, ns=100), dict(GEMM, ns=5),
                                        dict(MM, ns=1, outcome="unsupported")]
        out = build(recs, time_share=1.0, frequent=0)
        self.assertEqual([e["op"] for e in out], ["dot_general", "dot_general", "gemm_batched"])
        self.assertEqual(out[0]["c"], {"dims": [2, 4, 7], "strides": [1, 2, 8]})  # batch last
        self.assertEqual(out[0]["calls"], 1)
        self.assertEqual(out[1]["calls"], 3)
        self.assertEqual(out[1]["dtype"], "f64")
        self.assertEqual(out[2]["a"], {"dims": [2, 4, 5], "strides": [1, 2, 8]})
        self.assertAlmostEqual(sum(e["time_share"] for e in out), 1.0, places=5)

    def test_time_share_cut_then_most_frequent(self):
        recs = [dict(BMM, ns=1000)] + [dict(MM, ns=1)] * 50 + [dict(GEMM, ns=2)]
        out = build(recs, time_share=0.9, frequent=1)
        self.assertEqual([e["dtype"] for e in out], ["c64", "f64"])

    def test_cli_writes_a_corpus_file(self):
        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / "log.jsonl"
            log.write_text("\n".join(json.dumps(dict(MM, ns=3)) for _ in range(2)) + "\n")
            dst = Path(t) / "corpus.json"
            self.assertEqual(main([str(log), "-o", str(dst), "--source", '{"suite": "x"}']), 0)
            c = json.loads(dst.read_text())
            self.assertEqual(c["source"]["suite"], "x")
            self.assertEqual(c["entries"][0]["name"], "dot_general_000_f64")


if __name__ == "__main__":
    unittest.main()
