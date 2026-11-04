"""Print 1 when the selected cpu/perf_issues cases execute the Rust follow-up MWE.

scripts/run_perf_issues.sh uses this to decide whether it must build
mwe/cpu_followup into target/followup-current before collection (see #329).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_perf_issues as suite


def needs_followup_mwe(threads=None):
    _, cases, _, _, _, selected, _ = suite.plan(threads=threads)
    return any(cases[cid].get("kind") == "cpu_followup_mwe" or "followup_mwe" in cases[cid]
               for cid in selected)


def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print("1" if needs_followup_mwe(threads) else "0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
