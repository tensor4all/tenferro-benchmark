"""Descriptive command-wall observations; never a functional timing gate."""
import json
import math
import os
import time


def record_report_wall(run_dir):
    start = os.environ.get("BENCH_COMMAND_START_NS")
    report_start = os.environ.get("BENCH_REPORT_START_NS")
    if start is None or report_start is None:
        return
    now = time.monotonic_ns()
    total = (now - int(start)) / 1e9
    configured = os.environ.get("BENCH_COMMAND_BUDGET_SECONDS")
    budget = float(configured) if configured else None
    if budget is not None and (not math.isfinite(budget) or budget <= 0):
        raise ValueError("BENCH_COMMAND_BUDGET_SECONDS must be finite and positive")
    observation = {
        "command_seconds_before_final_report_write": total,
        "report_generation_seconds": (now - int(report_start)) / 1e9,
        "scope": "command startup through report generation; excludes final report file writes",
        "budget_seconds": budget,
        "budget_status": "unconfigured" if budget is None else
                         ("within_budget" if total <= budget else "over_budget; diagnostic only"),
    }
    (run_dir / "wall_command.json").write_text(json.dumps(observation, indent=2) + "\n")
