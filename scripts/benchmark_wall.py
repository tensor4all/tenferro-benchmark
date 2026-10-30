"""Descriptive command-wall observations; never a functional timing gate."""
import json
import math
import os
import time


def record_collection_wall(run_dir):
    """Record how long the collection command ran, as a collection fact.

    Called by the finalize step of a runner, before anything is rendered, so the
    observation covers startup, validation, build and collection only.  Rendering
    a page later never refreshes it.
    """
    start = os.environ.get("BENCH_COMMAND_START_NS")
    if start is None:
        return
    configured = os.environ.get("BENCH_COMMAND_BUDGET_SECONDS")
    budget = float(configured) if configured else None
    if budget is not None and (not math.isfinite(budget) or budget <= 0):
        raise ValueError("BENCH_COMMAND_BUDGET_SECONDS must be finite and positive")
    total = (time.monotonic_ns() - int(start)) / 1e9
    observation = {
        "command_seconds_through_collection": total,
        "scope": "command startup through collection; excludes report rendering and publication",
        "budget_seconds": budget,
        "budget_status": "unconfigured" if budget is None else
                         ("within_budget" if total <= budget else "over_budget; diagnostic only"),
    }
    (run_dir / "wall_command.json").write_text(json.dumps(observation, indent=2) + "\n")
