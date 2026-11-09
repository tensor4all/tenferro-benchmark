#!/usr/bin/env python3
"""Regime policy for the benchmark suites.

A *regime* is the workload class a case belongs to: `throughput`
(compute- or bandwidth-bound) or `latency` (dominated by per-call launch,
dispatch and synchronization cost).  The dashboard publishes one page per
regime and never mixes them, so a case's regime decides which page its bars
appear on.

Two rules are enforced here, and both fail closed:

1. **No implicit regime.**  A case's regime comes from its own declaration or
   from its suite's `defaults.regime`.  There is no fallback value: an
   undeclared case is an error, because the old fallback (`throughput`)
   silently promoted every undeclared case onto the throughput page, including
   cases whose measured time is per-call framework overhead.

2. **A throughput case must be large enough to be one.**  Every family has a
   documented floor on the case's declared working set (in scalar elements).
   A case that declares `throughput` below its family floor is an error, the
   same way an unknown `measure` value is: the declaration is checked at load
   time instead of being quietly published on the wrong page.

The floors are derived from the published NVIDIA A100 summaries in
`result/nvidia-gpu/gpu/*.summary.json`; `docs/regime-floors.md` carries the
table, the evidence for each value, and which values are provisional.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

THROUGHPUT = "throughput"
LATENCY = "latency"
REGIMES = (THROUGHPUT, LATENCY)

# How a case is timed.  This is a property of the collection mode, not of the
# workload class: `gpu/linalg_ad_throughput` times its cases in deferred-batch
# mode even when a case's regime is `latency` (a case can be per-call overhead
# and still be measured with one retirement per K calls).
MEASURES = ("deferred_batch", "per_call")


class RegimeError(ValueError):
    """Raised when a suite declares a regime that cannot be published."""


@dataclass(frozen=True)
class Floor:
    """Minimum declared working set for a `throughput` case in one family."""

    family: str
    elements: int
    basis: str
    provisional: bool = False


# Family floors, in scalar elements of the case's declared working set.
#
# `elements` is the size of the largest tensor the case declares (n for a 1-D
# elementwise case, prod(shape) for an instance record, m*n for a matmul, nnz
# for a sparse case).  The basis strings are the evidence: the size pair whose
# doubling behaviour shows where fixed per-call cost stops dominating.
FAMILY_FLOORS: dict[str, Floor] = {
    "dense": Floor(
        "dense",
        65_536,
        "dense_svd_f64_256 is 65,536 elements and takes 12.0 ms, ~900x the 13 µs "
        "dispatch scale, so a 256² dense factorisation is compute; "
        "dense_batched_matmul b16x32³ is 16,384 elements and takes 13 µs (latency).",
    ),
    "einsum": Floor(
        "einsum",
        1_048_576,
        "the only published einsum case is a 3072³ binary contraction "
        "(9.4e6 elements, 3.17 ms).",
        provisional=True,
    ),
    "elementwise": Floor(
        "elementwise",
        262_144,
        "the two published elementwise_chain points (1k = 69.6 µs, 1M = 99.6 µs in "
        "throughput mode) fit t = 69.6 µs + 27.9 ps * n, so doubling the work moves the "
        "time by 10% only above n ~ 2.8e5.  At 2^18 the gain is ~9.8%, just under that "
        "criterion: 2^18 is kept because it is the size the published reuse_contraction "
        "cases are measured at and moving them is a separate decision; 1k is a constant "
        "(1000x the elements buys 1.43x the time), and at 1M the fixed share is ~70%.",
        provisional=True,
    ),
    "fft": Floor(
        "fft",
        1_048_576,
        "only the 1M point is published (0.063-0.135 ms); the 1k point was never "
        "measured, so the floor sits at the only measured size.",
        provisional=True,
    ),
    "linalg": Floor(
        "linalg",
        65_536,
        "the same backend and dtype (tenferro-cuda-direct f64) at cholesky 32² "
        "(1,024 elements, 77.7 us) and 512² (262,144 elements, 419.4 us) fits "
        "1.31 ns/element plus a ~76 us fixed term, so 18% of the 512² call is still "
        "fixed; 2^16 keeps every 512² case and excludes the 32²/64x32 catalogue "
        "entries, and stays provisional because two sizes are all the evidence.",
        provisional=True,
    ),
    "linalg_ad": Floor(
        "linalg_ad",
        1_048_576,
        "QR JVP costs 0.68 ms per call at n=4 (16 elements) and 2.87 ms at n=256 "
        "(65,536), both single-call on tenferro-cuda-trace: 4,096x the elements "
        "buys 4.2x the time; a two-point fit puts about 24% of the 65,536-element "
        "call in a fixed term, and that term is the AD wrapper rather than the "
        "256² factorisation it wraps (tens of us).  (The deferred-batch suite "
        "reports 1.43 ms for the same case under a different timing contract, which "
        "is why the pair above is the single-call one.)  No AD case above 512² is "
        "measured, so 2^20 is a conservative bound extrapolated from two smaller "
        "sizes, not a measured crossing; no AD case currently clears it.",
        provisional=True,
    ),
    "permutation": Floor(
        "permutation",
        1_048_576,
        "every published pattern is 3.9e8-5.4e8 elements (5.0-8.4 ms); the catalogue "
        "is GPU-only and explicitly sized for steady-state timing.",
    ),
    "sparse": Floor(
        "sparse",
        1_048_576,
        "spmv at 6.7e7 nnz takes 2.26 ms and spmm at 4.2e6 nnz takes 16.3 ms; the "
        "floor is 2^20 nnz, two orders below the smallest published case.",
        provisional=True,
    ),
    "tensornetwork": Floor(
        "tensornetwork",
        1_048_576,
        "the only case is a whole-network contraction at 78 ms; the TensorNetworkBenchmarks "
        "fixture is an external submodule, so the element count comes from the "
        "problem's `size_elements` when the fixture is available and the floor is "
        "otherwise satisfied by `floor_exempt` with a reason.",
        provisional=True,
    ),
}

# Suites whose cases do not carry a `family` of their own (their cases come from
# an instance document, which has no family field).  A problem-level `family`
# always wins over this map.
SUITE_FAMILIES: dict[str, str] = {
    "gpu/dense": "dense",
    "gpu/dense_throughput": "dense",
    "gpu/einsum": "einsum",
    "gpu/elementwise": "elementwise",
    "gpu/fft": "fft",
    "gpu/isolated_ops": "elementwise",
    "gpu/nn": "elementwise",
    "gpu/structural": "elementwise",
    "gpu/reuse_contraction": "elementwise",
    "gpu/linalg": "linalg",
    "gpu/linalg_ad": "linalg_ad",
    "gpu/linalg_ad_latency": "linalg_ad",
    "gpu/linalg_ad_throughput": "linalg_ad",
    "gpu/linalg_jvp_vjp": "linalg_ad",
    "gpu/linalg_public_api": "linalg",
    "gpu/linalg_public_api_latency": "linalg",
    "gpu/linalg_public_api_throughput": "linalg",
    "gpu/lstsq": "linalg",
    "gpu/permutation": "permutation",
    "gpu/perf_issues": "linalg",
    "gpu/sparse": "sparse",
    "gpu/tensornetwork": "tensornetwork",
}

# Only the GPU manifests are policed here.  The CPU catalogues still carry
# sub-floor cases that were classified by the old fallback; moving them is a
# separate change (docs/regime-floors.md).
POLICED_SUITE_PREFIXES = ("gpu/",)


def is_policed(suite_id: str) -> bool:
    return suite_id.startswith(POLICED_SUITE_PREFIXES)


def family_for(suite_id: str, problem: dict[str, Any] | None = None) -> str | None:
    declared = (problem or {}).get("family")
    if isinstance(declared, str) and declared:
        return declared
    if suite_id in SUITE_FAMILIES:
        return SUITE_FAMILIES[suite_id]
    head = suite_id.split("/", 1)[-1]
    for candidate in (head, head.rsplit("_", 1)[0]):
        if candidate in FAMILY_FLOORS:
            return candidate
    return None


def _prod(values: Iterable[Any]) -> int | None:
    total = 1
    seen = False
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return None
        total *= value
        seen = True
    return total if seen else None


def _size_from_block(block: Any, rule: str) -> int | None:
    if not isinstance(block, dict):
        return None
    try:
        if rule == "matmul":
            # The working set is the largest tensor in the call, not the output:
            # an `m x k` by `k x n` matmul touches inputs that can dwarf `m x n`.
            sizes = [
                _prod([block["m"], block["n"]]),
                _prod([block["m"], block["k"]]),
                _prod([block["k"], block["n"]]),
            ]
            known = [size for size in sizes if size]
            return max(known) if known else None
        if rule == "batched_matmul":
            batch = block["batch"]
            sizes = [
                _prod([batch, block["m"], block["n"]]),
                _prod([batch, block["m"], block["k"]]),
                _prod([batch, block["k"], block["n"]]),
            ]
            known = [size for size in sizes if size]
            return max(known) if known else None
        if rule == "linalg":
            if isinstance(block.get("m"), int):
                rows, cols = block["m"], block["n"]
            else:
                rows = cols = block.get("n")
            sizes = [_prod([rows, cols])]
            rhs = block.get("rhs_cols")
            if isinstance(rhs, int):
                # A solve constructs the right-hand side too, and it can hold more
                # elements than the coefficient matrix.
                sizes.append(_prod([rows, rhs]))
                sizes.append(_prod([cols, rhs]))
            known = [size for size in sizes if size]
            return max(known) if known else None
        if rule == "linalg_ad":
            n = block.get("n")
            sizes = [_prod([n, n])]
            rhs = block.get("rhs_cols")
            if isinstance(rhs, int):
                sizes.append(_prod([n, rhs]))
            known = [size for size in sizes if size]
            return max(known) if known else None
        if rule == "elementwise_chain":
            n = block.get("n")
            return n if isinstance(n, int) else None
        if rule == "sparse":
            nnz = block.get("nnz")
            return nnz if isinstance(nnz, int) else None
        if rule == "einsum":
            shapes = block.get("shapes_rowmajor") or block.get("shapes")
            if not isinstance(shapes, list) or not shapes:
                return None
            # The working set is the largest operand, the same rule the catalogue
            # `shapes` path below uses.  Reading only `shapes[0]` let the operand
            # order decide whether a declaration cleared the floor.
            sizes = [size for size in (_prod(shape) for shape in shapes) if size]
            return max(sizes) if sizes else None
    except (KeyError, TypeError):
        return None
    return None


_SIZE_RULES = (
    ("matmul", "matmul"),
    ("batched_matmul", "batched_matmul"),
    ("linalg", "linalg"),
    ("linalg_ad", "linalg_ad"),
    ("elementwise_chain", "elementwise_chain"),
    ("sparse", "sparse"),
    ("einsum", "einsum"),
)


def case_elements(problem: dict[str, Any]) -> int | None:
    """Declared working set of one case, in scalar elements.

    An explicit `size_elements` wins (used where the shape lives outside the
    declaration, e.g. an external tensor-network fixture); otherwise the first
    recognised shape block decides.
    """
    explicit = problem.get("size_elements")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        return explicit
    for key, rule in _SIZE_RULES:
        if key in problem:
            size = _size_from_block(problem.get(key), rule)
            if size:
                return size
    # Instance records carry the element count directly (`n`) or a shape.
    n = problem.get("n")
    if isinstance(n, int) and not isinstance(n, bool) and n > 0:
        return n
    shape = problem.get("shape")
    if isinstance(shape, list):
        return _prod(shape)
    # Catalogue instances name the same field `shapes` (one entry per operand);
    # the largest operand is the working set.
    shapes = problem.get("shapes")
    if isinstance(shapes, list) and shapes:
        sizes = [_prod(item) for item in shapes if isinstance(item, list)]
        sizes = [size for size in sizes if size]
        if sizes:
            return max(sizes)
    # perf_issues records keep the shape in `params`: a full shape wins, then an
    # explicit m x n, then a lone `n` read as the side of a square matrix (the
    # `*_n512` cases are 512x512).
    params = problem.get("params")
    if isinstance(params, dict):
        if isinstance(params.get("shape"), list):
            return _prod(params["shape"])
        if isinstance(params.get("m"), int) and isinstance(params.get("n"), int):
            return _prod([params["m"], params["n"]])
        side = params.get("n")
        if isinstance(side, int) and not isinstance(side, bool) and side > 0:
            return _prod([side, side])
    return None


def _instances(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        rows = document
    elif isinstance(document, dict):
        rows = document.get("cases") or document.get("patterns") or []
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def declared_cases(
    suite: dict[str, Any], project_dir: Path, skips: list[str] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The suite's defaults and its cases, with the selector resolved.

    Inline problems are returned as written.  A `{source, include}` selector is
    resolved against its instance document, so every included case keeps the
    record's own `regime`, `n`/`shape` and family-facing fields.

    The selector fails closed.  A source that does not load, and an `include` id
    the loaded source does not contain, are errors: a broken catalogue used to
    produce no cases, so the suite validated with fewer cases than its author
    declared.  A source that may legitimately be absent opts in with
    `missing_source: skip`, and the skip is reported through `skips` so the
    validator names it on every run.
    """
    defaults = suite.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}
    suite_id = str(suite.get("suite_id") or "")
    problems = suite.get("problems")
    if isinstance(problems, list):
        return defaults, [row for row in problems if isinstance(row, dict)]
    if not isinstance(problems, dict):
        return defaults, []

    missing = problems.get("missing_source")
    if missing is not None and missing != "skip":
        raise RegimeError(
            f"{suite_id}: missing_source {missing!r} is not supported (only skip)"
        )
    source = problems.get("source")
    include = problems.get("include") or []
    exclude = set(problems.get("exclude") or [])
    if not isinstance(source, str) or not source or not isinstance(include, list):
        return defaults, []

    path = Path(source)
    if not path.is_absolute():
        path = project_dir / path
    # Absence of the declared source is the only thing the opt-in covers: a file
    # that disappears while its directory is read is a load failure, not a skip.
    if not path.exists():
        if problems.get("missing_source") != "skip":
            raise RegimeError(f"{suite_id}: include source {source!r} did not load: no such file or directory")
        if skips is not None:
            skips.append(f"{suite_id}: skipped include source {source!r} (missing_source: skip)")
        return defaults, []
    try:
        if path.is_dir():
            # A catalogue can point at a directory of per-instance files (the CPU
            # einsum suite); each file is one instance, named by `name`.
            document = []
            for entry in sorted(path.glob("*.json")):
                record = json.loads(entry.read_text(encoding="utf-8"))
                if isinstance(record, dict) and not record.get("id"):
                    record = {**record, "id": record.get("name") or entry.stem}
                if isinstance(record, dict):
                    document.append(record)
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RegimeError(
            f"{suite_id}: include source {source!r} did not load: {error}"
        ) from error
    records = {str(row["id"]): row for row in _instances(document) if row.get("id")}

    selected: list[dict[str, Any]] = []
    for entry in include:
        # `include` stays a list of ids: the GPU selection tooling reads it as a
        # set of strings, so an id-to-regime mapping is not an option here.
        if isinstance(entry, str):
            record = records.get(entry)
            if record is None:
                raise RegimeError(f"{suite_id}: include id {entry!r} is not in {source!r}")
            if entry not in exclude:
                selected.append(record)
    return defaults, selected


def resolve(
    suite_id: str,
    defaults: dict[str, Any],
    problem: dict[str, Any],
) -> tuple[str | None, str | None]:
    """(regime, measure) declared for one case; `None` where nothing is declared."""
    regime = problem.get("regime") or defaults.get("regime")
    if isinstance(regime, str):
        regime = regime.strip() or None
    else:
        regime = None
    measure = problem.get("measure") or defaults.get("measure")
    if not isinstance(measure, str) or not measure.strip():
        measure = None
    return regime, (measure.strip() if measure else None)


def regime_for_case(suite_id: str, problem: dict[str, Any]) -> str:
    """The regime a generated case has to declare to pass `check_case`.

    Suite generators call this instead of hard-coding a regime, so a case whose
    size changes crosses its family floor as the measurement it actually is.  A
    case below the floor, or one whose size cannot be derived, is `latency`:
    that is the conservative declaration, and `check_case` accepts it without a
    size.
    """
    floor = FAMILY_FLOORS.get(family_for(suite_id, problem) or "")
    elements = case_elements(problem)
    if floor is not None and elements is not None and elements >= floor.elements:
        return THROUGHPUT
    return LATENCY


def _fmt(value: int | None) -> str:
    return "unknown" if value is None else f"{value:,}"


def check_case(
    suite_id: str,
    case_id: str,
    regime: str | None,
    elements: int | None,
    family: str | None,
    floor_exempt: str | None,
) -> None:
    """Raise `RegimeError` unless this case may be published as declared."""
    if not is_policed(suite_id):
        return
    if regime not in REGIMES:
        raise RegimeError(
            f"{suite_id}: case {case_id!r} declares no regime. Add `regime: "
            f"throughput|latency` to the case (or `defaults.regime` to the suite): "
            "there is no implicit regime."
        )
    if regime == LATENCY:
        return
    floor = FAMILY_FLOORS.get(family or "")
    if floor is None:
        raise RegimeError(
            f"{suite_id}: case {case_id!r} declares `throughput` but its family "
            f"({family or 'unknown'!r}) has no documented floor. Add the family to "
            "FAMILY_FLOORS in scripts/regime_policy.py (with evidence) or declare "
            "the case `latency`."
        )
    if elements is None:
        if floor_exempt:
            return
        raise RegimeError(
            f"{suite_id}: case {case_id!r} declares `throughput` but its working set "
            "cannot be derived from the declaration. Add `size_elements`, or declare "
            "`floor_exempt` with the reason the floor cannot be applied."
        )
    if elements >= floor.elements:
        return
    suffix = " (provisional floor)" if floor.provisional else ""
    raise RegimeError(
        f"{suite_id}: case {case_id!r} declares `throughput` at {_fmt(elements)} "
        f"elements, below the {family} floor of {_fmt(floor.elements)} elements{suffix}. "
        "At this size the measurement is per-call framework overhead: declare the case "
        f"`latency` (see docs/regime-floors.md)."
    )


def suite_cases(
    suite: dict[str, Any], project_dir: Path
) -> list[tuple[str, str | None, int | None, str | None, str | None]]:
    """(case, regime, elements, family, floor_exempt) for every declared case."""
    if not isinstance(suite, dict) or not suite.get("suite_id"):
        return []
    suite_id = str(suite["suite_id"])
    defaults, cases = declared_cases(suite, project_dir)
    rows = []
    for problem in cases:
        case_id = problem.get("id")
        if not isinstance(case_id, str) or not case_id:
            continue
        regime, _measure = resolve(suite_id, defaults, problem)
        exempt = problem.get("floor_exempt")
        if not isinstance(exempt, str) or not exempt.strip():
            exempt = None
        rows.append(
            (
                case_id,
                regime,
                case_elements(problem),
                family_for(suite_id, problem),
                exempt,
            )
        )
    return rows


def check_suite(
    suite: dict[str, Any], project_dir: Path, skips: list[str] | None = None
) -> list[str]:
    """Validate one suite document; return the errors instead of raising."""
    errors: list[str] = []
    if not isinstance(suite, dict):
        return errors
    suite_id = str(suite.get("suite_id") or "")
    try:
        _defaults, cases = declared_cases(suite, Path(project_dir), skips)
    except RegimeError as exc:
        return [str(exc)]
    defaults = suite.get("defaults") or {}
    for problem in cases:
        case_id = str(problem.get("id") or "")
        regime, measure = resolve(suite_id, defaults, problem)
        if measure is not None and measure not in MEASURES:
            errors.append(
                f"{suite_id}: case {case_id!r} declares unknown measure {measure!r} "
                f"(expected one of {', '.join(MEASURES)})"
            )
        exempt = problem.get("floor_exempt")
        try:
            check_case(
                suite_id,
                case_id,
                regime,
                case_elements(problem),
                family_for(suite_id, problem),
                exempt if isinstance(exempt, str) and exempt.strip() else None,
            )
        except RegimeError as exc:
            errors.append(str(exc))
    return errors


def floor_table() -> list[dict[str, Any]]:
    """The floor table as data (used by the docs test and the banner tests)."""
    return [
        {
            "family": floor.family,
            "elements": floor.elements,
            "provisional": floor.provisional,
            "basis": floor.basis,
        }
        for floor in sorted(FAMILY_FLOORS.values(), key=lambda item: item.family)
    ]


def describe_sizes(sizes: Iterable[int | None]) -> str:
    """Human summary of a page's declared working sets."""
    # `sizes` is routinely a generator (`case.elements for case in cases`), so it
    # has to be materialised before both passes read it.
    values = list(sizes)
    known = sorted(size for size in values if isinstance(size, int))
    missing = sum(1 for size in values if not isinstance(size, int))
    if not known:
        return "none of them declares a working set"
    if known[0] == known[-1]:
        text = f"each {_fmt(known[0])} elements"
    else:
        text = f"from {_fmt(known[0])} to {_fmt(known[-1])} elements"
    if missing:
        text += f" ({missing} declare no size)"
    return text
