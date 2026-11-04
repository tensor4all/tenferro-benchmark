#!/usr/bin/env python3
"""Logical per operation traffic of a public API case, from its declared arrays.

A GPU record reports a duration per operation but nothing about how many bytes that
operation was declared to move, so no consumer of the record can put the duration next
to a rate. This module sizes that logical traffic: one read of every declared input and
one write of every declared output, taken from the arrays the suite itself persists
(their shape and dtype).

These are logical bytes, not device DRAM traffic. A contraction whose operands stay
resident in cache moves far less DRAM than the arrays suggest, and a kernel that
materialises intermediates moves more, so the number is a property of the case
interface, not a memory model. It is therefore recorded as a diagnostic next to the
timing and never used to decide whether a measurement is believable.

An accumulating case declares its destination among its inputs, so the destination read
is already counted. A case whose declared arrays do not determine the traffic is marked
unknown here instead of guessed.
"""

from __future__ import annotations

TRAFFIC_KIND = "logical_traffic"

BASIS = (
    "one read of every declared input array plus one write of every declared "
    "output array, sized from the suite shape and dtype; logical interface bytes "
    "only, so neither cache-resident operands nor intermediate allocations are "
    "represented"
)


def _total_nbytes(values):
    total = 0
    for value in values:
        size = getattr(value, "nbytes", None)
        if not isinstance(size, int) or size < 0:
            return None
        total += size
    return total


def case_traffic(inputs, outputs, *, op=""):
    """The logical traffic block a case carries into the manifest and every record.

    ``inputs`` and ``outputs`` are the arrays the fixture step persists, so the
    size is read from the same values every arm uploads, not from a second
    dtype table that could drift from them.
    """
    reads = _total_nbytes(inputs)
    writes = _total_nbytes(outputs)
    if reads is None or writes is None:
        raise TypeError("traffic needs arrays that expose an integer nbytes")
    block = {
        "kind": TRAFFIC_KIND,
        "bytes_per_operation": reads + writes,
        "reads_bytes": reads,
        "writes_bytes": writes,
        "destination_read_included": "_accum" in op,
        "basis": BASIS,
        "unknown_reason": None,
    }
    reason = None
    if not len(inputs):
        reason = "case declares no input array, so the bytes it must read are not derivable"
    elif not len(outputs):
        reason = "case declares no output array, so the bytes it must write are not derivable"
    elif reads + writes == 0:
        reason = "declared arrays are empty, so the operation moves no known traffic"
    if reason is not None:
        block["bytes_per_operation"] = None
        block["unknown_reason"] = reason
    return block
