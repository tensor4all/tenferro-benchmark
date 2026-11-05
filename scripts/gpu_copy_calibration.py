#!/usr/bin/env python3
"""Measure the achievable device copy rate on the benchmark GPU.

A GPU record states a duration per operation and the logical bytes the case must move.
Dividing one by the other gives a rate, and a rate is only readable next to something
it can be compared with. This times one plain device copy of a known size and records
the traffic rate that copy reached in the run metadata under ``gpu_memory_calibration``.

The copy moves ``buffer_bytes`` in and ``buffer_bytes`` out, so the recorded rate is the
traffic of both directions divided by the measured median time.

The rate is a reference for a recorded ratio, not a limit: logical array bytes are not
device DRAM traffic, and a case whose operands stay in cache moves far less DRAM than
its copy implies. ``scripts/run_gpu_public_api.sh`` runs this once per family run,
before any arm is timed, and ``scripts/gpu_bandwidth_guard.py`` records the ratio of
every record against it.
"""

from __future__ import annotations

import argparse
import datetime
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpu_bandwidth_guard import CALIBRATION_KEY  # noqa: E402

DEFAULT_BUFFER_BYTES = 64 << 20
METHOD = ("torch.Tensor.copy_ between two contiguous device buffers, one copy "
          "per sample, device synchronized before and after the clock")
TRAFFIC_MODEL = "one read plus one write of buffer_bytes per copy"


def copy_document(sample_ns, buffer_bytes, *, device_ordinal, device_name,
                  warmups, torch_version, cuda_version, timestamp,
                  method=METHOD, traffic_model=TRAFFIC_MODEL):
    """The run metadata block for a measured copy, from the raw sample times."""
    if not sample_ns:
        raise ValueError("a copy calibration needs at least one sample")
    if buffer_bytes <= 0:
        raise ValueError("a copy calibration needs a positive buffer size")
    traffic = 2 * buffer_bytes
    median_ns = statistics.median(float(value) for value in sample_ns)
    if median_ns <= 0:
        raise ValueError("a copy calibration needs a positive median time")
    rate = traffic / (median_ns / 1e9)
    return {
        "kind": "device_copy",
        "method": method,
        "device_ordinal": int(device_ordinal),
        "device_name": device_name,
        "buffer_bytes": int(buffer_bytes),
        "traffic_bytes_per_copy": int(traffic),
        "traffic_model": traffic_model,
        "warmups": int(warmups),
        "sample_ms": [float(value) / 1e6 for value in sample_ns],
        "median_ms": median_ns / 1e6,
        "bytes_per_second": rate,
        # ``torch.__version__`` is a ``TorchVersion``, a ``str`` subclass that
        # ``yaml.safe_dump`` cannot represent; this document is written to run.yaml.
        "torch_version": str(torch_version),
        "cuda_version": cuda_version,
        "timestamp": timestamp,
    }


def measure_copy(device, buffer_bytes, warmups, samples):
    """Time one device copy per sample; returns (sample_ns, device_name, versions)."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: the copy calibration must run on the benchmark GPU")
    if buffer_bytes % 4:
        raise SystemExit("--buffer-bytes must be a multiple of 4")
    torch.cuda.set_device(device)
    source = torch.ones(buffer_bytes // 4, dtype=torch.float32, device=f"cuda:{device}")
    destination = torch.empty_like(source)

    def one_copy():
        torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        destination.copy_(source)
        torch.cuda.synchronize(device)
        return time.perf_counter_ns() - start

    for _ in range(warmups):
        one_copy()
    values = [one_copy() for _ in range(samples)]
    name = torch.cuda.get_device_name(device)
    cuda_version = torch.version.cuda
    return values, name, torch.__version__, cuda_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--device", type=int,
                        default=int(os.environ.get("GPU_BENCH_DEVICE", "0")))
    parser.add_argument("--buffer-bytes", type=int, default=DEFAULT_BUFFER_BYTES)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        parser.error("--samples must be positive and --warmups must not be negative")
    values, name, torch_version, cuda_version = measure_copy(
        args.device, args.buffer_bytes, args.warmups, args.samples)
    document = copy_document(
        values, args.buffer_bytes,
        device_ordinal=args.device, device_name=name, warmups=args.warmups,
        torch_version=torch_version, cuda_version=cuda_version,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    metadata = yaml.safe_load(args.run_metadata.read_text()) or {}
    metadata[CALIBRATION_KEY] = document
    args.run_metadata.write_text(yaml.safe_dump(metadata, sort_keys=False))
    print(f"device copy rate on {name}: {document['bytes_per_second'] / 1e9:.1f} GB/s "
          f"({document['traffic_bytes_per_copy']} bytes per copy, "
          f"median {document['median_ms']:.3f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
