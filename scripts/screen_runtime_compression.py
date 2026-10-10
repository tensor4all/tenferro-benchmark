#!/usr/bin/env python3
"""Compare zstd levels on complete immutable hosted runtime tar streams."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo):
        raise SystemExit("Commit the screen and protocol before measurement")
    protocol = json.loads((repo / "result/nvidia-gpu/ci/runtime-compression-protocol.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, protocol["cpu_affinity"])
    result = {
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "protocol": protocol,
        "zstd": subprocess.check_output(["zstd", "--version"], text=True).strip(),
        "host": dict(zip(["sysname", "nodename", "release", "version", "machine"], os.uname())),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cases": [], "commands": [], "status": "running",
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def run(command):
        row = {"command": command, "load_start": os.getloadavg()}
        result["commands"].append(row)
        save()
        started = time.monotonic()
        try:
            subprocess.run(command, check=True, timeout=protocol["timeout_seconds_per_command"])
        finally:
            row["diagnostic_seconds"] = time.monotonic() - started
            save()

    save()
    try:
        for case in protocol["cases"]:
            prefix = f"gpu-runtime-{protocol['artifact_run']}-{case}-part"
            file_prefix = "runtime" if case == "common" else "sdk"
            path = args.output / case
            path.mkdir()
            original = path / "original.tar.zst"
            expected = (args.downloads / f"{prefix}00" / f"{file_prefix}.sha256").read_text().split()[0]
            with original.open("wb") as dest:
                for index in range(5):
                    part = args.downloads / f"{prefix}{index:02}" / f"{file_prefix}.part{index:02}"
                    if part.stat().st_size == 0:
                        raise ValueError(f"Empty part: {part}")
                    with part.open("rb") as src:
                        shutil.copyfileobj(src, dest)
            if digest(original) != expected:
                raise ValueError(f"Hosted archive checksum mismatch: {case}")
            tar = path / "original.tar"
            run(["zstd", "-d", str(original), "-o", str(tar)])
            tar_digest = digest(tar)
            row = {"case": case, "original_bytes": original.stat().st_size,
                   "original_sha256": expected, "tar_sha256": tar_digest, "levels": {}}
            result["cases"].append(row)
            for level in [protocol["baseline_compression_level"], protocol["candidate_compression_level"]]:
                compressed = path / f"level-{level}.tar.zst"
                decoded = path / f"level-{level}.tar"
                run(["zstd", f"-T{protocol['compression_threads']}", f"-{level}", str(tar), "-o", str(compressed)])
                run(["zstd", "-d", str(compressed), "-o", str(decoded)])
                if digest(decoded) != tar_digest:
                    raise ValueError(f"Uncompressed tar changed: {case}, level {level}")
                row["levels"][str(level)] = {"bytes": compressed.stat().st_size,
                                               "sha256": digest(compressed), "identical_tar": True}
                decoded.unlink()
                save()
            print(json.dumps(row), flush=True)
        by_case = {row["case"]: row for row in result["cases"]}
        result["combined"] = {}
        for sdk in ["cuda12.6", "cuda12.8"]:
            sizes = {str(level): sum(by_case[c]["levels"][str(level)]["bytes"] for c in ["common", sdk])
                     for level in [protocol["baseline_compression_level"], protocol["candidate_compression_level"]]}
            reduction = 1 - sizes[str(protocol["candidate_compression_level"])] / sizes[str(protocol["baseline_compression_level"])]
            result["combined"][sdk] = {"bytes": sizes, "reduction": reduction,
                                       "screen_pass": reduction >= protocol["screen_minimum_combined_byte_reduction"]}
        result["status"] = "PASS" if all(row["screen_pass"] for row in result["combined"].values()) else "FAIL"
    except Exception as error:
        result["status"] = "INCOMPLETE"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    print(json.dumps({"status": result["status"], "combined": result["combined"]}), flush=True)


if __name__ == "__main__":
    main()
