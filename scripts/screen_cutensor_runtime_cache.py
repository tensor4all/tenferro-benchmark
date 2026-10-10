#!/usr/bin/env python3
"""Compare two committed CI installers using one local vendor archive.

This measures stored bytes and content preservation, not execution time.
Network and privilege commands are replaced; extraction and validation execute
the actual scripts. Run sequentially with a fresh output directory.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            result[str(path.relative_to(root))] = {"link": os.readlink(path)}
        elif path.is_file():
            result[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size, "sha256": digest(path),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    binaries = output / "bin"
    binaries.mkdir()
    for name, body in {
        "curl": 'while [ "$1" != -o ]; do shift; done\ncp "$VENDOR_ARCHIVE" "$2"',
        "id": "echo 0",
    }.items():
        path = binaries / name
        path.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
        path.chmod(0o755)
    records = {}
    for arm, revision in [("baseline", protocol["baseline"]), ("candidate", args.candidate)]:
        source = subprocess.check_output([
            "git", "-C", str(args.repository), "show", f"{revision}:scripts/ci/install_cutensor.sh",
        ])
        script = output / f"{arm}.sh"
        script.write_bytes(source)
        root = output / arm
        with (output / f"{arm}.log").open("w") as log:
            subprocess.run(["bash", str(script), "2.6.0.4", str(root)], check=True,
                           env=dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}",
                                    VENDOR_ARCHIVE=str(args.archive.resolve())),
                           stdout=log, stderr=subprocess.STDOUT)
        packed = output / f"{arm}.tar.zst"
        subprocess.run(["tar", "--zstd", "-cf", str(packed), "-C", str(root), "."], check=True)
        entries = inventory(root)
        records[arm] = {"revision": revision, "files": entries,
                        "uncompressed_bytes": sum(v.get("bytes", 0) for v in entries.values()),
                        "compressed_bytes": packed.stat().st_size}
    expected = {k: v for k, v in records["baseline"]["files"].items()
                if k.startswith("lib/libcutensor.so")}
    content_equal = records["candidate"]["files"] == expected
    reductions = {metric: 1 - records["candidate"][metric] / records["baseline"][metric]
                  for metric in ("uncompressed_bytes", "compressed_bytes")}
    passed = content_equal and all(
        reduction >= protocol["acceptance"][metric + "_reduction_min"]
        for metric, reduction in reductions.items())
    result = {"protocol": protocol, "archive_sha256": digest(args.archive), "arms": records,
              "content_equal": content_equal, "reductions": reductions, "passed": passed,
              "zstd": subprocess.check_output(["zstd", "--version"], text=True).strip()}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"reductions": reductions, "content_equal": content_equal, "passed": passed}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
