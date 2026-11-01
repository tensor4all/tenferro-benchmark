"""Render measured coverage without treating omissions as successful execution."""
import argparse
import json
from pathlib import Path


def render(status, source=None):
    lines = ["# GPU routine coverage", "",
             f"Manifest version: {status['manifest_version']}; "
             f"effort: {status['effort']}.", "",
             "Measured coverage complete: " + ("yes" if status['complete'] else "no") + ".", "",
             "Verification-only, skipped, unsupported, missing and duplicate rows are not measured coverage. "
             "Noise flags are descriptive screens, not regression conclusions.", "",
             "| Classification | Count |", "|---|---:|"]
    if source is not None:
        lines[2:2] = [f"Source accounting: `{source}`.", ""]
    for category, count in status['counts'].items():
        lines.append(f"| {category} | {count} |")
    for category in ("expected", "selected", "executed", "skipped", "unsupported", "failed", "missing",
                     "verification_only", "raw_samples_unavailable", "noisy",
                     "duplicates", "unexpected"):
        lines.extend(["", f"## {category}", ""])
        for key in status.get(category, []):
            reason = status.get("reasons", {}).get(key)
            lines.append(f"- `{key}`" + (f": {reason}" if reason else ""))
        if not status.get(category):
            lines.append("None.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(json.loads(args.status.read_text()), args.status.as_posix()))


if __name__ == "__main__":
    main()
