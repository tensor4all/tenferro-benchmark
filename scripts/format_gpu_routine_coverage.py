"""Render measured coverage without treating omissions as successful execution."""
import argparse
import json
from pathlib import Path


def render(status):
    lines = ["# GPU routine coverage", "",
             f"Manifest version: {status['manifest_version']}; coverage: {status['coverage']}; "
             f"effort: {status['effort']}.", "",
             "Measured coverage complete: " + ("yes" if status['complete'] else "no") + ".", "",
             "Verification-only, unsupported, missing and unselected rows are not measured coverage. "
             "Noise flags are descriptive screens, not regression conclusions.", "",
             "| Classification | Count |", "|---|---:|"]
    for category, count in status['counts'].items():
        lines.append(f"| {category} | {count} |")
    for category in ("expected", "selected", "executed", "unsupported", "failed", "missing",
                     "verification_only", "raw_samples_unavailable", "noisy", "unselected",
                     "duplicates", "unexpected"):
        lines.extend(["", f"## {category}", ""])
        lines.extend([f"- `{key}`" for key in status[category]] or ["None."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(json.loads(args.status.read_text())))


if __name__ == "__main__":
    main()
