#!/usr/bin/env python3
"""Write the per-commit history page of the latest-generation results.

    python scripts/generate_history.py --repo . --output-dir _site \
        --profile zen5-cpu --profile nvidia-gpu

Adapted from ``campaign_toolkit/site.py`` (``history_pages`` / ``write_history``),
which the campaign repositories use for their ``history.html``. This repository's
layout differs (``result/<target_profile>/<device>/<suite>.md`` plus
``<suite>.summary.json``), so the git walk is kept and the record filter is its own.

Each entry is a first-parent commit on the checked-out branch that added, changed
or removed a maintained report of one of the listed profiles, newest first. It
lists the reports that commit changed and, from their summary sidecars at that
commit, the tenferro-rs revision measured and the collection time. Links point at
the commit, so they stay valid when ``main`` moves on.

Only the profiles passed with ``--profile`` appear: the history page, like the
landing page, never links an older hardware generation. Older results stay in
``result/`` and in git history.

It is a view over git history: it measures nothing and needs a checkout with
full history (``fetch-depth: 0``). With no matching history it writes nothing.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

CSS = """
body{font:14px/1.6 -apple-system,"Segoe UI",Roboto,sans-serif;max-width:72rem;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.45rem}h2{font-size:1.05rem;margin-top:2rem}
code{background:#f2f2f2;padding:0 .2rem}
.muted{color:#666;font-size:.9em}
@media (prefers-color-scheme: dark){body{background:#15181c;color:#e6e6e6}code{background:#262b31}.muted{color:#9aa4ad}a{color:#7ab7ff}}
"""

RESULT_DIR = "result"
RECORD_RE = re.compile(r"^result/(?P<profile>[^/]+)/(?P<device>cpu|gpu)/(?P<suite>[^/]+)\.md$")


def git(repo: pathlib.Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return out.stdout


def blob_base(repo: pathlib.Path) -> str:
    remote = git(repo, "remote", "get-url", "origin").strip().removesuffix(".git")
    for prefix in ("ssh://git@github.com/", "git@github.com:"):
        if remote.startswith(prefix):
            return "https://github.com/" + remote[len(prefix):]
    return remote or "https://github.com/tensor4all/tenferro-benchmark"


def records_at(repo: pathlib.Path, sha: str, profiles: tuple[str, ...]) -> dict[str, str]:
    """{repo-relative report path: blob id} of the latest-generation reports at a commit."""
    records: dict[str, str] = {}
    for profile in profiles:
        out = git(repo, "ls-tree", "-r", "-z", sha, "--", f"{RESULT_DIR}/{profile}/")
        for entry in out.split("\0"):
            if not entry:
                continue
            meta, path = entry.split("\t", 1)
            if RECORD_RE.match(path):
                records[path] = meta.split()[2]
    return records


def publication_commits(repo: pathlib.Path, profiles: tuple[str, ...]) -> list[tuple[str, int, str]]:
    """[(sha, epoch, subject)] newest first: first-parent commits touching a latest profile's results."""
    paths = [f"{RESULT_DIR}/{profile}/" for profile in profiles]
    out = git(repo, "log", "--first-parent", "--format=%H%x09%ct%x09%s", "--", *paths)
    commits = []
    for line in out.splitlines():
        sha, epoch, subject = line.split("\t", 2)
        commits.append((sha, int(epoch), subject))
    return commits


def summary_at(repo: pathlib.Path, sha: str, report: str) -> dict[str, Any]:
    path = report[: -len(".md")] + ".summary.json"
    text = git(repo, "show", f"{sha}:{path}")
    try:
        document = json.loads(text)
    except ValueError:
        return {}
    return document if isinstance(document, dict) else {}


def history_entries(repo: pathlib.Path, profiles: tuple[str, ...]) -> list[dict[str, Any]]:
    """Newest first; only commits that added, changed or removed a report are kept."""
    commits = publication_commits(repo, profiles)
    states = [records_at(repo, sha, profiles) for sha, _, _ in commits]
    entries = []
    for i, (sha, epoch, subject) in enumerate(commits):
        before = states[i + 1] if i + 1 < len(commits) else {}
        now = states[i]
        changed = sorted(path for path, blob in now.items() if before.get(path) != blob)
        removed = sorted(set(before) - set(now))
        if not changed and not removed:
            continue
        measured = []
        for path in changed:
            document = summary_at(repo, sha, path)
            if document.get("tenferro_commit"):
                measured.append(str(document["tenferro_commit"])[:12])
        entries.append(
            dict(sha=sha, epoch=epoch, subject=subject, changed=changed, removed=removed,
                 carried=sorted(set(now) - set(changed)),
                 tenferro=sorted(set(measured)))
        )
    return entries


def record_items(base: str, sha: str, paths: list[str]) -> str:
    items = []
    for path in paths:
        match = RECORD_RE.match(path)
        label = f"{match['profile']} {match['device']}/{match['suite']}" if match else path
        items.append(f'<li><a href="{html.escape(base)}/blob/{sha}/{html.escape(path)}">{html.escape(label)}</a></li>')
    return "<ul>" + "".join(items) + "</ul>"


def render(entries: list[dict[str, Any]], base: str, profiles: tuple[str, ...]) -> str:
    sections = []
    for entry in entries:
        sha = entry["sha"]
        when = datetime.datetime.fromtimestamp(entry["epoch"], datetime.timezone.utc)
        revs = "".join(f' <span class="muted">tenferro-rs <code>{html.escape(r)}</code></span>' for r in entry["tenferro"])
        parts = [
            f'<h2>{when:%Y-%m-%d %H:%M} UTC <a href="{html.escape(base)}/commit/{sha}"><code>{sha[:10]}</code></a></h2>',
            f"<p>{html.escape(entry['subject'])}{revs}</p>",
        ]
        if entry["changed"]:
            parts.append(f"<p><strong>New or changed ({len(entry['changed'])})</strong></p>"
                         + record_items(base, sha, entry["changed"]))
        if entry["removed"]:
            parts.append(f'<p class="muted">Removed in this commit: {len(entry["removed"])} report(s).</p>')
        if entry["carried"]:
            parts.append(f"<details><summary>Carried over ({len(entry['carried'])})</summary>"
                         + record_items(base, sha, entry["carried"]) + "</details>")
        sections.append("\n".join(parts))
    scope = ", ".join(f"<code>{html.escape(p)}</code>" for p in profiles)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>tenferro-benchmark: history</title>
<style>{CSS}</style></head><body>
<h1>History of published results</h1>
<p class="muted">Latest generation only ({scope}). Each entry is a commit on the main branch that
added or changed a report, newest first, with the reports present at that commit. Links are
permanent: they point at the commit. Older hardware generations stay in the repository and its git
history but are not listed here. <a href="index.html">Back to the current results</a>.</p>
{chr(10).join(sections)}
</body></html>
"""


def write_history(repo: pathlib.Path, output_dir: pathlib.Path, profiles: tuple[str, ...]) -> pathlib.Path | None:
    """Write ``<output_dir>/history.html``; return its path, or None with no matching history."""
    entries = history_entries(repo, profiles)
    if not entries:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "history.html"
    out.write_text(render(entries, blob_base(repo), profiles), encoding="utf-8")
    print(f"wrote {out} ({len(entries)} publication commits)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--profile", action="append", required=True, dest="profiles")
    args = parser.parse_args()
    write_history(args.repo, args.output_dir, tuple(args.profiles))
    return 0


if __name__ == "__main__":
    sys.exit(main())
