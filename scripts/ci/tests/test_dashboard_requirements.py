"""The GPU packaging step runs where Python packages cannot be fetched.

Two failures share this file. First, the pod restores a runtime with
`UV_OFFLINE=1` and ships no uv git cache, so a step that resolves
`scripts/dashboard-requirements.txt` there cannot clone the campaign-toolkit git
pin at all; run 41560144226 lost a whole weekly collection that way (the suite
passed, packaging failed, the pod was deleted). The packaging step therefore
runs the restored runtime interpreter. Second, this file and `uv.lock` and
`pyproject.toml` must name the same campaign-toolkit revision, or the hosted
workflows that do have network resolve a different toolkit from the one the
lock was tested against.
"""
from pathlib import Path
import re
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "uv.lock"
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "scripts" / "dashboard-requirements.txt"

NAME = "campaign-toolkit"
LOCK_ENTRY = re.compile(
    r'name = "' + re.escape(NAME) + r'"'
    r'.*?\nversion = "[^"]*"\n'
    r'source = \{ git = "[^"]*?rev=([0-9a-f]{40})',
    re.S,
)
REQUIREMENTS_ENTRY = re.compile(
    re.escape(NAME)
    + r" @ git\+https://github\.com/tensor4all/"
    + re.escape(NAME)
    + r"@([0-9a-f]{40})"
)
PYPROJECT_ENTRY = re.compile(
    re.escape(NAME) + r' = \{ git = "[^"]*", rev = "([0-9a-f]{40})" \}'
)


class DashboardRequirementsPinsTest(unittest.TestCase):
    def test_campaign_toolkit_revision_matches_uv_lock_and_pyproject(self):
        lock = LOCK_ENTRY.search(LOCK.read_text())
        self.assertIsNotNone(lock, "campaign-toolkit has no git source in uv.lock")
        pyproject = PYPROJECT_ENTRY.search(PYPROJECT.read_text())
        self.assertIsNotNone(pyproject, "pyproject.toml does not pin campaign-toolkit")
        requirements = REQUIREMENTS_ENTRY.search(REQUIREMENTS.read_text())
        self.assertIsNotNone(
            requirements, "dashboard-requirements.txt does not pin campaign-toolkit"
        )
        with self.subTest(source="pyproject.toml"):
            self.assertEqual(pyproject.group(1), lock.group(1))
        with self.subTest(source="scripts/dashboard-requirements.txt"):
            self.assertEqual(
                requirements.group(1),
                lock.group(1),
                "dashboard-requirements.txt names a campaign-toolkit revision that "
                "the prepared pod runtime does not have cached; packaging runs "
                "offline and will fail",
            )



class GpuPackagingRunsOfflineTest(unittest.TestCase):
    """The packaging step must not resolve requirements on the offline pod."""

    def test_gpu_packaging_step_does_not_invoke_uv_with_requirements(self):
        workflow = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "benchmark-runpod-gpu.yml").read_text()
        )
        steps = workflow["jobs"]["run-gpu-benchmark"]["steps"]
        package = next(
            step for step in steps if step.get("name") == "Package this collection's GPU results"
        )
        commands = [
            line.strip()
            for line in package["run"].splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        with self.subTest(step=package["name"]):
            self.assertFalse([cmd for cmd in commands if cmd.startswith("uv ")])
            self.assertIn("python scripts/ci/prepare_result_bundle.py", package["run"])

if __name__ == "__main__":
    unittest.main()
