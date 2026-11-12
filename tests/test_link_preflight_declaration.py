"""benchmarks/link-preflight.yaml has to name every library the build scripts link, or the link
preflight passes while the build still fails on the library it did not check. Nothing is linked here;
the preflight itself is tested in campaign-toolkit and run for real by the `validate` CI job."""
import pathlib
import re
import subprocess
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECLARATION = ROOT / "benchmarks/link-preflight.yaml"
# build script -> the arms that together mirror its link lines
SOURCES = {
    "build.rs": ["blas-mkl", "blas-openblas"],
    "mwe/cpu_followup/build.rs": ["followup-mwe", "followup-mwe-openblas"],
}


def linked_libraries(path):
    """Every `-l<name>`, `rustc-link-lib=dylib=<name>` and `for lib in [...]` entry of a build script."""
    text = (ROOT / path).read_text()
    names = set(re.findall(r"(?<![\w])-l([A-Za-z0-9_]+)", text))
    names |= set(re.findall(r"rustc-link-lib=dylib=([A-Za-z0-9_]+)", text))
    for array in re.findall(r"for lib in \[([^\]]*)\]", text):
        names |= set(re.findall(r'"([A-Za-z0-9_]+)"', array))
    return names


class DeclarationTest(unittest.TestCase):
    def setUp(self):
        self.arms = yaml.safe_load(DECLARATION.read_text())["arms"]

    def test_every_linked_library_is_declared(self):
        for path, arms in SOURCES.items():
            declared = {lib.lstrip(":") for arm in arms for lib in self.arms[arm]["libs"]}
            linked = linked_libraries(path)
            self.assertTrue(linked, f"found no library in {path}; the pattern no longer matches it")
            self.assertEqual(linked - declared, set(), f"{path} links libraries no arm of {DECLARATION.name} checks")

    def test_every_arm_names_its_source(self):
        for name, arm in self.arms.items():
            source = arm["source"].split()[0]
            self.assertTrue((ROOT / source).is_file(), f"arm {name}: source {source} does not exist")

    def test_run_all_checks_the_followup_mwe_when_perf_issues_builds_it(self):
        def arms(features, perf_issues, mklroot):
            env = {"PATH": "/usr/bin:/bin", **({"MKLROOT": mklroot} if mklroot else {})}
            out = subprocess.run(["bash", "-c", f"source scripts/link_preflight.sh; "
                                                f"link_preflight_arms_for_run {features} {perf_issues}"],
                                 cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
            return out.split()
        self.assertEqual(arms("native", 1, "/opt/mkl"), ["followup-mwe"])
        self.assertEqual(arms("native", 0, "/opt/mkl"), [])
        self.assertEqual(arms("native", 1, None), [])
        self.assertEqual(arms("blas-mkl", 1, "/opt/mkl"), ["blas-mkl", "followup-mwe"])
        self.assertEqual(arms("blas-openblas", 1, None), ["blas-openblas"])


if __name__ == "__main__":
    unittest.main()
