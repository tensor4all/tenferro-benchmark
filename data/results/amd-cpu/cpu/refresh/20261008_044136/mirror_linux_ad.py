from pathlib import Path
import shutil
for timestamp in ("20261008_044146", "20261008_053424"):
    source = Path("data/results/amd-cpu/cpu/einsum") / timestamp
    target = Path("data/results/linux-cpu/cpu/einsum") / timestamp
    shutil.copytree(source, target, dirs_exist_ok=True)
report = Path("result/amd-cpu/cpu/linalg_jvp_vjp.md").read_text()
report = report.replace("Target profile: `amd-cpu`", "Target profile: `linux-cpu`")
report = report.replace("data/results/amd-cpu/", "data/results/linux-cpu/")
report = report.replace("# CPU Linalg JVP/VJP Benchmark Results", "# Linux CPU Linalg JVP/VJP Benchmark Results", 1)
report += "\nThis Linux report alias mirrors the same sequential amd-cpu collection; no additional timing was performed.\n"
Path("result/linux-cpu/cpu/linalg_jvp_jvp.md").write_text(report)
