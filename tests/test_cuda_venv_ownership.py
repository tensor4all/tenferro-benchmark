"""Runtime regression for #234; requires Linux, Docker, and a non-root caller."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def main():
    if os.getuid() == 0:
        raise SystemExit("Run as a non-root host user to test host write permissions")
    config = json.loads((ROOT / ".devcontainer/cuda/devcontainer.json").read_text())
    image = os.environ.get("CUDA_VENV_TEST_IMAGE", "alpine:3.21")
    docker("image", "inspect", image)
    for existing in (False, True):
        # Spaces also exercise the initializeCommand argument-array semantics.
        with tempfile.TemporaryDirectory(prefix="cuda host workspace ") as tmp:
            workspace = Path(tmp)
            venv = workspace / ".venv"
            if existing:
                venv.mkdir(mode=0o750)
                (venv / "host-python").write_text("preserve host environment")
            before = venv.stat() if existing else None
            command = [arg.replace("${localWorkspaceFolder}", tmp)
                       for arg in config["initializeCommand"]]
            subprocess.run(command, check=True)
            subprocess.run(command, check=True)  # lifecycle may run it again
            volume = "cuda-venv-test-" + uuid.uuid4().hex
            container = volume + "-container"
            docker("volume", "create", volume)
            try:
                # Use the production mount declaration, substituting a unique volume.
                mount = config["mounts"][0].replace(
                    "tenferro-benchmark-cuda-venv-${devcontainerId}", volume
                ).replace("${containerWorkspaceFolder}", "/work")
                docker("run", "--name", container,
                       "--mount", f"type=bind,source={tmp},target=/work",
                       "--mount", mount, image, "sh", "-ec",
                       'chown -R "$1:$2" /work/.venv; '
                       'test ! -e /work/.venv/host-python; '
                       'echo container-only > /work/.venv/container-python',
                       "sh", str(os.getuid()), str(os.getgid()))
                for phase in ("created", "removed"):
                    if phase == "removed":
                        docker("rm", container)
                    # Access the underlying bind source directly, never the overlay.
                    stat = venv.stat()
                    assert (stat.st_uid, stat.st_gid) == (os.getuid(), os.getgid())
                    if before:
                        assert (stat.st_ino, stat.st_mode) == (before.st_ino, before.st_mode)
                        assert (venv / "host-python").read_text() == "preserve host environment"
                    assert not (venv / "container-python").exists()
                    (venv / "host-write").write_text(phase)
                    assert (venv / "host-write").read_text() == phase
                # The separate environment survives container removal in its volume.
                assert docker("run", "--rm", "--mount",
                              f"type=volume,source={volume},target=/venv", image,
                              "cat", "/venv/container-python") == "container-only"
                print(f"PASS: {'existing' if existing else 'fresh'} host .venv")
            finally:
                subprocess.run(["docker", "rm", "-f", container],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                docker("volume", "rm", volume)


if __name__ == "__main__":
    main()
