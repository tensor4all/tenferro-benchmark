#!/usr/bin/env python3
"""Cargo-owned provenance receipt for the Rust small-work binary."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

RECEIPT_VERSION = 2
ARTIFACT_KINDS = ("bin", "lib-test")


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _git_state(path: Path) -> dict[str, Any]:
    """Return a complete, fail-closed checkout identity."""
    state: dict[str, Any] = {
        "path": str(path.resolve()), "head": None, "dirty": None,
        "untracked": None, "status_sha256": None, "tracked_file_count": None,
        "tracked_files_sha256": None,
    }
    try:
        head_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                                     check=True, capture_output=True, text=True, timeout=10)
        head = head_result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        state["error"] = f"HEAD: {type(exc).__name__}"
        return state
    if len(head) != 40 or any(char not in "0123456789abcdefABCDEF" for char in head):
        state["error"] = "HEAD is not a full Git object id"
        return state
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                cwd=path, check=True, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        state.update({"head": head, "error": f"status: {type(exc).__name__}"})
        return state
    try:
        tracked = subprocess.run(["git", "ls-files", "-z"], cwd=path, check=True,
                                 capture_output=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        state.update({"head": head, "dirty": bool(status.strip()),
                      "untracked": any(line.startswith("??") for line in status.splitlines()),
                      "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
                      "error": f"tracked membership: {type(exc).__name__}"})
        return state
    # git -z uses a NUL byte, not the two-character string ``\\0``.
    files = [item for item in tracked.split(b"\0") if item]
    state.update({"head": head, "dirty": bool(status.strip()),
                  "untracked": any(line.startswith("??") for line in status.splitlines()),
                  "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
                  "tracked_file_count": len(files),
                  "tracked_files_sha256": hashlib.sha256(tracked).hexdigest()})
    return state


def _config_and_toolchain_hashes(
    root: Path, cargo_home: str | None
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Hash Cargo's search chain and rust-toolchain selectors, not contents."""
    root = root.resolve()
    ancestors = [root, *root.parents]
    config_paths: list[Path] = []
    toolchain_paths: list[Path] = []
    for ancestor in ancestors:
        config_paths.extend(ancestor / ".cargo" / name for name in ("config", "config.toml"))
        toolchain_paths.extend(ancestor / name for name in ("rust-toolchain", "rust-toolchain.toml"))
    cargo_home = cargo_home or os.path.join("~", ".cargo")
    if cargo_home:
        cargo_root = Path(cargo_home).expanduser()
        if not cargo_root.is_absolute():
            cargo_root = (root / cargo_root).resolve()
        config_paths.extend(cargo_root / name for name in ("config", "config.toml"))
    # Keep absent candidates as null so a newly-created config is observable.
    configs = {str(path.resolve()): _sha256(path) for path in dict.fromkeys(config_paths)}
    toolchains = {str(path.resolve()): _sha256(path) for path in dict.fromkeys(toolchain_paths)}
    return configs, toolchains


def _file_hashes(root: Path, *, config_root: Path | None = None) -> dict[str, Any]:
    names = ["Cargo.toml", "Cargo.lock"]
    files = {str((root / name).resolve()): _sha256(root / name) for name in names}
    # Cargo resolves configuration from the build cwd, not from each path
    # dependency.  Hash that one search chain plus the actual CARGO_HOME.
    search_root = config_root or root
    configs, toolchains = _config_and_toolchain_hashes(search_root, os.environ.get("CARGO_HOME"))
    files["configs"] = configs
    files["toolchain"] = toolchains
    files["unreadable"] = sorted(
        path for path, digest in (*configs.items(), *toolchains.items())
        if digest is None and Path(path).exists()
    )
    return files


def _metadata_summary(metadata: Mapping[str, Any], library: Path,
                      benchmark: Path | None = None) -> dict[str, Any]:
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise ValueError("cargo metadata has no packages")
    library = library.resolve()
    benchmark_manifest = (benchmark.resolve() / "Cargo.toml") if benchmark is not None else None
    resolve = metadata.get("resolve")
    resolve = resolve if isinstance(resolve, Mapping) else {}
    root_id = resolve.get("root")
    if root_id is not None and (not isinstance(root_id, str) or not root_id):
        raise ValueError("cargo metadata has an invalid resolved root package")
    nodes = resolve.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("cargo metadata has no resolved dependency graph")
    package_ids = {
        package.get("id") for package in packages
        if isinstance(package, Mapping) and isinstance(package.get("id"), str)
    }
    if len(package_ids) != len(packages):
        raise ValueError("cargo metadata contains duplicate or unknown package identities")
    node_features: dict[str, list[str]] = {}
    resolved_nodes: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), str):
            raise ValueError("cargo resolve node identity is unknown")
        node_id = node["id"]
        if node_id not in package_ids:
            raise ValueError("cargo resolve references an unknown package")
        features = node.get("features")
        if not isinstance(features, list) or any(not isinstance(v, str) for v in features):
            raise ValueError("cargo resolve node has unknown features")
        dependencies = node.get("dependencies", [])
        deps = node.get("deps", [])
        if not isinstance(dependencies, list) or any(not isinstance(v, str) or v not in package_ids for v in dependencies):
            raise ValueError("cargo resolve dependency edges are unknown")
        if not isinstance(deps, list):
            raise ValueError("cargo resolve dependency edges are unknown")
        for dep in deps:
            if not isinstance(dep, Mapping) or not isinstance(dep.get("pkg"), str) or dep["pkg"] not in package_ids:
                raise ValueError("cargo resolve dependency package identity is unknown")
        normalized = sorted(set(features))
        if node_id in node_features:
            raise ValueError("cargo resolve contains duplicate package nodes")
        node_features[node_id] = normalized
        resolved_nodes.append({"id": node_id, "features": normalized,
                               "dependencies": sorted(dependencies),
                               "deps": sorted(deps, key=lambda dep: json.dumps(dep, sort_keys=True))})
    if set(node_features) != package_ids:
        raise ValueError("cargo resolve does not describe every package")
    resolved: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, Mapping):
            raise ValueError("cargo metadata contains an invalid package")
        manifest = package.get("manifest_path")
        name = package.get("name")
        package_id = package.get("id")
        if not isinstance(manifest, str) or not isinstance(name, str) or not isinstance(package_id, str):
            raise ValueError("cargo metadata package identity is unknown")
        if package_id not in node_features:
            raise ValueError(f"package is absent from Cargo resolve: {name}")
        manifest_path = Path(manifest).resolve()
        if not manifest_path.is_file():
            raise ValueError(f"cargo package manifest is unavailable: {manifest}")
        inside_library = False
        try:
            manifest_path.relative_to(library)
            inside_library = True
        except ValueError:
            pass
        is_root = (root_id is not None and package_id == root_id and
                   benchmark_manifest is not None and manifest_path == benchmark_manifest)
        is_tenferro = name.startswith("tenferro")
        if is_tenferro and not is_root and not inside_library:
            raise ValueError(f"tenferro dependency resolves outside claimed checkout: {manifest}")
        # A path package is mutable even when it has a non-tenferro name.  Its
        # checkout identity is therefore part of the receipt and must be known
        # and clean; registry packages need no mutable source assertion.
        path_package = package.get("source") is None or package_id.startswith("path+")
        source_identity = None
        if path_package and not is_root:
            # Every mutable path package is provenance-bearing, including a
            # non-tenferro dependency and workspace subcrate.  Run git from
            # the manifest directory so nested workspace packages resolve to
            # their actual repository root.
            source_identity = _git_state(manifest_path.parent)
            if (source_identity.get("error") or not source_identity.get("head") or
                    source_identity.get("dirty") is not False or
                    source_identity.get("untracked") is not False):
                raise ValueError(f"mutable path dependency identity is unknown or dirty: {manifest}")
        raw_targets = package.get("targets", [])
        targets: list[dict[str, Any]] = []
        if not isinstance(raw_targets, list):
            raise ValueError("cargo package targets are unknown")
        for target in raw_targets:
            if not isinstance(target, Mapping) or not isinstance(target.get("name"), str):
                raise ValueError("cargo package target identity is unknown")
            kind = target.get("kind")
            src_path = target.get("src_path")
            if not isinstance(kind, list) or any(not isinstance(value, str) for value in kind):
                raise ValueError("cargo package target kind is unknown")
            if not isinstance(src_path, str):
                raise ValueError("cargo package target source is unknown")
            targets.append({"name": target["name"], "kind": kind,
                            "src_path": str(Path(src_path).resolve())})
        resolved.append({"id": package_id, "name": name,
                         "manifest_path": str(manifest_path),
                         "source": package.get("source"),
                         "features": node_features[package_id], "root": is_root,
                         "source_identity": source_identity, "targets": targets})
    if (benchmark_manifest is not None and
            any(Path(item["manifest_path"]) == benchmark_manifest for item in resolved) and
            not any(item["root"] for item in resolved)):
        raise ValueError("cargo metadata root package is not exact benchmark manifest")
    resolved.sort(key=lambda item: (item["name"], item["manifest_path"]))
    members = metadata.get("workspace_members", [])
    if not isinstance(members, list) or any(not isinstance(v, str) for v in members):
        raise ValueError("cargo metadata workspace members are unknown")
    resolved_nodes.sort(key=lambda item: item["id"])
    encoded = json.dumps({"packages": resolved, "nodes": resolved_nodes,
                          "workspace_members": members},
                         sort_keys=True, separators=(",", ":")).encode()
    return {"packages": resolved, "nodes": resolved_nodes,
            "workspace_members": members, "root_id": root_id,
            "sha256": hashlib.sha256(encoded).hexdigest()}


def _cargo_identity(cargo: str, cwd: Path | None = None, *, argv: Sequence[str] | None = None,
                    env: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """Resolve Cargo and the rustc Cargo will select for this build."""
    cwd = cwd or Path.cwd()
    process_env = dict(os.environ if env is None else env)
    result: dict[str, str | None] = {"cargo": None, "rustc": None}
    try:
        result["cargo"] = subprocess.run([cargo, "--version", "--verbose"], cwd=cwd,
                                          env=process_env, check=True, capture_output=True,
                                          text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    rustc = process_env.get("RUSTC") or "rustc"
    rustc_command = [rustc]
    if not process_env.get("RUSTC"):
        selected = process_env.get("RUSTUP_TOOLCHAIN")
        if argv:
            selected = next((token[1:] for token in argv if token.startswith("+") and len(token) > 1), selected)
        if selected:
            process_env["RUSTUP_TOOLCHAIN"] = selected
        resolved = shutil.which(rustc)
        if resolved:
            rustc_command = [resolved]
    try:
        result["rustc"] = subprocess.run(rustc_command + ["--version", "--verbose"], cwd=cwd,
                                          env=process_env, check=True, capture_output=True,
                                          text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _metadata_argv(cargo: str, features: Sequence[str], no_default_features: bool,
                   target: str | None) -> list[str]:
    argv = [cargo, "metadata", "--format-version", "1", "--locked"]
    if no_default_features:
        argv.append("--no-default-features")
    if features:
        argv.extend(["--features", ",".join(sorted(set(features)))])
    if target:
        argv.extend(["--filter-platform", target])
    return argv


def collect_identity(benchmark: Path, library: Path, *, binary: Path | None = None,
                     metadata: Mapping[str, Any] | None = None,
                     cargo: str = "cargo", argv: Sequence[str] | None = None,
                     profile: str | None = None, features: Sequence[str] = (),
                     no_default_features: bool = True, target: str | None = None,
                     build_project: Path | None = None, package: str | None = None,
                     target_name: str | None = None, artifact_kind: str = "bin") -> dict[str, Any]:
    """Collect every input identity used by preparation and promotion gates."""
    benchmark, library = benchmark.resolve(), library.resolve()
    build_project = (build_project or benchmark).resolve()
    selectors = {"build_project": str(build_project), "package": package,
                 "target_name": target_name, "artifact_kind": artifact_kind}
    errors: list[str] = []
    if artifact_kind not in ARTIFACT_KINDS:
        errors.append("artifact kind is unknown")
    benchmark_git, library_git = _git_state(benchmark), _git_state(library)
    for label, state in (("benchmark", benchmark_git), ("library", library_git)):
        if state.get("error") or not state.get("head"):
            errors.append(f"{label} git identity is unknown")
        if state.get("dirty") is not False or state.get("untracked") is not False:
            errors.append(f"{label} checkout cleanliness is unknown or dirty")
    process_env = dict(os.environ)
    metadata_command = _metadata_argv(cargo, features, no_default_features, target)
    try:
        metadata_value = metadata
        if metadata_value is None:
            metadata_value = json.loads(subprocess.run(
                metadata_command, cwd=build_project, env=process_env, check=True, capture_output=True,
                text=True, timeout=60).stdout)
        dependencies = _metadata_summary(metadata_value, library,
                                         benchmark if build_project == benchmark else None)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        dependencies = {"packages": [], "sha256": None}
        errors.append(f"cargo dependency identity unavailable: {type(exc).__name__}")
    files = {"benchmark": _file_hashes(benchmark, config_root=build_project),
             "library": _file_hashes(library, config_root=build_project),
             "build_project": _file_hashes(build_project, config_root=build_project)}
    for label, file_hashes in files.items():
        if file_hashes.get("unreadable"):
            errors.append(f"{label} Cargo configuration identity is unreadable")
    binary_info: dict[str, Any] | None = None
    if binary is not None:
        binary = binary.resolve()
        try:
            size = binary.stat().st_size
        except OSError:
            size = None
        binary_info = {"path": str(binary), "sha256": _sha256(binary), "size": size}
        if binary_info["sha256"] is None:
            errors.append("benchmark binary identity is unavailable")
    allowed = {"CARGO_HOME", "CARGO_BUILD_JOBS", "CARGO_INCREMENTAL", "RUSTFLAGS",
               "RUSTC_WRAPPER", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
               "OMP_DYNAMIC", "TENFERRO_CPU_BACKEND_KIND", "TENFERRO_OPT_DOT_DECOMPOSER",
               "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "VECLIB_NUM_THREADS", "BENCHMARK_TARGET_PROFILE", "CARGO_TARGET_DIR",
               "RUSTC", "RUSTUP_TOOLCHAIN", "CARGO_PROFILE_RELEASE_LTO", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS"}
    toolchain = _cargo_identity(cargo, build_project, argv=argv, env=process_env)
    if toolchain.get("cargo") is None or toolchain.get("rustc") is None:
        errors.append("Cargo/rustc identity is unknown")
    override_keys = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS",
                     "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
                     "CARGO_PROFILE_RELEASE_LTO", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS")
    # Keep override values closed; the hash is sufficient to detect changes.
    environment = {key: process_env[key] for key in sorted(process_env)
                   if key in allowed and key not in override_keys}
    overrides: dict[str, Any] = {}
    for key in override_keys:
        if key in process_env:
            value = process_env[key]
            overrides[key] = {"present": True, "sha256": hashlib.sha256(value.encode()).hexdigest(),
                              "length": len(value)}
    unknown = sorted(
        key for key in process_env
        if key.startswith(("CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_TARGET_"))
        and key not in allowed
    )
    if unknown:
        errors.append("unrecognized Cargo build overrides: " + ",".join(unknown))
    if process_env.get("RUSTFLAGS"):
        errors.append("RUSTFLAGS override is unsupported")
    if process_env.get("CARGO_ENCODED_RUSTFLAGS"):
        errors.append("encoded Cargo rustflags override is unsupported")
    if process_env.get("CARGO_BUILD_RUSTFLAGS"):
        errors.append("Cargo build rustflags override is unsupported")
    for key in ("CARGO_PROFILE_RELEASE_LTO", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS"):
        if process_env.get(key):
            errors.append(f"{key} override is unsupported")
    if process_env.get("RUSTC_WRAPPER"):
        errors.append("rustc wrapper override is unsupported")
    if process_env.get("RUSTC_WORKSPACE_WRAPPER"):
        errors.append("workspace rustc wrapper override is unsupported")
    return {
        "benchmark": benchmark_git, "library": library_git,
        "files": files,
        "dependencies": dependencies, "toolchain": toolchain,
        "environment": environment, "overrides": overrides, "unknown_environment_keys": unknown,
        "metadata_argv": metadata_command, "binary": binary_info, "argv": list(argv) if argv is not None else None,
        "profile": profile, "features": sorted(set(features)),
        "no_default_features": no_default_features, "target": target,
        "selectors": selectors, "build_project": str(build_project),
        "package": package, "target_name": target_name, "artifact_kind": artifact_kind,
        "errors": errors,
    }


def _input_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    states = {}
    for label in ("benchmark", "library"):
        state = identity.get(label, {})
        states[label] = {key: state.get(key) for key in
                         ("path", "head", "dirty", "untracked", "status_sha256",
                          "tracked_file_count", "tracked_files_sha256", "error")}
    return {"states": states, "files": identity.get("files"),
            "dependencies": identity.get("dependencies"), "toolchain": identity.get("toolchain"),
            "environment": identity.get("environment"),
            "unknown_environment_keys": identity.get("unknown_environment_keys"),
            "overrides": identity.get("overrides"),
            "metadata_argv": identity.get("metadata_argv"),
            "profile": identity.get("profile"), "features": identity.get("features"),
            "no_default_features": identity.get("no_default_features"),
            "target": identity.get("target"), "selectors": identity.get("selectors"),
            "build_project": identity.get("build_project"), "package": identity.get("package"),
            "target_name": identity.get("target_name"), "artifact_kind": identity.get("artifact_kind")}


def _profile_known(profile: Any) -> bool:
    if not isinstance(profile, Mapping) or "debug" in profile:
        return False
    required = ("opt_level", "debuginfo", "debug_assertions", "overflow_checks", "test")
    if set(profile) != set(required):
        return False
    return (isinstance(profile["opt_level"], (str, int)) and not isinstance(profile["opt_level"], bool)
            and isinstance(profile["debuginfo"], int) and not isinstance(profile["debuginfo"], bool)
            and all(isinstance(profile[key], bool) for key in required[2:]))


def _profile_is_debug(profile: Any) -> bool:
    # Cargo's test=true marks a harness, not an unoptimized build.
    return (_profile_known(profile) and
            (profile.get("debug_assertions") is True or
             profile.get("opt_level") in ("0", 0) or profile.get("debuginfo", 0) != 0))


def _profile_matches_requested(profile: Any, requested: Any) -> bool:
    if requested == "release":
        return (_profile_known(profile) and profile.get("debug_assertions") is False and
                profile.get("opt_level") not in ("0", 0) and profile.get("debuginfo") == 0)
    if requested == "debug":
        return _profile_known(profile) and _profile_is_debug(profile)
    return False


def _events(stdout: str) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            events.append(value)
    return events


def _run_cargo(argv: Sequence[str], cwd: Path, timeout: float = 300.0) -> dict[str, Any]:
    """Run Cargo in its own group and retain output even when it times out."""
    from small_work import _terminate_process_group
    process = subprocess.Popen(list(argv), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=(os.name == "posix"))
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return {"returncode": process.returncode, "stdout": stdout or "", "stderr": stderr or "",
                "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        partial_out, partial_err = exc.stdout or "", exc.stderr or ""
        _terminate_process_group(process)
        out, err = process.communicate()
        def text(value: Any) -> str:
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        return {"returncode": process.returncode, "stdout": text(out or partial_out),
                "stderr": text(err or partial_err), "timed_out": True}


def _previous_receipt_matches(previous: Mapping[str, Any] | None, before: Mapping[str, Any],
                              binary: Path, profile: str, features: Sequence[str],
                              no_default_features: bool, target: str | None,
                              build_project: Path, package: str | None,
                              target_name: str, artifact_kind: str) -> bool:
    if (not isinstance(previous, Mapping) or previous.get("status") != "verified" or
            previous.get("errors") != []):
        return False
    old_after = previous.get("after")
    old_build = previous.get("build")
    if (not isinstance(old_after, Mapping) or not isinstance(old_build, Mapping) or
            old_after.get("errors") != []):
        return False
    if _input_identity(old_after) != _input_identity(before):
        return False
    old_binary = old_after.get("binary")
    if not isinstance(old_binary, Mapping) or old_binary.get("sha256") != _sha256(binary):
        return False
    artifact = old_build.get("artifact")
    return (old_build.get("finished") is True and old_build.get("returncode") == 0 and
            old_build.get("requested_profile") == profile and
            sorted(old_build.get("requested_features", [])) == sorted(set(features)) and
            old_build.get("no_default_features") is no_default_features and
            old_build.get("target_triple") == target and
            old_build.get("build_project") == str(build_project.resolve()) and
            old_build.get("package") == package and old_build.get("target_name") == target_name and
            old_build.get("artifact_kind") == artifact_kind and isinstance(artifact, Mapping) and
            isinstance(artifact.get("fresh"), bool))


def _selected_package_targets(dependencies: Mapping[str, Any], package: str | None,
                              target_name: str, expected_kind: list[str]) -> tuple[list[Mapping[str, Any]], Any, list[Mapping[str, Any]]]:
    packages = [item for item in dependencies.get("packages", []) if isinstance(item, Mapping)]
    selected = ([item for item in packages if item.get("id") == package or item.get("name") == package]
                if package is not None else [item for item in packages if item.get("root") is True])
    if package is None and len(selected) != 1:
        selected = [item for item in packages if any(
            isinstance(target, Mapping) and target.get("name") == target_name and
            target.get("kind") == expected_kind for target in item.get("targets", []))]
    selected_id = selected[0].get("id") if len(selected) == 1 else None
    selected_targets = [target for item in selected for target in item.get("targets", [])
                        if isinstance(target, Mapping) and target.get("name") == target_name and
                        target.get("kind") == expected_kind]
    return selected, selected_id, selected_targets


def prepare_cargo(*, benchmark: Path, library: Path, binary: Path | None, receipt: Path,
                  features: Sequence[str] = (), profile: str = "release", cargo: str = "cargo",
                  no_default_features: bool = True, target: str | None = None,
                  previous_receipt: Mapping[str, Any] | Path | None = None,
                  build_project: Path | None = None, package: str | None = None,
                  target_name: str | None = None, artifact_kind: str = "bin") -> dict[str, Any]:
    """Build a selected Cargo artifact with JSON events and write a receipt."""
    build_project = (build_project or benchmark).resolve()
    if target_name is None and binary is not None:
        target_name = binary.stem
    argv = [cargo, "test" if artifact_kind == "lib-test" else "build",
            "--locked", "--message-format=json-render-diagnostics"]
    if artifact_kind == "lib-test":
        argv.extend(["--no-run", "--lib"])
    elif artifact_kind == "bin":
        argv.extend(["--bin", target_name or ""])
    if package:
        argv.extend(["-p", package])
    if no_default_features:
        argv.append("--no-default-features")
    if features:
        argv.extend(["--features", ",".join(sorted(set(features)))])
    if target:
        argv.extend(["--target", target])
    if profile == "release":
        argv.append("--release")
    elif profile != "debug":
        argv.extend(["--profile", profile])
    previous: Mapping[str, Any] | None = None
    if isinstance(previous_receipt, Path):
        try:
            value = json.loads(previous_receipt.read_text(encoding="utf-8"))
            previous = value if isinstance(value, Mapping) else None
        except (OSError, json.JSONDecodeError):
            previous = None
    elif isinstance(previous_receipt, Mapping):
        previous = previous_receipt
    before = collect_identity(benchmark, library, binary=binary, cargo=cargo, argv=argv,
                              profile=profile, features=features,
                              no_default_features=no_default_features, target=target,
                              build_project=build_project, package=package,
                              target_name=target_name, artifact_kind=artifact_kind)
    # The requested output may not exist before a first build; that is not a
    # source-provenance failure.  The after-identity check owns this decision.
    errors = [error for error in before.get("errors", [])
              if error != "benchmark binary identity is unavailable"]
    if artifact_kind not in ARTIFACT_KINDS:
        errors.append("artifact kind is unknown")
    if not target_name:
        errors.append("Cargo target name is required")
    if artifact_kind == "lib-test" and not package:
        errors.append("Cargo package is required for lib-test")
    build_result = {"returncode": None, "stdout": "", "stderr": "", "timed_out": False}
    benchmark_files = before.get("files", {}).get("build_project", {})
    required_lock = (isinstance(benchmark_files, Mapping) and
                     any(path.endswith("/Cargo.lock") and digest is not None
                         for path, digest in benchmark_files.items()))
    required_manifest = (isinstance(benchmark_files, Mapping) and
                         any(path.endswith("/Cargo.toml") and digest is not None
                             for path, digest in benchmark_files.items()))
    known_clean = (not errors and
                   all(before.get(label, {}).get("dirty") is False and
                       before.get(label, {}).get("untracked") is False and
                       not before.get(label, {}).get("error")
                       for label in ("benchmark", "library")) and
                   required_manifest and required_lock and
                   before.get("dependencies", {}).get("sha256") is not None and
                   before.get("toolchain", {}).get("cargo") is not None and
                   before.get("toolchain", {}).get("rustc") is not None and
                   not before.get("unknown_environment_keys"))
    if not required_lock or not required_manifest:
        errors.append("Cargo manifest or lockfile identity is unavailable")
    if known_clean:
        try:
            build_result = _run_cargo(argv, build_project)
        except OSError as exc:
            build_result = {"returncode": None, "stdout": "", "stderr": str(exc), "timed_out": False}
            errors.append(f"cargo build launch failed: {type(exc).__name__}")
    else:
        errors.append("cargo build skipped because source provenance is not known clean")
    stdout, stderr = build_result["stdout"], build_result["stderr"]
    events = _events(stdout)
    expected_kind = ["lib"] if artifact_kind == "lib-test" else ["bin"]
    artifacts = [event for event in events if event.get("reason") == "compiler-artifact"
                 and isinstance(event.get("target"), Mapping)
                 and event["target"].get("name") == target_name
                 and event["target"].get("kind") == expected_kind
                 and event.get("executable")]
    finished = any(event.get("reason") == "build-finished" and event.get("success") is True for event in events)
    artifact = artifacts[0] if len(artifacts) == 1 else None
    if artifact is not None:
        artifact_path = Path(str(artifact["executable"]))
        executable = (artifact_path if artifact_path.is_absolute() else build_project / artifact_path).resolve()
    elif binary is not None:
        executable = binary.resolve()
    else:
        executable = build_project / "target" / ("release" if profile == "release" else "debug") / (target_name or "unknown")
    after = collect_identity(benchmark, library, binary=executable, cargo=cargo, argv=argv,
                             profile=profile, features=features,
                             no_default_features=no_default_features, target=target,
                             build_project=build_project, package=package,
                             target_name=target_name, artifact_kind=artifact_kind)
    if build_result["timed_out"]:
        errors.append("cargo build timed out")
    if build_result["returncode"] != 0:
        errors.append("cargo build failed")
    if not finished:
        errors.append("successful cargo build-finished event is missing")
    if len(artifacts) != 1:
        errors.append("matching cargo compiler-artifact event is missing or ambiguous")
    if _input_identity(before) != _input_identity(after):
        errors.append("inputs changed during cargo preparation")
    errors.extend(str(error) for error in after.get("errors", []))
    fresh_reused = False
    if artifact is not None:
        artifact_executable = Path(str(artifact["executable"]))
        artifact_executable = (artifact_executable if artifact_executable.is_absolute()
                               else build_project / artifact_executable).resolve()
        if binary is not None and artifact_executable != binary.resolve():
            errors.append("compiler-artifact executable does not match requested binary")
        package_id = artifact.get("package_id")
        members = before.get("dependencies", {}).get("workspace_members", [])
        selected, selected_id, selected_targets = _selected_package_targets(
            before.get("dependencies", {}), package, target_name or "", expected_kind)
        if (not isinstance(package_id, str) or package_id not in members or
                (len(selected) == 1 and package_id != selected_id)):
            errors.append("compiler-artifact package identity does not match selected package")
        target_info = artifact.get("target")
        if (not isinstance(target_info, Mapping) or target_info.get("kind") != expected_kind or
                target_info.get("name") != target_name):
            errors.append("compiler-artifact target is not the requested artifact")
        if artifact_kind == "lib-test" and len(selected_targets) != 1:
            errors.append("selected lib target is missing or ambiguous")
        if len(selected_targets) == 1 and isinstance(target_info, Mapping):
            if target_info.get("src_path") and Path(str(target_info["src_path"])).resolve() != Path(str(selected_targets[0]["src_path"])).resolve():
                errors.append("compiler-artifact target source does not match selected target")
        if artifact_kind == "bin" and isinstance(artifact.get("profile"), Mapping) and artifact["profile"].get("test") is not False:
            errors.append("binary compiler-artifact has test profile")
        if artifact_kind == "lib-test" and isinstance(artifact.get("profile"), Mapping) and artifact["profile"].get("test") is not True:
            errors.append("lib-test compiler-artifact is not a test harness")
        actual_features = artifact.get("features")
        if (not isinstance(actual_features, list) or any(not isinstance(v, str) for v in actual_features)
                or sorted(actual_features) != sorted(set(features))):
            errors.append("compiler-artifact enabled features do not match requested features")
        artifact_profile = artifact.get("profile")
        if not _profile_known(artifact_profile):
            errors.append("compiler-artifact profile identity is missing or unknown")
        elif not _profile_matches_requested(artifact_profile, profile):
            errors.append("compiler-artifact profile does not match requested profile")
        if not isinstance(artifact.get("fresh"), bool):
            errors.append("compiler-artifact freshness is unknown")
        elif artifact.get("fresh"):
            fresh_reused = _previous_receipt_matches(
                previous, before, executable, profile, features, no_default_features, target,
                build_project, package, target_name or "", artifact_kind)
            if not fresh_reused:
                errors.append("fresh cached artifact lacks a matching verified previous receipt")
    if after.get("binary", {}).get("sha256") is None:
        errors.append("built executable hash is unavailable")
    build = {"argv": argv, "cwd": str(build_project), "target": target_name,
             "target_name": target_name, "artifact_kind": artifact_kind,
             "build_project": str(build_project), "package": package,
             "requested_profile": profile, "requested_features": sorted(set(features)),
             "no_default_features": no_default_features, "target_triple": target,
             "artifact": artifact, "executable": str(executable), "stdout": stdout,
             "stderr": stderr, "returncode": build_result["returncode"],
             "timed_out": build_result["timed_out"], "finished": finished,
             "fresh_reused": fresh_reused,
             "previous_receipt_after": previous.get("after") if fresh_reused and previous else None}
    result = {"schema_version": RECEIPT_VERSION,
              "status": "verified" if not errors else "unverified",
              "before": before, "after": after, "build": build, "errors": sorted(set(errors))}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)
    return result


def verify_preparation_receipt(receipt: Mapping[str, Any], *, benchmark: Path, library: Path,
                               binary: Path | None = None, require_timing: bool = False,
                               cargo: str = "cargo", build_project: Path | None = None,
                               package: str | None = None, target_name: str | None = None,
                               artifact_kind: str = "bin") -> list[str]:
    errors: list[str] = []
    if receipt.get("schema_version") != RECEIPT_VERSION or receipt.get("status") != "verified":
        errors.append("preparation receipt is not verified")
    if receipt.get("errors") != []:
        errors.append("preparation receipt contains build errors")
    build = receipt.get("build")
    after = receipt.get("after")
    if not isinstance(build, Mapping) or not isinstance(after, Mapping):
        return errors + ["preparation receipt is incomplete"]
    if any(key not in build for key in ("artifact_kind", "build_project", "target_name", "package")):
        errors.append("preparation receipt build selectors are incomplete")
    if build.get("finished") is not True or build.get("returncode") != 0 or build.get("timed_out") is True:
        errors.append("receipt does not prove a successful Cargo build")
    recorded_project = build.get("build_project")
    actual_project = (build_project or (Path(str(recorded_project)) if recorded_project else benchmark)).resolve()
    recorded_package = build.get("package")
    recorded_target = build.get("target_name", build.get("target"))
    recorded_kind = build.get("artifact_kind", "bin")
    if recorded_kind not in ARTIFACT_KINDS:
        errors.append("receipt artifact kind is unknown")
    if package is None:
        package = recorded_package
    if target_name is None:
        target_name = recorded_target
    expected_project = (Path(str(recorded_project)).resolve() if recorded_project else benchmark.resolve())
    if actual_project != expected_project or build.get("cwd") != str(expected_project):
        errors.append("receipt build project does not match requested project")
    if package != recorded_package or target_name != recorded_target or artifact_kind != recorded_kind:
        errors.append("receipt artifact selectors do not match requested selectors")
    executable = binary or Path(str(build.get("executable", "")))
    if not executable.is_absolute():
        executable = actual_project / executable
    executable = executable.resolve()
    requested_profile = build.get("requested_profile")
    if requested_profile not in ("debug", "release"):
        errors.append("receipt build profile is unknown")
    requested_no_default = build.get("no_default_features")
    if not isinstance(requested_no_default, bool):
        errors.append("receipt no-default-features setting is unknown")
    features = build.get("requested_features", [])
    if not isinstance(features, list) or any(not isinstance(v, str) for v in features):
        errors.append("receipt build features are unknown")
        features = []
    current = collect_identity(benchmark, library, binary=executable, cargo=cargo,
                               argv=build.get("argv") if isinstance(build.get("argv"), list) else None,
                               profile=requested_profile, features=features,
                               no_default_features=requested_no_default is True,
                               target=build.get("target_triple"), build_project=actual_project,
                               package=recorded_package, target_name=recorded_target,
                               artifact_kind=recorded_kind)
    errors.extend(str(error) for error in current.get("errors", []))
    for phase, identity in (("before", receipt.get("before")), ("after", after), ("current", current)):
        identity_errors = identity.get("errors") if isinstance(identity, Mapping) else None
        allowed_before = ["benchmark binary identity is unavailable"]
        if not isinstance(identity_errors, list) or any(
                error not in (allowed_before if phase == "before" else [])
                for error in identity_errors):
            errors.append(f"{phase} provenance identity contains errors")
    for label in ("benchmark", "library"):
        state = current.get(label, {})
        if state.get("dirty") is not False or state.get("untracked") is not False:
            errors.append(f"{label} checkout is dirty or unknown")
        if not state.get("head"):
            errors.append(f"{label} checkout HEAD is unknown")
    if _input_identity(current) != _input_identity(after):
        errors.append("preparation inputs are stale or changed")
    expected_hash = (after.get("binary") or {}).get("sha256") if isinstance(after.get("binary"), Mapping) else None
    if not expected_hash or current.get("binary", {}).get("sha256") != expected_hash:
        errors.append("benchmark binary was substituted or changed")
    artifact = build.get("artifact")
    artifact_profile = artifact.get("profile") if isinstance(artifact, Mapping) else None
    if not isinstance(artifact, Mapping) or not artifact.get("executable"):
        errors.append("compiler-artifact evidence is missing")
    elif executable.resolve() != (Path(str(artifact["executable"])) if Path(str(artifact["executable"])).is_absolute()
                                  else actual_project / str(artifact["executable"])).resolve():
        errors.append("receipt binary does not match compiler-artifact executable")
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("fresh"), bool):
        errors.append("compiler-artifact freshness is unknown")
    elif artifact.get("fresh") is True:
        previous_after = build.get("previous_receipt_after")
        old_binary = previous_after.get("binary") if isinstance(previous_after, Mapping) else None
        current_binary = current.get("binary")
        if (build.get("fresh_reused") is not True or
                not isinstance(previous_after, Mapping) or
                _input_identity(previous_after) != _input_identity(after) or
                not isinstance(old_binary, Mapping) or not isinstance(current_binary, Mapping) or
                old_binary.get("sha256") != current_binary.get("sha256")):
            errors.append("fresh cached artifact lacks a matching verified previous receipt")
    if not _profile_known(artifact_profile):
        errors.append("artifact profile identity is missing or unknown")
    elif not _profile_matches_requested(artifact_profile, requested_profile):
        errors.append("artifact profile does not match requested profile")
    if isinstance(artifact, Mapping):
        actual_features = artifact.get("features")
        if not isinstance(actual_features, list) or sorted(actual_features) != sorted(set(features)):
            errors.append("artifact enabled features do not match receipt")
        expected_kind = ["lib"] if recorded_kind == "lib-test" else ["bin"]
        target_info = artifact.get("target")
        if (not isinstance(target_info, Mapping) or target_info.get("kind") != expected_kind or
                target_info.get("name") != recorded_target):
            errors.append("artifact target identity does not match receipt")
        profile_test = artifact_profile.get("test") if isinstance(artifact_profile, Mapping) else None
        if recorded_kind == "bin" and profile_test is not False:
            errors.append("binary artifact has test profile")
        if recorded_kind == "lib-test" and profile_test is not True:
            errors.append("lib-test artifact is not a test harness")
        package_id = artifact.get("package_id")
        members = current.get("dependencies", {}).get("workspace_members", [])
        selected, selected_id, selected_targets = _selected_package_targets(
            current.get("dependencies", {}), recorded_package, recorded_target, expected_kind)
        if (not isinstance(package_id, str) or package_id not in members or
                (len(selected) == 1 and package_id != selected_id)):
            errors.append("artifact package identity does not match selected package")
        if recorded_kind == "lib-test" and len(selected_targets) != 1:
            errors.append("selected lib target is missing or ambiguous")
        if len(selected_targets) == 1 and isinstance(target_info, Mapping) and target_info.get("src_path"):
            if Path(str(target_info["src_path"])).resolve() != Path(str(selected_targets[0]["src_path"])).resolve():
                errors.append("artifact target source does not match selected target")
    if require_timing and _profile_is_debug(artifact_profile):
        errors.append("debug artifact cannot provide release timing evidence")
    return sorted(set(errors))


def receipt_is_verified(*args: Any, **kwargs: Any) -> bool:
    return not verify_preparation_receipt(*args, **kwargs)
