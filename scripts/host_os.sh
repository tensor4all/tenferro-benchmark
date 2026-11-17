#!/usr/bin/env bash

# Host operating system of the machine a run is collected on. Runners use it to
# pick the default target profile; BENCHMARK_HOST_OS overrides the probe, which
# the fixture tests use to exercise the macOS path off macOS.

benchmark_host_os() {
    if [[ -n "${BENCHMARK_HOST_OS:-}" ]]; then
        printf '%s\n' "$BENCHMARK_HOST_OS"
    else
        uname -s
    fi
}
