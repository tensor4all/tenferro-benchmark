# Issue 95: small-work suite

## Context

Parent: tensor4all/tenferro-rs#1758. The existing public_api suite's operation
coverage and thin aliases do not establish equal public-boundary cost. Preserve
that suite and add explicit concrete/eager/prepared/compiled timing contracts.
Design: [small-work-design](../small-work-design.md).
Benchmark baseline: `ce729ecfa040b4c825850e0bcf0907142bdac58f`.
Library baseline: `0457a2ed0aeea21b14f4297f7f4731e09b3a0507`.

## Pre-implementation gate

DeepSeek V4 Flash, read-only design review round 1: **Correct-to-merge**, no
Critical/Important findings, four Minor pins incorporated before implementation:
worktree-local extern pin, strict run-schema/metadata integration, distinct API tier
keys and new-runner-specific timing guards. Reviewed design, public runner,
publication gate, suite/coverage manifests, result/run schemas, extern setup and
layout/timing tests. Implementation is assigned to Luna and has not started.

## Evidence and constraints

The unpushed prototype object `30f446761d08bd085da356b21f6a509651776b10` was not
available in this host's tenferro git database. Do not imply the prototype was read
or reproduced; owning-library component probes need explicit integration evidence.

A preliminary three-second `/proc/stat` observation found every allowed CPU busy
(minimum 83.85% across CPUs 0–63). This is not a frozen benchmark protocol or a
measurement result. No timings were collected; wait for valid idle resources and
retain run-level INCONCLUSIVE diagnostics if unavailable at collection time.
Docker is available; a devcontainer CLI was not found on PATH. Existing containers
must not be modified or stopped without identifying a task-owned environment.

## Remaining

Implement approved design, test validators/runner/report and migrated dependency
consumers, integrate #1759 case contracts and owning-library probes, obtain actual
valid raw evidence, run Flash full-diff review and relevant local gates, publish PR
and verify CI/merge. This worklog does not claim #95 or #1758 complete.
