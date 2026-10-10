#!/usr/bin/env bash
# Human-triggered registry publication of the already validated image artifact.
# The default verifies readiness only; --execute is invoked by the manual publish job.
set -euo pipefail
mode="${1:---check}"
[[ "$mode" == --check || "$mode" == --execute ]] || { echo 'Use --check or --execute' >&2; exit 2; }
repo=tensor4all/tenferro-benchmark
run=38035993659
source=8388132a874d2df42a8482e26a452d1a1735ee5d
image=ghcr.io/tensor4all/tenferro-ci-prepared-runner:356a2011-8388132
record=$(gh run view "$run" --repo "$repo" --json status,conclusion,headSha)
printf '%s' "$record" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["status"]=="completed" and d["conclusion"]=="success" and d["headSha"]==sys.argv[1], d' "$source"
count=$(gh api "repos/$repo/actions/runs/$run/artifacts" --jq '[.artifacts[] | select(.name == "runpod-prepared-runner-image" and .expired == false)] | length')
[[ "$count" == 1 ]] || { echo 'Validated image artifact missing or expired' >&2; exit 1; }
echo "Validated source run $run is ready for $image"
[[ "$mode" == --execute ]] || exit 0
# This path is only dispatched by the human publication handoff.
directory=$(mktemp -d)
trap 'rm -rf "$directory"' EXIT
gh run download "$run" --repo "$repo" --name runpod-prepared-runner-image --dir "$directory"
[[ "$(cat "$directory/source-commit.txt")" == "$source" ]]
(cd "$directory" && sha256sum --check prepared-runner.sha256)
zstd -dc "$directory/prepared-runner.tar.zst" | docker load
expected=$(python3 - "$directory/image-inspect.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))[0]['Id'])
PY
)
[[ "$(docker image inspect tenferro-prepared-runner:validated --format '{{.Id}}')" == "$expected" ]]
docker tag "$expected" "$image"
printf '%s' "$GH_TOKEN" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin
docker push "$image"
echo "Published $image; a new GHCR package must be made public by its owner before anonymous RunPod pulls."
