FROM ghcr.io/actions/actions-runner:2.337.0@sha256:e5496277be5d09bc968b3d64911b74e219ac4a3f2edce956a3ecf9271bea1ef4
USER root
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git jq zstd binutils libssl3t64 \
 && rm -rf /var/lib/apt/lists/*
COPY payload/ /
ENV RUNNER_ALLOW_RUNASROOT=1
ENV PATH="/opt/ci-cost-runtime/bin:${PATH}"
USER runner
