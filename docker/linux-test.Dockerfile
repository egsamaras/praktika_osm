# Linux test image for Praktika: Ubuntu 24.04 on aarch64 (the base and CPU family of NVIDIA DGX
# OS 7, as on a DGX Spark), the distro's own python3.12, uv from its pinned release tarball, the
# project installed from uv.lock with the dev extra and WITHOUT the mac extra, run as a non-root
# user. It runs the test suite on Linux; it has no GPU and proves nothing about speed. On an
# x86_64 machine, build it with emulation (docker buildx and QEMU) or adapt the uv tarball and
# its checksum.
#
# Build (from the repository root; docker/linux-test.Dockerfile.dockerignore applies):
#   docker build --platform linux/arm64 -f docker/linux-test.Dockerfile -t praktika-linux-test .
# Test (the 3.7 MB speech fixture is kept out of the image; 11 VAD/router tests read it, so
# mount it read-only, otherwise those 11 fail with "Error opening ... synthetic_meeting.wav"):
#   docker run --rm \
#     -v "$PWD/tests/fixtures/synthetic_meeting.wav:/app/tests/fixtures/synthetic_meeting.wav:ro" \
#     praktika-linux-test
# The default command is: uv run --frozen --no-sync pytest -p no:cacheprovider -rs (-q comes from
# addopts; a second -q would hide the summary line)
FROM ubuntu:24.04

ARG UV_VERSION=0.12.5
# sha256 of uv-aarch64-unknown-linux-gnu.tar.gz, from the release's .sha256 file and sha256.sum.
ARG UV_SHA256=9bf43b4d1a07665bf64d4c4e710930b382321a785e0eb10aac07f46471f86a31

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/praktika-venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv ffmpeg util-linux ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    arch="$(uname -m)"; [ "$arch" = "aarch64" ] || { echo "expected aarch64, got $arch" >&2; exit 1; }; \
    tarball="uv-aarch64-unknown-linux-gnu.tar.gz"; \
    curl -fsSL -o "/tmp/$tarball" \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${tarball}"; \
    echo "${UV_SHA256}  /tmp/${tarball}" | sha256sum -c -; \
    tar -xzf "/tmp/$tarball" -C /tmp; \
    install -m 0755 /tmp/uv-aarch64-unknown-linux-gnu/uv /tmp/uv-aarch64-unknown-linux-gnu/uvx /usr/local/bin/; \
    rm -rf /tmp/uv-aarch64-unknown-linux-gnu "/tmp/$tarball"; \
    uv --version

WORKDIR /app
# Dependencies first (cached layer), then the project itself. The uv cache lives in a BuildKit
# cache mount so it never lands in an image layer; `docker builder prune` removes it.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev \
 && /opt/praktika-venv/bin/python -c "import sys, platform; print(sys.version, platform.machine())"

# Non-root runtime user. /app and the venv stay root-owned (read-only to the service account, as
# on a production server); the account's home is the only writable place besides /tmp.
RUN useradd --uid 10001 --create-home --shell /bin/bash praktika
USER praktika
ENV PATH="/opt/praktika-venv/bin:${PATH}" \
    HOME=/home/praktika
CMD ["uv", "run", "--frozen", "--no-sync", "pytest", "-p", "no:cacheprovider", "-rs"]
