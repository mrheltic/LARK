#!/bin/bash
# ============================================================
#  KrakenSDR Docker – Image build
#
#  Usage: bash docker/build.sh [OPTIONS]
#
#  Options:
#    --no-cache          Force full rebuild, skip Docker cache
#    --build-flags FLAGS Override C/C++ optimisation flags
#                        (default: "-O3 -march=native -ffast-math -pipe")
#    --kfr-version VER   Override KFR DSP library version (default: 5.0.2)
#    --push              After a successful build, push to Docker Hub
#                        (docker login required beforehand)
#
#  The image is tagged locally and, with --push, on Docker Hub too.
#  Example tags after build:
#    krakensdr-docker:latest            (local)
#    krakensdr-docker:a3f7c92           (local)
#    mrheltic/krakensdr-docker:latest   (Docker Hub, only with --push)
#    mrheltic/krakensdr-docker:a3f7c92  (Docker Hub, only with --push)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

IMAGE="krakensdr-docker"
DOCKERHUB_USER="mrheltic"
NO_CACHE=""
BUILD_FLAGS="-O3 -march=native -ffast-math -pipe"
KFR_VERSION="5.0.2"
PUSH_TO_HUB=""

for arg in "$@"; do
    case $arg in
        --no-cache)        NO_CACHE="--no-cache" ;;
        --build-flags)     shift; BUILD_FLAGS="$1" ;;
        --kfr-version)     shift; KFR_VERSION="$1" ;;
        --push)            PUSH_TO_HUB="1" ;;
        --help|-h)
            sed -n '2,23p' "$0"
            exit 0
            ;;
    esac
done

# Derive a short commit hash for image labelling and tagging.
# Falls back to "dev" when building outside a git repository.
GIT_COMMIT=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "dev")

echo "╔══════════════════════════════════════════════════════╗"
echo "║         KrakenSDR Docker – Build                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "[*] Build context : $ROOT_DIR"
echo "[*] Target image  : ${IMAGE}:latest  +  ${IMAGE}:${GIT_COMMIT}"
echo "[*] Commit        : ${GIT_COMMIT}"
echo "[*] Build flags   : ${BUILD_FLAGS}"
echo "[*] KFR version   : ${KFR_VERSION}"
[ -n "$NO_CACHE" ] && echo "[*] Mode          : --no-cache"
[ -n "$PUSH_TO_HUB" ] && echo "[*] Docker Hub    : ${DOCKERHUB_USER}/${IMAGE}"
echo ""

if [ ! -f "$ROOT_DIR/docker/Dockerfile" ]; then
    echo "[✗] Dockerfile not found at docker/Dockerfile"
    exit 1
fi

# Resolve critical hosts on the host machine and inject them into build
# containers via --add-host. This avoids DNS failures inside Docker build
# layers on hosts where container DNS is broken.
BUILD_ADD_HOSTS=()
for host in archive.ubuntu.com security.ubuntu.com github.com pypi.org files.pythonhosted.org pypi.python.org; do
    ip=$(getent ahostsv4 "$host" 2>/dev/null | awk 'NR==1 {print $1}')
    if [ -n "$ip" ]; then
        BUILD_ADD_HOSTS+=(--add-host "$host:$ip")
        echo "[*] add-host      : ${host} -> ${ip}"
    else
        echo "[!] add-host skip : could not resolve ${host} on host DNS"
    fi
done

# Build – tag with both :latest and the git commit hash
docker build \
    $NO_CACHE \
    "${BUILD_ADD_HOSTS[@]}" \
    --tag "${IMAGE}:latest" \
    --tag "${IMAGE}:${GIT_COMMIT}" \
    --file docker/Dockerfile \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg GIT_COMMIT="${GIT_COMMIT}" \
    --build-arg BUILD_FLAGS="${BUILD_FLAGS}" \
    --build-arg KFR_VERSION="${KFR_VERSION}" \
    .

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  [✓] Build completata                                ║"
echo "║                                                      ║"
echo "║  Tags: ${IMAGE}:latest                               ║"
echo "║        ${IMAGE}:${GIT_COMMIT}                        ║"
echo "║                                                      ║"
echo "║  Start with: bash docker/run.sh synthetic            ║"
echo "║              bash docker/run.sh test                 ║"
echo "║              bash docker/run.sh bash                 ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── Optional Docker Hub push ─────────────────────────────────────────────────
if [ -n "$PUSH_TO_HUB" ]; then
    echo ""
    echo "[*] Tagging per Docker Hub: ${DOCKERHUB_USER}/${IMAGE}"
    docker tag "${IMAGE}:latest"      "${DOCKERHUB_USER}/${IMAGE}:latest"
    docker tag "${IMAGE}:${GIT_COMMIT}" "${DOCKERHUB_USER}/${IMAGE}:${GIT_COMMIT}"

    echo "[*] Push → Docker Hub …"
    docker push "${DOCKERHUB_USER}/${IMAGE}:latest"
    docker push "${DOCKERHUB_USER}/${IMAGE}:${GIT_COMMIT}"

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  [✓] Push completata su Docker Hub                   ║"
    echo "║                                                      ║"
    echo "║  Pull: docker pull ${DOCKERHUB_USER}/${IMAGE}:latest ║"
    echo "╚══════════════════════════════════════════════════════╝"
fi
