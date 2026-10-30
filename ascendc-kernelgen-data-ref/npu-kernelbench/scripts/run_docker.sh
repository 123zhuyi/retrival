#!/bin/bash
set -e

# =============================================================================
# Run npu-kernelbench in Docker with NPU device passthrough.
#
# Usage:
#   ./scripts/run_docker.sh build                # Build the Docker image
#   ./scripts/run_docker.sh -- <command>         # Run a command in container
#   ./scripts/run_docker.sh -- bash              # Interactive shell
#   ./scripts/run_docker.sh eval ...             # Run eval directly
# =============================================================================

IMAGE_NAME="${NPU_KB_IMAGE:-npu-kernelbench:latest}"
CONTAINER_NAME="npu-kb-$(date +%s)"

# Collect all NPU devices on the host
NPU_DEVICES=""
for dev in /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    if [ -e "$dev" ]; then
        NPU_DEVICES="$NPU_DEVICES --device=$dev"
    fi
done

build() {
    echo "Building ${IMAGE_NAME}..."
    docker build \
        --build-arg HOST_UID=$(id -u) \
        --build-arg HOST_GID=$(id -g) \
        -t "${IMAGE_NAME}" \
        -f docker/Dockerfile \
        .
    echo "Build complete: ${IMAGE_NAME}"
}

run() {
    echo "Starting container: ${CONTAINER_NAME}"
    docker run \
        --rm \
        --name "${CONTAINER_NAME}" \
        --network host \
        --ipc host \
        --privileged \
        ${NPU_DEVICES} \
        -v /usr/local/Ascend:/usr/local/Ascend:ro \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
        -v $(pwd)/data:/npu-kernelbench/data \
        -v $(pwd)/scripts:/npu-kernelbench/scripts \
        -e ASCEND_HOME=/usr/local/Ascend \
        "${IMAGE_NAME}" \
        "$@"
}

case "${1:-}" in
    build)
        build
        ;;
    --)
        shift
        run "$@"
        ;;
    *)
        # Default: run the command directly
        run "$@"
        ;;
esac
