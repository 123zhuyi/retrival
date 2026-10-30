# =============================================================================
# NPU KernelBench Docker image — Ascend NPU kernel evaluation framework.
# Build: docker build -t npu-kernelbench:latest .
# Run:   docker run --device=/dev/davinci0 --device=/dev/davinci_manager ...
# =============================================================================

FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    ASCEND_BASE=/usr/local/Ascend

# System packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    curl git vim python3 python3-dev python3-pip \
    build-essential g++ gcc cmake make && \
    apt-get clean

# Install Python 3.11 (matching target environment)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    apt-get clean

ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=npu-kb

# Create user
RUN if [ "${HOST_UID}" != "0" ]; then \
        groupadd -f -g ${HOST_GID} ${HOST_USER} && \
        useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER}; \
    fi && \
    mkdir -p /npu-kernelbench && \
    chown -R ${HOST_UID}:${HOST_GID} /npu-kernelbench

# Copy and set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to user
WORKDIR /npu-kernelbench
USER ${HOST_UID}:${HOST_GID}

# Install Python dependencies via pip
COPY pyproject.toml README.md ./
RUN python3 -m pip install --user --no-cache-dir \
    torch numpy pydantic click rich && \
    python3 -m pip install --user --no-cache-dir torch_npu

# Copy source code (changes frequently — do this last)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY bench_config.example.json ./

# Install the project itself in development mode
RUN python3 -m pip install --user -e .

# Runtime environment
ENV PYTHONPATH="/npu-kernelbench/src:${PYTHONPATH}" \
    PATH="/root/.local/bin:${PATH}"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "npu_kernelbench.cli.main", "--help"]
