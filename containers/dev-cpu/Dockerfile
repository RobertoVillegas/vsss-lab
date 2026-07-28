FROM ubuntu:26.04@sha256:3131b4cc82a783df6c9df078f86e01819a13594b865c2cad47bd1bca2b7063bb

ARG TARGETARCH
ARG MISE_VERSION=2026.7.14
ENV DEBIAN_FRONTEND=noninteractive \
    MISE_DATA_DIR=/opt/mise \
    MISE_CONFIG_DIR=/opt/mise/config \
    PATH=/opt/mise/shims:/opt/mise/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && case "$TARGETARCH" in amd64) arch=x64 ;; arm64) arch=arm64 ;; *) exit 2 ;; esac \
    && asset="mise-v${MISE_VERSION}-linux-${arch}" \
    && curl -fsSLO "https://github.com/jdx/mise/releases/download/v${MISE_VERSION}/${asset}" \
    && curl -fsSLO "https://github.com/jdx/mise/releases/download/v${MISE_VERSION}/SHASUMS256.txt" \
    && grep "  ./${asset}$" SHASUMS256.txt | sha256sum --check --strict \
    && install -m 0755 "${asset}" /usr/local/bin/mise \
    && rm "${asset}" SHASUMS256.txt

RUN useradd --create-home --uid 10001 --shell /bin/bash vsss \
    && mkdir -p /opt/mise/config /workspace \
    && chown -R vsss:vsss /opt/mise /workspace

WORKDIR /workspace
COPY --chown=vsss:vsss . .
USER vsss
RUN mise trust mise.toml && mise install && mise run bootstrap
CMD ["mise", "run", "test"]
