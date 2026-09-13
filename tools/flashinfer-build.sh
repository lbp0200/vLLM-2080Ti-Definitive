#!/usr/bin/env bash
# This script is used to build FlashInfer wheels with AOT kernels

set -ex

# FlashInfer configuration
FLASHINFER_GIT_REPO="https://github.com/flashinfer-ai/flashinfer.git"
BUILD_WHEEL="${BUILD_WHEEL:-true}"

if [[ -z "${FLASHINFER_GIT_REF}" ]]; then
    echo "❌ FLASHINFER_GIT_REF must be specified" >&2
    exit 1
fi

if [[ -z "${CUDA_VERSION}" ]]; then
    echo "❌ CUDA_VERSION must be specified" >&2
    exit 1
fi

echo "🏗️  Building FlashInfer ${FLASHINFER_GIT_REF} for CUDA ${CUDA_VERSION}"

# Clone FlashInfer
git clone --depth 1 --recursive --shallow-submodules \
    --branch "${FLASHINFER_GIT_REF}" \
    ${FLASHINFER_GIT_REPO} flashinfer

# This fork ships FlashInfer AOT kernels for RTX 2080 Ti/Turing only.
FI_TORCH_CUDA_ARCH_LIST="7.5"

echo "🏗️ Building FlashInfer AOT for arches: ${FI_TORCH_CUDA_ARCH_LIST}"

pushd flashinfer
    # Make sure the wheel is built for the correct CUDA version
    export UV_TORCH_BACKEND=cu$(echo "$CUDA_VERSION" | cut -d. -f1,2 | tr -d '.')

    # Build AOT kernels
    export TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}"
    export FLASHINFER_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}"
    python3 -m flashinfer.aot
    
    if [[ "${BUILD_WHEEL}" == "true" ]]; then
        # Build wheel for distribution
        uv build --no-build-isolation --wheel --out-dir ../flashinfer-dist .
        echo "✅ FlashInfer wheel built successfully in flashinfer-dist/"
    else
        # Install directly (for Dockerfile)
        uv pip install --system --no-build-isolation --force-reinstall .
        echo "✅ FlashInfer installed successfully"
    fi
popd

# Cleanup
rm -rf flashinfer
