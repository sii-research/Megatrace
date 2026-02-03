#!/bin/bash
# Build script for Megatrace NCCL interceptor
# Supports both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs

# Default GPU vendor (NVIDIA for backward compatibility)
GPU_VENDOR="${GPU_VENDOR:-nvidia}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-vendor)
            GPU_VENDOR="$2"
            shift 2
            ;;
        --nvidia)
            GPU_VENDOR="nvidia"
            shift
            ;;
        --amd)
            GPU_VENDOR="amd"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu-vendor VENDOR   Set GPU vendor (nvidia|amd)"
            echo "  --nvidia              Build for NVIDIA GPUs (CUDA) [default]"
            echo "  --amd                 Build for AMD GPUs (ROCm/HIP)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  GPU_VENDOR            GPU vendor (nvidia|amd)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Normalize GPU vendor name
GPU_VENDOR=$(echo "$GPU_VENDOR" | tr '[:upper:]' '[:lower:]')

# Set build flags based on GPU vendor
if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "cuda" ]; then
    echo "Building for NVIDIA GPUs (CUDA)..."
    GPU_DEFINE="-DMEGATRACE_GPU_NVIDIA"
    if [ -z "$CPATH" ]; then
        export CPATH=/usr/local/cuda/include:$CPATH
    else
        export CPATH=/usr/local/cuda/include:$CPATH
    fi
    LIBS="-ldl -fPIC -lpthread"
    OUTPUT="nccl_intercept.so"
elif [ "$GPU_VENDOR" = "amd" ] || [ "$GPU_VENDOR" = "rocm" ] || [ "$GPU_VENDOR" = "hip" ]; then
    echo "Building for AMD GPUs (ROCm/HIP)..."
    GPU_DEFINE="-DMEGATRACE_GPU_AMD"
    if [ -z "$ROCM_PATH" ]; then
        export ROCM_PATH=/opt/rocm
    fi
    if [ -z "$CPATH" ]; then
        export CPATH=$ROCM_PATH/include:$CPATH
    else
        export CPATH=$ROCM_PATH/include:$CPATH
    fi
    LIBS="-ldl -fPIC -lpthread -L$ROCM_PATH/lib -lhip_hcc"
    OUTPUT="rccl_intercept.so"
else
    echo "Error: Unknown GPU vendor: $GPU_VENDOR"
    echo "Supported vendors: nvidia, amd"
    exit 1
fi

# Select source files based on GPU vendor
if [ "$GPU_VENDOR" = "nvidia" ] || [ "$GPU_VENDOR" = "cuda" ]; then
    SOURCES="nccl_intercept.cc intercept_common.cc ring_log.cc stream_watchdog.cc"
elif [ "$GPU_VENDOR" = "amd" ] || [ "$GPU_VENDOR" = "rocm" ] || [ "$GPU_VENDOR" = "hip" ]; then
    SOURCES="rccl_intercept.cc intercept_common.cc ring_log.cc"
else
    echo "Error: Unknown GPU vendor: $GPU_VENDOR"
    exit 1
fi

# Build command
echo "Compiling with flags: $GPU_DEFINE"
echo "Source files: $SOURCES"
g++ -shared -o "$OUTPUT" $SOURCES $GPU_DEFINE $LIBS

if [ $? -eq 0 ]; then
    echo "Build successful! Output: $OUTPUT"
    echo "GPU Vendor: $GPU_VENDOR"
    echo "GPU API: $(echo $GPU_VENDOR | tr '[:lower:]' '[:upper:]')"
else
    echo "Build failed!"
    exit 1
fi
