#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
OP_DIR="${SCRIPT_DIR}/../.."
MOCK_MODE=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --mock)
            MOCK_MODE=true
            ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            echo "Usage: $0 [--mock]"
            exit 1
            ;;
    esac
done

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "[ERROR] ASCEND_HOME_PATH not set"; exit 1
fi

echo "========================================"
if [ "$MOCK_MODE" = true ]; then
    echo "Pdist ST Test (Mock - CPU Golden Only)"
else
    echo "Pdist ST Test (Real - NPU)"
fi
echo "========================================"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [ "$MOCK_MODE" = true ]; then
    # Mock mode: compile without ACL headers, verify CPU golden correctness
    g++ -std=c++17 -O2 \
        -DUSE_MOCK_ACLNN \
        -o test_pdist_st_mock \
        ${SCRIPT_DIR}/test_aclnn_pdist.cpp

    echo "[INFO] Mock compilation successful"
    echo ""
    ./test_pdist_st_mock
else
    # Real mode: compile with ACL headers, run on NPU
    CUST_LIB_DIR="${ASCEND_HOME_PATH}/vendors/custom_math/op_api/lib"
    DRIVER_LIB_DIR="/usr/local/Ascend/driver/lib64/driver"

    g++ -std=c++17 -O2 \
        -I${ASCEND_HOME_PATH}/include \
        -I${ASCEND_HOME_PATH}/aarch64-linux/include \
        -I${ASCEND_HOME_PATH}/aarch64-linux/include/aclnn \
        -I${OP_DIR}/op_api \
        -L${CUST_LIB_DIR} \
        -L${ASCEND_HOME_PATH}/lib64 \
        -L${DRIVER_LIB_DIR} \
        -o test_pdist_st \
        ${SCRIPT_DIR}/test_aclnn_pdist.cpp \
        -lcust_opapi -lascendcl -lnnopbase -lopapi -lacl_op_compiler \
        -Wl,-rpath,${CUST_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${DRIVER_LIB_DIR}

    echo "[INFO] Real compilation successful"
    echo ""

    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
    ./test_pdist_st
fi
