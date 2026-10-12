#!/bin/bash
# Run ATK accuracy/performance on FusedMulAddN (aclnn pyaclnn vs cpu golden).
# Bridges ATK's pyaclnn backend to the repo-built custom_math package.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "$HERE/../../../../../.." && pwd)"   # -> worktree root (feat+fused-mul-add-n)

# CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/driver/bin/setenv.bash ] && source /usr/local/Ascend/driver/bin/setenv.bash

# Bridge: ATK's pyaclnn resolves a custom op via ${ASCEND_CUSTOM_OPP_PATH}/vendors/<name>/op_api/lib/
# libcust_opapi.so (vendor names from vendors/config.ini). So ASCEND_CUSTOM_OPP_PATH must be the PARENT
# of vendors/ (NOT the vendor dir itself), and a vendors/config.ini must list the vendor.
# NOTE: the system opp/vendors/custom_math is a DIFFERENT op (bitwise_not) — do NOT rely on it.
CUSTOM_OPP="$WORKTREE_ROOT/build_out/installed_custom_math/packages"
[ -f "$CUSTOM_OPP/vendors/config.ini" ] || echo "load_priority=custom_math" > "$CUSTOM_OPP/vendors/config.ini"
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP"
export LD_LIBRARY_PATH="$CUSTOM_OPP/vendors/custom_math/op_api/lib:$LD_LIBRARY_PATH"

DEV="${DEV:-0}"
TASK="${TASK:-accuracy}"
cd "$HERE"

echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
echo "task=$TASK dev=$DEV"

case "$TASK" in
  accuracy)
    atk node --backend pyaclnn --devices "$DEV" \
        node --backend cpu \
        task -c atk_aclnnFusedMulAddN.json -p executor_aclnnFusedMulAddN.py --task accuracy -cp
    ;;
  kernel)
    atk node --backend kernel --devices "$DEV" \
        node --backend cpu \
        task -c atk_aclnnFusedMulAddN.json -p executor_aclnnFusedMulAddN.py --task accuracy
    ;;
  performance)
    export ATK_BIND_CPU_TYPE=2
    atk node --backend pyaclnn --devices "$DEV" \
        node --backend npu --devices "$((DEV+1))" \
        task -c atk_aclnnFusedMulAddN.json -p executor_aclnnFusedMulAddN.py --task performance_device
    ;;
  *)
    echo "unknown TASK=$TASK (accuracy|kernel|performance)"; exit 2 ;;
esac
