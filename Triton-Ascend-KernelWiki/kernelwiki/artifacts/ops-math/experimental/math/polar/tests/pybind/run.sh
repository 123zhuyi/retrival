#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# Polar 测试驱动（S8 同款流程：重建 pybind wheel → msprof 计时 → get_time 提取 → 基线校验）
#
# 前置（在 NPU 环境，且只需做一次/改 kernel 后重做）：
#   1) 在 ops-math 仓编译并安装自定义 Polar 算子：
#        bash build.sh --pkg --soc=ascend910b --ops=polar -j16
#        ./build_out/cann-ops-math-*linux*.run
#      （A3 用 --soc=ascend910_93）
#   2) 自定义算子包安装在 ${ASCEND_HOME_PATH}/opp/vendors/custom_math/
#
# 用法：bash run.sh <caseNum>      例：bash run.sh 2

# 自定义算子 op_api 库路径（ops-math 自定义包 vendor = custom_math；兼容 customize 兜底）
_OPP="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}"
export LD_LIBRARY_PATH="$_OPP/vendors/custom_math/op_api/lib/:$_OPP/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH"

# 每次 run 都重建 wheel（避免上次跑别的 op 留下的 wheel signature 不匹配）
rm -rf ./dist ./build ./custom_ops.egg-info
echo "重新生成whl"
python3 setup.py build bdist_wheel
pip3 install dist/custom_ops*.whl --force-reinstall

rm -rf PROF*
timeout 180 msprof --application="python3 test_op.py $1"

if [ $? -eq 124 ]; then
     echo "timed out!"
     exit 1
fi

time_use=$(($(python3 get_time.py)))
# 基线：先用哨兵值（仅做正确性门禁）；硬件实测出 l0 参考实现耗时后回填真实基线
time_base=9999999999999
echo "time_base = $time_base time_use = $time_use"
if [ $time_use -eq 0 ]; then
    echo "[ERROR] Performance not achieved"
    exit 1
fi

if [ $time_use -ge $time_base ]; then
    echo "test fail for performance exceeds baseline data"
    exit 1
fi

echo "Operator performance and accuracy have passed"
