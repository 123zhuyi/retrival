# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 需要将 Tiling 依赖的源码加入编译（实验态 flat 工程：op_host/ 下扁平 tiling + infershape）。
    set(fused_mul_add_n_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/fused_mul_add_n_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/fused_mul_add_n_infershape.cpp
        )
    # 使用 AddOpTestCase
    # param1：算子名称（kernel 命名 fused_mul_add_n）
    # param2：soc 版本（本算子 A2 数值验证目标 ascend910b）
    # param3：自定义编译选项
    #   ⚠ A2 kernel `op_kernel/fused_mul_add_n.cpp` 按 **TilingKey** 在运行期选择 dtype 模板
    #     （0=fp32/1=fp16/2=int32/3=int16/4=bf16），入口签名与 DTYPE_X 编译宏无关
    #     （已核对源码不引用 DTYPE_X）。故单份编译产物即可覆盖全部 5 dtype，UT 在用例内
    #     按 ICPU_SET_TILING_KEY 切换 dtype。-DDTYPE_X=float 仅为占位保留，不影响分发。
    # param4：该算子依赖的所有 tiling 源码文件
    AddOpTestCase(fused_mul_add_n "ascend910b" "-DDTYPE_X=float" "${fused_mul_add_n_tiling_files}")
endif()
