# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    set(grouped_matmul_finalize_routing_depends gmm/grouped_matmul CACHE INTERNAL "Dependencies for grouped_matmul_finalize_routing.")
    target_sources(op_host_aclnnExc PRIVATE
        grouped_matmul_finalize_routing_def.cpp
    )
    add_ops_compile_options(
        OP_NAME GroupedMatmulFinalizeRouting
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
    )
endif()
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE grouped_matmul_finalize_routing ACLNNTYPE aclnn_exclude)
endif()