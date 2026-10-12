# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# Operator-root source wiring:
# add_all_modules_sources globs op_host (def/infershape/tiling), op_api (L0 sort_with_index.cpp +
# L2 aclnn_sort_with_index.cpp), op_graph (proto + graph_infer) under THIS operator root, plus the
# UT cases under tests/ut/{op_host,op_api}.
#
# ACLNNTYPE=aclnn_exclude: the L2 wrapper aclnn_sort_with_index.cpp is HAND-WRITTEN, so its public
# signature matches docs/aclnnSortWithIndex.md. aclnn_exclude tells the build NOT to auto-generate
# an aclnn_*.cpp from the op definition (which would collide with the hand-written one), while still
# packaging the public header + cust_opapi lib.
add_all_modules_sources(OPTYPE sort_with_index ACLNNTYPE aclnn_exclude)
