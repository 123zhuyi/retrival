/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ragged_bin_count_tiling_check_support_arch35.cpp
 * \brief Register RaggedBinCount CheckSupport in the Ascend 950 tiling library.
 */

#include "../ragged_bin_count_check_support.h"

namespace ops {
static int g_RaggedBinCount_register_check_support = []() {
    optiling::OpCheckFuncHelper(FUNC_CHECK_SUPPORTED, "RaggedBinCount", CheckSupport4RaggedBinCount);
    return 0;
}();
} // namespace ops
