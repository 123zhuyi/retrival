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
 * \file tensor_redirect_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_TENSOR_REDIRECT_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_TENSOR_REDIRECT_H_

#include <cstdint>

namespace optiling {

// TilingData 为 plain struct，定义在 tiling_data.h

struct TensorRedirectTilingParam {
    int64_t totalCoreNum = 0;
    int64_t ubSize = 0;
    int64_t uo = 0; // 总循环块数
    int64_t usedCoreNum = 0;
    int64_t bytesForOneData = 0; // 元素字节宽 ∈ {1,2,4,8}
    int64_t ubFactor = 0;
    int64_t tailBlockTailUbFactor = 0;
    int64_t blockFactor = 0;
    int64_t tailBlockFactor = 0;
};

struct TensorRedirectCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t libApiWorkspaceSize = 0;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_TENSOR_REDIRECT_H_
