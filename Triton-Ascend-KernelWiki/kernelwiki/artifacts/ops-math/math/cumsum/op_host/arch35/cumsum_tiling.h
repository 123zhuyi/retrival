/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cumsum.h
 * \brief calc corenum and threadnum for AscendC kernel
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CUMSUM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CUMSUM_H_
#include <cstdint>

namespace optiling {
struct CumsumCompileInfo {
    int64_t core_num = 0;
    int64_t ub_size = 0;
    int64_t support_atomic = 0;
    int64_t isSupportTransposeMode = 0;
    int64_t exclusive = 0;
    int64_t reverse = 0;
    int64_t maxLastDimNum = 0;
    int64_t blockSize = 0;
    uint32_t clSize = 0;
    uint32_t vRegSize = 0;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CUMSUM_H_
