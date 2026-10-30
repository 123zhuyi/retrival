/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file grouped_bias_add_grad_tiling_arch35.h
\brief tiling for grouped_bias_add_grad arch35
*/
#ifndef GROUPED_BIAS_ADD_GRAD_TILING_ARCH35_H
#define GROUPED_BIAS_ADD_GRAD_TILING_ARCH35_H

#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

// Constants for arch35 tiling
constexpr int64_t ALIGN_32_BYTE = 32;
constexpr int64_t BUFFER_NUM_ARCH35 = 4;   // 4 buffer
constexpr int64_t GRAD_Y_INPUT_IDX = 0;
constexpr int64_t GROUP_IDX_INPUT_IDX = 1;
constexpr int64_t GRAD_BIAS_OUTPUT_IDX = 0;
constexpr int64_t ATTR_GROUP_IDX_TYPE_IDX = 0;
constexpr int64_t ARA_DIM_NUM = 3;
constexpr int64_t RA_DIM_NUM = 2;
constexpr int64_t QUE_NUM = 2;
constexpr int64_t USE_TEMP_CACHELINE_NUM = 4;
constexpr int64_t INPUT_MAX_GROUP = 2048;
constexpr int64_t TEMP_BUF_SIZE = 20 * 1024;
constexpr int64_t MAX_OUT_SIZE = 8192;
constexpr int64_t EMPTY_WORKSPACE_SIZE = 4096;
constexpr int64_t CACHELINE_DEFINE = 512;

// Compile info for arch35
struct GroupedBiasAddGradCompileInfoArch35 {
    uint32_t coreNum;
    uint32_t ubSize;
    uint32_t blockSize;
    uint32_t clSize;
    uint32_t vRegSize;
};

} // namespace optiling

#endif // GROUPED_BIAS_ADD_GRAD_TILING_ARCH35_H