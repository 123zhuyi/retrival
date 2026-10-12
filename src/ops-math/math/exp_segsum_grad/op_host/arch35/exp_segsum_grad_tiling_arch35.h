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
 * \file exp_segsum_grad_tiling_arch35.h
 * \brief arch35 / Ascend950 tiling class for ExpSegsumGrad.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_EXP_SEGSUM_GRAD_TILING_ARCH35_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_EXP_SEGSUM_GRAD_TILING_ARCH35_H

#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "../../op_kernel/arch35/exp_segsum_grad_tiling_data.h"

namespace optiling {

// Empty marker struct used only for TilingParse registration. The arch35 tiling
// function reads platform information directly from gert::TilingContext.
struct ExpSegsumGradCompileInfoArch35 {};

class ExpSegsumGradTilingArch35 {
public:
    explicit ExpSegsumGradTilingArch35(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ParseInputAttrs();
    void GetNeedCoreNum(uint32_t coreNumPlatform);
    void GetTilingKey(uint64_t ubSizePlatform);
    uint8_t GetDataTypeSize() const;
    void FillTilingData(ExpSegsumGradTilingDataArch35* tiling);

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const
    {
        return (b != 0) ? ((a + b - 1) / b) : a;
    }

private:
    gert::TilingContext* tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;

    int64_t slideSize = 0;
    int64_t batches = 1;
    int64_t tailDimLength = 1;
    int32_t batchStart[EXP_SEGSUM_GRAD_MAX_CORE_ARCH35] = {0};
    int32_t batchEnd[EXP_SEGSUM_GRAD_MAX_CORE_ARCH35] = {0};
    uint32_t needCoreNum = 0;
    uint32_t tilingKey = 0;
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_EXP_SEGSUM_GRAD_TILING_ARCH35_H
