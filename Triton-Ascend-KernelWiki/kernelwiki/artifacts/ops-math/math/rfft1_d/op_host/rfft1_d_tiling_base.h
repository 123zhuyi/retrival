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
 * \file rfft1d_tiling_base.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_RFFT1D_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_RFFT1D_BASE_H_

#include "rfft1_d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Rfft1DTilingData)
TILING_DATA_FIELD_DEF(int32_t, length);
TILING_DATA_FIELD_DEF(uint8_t, isBluestein);
TILING_DATA_FIELD_DEF(int32_t, lengthPad);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, factors);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, prevRadices);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, nextRadices);
TILING_DATA_FIELD_DEF_ARR(uint8_t, 3, prevRadicesAlign);
TILING_DATA_FIELD_DEF(int32_t, outLength);
TILING_DATA_FIELD_DEF(uint32_t, tailSize);

TILING_DATA_FIELD_DEF(uint32_t, batchesPerCore);
TILING_DATA_FIELD_DEF(uint32_t, leftOverBatches);

TILING_DATA_FIELD_DEF(uint32_t, tmpLenPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, tmpSizePerBatch);
TILING_DATA_FIELD_DEF(uint32_t, matmulTmpsLen);
TILING_DATA_FIELD_DEF(uint32_t, matmulTmpsSize);

TILING_DATA_FIELD_DEF(int32_t, normal);

TILING_DATA_FIELD_DEF(uint32_t, dftRealOverallSize);
TILING_DATA_FIELD_DEF(uint32_t, dftImagOverallSize);
TILING_DATA_FIELD_DEF(uint32_t, twiddleOverallSize);
TILING_DATA_FIELD_DEF(uint32_t, fftMatrOverallSize);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, dftRealOffsets);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, dftImagOffsets);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, twiddleOffsets);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Rfft1D, Rfft1DTilingData);

enum class NORM_VALUES : int
{
    BACKWARD = 1,
    FORWARD = 2,
    ORTHO = 3
};

using Ops::Base::TilingBaseClass;

class Rfft1DBaseTiling : public TilingBaseClass {
public:
    explicit Rfft1DBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;

protected:
    uint32_t ubSize;
    uint32_t coreNum;
    uint32_t batches;
    int64_t length;
    int64_t normal;
    ge::DataType dtype;
};
} // namespace optiling

#endif
