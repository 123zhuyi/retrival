// ----------------------------------------------------------------------------
// Copyright (c) Huawei Device Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include "arch35/masked_scale_kernel.h"

using namespace NsMaskedScale;

namespace {
template <int dtypeKey>
struct MaskedScaleDtype;

template <>
struct MaskedScaleDtype<MASKED_SCALE_TPL_FP16> {
    using Type = half;
};

template <>
struct MaskedScaleDtype<MASKED_SCALE_TPL_FP32> {
    using Type = float;
};

template <>
struct MaskedScaleDtype<MASKED_SCALE_TPL_BF16> {
    using Type = bfloat16_t;
};

template <>
struct MaskedScaleDtype<MASKED_SCALE_TPL_UINT8> {
    using Type = uint8_t;
};

template <>
struct MaskedScaleDtype<MASKED_SCALE_TPL_INT8> {
    using Type = int8_t;
};
} // namespace

template <int TYPE_X, int TYPE_MASK>
__global__ __aicore__ void masked_scale(GM_ADDR self, GM_ADDR mask, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace != nullptr) {
        AscendC::SetSysWorkspace(workspace);
    }
    REGISTER_TILING_DEFAULT(MaskedScaleTilingData);
    GET_TILING_DATA_WITH_STRUCT(MaskedScaleTilingData, tilingData, tiling);
    using SelfT = typename MaskedScaleDtype<TYPE_X>::Type;
    using MaskT = typename MaskedScaleDtype<TYPE_MASK>::Type;
    MaskedScale<SelfT, MaskT> op;
    op.Init(self, mask, y, &tilingData);
    op.Process();
}
