// ----------------------------------------------------------------------------
// Copyright (c) Huawei Device Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#ifndef MASKED_SCALE_TILING_DATA_H
#define MASKED_SCALE_TILING_DATA_H

#include <cstdint>

struct MaskedScaleTilingData {
    uint32_t dim0;
    uint32_t coreNum;
    uint32_t blockFormer;
    uint32_t blockNum;
    uint32_t ubFormer;
    uint32_t selfDtype;
    uint32_t maskDtype;
    uint32_t branchKey;
    uint32_t subCaseKey;
    float scaleFloat;
    uint32_t bufferNum;

    void Init()
    {
        dim0 = coreNum = blockFormer = blockNum = ubFormer = 0U;
        selfDtype = maskDtype = branchKey = subCaseKey = 0U;
        scaleFloat = 0.0f;
        bufferNum = 0U;
    }
};

#endif // MASKED_SCALE_TILING_DATA_H
