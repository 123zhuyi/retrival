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
 * \file as_strided_merge_axis_tiling_arch35.h
 * \brief as_strided_merge_axis_tiling_arch35
 */

#include "as_strided_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/const_util.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
constexpr size_t IN_X = 0;
constexpr size_t IN_SIZE = 1;
constexpr size_t IN_STRIDE = 2;
constexpr size_t IN_OFFSET = 3;

static bool GetSizeAndStride(gert::TilingContext* context, AsStridedRunInfo& runInfo) {
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context, IN_SIZE, runInfo.outputSize),
                    OP_LOGE(context, "get const of size failed"), return false);
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context, IN_STRIDE, runInfo.outputStride),
                    OP_LOGE(context, "get const of stride failed"), return false);

    OP_CHECK_IF(runInfo.outputSize.GetDimNum() != runInfo.outputStride.GetDimNum(),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context->GetNodeName(), "output_size, output_stride",
                        (std::to_string(runInfo.outputSize.GetDimNum()) + ", " + 
                         std::to_string(runInfo.outputStride.GetDimNum())).c_str(),
                        "The shape dimensions of output_size and output_stride must be the same."),
                    return false);

    const int64_t oriSizeLen = runInfo.outputSize.GetDimNum();
    OP_CHECK_IF(
        oriSizeLen == 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "output_size",
            std::to_string(oriSizeLen).c_str(),
            "The shape dim of output_size must be greater than 0."),
        return false);

    return true;
}

static void MergeAxisRule1(AsStridedRunInfo& runInfo) {
    // merge rule 1: delete axis which size is 1
    const int64_t oriSizeLen = runInfo.outputSize.GetDimNum();
    size_t sizeIdx = 0;
    for (int64_t i = 0; i < oriSizeLen; i++) {
        if (runInfo.outputSize.GetDim(i) != 1) {
            runInfo.outputSize.SetDim(sizeIdx, runInfo.outputSize.GetDim(i));
            runInfo.outputStride.SetDim(sizeIdx, runInfo.outputStride.GetDim(i));
            sizeIdx += 1UL;
        }
    }

    runInfo.outputSize.SetDimNum(sizeIdx);
    runInfo.outputStride.SetDimNum(sizeIdx);
}

static void MergeAxisRule2(AsStridedRunInfo& runInfo) {
    // merge rule 2: merge the axes with continuous stride by 0
    const size_t sizeIdx = runInfo.outputSize.GetDimNum();
    size_t cuMergeSizeIdx = 0;
    for (size_t i = 1; i < sizeIdx; i++) {
        const int64_t strideValue = runInfo.outputStride.GetDim(i);
        const int64_t previousStrideValue = runInfo.outputStride.GetDim(cuMergeSizeIdx);
        if (strideValue == 0 && previousStrideValue == 0) {
            runInfo.outputSize.SetDim(cuMergeSizeIdx,
                                    runInfo.outputSize.GetDim(cuMergeSizeIdx) * runInfo.outputSize.GetDim(i));
            continue;
        }

        cuMergeSizeIdx += 1UL;
        runInfo.outputSize.SetDim(cuMergeSizeIdx, runInfo.outputSize.GetDim(i));
        runInfo.outputStride.SetDim(cuMergeSizeIdx, runInfo.outputStride.GetDim(i));
    }
    runInfo.outputSize.SetDimNum(cuMergeSizeIdx + 1UL);
    runInfo.outputStride.SetDimNum(cuMergeSizeIdx + 1UL);
}

static void MergeAxisRule3(AsStridedRunInfo& runInfo) {
    // merge rule 3: merge dims that is continuous stride except last dim
    const size_t outDimNum = runInfo.outputSize.GetDimNum();

    const size_t cuMergeSizeIdx = outDimNum - 1;
    int64_t lastDimMergeValue =
        (runInfo.outputStride.GetDim(cuMergeSizeIdx) * runInfo.outputSize.GetDim(cuMergeSizeIdx));
    int64_t lastDimMergeSize = runInfo.outputSize.GetDim(cuMergeSizeIdx);
    int64_t lastDimMergeStride = runInfo.outputStride.GetDim(cuMergeSizeIdx);
    size_t lastDimMergeDim = cuMergeSizeIdx;
    for (size_t i = 0; i < cuMergeSizeIdx; i++) {
        if (lastDimMergeValue != runInfo.outputStride.GetDim(cuMergeSizeIdx - 1UL - i)) {
            runInfo.outputSize.SetDim(lastDimMergeDim, lastDimMergeSize);
            runInfo.outputStride.SetDim(lastDimMergeDim, lastDimMergeStride);
            lastDimMergeDim--;
            lastDimMergeValue =
                (runInfo.outputStride.GetDim(cuMergeSizeIdx - 1 - i) * runInfo.outputSize.GetDim(cuMergeSizeIdx - 1 - i));
            lastDimMergeSize = runInfo.outputSize.GetDim(cuMergeSizeIdx - 1UL - i);
            lastDimMergeStride = runInfo.outputStride.GetDim(cuMergeSizeIdx - 1UL - i);
            continue;
        }
        lastDimMergeValue *= runInfo.outputSize.GetDim(cuMergeSizeIdx - 1UL - i);
        lastDimMergeSize *= runInfo.outputSize.GetDim(cuMergeSizeIdx - 1UL - i);
    }
    runInfo.outputSize.SetDim(lastDimMergeDim, lastDimMergeSize);
    runInfo.outputStride.SetDim(lastDimMergeDim, lastDimMergeStride);

    if (lastDimMergeDim != 0UL) {
        for (size_t i = lastDimMergeDim, j = 0; i < outDimNum; i++, j++) {
            runInfo.outputSize[j] = runInfo.outputSize[i];
            runInfo.outputStride[j] = runInfo.outputStride[i];
        }
        runInfo.outputSize.SetDimNum(outDimNum - lastDimMergeDim);
        runInfo.outputStride.SetDimNum(outDimNum - lastDimMergeDim);
    }
}

static void MergeAxis(AsStridedRunInfo& runInfo) {
    // merge rule 1: delete axis which size is 1
    MergeAxisRule1(runInfo);

    // 规则1会将全1的shape，如[1, 1, 1]合为空的[]，但规则2能够恢复一维[1]
    // merge rule 2: merge the axes with continuous stride by 0
    /***
    *** stride[0, 0, 1, 0, 0] -> stride[0, 1, 0]
    *** size[1, 2, 3, 4, 5] -> size[2, 3, 20]
    ***/
    MergeAxisRule2(runInfo);

    // merge rule 3: merge dims that is continuous stride
    /***
    *** stride[2, 18, 6, 3] -> stride[2, 3]
    *** size[3, 4, 3, 2] -> size[3, 24]

    *** stride[2, 18, 6, 3, 1] -> stride[2, 3, 1]
    *** size[3, 4, 3, 2, 10] -> size[3, 24, 10]
    ***/
    MergeAxisRule3(runInfo);
}
} // namespace