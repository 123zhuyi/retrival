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
 * \file strided_slice_grad_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDEDSLICEGRAD_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDEDSLICEGRAD_H_

#include <cstdint>
#include <vector>
#include <string>
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "log/log.h"
#include "op_api/op_util.h"
#include "conversion/strided_slice/op_host/strided_slice_util.h"

namespace optiling
{
constexpr int64_t TILING_ARRAY_LEN_EIGHT = 8;

BEGIN_TILING_DATA_DEF(StridedSliceGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumForClear);           // 清零使用核数
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNumForClear);  // 清零物理总核数
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNumForClear);    // 清零物理总核数
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);          // 非尾核的元素个数
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);            // 尾核的元素个数
TILING_DATA_FIELD_DEF(int64_t, tailAxisOuter);                 // 尾轴的次数
TILING_DATA_FIELD_DEF(int64_t, tailAxisInner);                 // 尾轴的非尾loop
TILING_DATA_FIELD_DEF(int64_t, tailAxisTail);                  // 尾轴的尾loop
TILING_DATA_FIELD_DEF(int64_t, inputDimNum);                   // 输入的元素个数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);                   // 使用核数
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);                  // 物理总核数
TILING_DATA_FIELD_DEF(int64_t, bufferSize);                    // ub中可用buffer大小
TILING_DATA_FIELD_DEF(int64_t, splitUbAxisNum);
TILING_DATA_FIELD_DEF(int64_t, bytesForOneData);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);                                          // tilingKey的大小
TILING_DATA_FIELD_DEF(int64_t, workspaceSize);                                      // 总workspace的大小
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, begin);                  // 输入begin的vector
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, end);                    // 输入end的vector
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, strides);                // 输入strides的vector
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, inputShape);             // 输入dyShape的vector
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, outputShape);            // 输出inShape的vector
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, fusedOutputInnerShape);  // 输出inshape的融合轴vector
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, fusedSliceInnerShape);   // 输入dyShape的融合轴vector
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(StridedSliceGrad, StridedSliceGradTilingData)

struct StridedSliceGradParamList {
    ge::DataType IndexDtype;
    ge::DataType dtype;
    std::vector<int64_t> fusedInShape;
    std::vector<int64_t> fusedDyShape;
    std::vector<int64_t> inShapeVec;
    std::vector<int64_t> beginVec;
    std::vector<int64_t> endVec;
    std::vector<int64_t> stridesVec;
    std::vector<int64_t> dyShapeVec;

    gert::Shape inShape;
    gert::Shape begin;
    gert::Shape end;
    gert::Shape strides;
    gert::Shape dyShape;
    int64_t beginMask;
    int64_t endMask;
    int64_t ellipsisMask;
    int64_t newAxisMask;
    int64_t shrinkAxisMask;
    bool beginValid;
    bool endValid;
    bool strideValid;

    int64_t tailAxisOuter;
    int64_t tailAxisInner;
    int64_t tailAxisTail;
    int64_t inputDimNum;
    int64_t usedCoreNumForClear;
    int64_t normalCoreProcessNumForClear;
    int64_t tailCoreProcessNumForClear;
    int64_t splitUbAxisNum;
    int64_t u_o;
    int64_t caluMode;
    size_t realUbDim;
    int64_t ubFactor;
    int64_t fusedBlockDims;
    int64_t blockFactor;
    int64_t bufferSize;
    int64_t bytesForOneData;
    int64_t maxUbAvailable;
    int64_t inShapeOutSize;
    int64_t dyShapeOutSize;
    int64_t dyTailAxisLen;
    int64_t dyFusedFront;
    int64_t inTailAxisLen;
    int64_t inFusedFront;
    int64_t normalCoreProcessNum;
    int64_t tailCoreProcessNum;
    int64_t usedCoreNum;
    int64_t totalCoreNum;
    int64_t hardwareUbSize;
    int64_t workspaceSize;
    int64_t tilingKey;
};

struct StridedSliceGradCompileInfo {
  int64_t coreNum = 0;
  int64_t ubSize = 0;
};

struct SliceParametersRuntime {
    gert::Shape inputShape;
    gert::Shape outputShape;
    ops::QuickVector beginList;
    ops::QuickVector endList;
    ops::QuickVector strideList;

    std::string to_string() const
    {
        std::string result = "inputShape:" + Ops::Base::ToString(inputShape);
        result += " outputShape:" + Ops::Base::ToString(outputShape);
        result += " beginList:" + Ops::Base::ToString(beginList);
        result += " endList:" + Ops::Base::ToString(endList);
        result += " strideList:" + Ops::Base::ToString(strideList);
        return result;
    }
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDEDSLICEGRAD_H_