/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_OFFSETS = 1;
static constexpr size_t INPUT_IDX_SIZE = 2;
static constexpr size_t OUTPUT_IDX_Y = 0;

static ge::graphStatus Infershape4SliceWithAxes(gert::InferShapeContext* context)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto axesList = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, axesList);

    auto sizeTensor = context->GetInputTensor(INPUT_IDX_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeTensor);

    size_t dimNum = xShape->GetDimNum();
    yShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }

    size_t axesLen = axesList->GetSize();
    const int64_t* axesData = axesList->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, axesData);
    DataType sizeDataType = sizeTensor->GetDataType();

    for (size_t i = 0; i < axesLen; ++i) {
        int64_t axis = axesData[i];
        int64_t sliceSize = 0;
        if (sizeDataType == ge::DT_INT32) {
            const int32_t* data = sizeTensor->GetData<int32_t>();
            OP_CHECK_NULL_WITH_CONTEXT(context, data);
            sliceSize = static_cast<int64_t>(data[i]);
        } else {
            const int64_t* data = sizeTensor->GetData<int64_t>();
            OP_CHECK_NULL_WITH_CONTEXT(context, data);
            sliceSize = data[i];
        }
        if (sliceSize == -1) {
            int64_t offsetVal = 0;
            auto offsetTensor = context->GetInputTensor(INPUT_IDX_OFFSETS);
            if (offsetTensor != nullptr) {
                if (sizeDataType == ge::DT_INT32) {
                    const int32_t* oData = offsetTensor->GetData<int32_t>();
                    if (oData != nullptr) {
                        offsetVal = static_cast<int64_t>(oData[i]);
                    }
                } else {
                    const int64_t* oData = offsetTensor->GetData<int64_t>();
                    if (oData != nullptr) {
                        offsetVal = oData[i];
                    }
                }
            }
            int64_t dimSize = xShape->GetDim(axis);
            sliceSize = (dimSize >= 0) ? (dimSize - offsetVal) : -1;
        }
        yShape->SetDim(axis, sliceSize);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SliceWithAxes)
    .InferShape(Infershape4SliceWithAxes)
    .InputsDataDependency({INPUT_IDX_OFFSETS, INPUT_IDX_SIZE});
} // namespace ops
