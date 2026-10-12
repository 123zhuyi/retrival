/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GET_SHAPE_OP_KERNEL_ARCH35_GET_SHAPE_H_
#define OPS_GET_SHAPE_OP_KERNEL_ARCH35_GET_SHAPE_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "get_shape_tiling_data.h"

namespace NsGetShape {

using namespace AscendC;

class KernelGetShape {
public:
    __aicore__ inline KernelGetShape() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const GetShapeTilingData& tilingData);

    __aicore__ inline void Process();

private:
    TPipe pipe_;
    TQue<QuePosition::VECOUT, GetShapeConst::BUFFER_NUM> outQueue_;
    GlobalTensor<int32_t> yGm_;
    ListTensorDesc inputList_;
    uint32_t inputNum_ = 0;
    int64_t totalDimNum_ = 0;
    int32_t shapeValues_[GetShapeConst::MAX_TOTAL_DIM] = {0};
};

__aicore__ inline void KernelGetShape::Init(GM_ADDR x, GM_ADDR y, const GetShapeTilingData& tilingData)
{
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));

    pipe_.InitBuffer(outQueue_, GetShapeConst::BUFFER_NUM, GetShapeConst::MAX_TOTAL_DIM * sizeof(int32_t));

    inputNum_ = static_cast<uint32_t>(tilingData.inputNum);
    if (inputNum_ == 0) {
        return;
    }

    totalDimNum_ = 0;
    for (uint32_t i = 0; i < inputNum_; ++i) {
        __gm__ int64_t* xGmPtr = inputList_.GetDataPtr<int64_t>(i);
        GlobalTensor<int64_t> xGm;
        xGm.SetGlobalBuffer(xGmPtr);

        int64_t dimNum = xGm.GetValue(3);
        if (dimNum < 0 || dimNum > GetShapeConst::MAX_DIM_PER_TENSOR) {
            inputNum_ = 0;
            totalDimNum_ = 0;
            return;
        }
        if (totalDimNum_ + dimNum > GetShapeConst::MAX_TOTAL_DIM) {
            inputNum_ = 0;
            totalDimNum_ = 0;
            return;
        }
        for (int64_t d = 0; d < dimNum; ++d) {
            shapeValues_[totalDimNum_ + d] = static_cast<int32_t>(xGm.GetValue(4 + d));
        }
        totalDimNum_ += dimNum;
    }
}

__aicore__ inline void KernelGetShape::Process()
{
    if (totalDimNum_ <= 0 || totalDimNum_ > GetShapeConst::MAX_TOTAL_DIM) {
        return;
    }

    LocalTensor<int32_t> yUb = outQueue_.AllocTensor<int32_t>();

    for (int64_t i = 0; i < totalDimNum_; ++i) {
        yUb.SetValue(i, shapeValues_[i]);
    }

    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evt);
    WaitFlag<HardEvent::S_MTE3>(evt);
    GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE3>(evt);

    outQueue_.EnQue(yUb);
    LocalTensor<int32_t> yOut = outQueue_.DeQue<int32_t>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(totalDimNum_ * sizeof(int32_t)), 0,
                                    0, 0};
    DataCopyPad(yGm_, yOut, copyParams);

    outQueue_.FreeTensor(yOut);
}

} // namespace NsGetShape

#endif // OPS_GET_SHAPE_OP_KERNEL_ARCH35_GET_SHAPE_H_
