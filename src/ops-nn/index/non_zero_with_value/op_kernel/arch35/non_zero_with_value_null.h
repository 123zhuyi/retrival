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
 * \file non_zero_with_value_null.h
 * \brief 空张量(numel==0):count=0,value/index 不写。TilingKey 1000。
 */
#ifndef NON_ZERO_WITH_VALUE_NULL_H
#define NON_ZERO_WITH_VALUE_NULL_H

#include "kernel_operator.h"

namespace NonZeroWithValue {
using namespace AscendC;

template <typename TValue, typename TIndex>
class NonZeroWithValueNull {
public:
    __aicore__ inline NonZeroWithValueNull() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR index, GM_ADDR count, GM_ADDR workspace,
                                const NonZeroWithValueTilingData* tilingData)
    {
        (void)x;
        (void)value;
        (void)index;
        (void)workspace;
        (void)tilingData;
        countGm_.SetGlobalBuffer((__gm__ int32_t*)count, 1);
    }

    __aicore__ inline void Process()
    {
        // 仅 0 号核写 count=0
        if (GetBlockIdx() != 0) {
            return;
        }
        pipe_.InitBuffer(outQue_, 1, ONE_BLOCK_BYTES);
        LocalTensor<int32_t> cntLocal = outQue_.AllocTensor<int32_t>();
        cntLocal.SetValue(0, 0);
        outQue_.EnQue(cntLocal);
        LocalTensor<int32_t> cntOut = outQue_.DeQue<int32_t>();
        DataCopyExtParams cntParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(countGm_, cntOut, cntParams);
        outQue_.FreeTensor(cntOut);
    }

private:
    static constexpr int32_t ONE_BLOCK_BYTES = 32;
    TPipe pipe_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    GlobalTensor<int32_t> countGm_;
};
} // namespace NonZeroWithValue
#endif // NON_ZERO_WITH_VALUE_NULL_H
