/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAST_EXPAND_H
#define CAST_EXPAND_H

#include "cast_base.h"

namespace AscendC {

// int8/uint8 -> {half, float, int32, int16} expansion path.
// Requires a half-width scratch buffer for 1-byte -> wide conversions.
template <typename T, typename U>
class CastExpand : public CastBase<T, U> {
    using Base = CastBase<T, U>;
    using Base::pipe;
    using Base::ubProcessNum;
    using Base::xGm;
    using Base::xLocal;
    using Base::xQue;
    using Base::yGm;
    using Base::yLocal;
    using Base::yQue;

public:
    __aicore__ inline CastExpand() {}
    __aicore__ inline CastExpand(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const CastTilingData& tiling)
    {
        this->InitParams(tiling);
        this->SetGmAddr(x, y, workspace);
        this->InitIoBuffers();
        if constexpr (!std::is_same_v<U, half>) {
            pipe.InitBuffer(halfBuf, ubProcessNum * sizeof(half));
            halfLocal = halfBuf.template Get<half>();
        }
    }

    __aicore__ inline void Process() { this->RunProcess(this); }

    __aicore__ inline void Compute(int32_t length)
    {
        xQue.template DeQue<T>(xLocal);
        yQue.template AllocTensor<U>(yLocal);
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            if constexpr (std::is_same_v<U, half>) {
                Cast(yLocal, xLocal, RoundMode::CAST_NONE, length);
            } else if constexpr (std::is_same_v<U, float>) {
                Cast(halfLocal, xLocal, RoundMode::CAST_NONE, length);
                Cast(yLocal, halfLocal, RoundMode::CAST_NONE, length);
            } else if constexpr (std::is_same_v<U, int32_t> || std::is_same_v<U, int16_t>) {
                Cast(halfLocal, xLocal, RoundMode::CAST_NONE, length);
                Cast(yLocal, halfLocal, RoundMode::CAST_RINT, length);
            }
        }
        xQue.template FreeTensor<T>(xLocal);
        yQue.template EnQue<U>(yLocal);
    }

private:
    TBuf<TPosition::VECCALC> halfBuf;
    LocalTensor<half> halfLocal;
};

} // namespace AscendC

#endif // CAST_EXPAND_H
