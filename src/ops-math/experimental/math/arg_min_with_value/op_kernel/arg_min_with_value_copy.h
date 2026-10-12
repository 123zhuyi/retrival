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
 * \file arg_min_with_value_copy.h
 * \brief COPY pattern: the reduce axis has size 1, so every output equals its single input element and
 *        every index is 0. Pure data movement — load a tile of input, write it straight to values, and
 *        write a zero-filled index tile. No reduce, no fp32 cast.
 */
#ifndef ARG_MIN_WITH_VALUE_COPY_H
#define ARG_MIN_WITH_VALUE_COPY_H

#include "arg_min_with_value_base.h"

namespace ArgWithValueNs {
using namespace AscendC;

template <typename T, bool IS_MIN>
class ArgCopy : public ArgBase<T, IS_MIN> {
public:
    __aicore__ inline ArgCopy() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values,
                                __tiling_data_ptr__ ArgMinWithValueTilingData* t)
    {
        this->InitBase(x, indice, values, t);
        tile_ = t->rowTile;
        if (tile_ == 0)
            tile_ = 1;
        indexAddr_ = this->RoundUp(tile_ * sizeof(T), 32u);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t done = 0; done < this->oLen_; done += tile_) {
            uint32_t n = (this->oLen_ - done) < tile_ ? (this->oLen_ - done) : tile_;
            uint32_t off = this->oStart_ + done;
            __ubuf__ T* xUb = reinterpret_cast<__ubuf__ T*>(LocalTensor<T>(TPosition::VECCALC, 0, tile_).GetPhyAddr());
            __ubuf__ int32_t* indexUb = reinterpret_cast<__ubuf__ int32_t*>(
                LocalTensor<int32_t>(TPosition::VECCALC, indexAddr_, this->RoundUp(tile_, 8u)).GetPhyAddr());
            __gm__ T* xGm = this->xGm_ + off;
            __gm__ T* valueGm = this->valuesGm_ + off;
            __gm__ int32_t* indexGm = this->indiceGm_ + off;
            if constexpr (sizeof(T) == 2u) {
                copy_gm_to_ubuf_align_b16(xUb, xGm, 0, 1, n * sizeof(T), 0, 0, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b32(xUb, xGm, 0, 1, n * sizeof(T), 0, 0, 0, 0);
            }
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            if constexpr (sizeof(T) == 2u) {
                copy_ubuf_to_gm_align_b16(valueGm, xUb, 0, 1, n * sizeof(T), 0, 0, 0, 0);
            } else {
                copy_ubuf_to_gm_align_b32(valueGm, xUb, 0, 1, n * sizeof(T), 0, 0, 0, 0);
            }
            set_mask_count();
            set_vector_mask(0, n);
            vector_dup(indexUb, static_cast<int32_t>(0), 1, 1, 1, 8, 0);
            set_mask_norm();
            set_vector_mask(~0ULL, ~0ULL);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm_align_b32(indexGm, indexUb, 0, 1, n * sizeof(int32_t), 0, 0, 0, 0);
            pipe_barrier(PIPE_MTE3);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }

private:
    uint32_t tile_ = 1;
    uint32_t indexAddr_ = 0;
};
} // namespace ArgWithValueNs
#endif // ARG_MIN_WITH_VALUE_COPY_H
