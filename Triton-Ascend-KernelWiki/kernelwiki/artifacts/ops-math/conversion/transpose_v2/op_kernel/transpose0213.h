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
 * \file transpose0213.h
 * \brief
 */

#ifndef ASCEND_TRANSPOSE0213_H
#define ASCEND_TRANSPOSE0213_H

#include "transpose102.h"

namespace TransposeV2 {
template <typename T>
class Transpose0213 : public Transpose102<T, false>
{
public:
    __aicore__ inline Transpose0213(AscendC::TPipe* p) : Transpose102<T, false>(p)
    {}

    __aicore__ inline void Init0213(__gm__ uint8_t* src, __gm__ uint8_t* dst,
                                    const TransposeV2TilingData* __restrict__ tiling_data)
    {
        this->InitCommon(tiling_data);
        this->srcGlobal.SetGlobalBuffer(
            (__gm__ T*)src + this->startIdx * this->dim1Len * this->dim2Len * this->dim3Len);
        this->dstGlobal.SetGlobalBuffer(
            (__gm__ T*)dst + this->startIdx * this->dim1Len * this->dim2Len * this->dim3Len);
        this->dim1Once = this->dim1OnceMax;
        this->dim2Once = this->dim2OnceMax;
        this->dim1Loop = this->dim1Len / this->dim1OnceMax;
        this->dim1Tail = this->dim1Len % this->dim1OnceMax;
        this->dim2Loop = this->dim2Len / this->dim2OnceMax;
        this->dim2Tail = this->dim2Len % this->dim2OnceMax;
    }

    __aicore__ inline void Process0213CopyMode()
    {
        for (uint64_t i = 0; i < this->tasksPerCore; ++i) {
            if (this->subMode == 0) {
                this->dim1Once = this->dim1OnceMax;
                this->ProcessSubMode0(i);
            } else {
                this->dim2Once = this->dim2OnceMax;
                this->ProcessSubMode1(i);
            }
        }
    }
};
} // namespace TransposeV2
#endif