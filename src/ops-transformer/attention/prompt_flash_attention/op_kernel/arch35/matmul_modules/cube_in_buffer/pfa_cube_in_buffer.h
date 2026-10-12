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
 * \file pfa_cube_in_buffer.h
 * \brief
 */
#ifndef PFA_CUBE_IN_BUFFER_H
#define PFA_CUBE_IN_BUFFER_H

#include "../pfa_policy_data.h"

constexpr int32_t ALIGN_16_MASK = 0xFFFFFFF0;
constexpr int32_t ALIGN_16_OFFSET = 15; // 向上16对齐

__aicore__ inline int32_t Align16Func(int32_t data) {
    return (data + ALIGN_16_OFFSET) & ALIGN_16_MASK;
}

namespace AscendC {
namespace Impl {
namespace Detail {
template<typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class PFACubeInBuffer {
    using SrcT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline PFACubeInBuffer() {}
    __aicore__ inline ~PFACubeInBuffer() {}

    __aicore__ inline void Init(const MatmulTiling<MM_CFG>& cubeTiling, int32_t baseSize, int32_t cacheSize,
                                int32_t reduceAxisCnt) {}

    __aicore__ inline void Destroy() {
        tscmGlobalPFA->localScm[tscmIndex_].FreeTensor(cacheHead_);
    }

    __aicore__ inline LocalTensor<SrcT> AllocTensor(int32_t iterIndex) {
        cacheHead_ = tscmGlobalPFA->localScm[iterIndex].template AllocTensor<SrcT>();
        tscmIndex_ = iterIndex;
        return cacheHead_;
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<SrcT>& tensor = LocalTensor<SrcT>{}) {}

    __aicore__ inline void Reset() {
        tscmGlobalPFA->localScm[tscmIndex_].FreeTensor(cacheHead_);
    }

    __aicore__ inline LocalTensor<SrcT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1) {
        return cacheHead_;
    }

    __aicore__ inline void EnQue(LocalTensor<SrcT>& tensor) {
        (void)tscmGlobalPFA->localScm[tscmIndex_].EnQue(tensor);
    }

    __aicore__ inline void DeQue() {
        (void)tscmGlobalPFA->localScm[tscmIndex_].DeQue();
    }

    __aicore__ inline void SetOrgAddr(__gm__ SrcT* gmAddr) {}

    __aicore__ inline uint64_t GetBufferHeadAddr()
    {
        return GetTQueHeadAddr(tscmGlobalPFA->localScm[tscmIndex_]);
    }

private:
    LocalTensor<SrcT> cacheHead_;
    int32_t tscmIndex_;
};

}
}
}
#endif // PFA_CUBE_IN_BUFFER_H