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
 * \file fag_cube_in_buffer.h
 * \brief
 */
#ifndef FAG_CUBE_IN_BUFFER_PRE_H
#define FAG_CUBE_IN_BUFFER_PRE_H

#include "../fag_flag_data.h"
namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG> class FagCubeInBufferS1S2Pre
{
    MATMUL_USE_MODULE(Context);
    using SRC_T = typename INPUT_TYPE::T;
    using SrcT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline FagCubeInBufferS1S2Pre() {}

    __aicore__ inline void Init(int32_t baseSize, int32_t cacheSize, int32_t reduceAxisCnt) {}

    __aicore__ inline void Destroy()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            if (MATMUL_MODULE(Context)->rightMatrixEncodingTableIdx == 1) {
                LocalTensor<SrcT> tmp;
                // release v
                uint8_t tscmIndex = MM1_ENCODING_TABLE[MATMUL_MODULE(Context)->rightMatrixEncodingTableIdx];
                tmp.SetAddr(gTscmArray[tscmIndex].srcAddr);
                gTscmArray[tscmIndex].localQue.FreeTensor(tmp);
            } else {
                return;
            }
        }
    }

    __aicore__ inline LocalTensor<SrcT> AllocTensor() {}

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<SrcT> &tensor = LocalTensor<SrcT>{}) {}

    __aicore__ inline void Reset() {}

    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1) { return false; }

    __aicore__ inline LocalTensor<SrcT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        return LocalTensor<SrcT>{};
    }

    __aicore__ inline void SetOrgAddr(__gm__ SrcT *gmAddr) {}

    __aicore__ inline void EnQue(LocalTensor<SrcT> &tensor, int32_t iterIndex = -1) {}

    __aicore__ inline void DeQue(int32_t iterIndex = -1) {}

    __aicore__ inline int32_t GetIterIndex(int32_t curRow, int32_t curCol) { return 0; }
};

}
}
}
#endif