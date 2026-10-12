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
 * \file fa_pa_mm1_copy_left.h
 * \brief
 */
#ifndef FA_PA_MM1_COPY_LEFT_H
#define FA_PA_MM1_COPY_LEFT_H

#include "../../pfa_policy_data.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FaPaMm1CopyLeft {
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    inline __aicore__ FaPaMm1CopyLeft() = default;
    inline __aicore__ ~FaPaMm1CopyLeft() = default;

    __aicore__ inline void Init()
    {
        nd2nzParams_.ndNum = 1;
        nd2nzParams_.srcNdMatrixStride = 0;
        nd2nzParams_.dstNzNStride = 1;
        nd2nzParams_.dstNzMatrixStride = 1;
    }

    __aicore__ inline void SetSplitCount(int32_t totalRow, int32_t totalCol)
    {
    }

    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx)
    {
    }

    __aicore__ inline void SetInput(const LocalTensor<SrcT> &localMatrix, bool isTranspose)
    {
        srcAddr_ = localMatrix.address_;
    }

    __aicore__ inline void SetInput(const GlobalTensor<SrcT> &globalMatrix, bool isTranspose)
    {
        MATMUL_MODULE(MatmulTensorInfo)->template SetGlobalTensor<false>(globalMatrix, isTranspose);
        srcGlobalAddr_ = globalMatrix.address_;
    }

    __aicore__ inline void SetBatchNum(int32_t batchA)
    {
    }

    __aicore__ inline LocalTensor<TransT> LoadData(int curRow, int curCol, int tileHeight, int tileWidth,
        int batchNum = -1)
    {
        LocalTensor<TransT> l1;
        GlobalTensor<TransT> aGlobal;
        FaPaPolicyData flag = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
        if (flag.splitD == 1) {
            l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(curCol & 1);
            aGlobal.SetGlobalBuffer(srcGlobalAddr_ + (curCol * MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth()));
            nd2nzParams_.nValue = tileHeight - flag.mOrNAdditionalSize;
            nd2nzParams_.dValue = tileWidth;
            nd2nzParams_.srcDValue = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
            nd2nzParams_.dstNzC0Stride = Align16Func(tileHeight);
            DataCopy(l1, aGlobal, nd2nzParams_);
            MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
            MATMUL_MODULE(CubeInBuffer)->DeQue();
        } else {
            if (!flag.reuseLeft) {
                l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.leftBufIdx);
                aGlobal.SetGlobalBuffer(srcGlobalAddr_);
                nd2nzParams_.nValue = tileHeight - flag.mOrNAdditionalSize;
                nd2nzParams_.dValue = tileWidth;
                nd2nzParams_.srcDValue = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
                nd2nzParams_.dstNzC0Stride = Align16Func(tileHeight);
                DataCopy(l1, aGlobal, nd2nzParams_);
                MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
                MATMUL_MODULE(CubeInBuffer)->DeQue();
            } else {
                l1 = MATMUL_MODULE(CubeInBuffer)->GetBuffer(flag.leftBufIdx);
            }
        }
        return l1;
    }
    __aicore__ inline void ClearLoadData(const LocalTensor<TransT> &l1Matrix = LocalTensor<TransT>{}, int32_t curRow = 0,
        int32_t curCol = 0)
    {
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            MATMUL_MODULE(CubeInBuffer)->Reset();
        }
    }
    __aicore__ inline void Destroy()
    {
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            MATMUL_MODULE(CubeInBuffer)->Destroy();
        }
    }
    __aicore__ inline void Reset()
    {
        MATMUL_MODULE(CubeInBuffer)->Reset();
    }
    __aicore__ inline uint64_t GetBufferHeadAddr()
    {
        return MATMUL_MODULE(CubeInBuffer)->GetBufferHeadAddr();
    }

private:
    TBuffAddr srcAddr_{};
    __gm__ SrcT *srcGlobalAddr_ = nullptr;
    Nd2NzParams nd2nzParams_{};
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // FA_PA_MM1_COPY_LEFT_H