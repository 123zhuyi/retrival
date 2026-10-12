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
 * \file fa_pa_mm2_copy_right.h
 * \brief
 */
#ifndef FA_PA_MM2_COPY_RIGHT_H
#define FA_PA_MM2_COPY_RIGHT_H

#include "../../pfa_policy_data.h"

constexpr int32_t ZERO_CUBE_RIGHT = 0;
constexpr int32_t ONE_CUBE_RIGHT = 1;
constexpr int32_t EIGHT_CUBE_RIGHT = 8;
constexpr int32_t SIXTEEN_CUBE_RIGHT = 16;
constexpr int32_t THIRTY_TWO_CUBE_RIGHT = 32;

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FaPaMm2CopyRight {
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulContext);
    MATMUL_USE_MODULE(MatmulShapeTiling);

    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

public:
    inline __aicore__ FaPaMm2CopyRight() = default;
    inline __aicore__ ~FaPaMm2CopyRight() = default;

    __aicore__ inline void Init()
    {
        if constexpr (IsSameType<SrcT, int8_t>::value ||
            IsSameType<SrcT, fp8_e4m3fn_t>::value || IsSameType<SrcT, hifloat8_t>::value) {
            alignNum_ = THIRTY_TWO_CUBE_RIGHT; // 此处对齐方式需要与matmul中L1到L0A/B逻辑保持一致,matmul中tileHeight参数在转置时按照16对齐,非转置时按照c0size对齐.
        } else if constexpr (IsSameType<SrcT, float>::value) {
            alignNum_ = EIGHT_CUBE_RIGHT; // float32类型，32B对齐需要有8个数
        } else {
            alignNum_ = SIXTEEN_CUBE_RIGHT; // FP16, BF16
        }
        nd2nzParams_.ndNum = ONE_CUBE_RIGHT;
        nd2nzParams_.srcNdMatrixStride = ZERO_CUBE_RIGHT;
        nd2nzParams_.dstNzNStride = ONE_CUBE_RIGHT;
        nd2nzParams_.dstNzMatrixStride = ONE_CUBE_RIGHT;
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

    __aicore__ inline void SetBatchNum(int32_t batchNum)
    {
    }

    __aicore__ inline LocalTensor<TransT> LoadData(int curRow, int curCol, int tileHeight, int tileWidth,
        int batchNum = 1)
    {
        FaPaPolicyData flag = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
        LocalTensor<TransT> l1;
        GlobalTensor<SrcT> src;
        // mm2 右矩阵 D/H对应width col, S2对应height row
        int64_t s2BaseOffset = curRow * MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
        int64_t s2AllOffset = flag.s2SingleOffset + s2BaseOffset;
        int64_t copyFinishRowCnt = 0;
        int64_t blockTableIdx = 0;
        int64_t offsetInBlock = 0;
        int64_t blockId = 0;
        int64_t copyRowCnt = 0;
        int64_t curOffset = 0;
        if (flag.splitD == 1) {
            l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(4 + (curCol & 1)); // l1内存基础偏移4
        } else {
            l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.rightBufIdx);
        }
        int64_t needCopyS2 = tileHeight;
        while (copyFinishRowCnt < needCopyS2) {
            blockTableIdx = s2AllOffset / flag.blockSize;
            offsetInBlock = s2AllOffset % flag.blockSize;
            blockId = *(reinterpret_cast<__gm__ int32_t *>(flag.blockTableAddr) + flag.bIdx * flag.blockTableDim2 + blockTableIdx);
            copyRowCnt = flag.blockSize - offsetInBlock; // the data can be moved in the current block.
            if (copyFinishRowCnt + copyRowCnt > needCopyS2) { // moving all the current block data will exceed the scope.
                copyRowCnt = needCopyS2 - copyFinishRowCnt;
            }
            if (flag.isLayoutBSH == 1) {
                curOffset = (blockId * flag.blockSize + offsetInBlock) * flag.kvHeadNum * flag.kvD +
                    flag.nIdx * flag.kvD;
            } else {
                curOffset = blockId * flag.blockSize * flag.kvHeadNum * flag.kvD +
                    flag.nIdx * flag.blockSize * flag.kvD + offsetInBlock * flag.kvD;
            }
            if (flag.splitD == 1) {
                curOffset += static_cast<int64_t>(curCol) * static_cast<int64_t>(MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth());
            }
            src.SetGlobalBuffer((__gm__ SrcT *)flag.tensorBAddr, flag.paBlockNumSum * flag.blockSize * flag.kvHeadNum * flag.kvD);
            nd2nzParams_.nValue = copyRowCnt;
            nd2nzParams_.dValue = tileWidth;
            nd2nzParams_.srcDValue = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
            nd2nzParams_.dstNzC0Stride = Ceil(tileHeight, alignNum_) * alignNum_;
            DataCopy(l1[copyFinishRowCnt * alignNum_], src[curOffset], nd2nzParams_);
            copyFinishRowCnt += copyRowCnt;
            s2AllOffset += copyRowCnt;
        }
        MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
        MATMUL_MODULE(CubeInBuffer)->DeQue();

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
    uint32_t alignNum_{16};
    TBuffAddr srcAddr_{};
    __gm__ SrcT *srcGlobalAddr_ = nullptr;
    Nd2NzParams nd2nzParams_{};
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // FA_PA_MM2_COPY_RIGHT_H