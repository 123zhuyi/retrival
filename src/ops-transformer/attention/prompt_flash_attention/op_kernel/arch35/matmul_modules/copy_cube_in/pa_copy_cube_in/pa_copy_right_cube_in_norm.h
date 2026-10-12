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
 * \file pa_copy_right_cube_in_norm.h
 * \brief
 */
#ifndef PA_COPY_RIGHT_CUBE_IN_NORM_H
#define PA_COPY_RIGHT_CUBE_IN_NORM_H

#include "../../pfa_policy_data.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class PACopyRightCubeInNorm {
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulContext);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;
public:
    using BASE_MODULE = AscendC::Impl::Detail::CopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE>;
    inline __aicore__ PACopyRightCubeInNorm() = default;
    inline __aicore__ ~PACopyRightCubeInNorm() = default;

    __aicore__ inline void Init() {
        baseK_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
        baseN_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth();
        orgKb_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgHeight();
        orgN_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
        totalCol_ = MATMUL_MODULE(CopyCubeInParams)->GetTotalCol();
        totalRow_ = MATMUL_MODULE(CopyCubeInParams)->GetTotalRow();
        int32_t cacheSize = INPUT_TYPE::TAG == InputTypeTag::A ?
                        MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetDepthA1() :
                        MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetDepthB1();
        int32_t reduceAxisCnt = INPUT_TYPE::TAG == InputTypeTag::A ? totalCol_ : totalRow_;
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            int32_t baseSize = baseK_ * baseN_;
            MATMUL_MODULE(CubeInBuffer)->Init(baseSize, cacheSize);
        }
    }

    __aicore__ inline void SetSplitCount(int32_t totalRow, int32_t totalCol) {}

    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx) {}

    __aicore__ inline void SetInput(const LocalTensor<SrcT>& localMatrix, bool isTranspose) {
        address_ = localMatrix.address_;
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            MATMUL_MODULE(CubeInBuffer)->Reset();
        }
        if constexpr (INPUT_TYPE::isTrans) {
            isTranspose_ = isTranspose;
        }
    }

    __aicore__ inline void SetInput(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose) {
        MATMUL_MODULE(MatmulTensorInfo)->template SetGlobalTensor<false>(globalMatrix, isTranspose);
        srcGlobalAddr_ = globalMatrix.address_;
        MATMUL_MODULE(CubeInBuffer)->Reset();
        if constexpr (INPUT_TYPE::isTrans) {
            isTranspose_ = isTranspose;
        }
    }

    __aicore__ inline void SetBatchNum(int32_t batchNum) {}

    template <class SrcT>
    static __aicore__ void CopyND2NZ(const LocalTensor<SrcT>& dst, const GlobalTensor<SrcT>& src, 
                                    const int row, const int col, const int height,
                                    const int width, const int gCol, const int ndNum = 1, 
                                    const int srcNdMatrixStride = 0, const int dstNzMatrixStride = 1, 
                                    const bool kAlignToC0Size = false, const int dstNzC0Stride = 0, 
                                    const bool isTranspose = false) 
    {
        int64_t srcOffset = (int64_t)row * (int64_t)gCol + (int64_t)col;
        int32_t alignNum = 16;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = ndNum;
        nd2nzParams.nValue = height;
        nd2nzParams.dValue = width;
        nd2nzParams.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzParams.srcDValue = gCol;
        if (kAlignToC0Size) {
            if constexpr (IsSameType<SrcT, int8_t>::value ||
                IsSameType<SrcT, fp8_e4m3fn_t>::value || IsSameType<SrcT, hifloat8_t>::value) {
                if (isTranspose) {
                    alignNum = 16; // 此处对齐方式需要与matmul中L1到L0A/B逻辑保持一致，matmul中tileHeight参数在转置时按照16对齐，非转置时按照c0size对齐。
                } else {
                    alignNum = 32; // 当数据类型时fp8/int8时，c0size=32；当数据类型为fp16/bf16时，c0size=16。
                }
            } else if constexpr (IsSameType<SrcT, float>::value) {
                alignNum = 8; // float32类型，32B对齐需要有8个数
            }
        }
        nd2nzParams.dstNzC0Stride = Ceil(dstNzC0Stride, alignNum) * alignNum;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = dstNzMatrixStride;
        DataCopy(dst, src[srcOffset], nd2nzParams);
    }

    __aicore__ inline LocalTensor<TransT>
    LoadData(int curRow, int curCol, int tileHeight, int tileWidth, int batchNum = 1)
    {
        PAFlagData flag = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
        int posL1 = this->GetIterIndex(curRow, curCol);
        MATMUL_MODULE(MatmulContext);
        LocalTensor<TransT> l1;
        uint32_t g_col = 0;
        if (flag.isbmm1) {
            g_col = MATMUL_MODULE(CopyCubeInParams)->GetOrgHeight();
        } else {
            g_col = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
        }
        int64_t s2BaseOffset = 0;
        if (flag.isbmm1) {
            // bmm1 s2对应N轴
            s2BaseOffset = curCol * MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth();
        } else {
            // bmm2 s2对应K轴
            s2BaseOffset = curRow * MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
        }
        int64_t s2AllOffset = flag.s2SingleOffset + s2BaseOffset;
        uint32_t colElementCnt = 16; // ND2NZ,32B为一个block,16个FP16数字
        int64_t copyFinishRowCnt = 0;
        uint64_t blockTableIdx = 0;
        uint64_t offsetInBlock = 0;
        int64_t blockRowOffsetInSingle = 0;
        uint32_t blockId = 0;
        int64_t copyRowCnt = 0;
        int64_t curOffset = 0;
        uint32_t baseRowOffsetInSingle = 0;
        uint32_t baseColOffsetInSingle = 0;
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            if (MATMUL_MODULE(CubeInBuffer)->Hit(posL1)) {
                l1 = MATMUL_MODULE(CubeInBuffer)->GetBuffer(posL1);
            } else {
                l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(posL1);
                // bmm1的时候，s2轴对应N轴,bmm2的时候s2对应K轴
                int64_t needCopyS2 = 0;
                if (flag.isbmm1) {
                    needCopyS2 = tileWidth;
                } else {
                    needCopyS2 = tileHeight;
                }
                while (copyFinishRowCnt < needCopyS2) {
                    blockTableIdx = s2AllOffset / flag.blockSize;
                    offsetInBlock = s2AllOffset % flag.blockSize;
                    blockRowOffsetInSingle = blockTableIdx * flag.blockSize - flag.s2SingleOffset;
                    blockId = *(reinterpret_cast<__gm__ int32_t*>(flag.blockTableAddr) + flag.bIdx * flag.blockTableDim2 + blockTableIdx);
                    copyRowCnt = flag.blockSize - offsetInBlock;
                    if (copyFinishRowCnt + copyRowCnt > needCopyS2) {
                        copyRowCnt = needCopyS2 - copyFinishRowCnt;
                    }
                    if (flag.isLayoutBSH == 1) {
                        curOffset = blockId * flag.blockSize * flag.kvHeadNum * flag.kvD + flag.nIdx * flag.kvD;
                    } else {
                        curOffset = blockId * flag.blockSize * flag.kvHeadNum * flag.kvD + flag.nIdx * flag.blockSize * flag.kvD;
                    }
                    GlobalTensor<SrcT> src;
                    src.SetGlobalBuffer((__gm__ SrcT *)flag.tensorBAddr, flag.paBlockNumSum * flag.blockSize * flag.kvHeadNum * flag.kvD);
                    if (flag.isbmm1) {
                        baseRowOffsetInSingle = curCol * MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth();
                        baseColOffsetInSingle = curRow * MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
                    } else {
                        baseRowOffsetInSingle = curRow * MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
                        baseColOffsetInSingle = curCol * MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth();
                    }
                    if (blockRowOffsetInSingle > baseRowOffsetInSingle) {
                        baseRowOffsetInSingle = 0;
                    } else {
                        baseRowOffsetInSingle -= blockRowOffsetInSingle;
                    }
                    if (flag.isbmm1) {
                        CopyND2NZ(l1[copyFinishRowCnt * colElementCnt], src[curOffset], baseRowOffsetInSingle,
                                    baseColOffsetInSingle, copyRowCnt, tileHeight, g_col, 1, 0, 1, true, tileWidth, true);
                    } else {
                        CopyND2NZ(l1[copyFinishRowCnt * colElementCnt], src[curOffset], baseRowOffsetInSingle,
                                    baseColOffsetInSingle, copyRowCnt, tileWidth, g_col, 1, 0, 1, true, tileHeight, false);
                    }
                    copyFinishRowCnt += copyRowCnt;
                    s2AllOffset += copyRowCnt;
                }
                MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
                MATMUL_MODULE(CubeInBuffer)->DeQue();
            }
        } else {
            l1.SetAddr(address_);
        }
        return l1;
    }
    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& l1Matrix = LocalTensor<TransT>{},
                                         int32_t curRow = 0, int32_t curCol = 0) 
    {
        if constexpr (!PhyPosIsUB(INPUT_TYPE::pos) && !PhyPosIsL1(INPUT_TYPE::pos)) {
            int posL1 = this->GetIterIndex(curRow, curCol);
            MATMUL_MODULE(CubeInBuffer)->FreeTensor(posL1, l1Matrix);
        }
    }
    __aicore__ inline void Destroy()
    {
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos))
        {
            MATMUL_MODULE(CubeInBuffer)->Destroy();
        }
    }
    __aicore__ inline void Reset() {
        MATMUL_MODULE(CubeInBuffer)->Reset();
    }
    __aicore__ inline uint64_t GetBufferHeadAddr()
    {
        return MATMUL_MODULE(CubeInBuffer)->GetBufferHeadAddr();
    }

private:
    int32_t baseK_{0};
    int32_t baseN_{0};
    int32_t orgN_{0};
    int32_t orgKb_{0};
    int32_t totalCol_{0};
    int32_t totalRow_{0};
    TBuffAddr address_{};
    bool isTranspose_{false};
    __gm__ SrcT* srcGlobalAddr_ = nullptr;

    __aicore__ inline int32_t GetIterIndex(int32_t curRow, int32_t curCol)
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            return curCol % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa();
        }
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            return curRow % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
        }
    }
};

}
}
}
#endif // PA_COPY_RIGHT_CUBE_IN_NORM_H