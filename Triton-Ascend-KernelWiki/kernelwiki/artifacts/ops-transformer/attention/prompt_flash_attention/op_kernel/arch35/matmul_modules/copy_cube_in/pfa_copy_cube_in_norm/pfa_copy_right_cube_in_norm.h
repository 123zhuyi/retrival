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
 * \file pfa_norm_copy_right_cube_in.h
 * \brief
 */

#ifndef PFA_NORM_COPY_RIGHT_CUBE_H
#define PFA_NORM_COPY_RIGHT_CUBE_H

#include "../../pfa_policy_data.h"
#include "../../cube_in_buffer/pfa_cube_in_buffer.h"
namespace AscendC {
namespace Impl {
namespace Detail {
template<typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class PFACopyRightCubeIn {
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;
public:
    using InputType = INPUT_TYPE;
    __aicore__ inline PFACopyRightCubeIn() = default;
    __aicore__ inline ~PFACopyRightCubeIn() = default;

    __aicore__ inline void Init() {
        baseHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
        baseWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth();
        orgHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgHeight();
        orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
        isTranspose_ = INPUT_TYPE::isTrans;
    }

    __aicore__ inline void SetSplitCount(int32_t totalRow, int32_t totalCol) {}

    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx) {}

    __aicore__ inline void SetInput(const LocalTensor<SrcT>& localMatrix, bool isTranspose) {
        srcAddr_ = localMatrix.address_;
        if constexpr (INPUT_TYPE::isTrans) {
            isTranspose_ = isTranspose;
        }
    }

    __aicore__ inline void SetInput(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose) {
        MATMUL_MODULE(MatmulTensorInfo)->template SetGlobalTensor<false>(globalMatrix, isTranspose);
        srcGlobalAddr_ = globalMatrix.address_;
        if constexpr (INPUT_TYPE::isTrans) {
            isTranspose_ = isTranspose;
        }
    }

    __aicore__ inline void SetBatchNum(int32_t batchA) {}

    // rignt matrix dont need reuse
    __aicore__ inline LocalTensor<TransT>
    LoadData(int curRow, int curCol, int tileHeight, int tileWidth, int batchNum = -1) {
        if constexpr (INPUT_TYPE::isTrans) {
            if (isTranspose_) {
                tileHeight = tileHeight ^ tileWidth;
                tileWidth = tileHeight ^ tileWidth;
                tileHeight = tileHeight ^ tileWidth;
                orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<true>();
            }
        } else {
            orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
        }
        LocalTensor<TransT> l1;
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            PFAMatmulPolicyData flag = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
            l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.rightBufIdx);
            GlobalTensor<TransT> aGlobal;
            aGlobal.SetGlobalBuffer(srcGlobalAddr_);
            Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = tileHeight;
            nd2nzParams.dValue = tileWidth;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = orgWidth_;
            if constexpr ((IsSameType<SrcT, int8_t>::value ||
                IsSameType<SrcT, fp8_e4m3fn_t>::value || IsSameType<SrcT, hifloat8_t>::value) && !INPUT_TYPE::isTrans) {
                nd2nzParams.dstNzC0Stride = Ceil(tileHeight, 32) * 32; // int8类型，32B对齐需要有32个数
            } else {
                nd2nzParams.dstNzC0Stride = Ceil(tileHeight, 16) * 16; // fp16类型，32B对齐需要有16个数
            }
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            DataCopy(l1, aGlobal, nd2nzParams);
            MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
            MATMUL_MODULE(CubeInBuffer)->DeQue();
        } else {
            l1.SetAddr(srcAddr_);
        }
        return l1;
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& tensor = LocalTensor<TransT>{},
                                         int32_t curRow = 0, int32_t curCol = 0) {}

    __aicore__ inline void Destroy() {
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
    __gm__ SrcT* srcGlobalAddr_ = nullptr;
    int32_t baseHeight_{0};
    int32_t baseWidth_{0};
    int32_t orgHeight_{0};
    int32_t orgWidth_{0};
    int32_t tileWidth_{0};
    bool isTranspose_{false};
};
}
}
}

#endif // PFA_NORM_COPY_RIGHT_CUBE_H
