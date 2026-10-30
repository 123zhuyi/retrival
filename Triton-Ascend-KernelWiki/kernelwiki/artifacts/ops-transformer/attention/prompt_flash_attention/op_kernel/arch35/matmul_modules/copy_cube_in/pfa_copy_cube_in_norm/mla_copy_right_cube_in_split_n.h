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
 * \file mla_copy_right_cube_in_split_n.h
 * \brief
 */

#ifndef MLA_COPY_RIGHT_CUBE_IN_SPLIT_N_H
#define MLA_COPY_RIGHT_CUBE_IN_SPLIT_N_H

#include "../../pfa_policy_data.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template<typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class MLACopyRightCubeInSplitN {
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;
public:
    using InputType = INPUT_TYPE;
    __aicore__ inline MLACopyRightCubeInSplitN() = default;
    __aicore__ inline ~MLACopyRightCubeInSplitN() = default;

    __aicore__ inline void Init() {
        baseHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
        baseWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth();
        orgHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgHeight();
        orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
        isTranspose_ = INPUT_TYPE::isTrans;
        nd2nzParams_.ndNum = 1;
        nd2nzParams_.srcNdMatrixStride = 0;
        nd2nzParams_.dstNzNStride = 1;
        nd2nzParams_.dstNzMatrixStride = 0;
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

    // left matrix check reuse or not
    __aicore__ inline LocalTensor<TransT>
    LoadData(int curRow, int curCol, int tileHeight, int tileWidth, int batchNum = -1) {
        if constexpr (INPUT_TYPE::isTrans) {
            if (isTranspose_) {
                tileHeight = tileHeight ^ tileWidth;
                tileWidth = tileHeight ^ tileWidth;
                tileHeight = tileHeight ^ tileWidth;
                orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<true>();
                baseWidth_ = MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<true>();
            }
        } else {
            orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
            baseWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth();
        }
        LocalTensor<TransT> l1;
        l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(4 + (callTimes_ & 1));
        GlobalTensor<TransT> aGlobal;
        aGlobal.SetGlobalBuffer(srcGlobalAddr_ + (callTimes_) * baseWidth_);
        nd2nzParams_.nValue = tileHeight;
        nd2nzParams_.dValue = tileWidth;
        nd2nzParams_.srcDValue = orgWidth_;
        if constexpr (IsSameType<SrcT, int8_t>::value ||
            IsSameType<SrcT, fp8_e4m3fn_t>::value || IsSameType<SrcT, hifloat8_t>::value) {
            nd2nzParams_.dstNzC0Stride = Ceil(tileHeight, 32) * 32; // int8/fp8类型，32B对齐需要有32个数
        } else {
            nd2nzParams_.dstNzC0Stride = Ceil(tileHeight, 16) * 16; // fp16类型，32B对齐需要有16个数
        }
        DataCopy(l1, aGlobal, nd2nzParams_);
        MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
        MATMUL_MODULE(CubeInBuffer)->DeQue();

        ++callTimes_;

        return l1;
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& tensor = LocalTensor<TransT>{},
                                         int32_t curRow = 0, int32_t curCol = 0) {
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            MATMUL_MODULE(CubeInBuffer)->Reset();
        }
    }

    __aicore__ inline void Destroy() {
        callTimes_ = 0;
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
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
    TBuffAddr srcAddr_{};
    __gm__ SrcT* srcGlobalAddr_ = nullptr;
    Nd2NzParams nd2nzParams_{};
    uint32_t callTimes_{0};
    int32_t baseHeight_{0};
    int32_t baseWidth_{0};
    int32_t orgHeight_{0};
    int32_t orgWidth_{0};
    bool isTranspose_{false};
};
}
}
}

#endif // MLA_COPY_RIGHT_CUBE_IN_SPLIT_N_H
