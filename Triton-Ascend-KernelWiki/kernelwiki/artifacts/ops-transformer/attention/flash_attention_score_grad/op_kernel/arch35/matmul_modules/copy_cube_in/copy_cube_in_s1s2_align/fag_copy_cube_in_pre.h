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
 * \file fag_copy_cube_in_pre.h
 * \brief
 */

#ifndef FAG_COPY_CUBE_IN_PRE_H
#define FAG_COPY_CUBE_IN_PRE_H

#include "../../cube_in_buffer/fag_cube_in_buffer_pre.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG>
class FagCopyCubeInS1S2Pre : public CopyCubeInBase<IMPL, MM_CFG, INPUT_TYPE> {
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyUtils, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    using SRC_T = typename INPUT_TYPE::T;
    using TRANS_T = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline FagCopyCubeInS1S2Pre()
    {
    }

    __aicore__ inline void Init()
    {
        totalRow_ = MATMUL_MODULE(CopyCubeInParams)->GetTotalRow();
        totalCol_ = MATMUL_MODULE(CopyCubeInParams)->GetTotalCol();
        isTranspose_ = INPUT_TYPE::isTrans;
        iskRowDirec_ = INPUT_TYPE::TAG == InputTypeTag::A && INPUT_TYPE::isTrans ||
                       INPUT_TYPE::TAG == InputTypeTag::B && !INPUT_TYPE::isTrans;
    }

    __aicore__ inline void SetInput(const LocalTensor<SRC_T> &localMatrix, bool isTranspose)
    {
        isTranspose_ = isTranspose;
        MATMUL_MODULE(MatmulTensorInfo)->SetLocalTensor(localMatrix, isTranspose);
    }

    __aicore__ inline void SetInput(const GlobalTensor<SRC_T> &globalMatrix, bool isTranspose)
    {
        isTranspose_ = isTranspose;
        MATMUL_MODULE(MatmulTensorInfo)->SetGlobalTensor(globalMatrix, isTranspose);
    }

    __aicore__ inline LocalTensor<SRC_T> LoadData(int curRow, int curCol, int tileHeight, int tileWidth,
                                                  int batchNum = 1)
    {
        // 获取用户设置的flag
        FagTscmFlagData flag = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
        uint8_t leftMatrixEncodingTableIdx = flag.leftMatrixEncodingTableIdx;
        uint8_t kvNeedCopy = flag.kvNeedCopy;
        uint8_t rightMatrixEncodingTableIdx = flag.rightMatrixEncodingTableIdx;
        uint8_t tscmIndex = 0;
        LocalTensor<SRC_T> l1;
        int32_t orgHeight = MATMUL_MODULE(CopyCubeInParams)->GetOrgHeight();
        int32_t orgWidth = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();

        if constexpr (INPUT_TYPE::isTrans) {
            if (isTranspose_) {
                // Init传参时orgHeight_、orgWidth_、tileHeight、tileWidth等参数需要翻转传入
                curRow = curRow ^ curCol;
                curCol = curRow ^ curCol;
                curRow = curRow ^ curCol;
                tileHeight = tileHeight ^ tileWidth; // SingleM
                tileWidth = tileHeight ^ tileWidth;  // SingleN
                tileHeight = tileHeight ^ tileWidth;
                orgHeight = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
                orgWidth = MATMUL_MODULE(CopyCubeInParams)->GetOrgHeight();
            }
        }

        MATMUL_MODULE(Context)->leftMatrixEncodingTableIdx = static_cast<int32_t>(leftMatrixEncodingTableIdx);
        MATMUL_MODULE(Context)->rightMatrixEncodingTableIdx = static_cast<int32_t>(rightMatrixEncodingTableIdx);

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            tscmIndex = MM1_ENCODING_TABLE[leftMatrixEncodingTableIdx];
        } else {
            tscmIndex = MM1_ENCODING_TABLE[rightMatrixEncodingTableIdx];
            if (!kvNeedCopy) {
                l1 = gTscmArray[tscmIndex].localQue.template AllocTensor<SRC_T>();
                gTscmArray[tscmIndex].srcAddr = l1.address_;
                return l1;
            }
        }
        auto tscm = gTscmArray[tscmIndex].localQue;
        l1 = tscm.template AllocTensor<SRC_T>();
        gTscmArray[tscmIndex].srcAddr = l1.address_;

        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            Nd2NzParams intriParams;
            intriParams.ndNum = 1;
            intriParams.nValue = tileHeight; // left matrix: m
            intriParams.dValue = tileWidth;                         // left matrix: k;
            intriParams.srcNdMatrixStride = 0;
            intriParams.srcDValue = orgWidth;
            // L1->L0A，L1上的src stride为C0对齐后的SingleM
            intriParams.dstNzC0Stride = (tileHeight + C0_SIZE - 1) >> 4 << 4;
            intriParams.dstNzNStride = 1;
            intriParams.dstNzMatrixStride = 0;
            GlobalTensor<SRC_T> aGlobal;
            aGlobal.SetGlobalBuffer(MATMUL_MODULE(MatmulTensorInfo)->GetGlobalTensor().address_);
            DataCopy(l1, aGlobal, intriParams);
        } else {
            Nd2NzParams intriParams;
            intriParams.ndNum = 1;
            intriParams.nValue = tileHeight; // right matrix: m
            intriParams.dValue = tileWidth;  // right matrix: k;
            intriParams.srcNdMatrixStride = 0;
            intriParams.srcDValue = orgWidth;
            intriParams.dstNzC0Stride = (tileHeight + C0_SIZE - 1) >> 4 << 4;
            intriParams.dstNzNStride = 1;
            intriParams.dstNzMatrixStride = 0;
            GlobalTensor<SRC_T> bGlobal;
            bGlobal.SetGlobalBuffer(MATMUL_MODULE(MatmulTensorInfo)->GetGlobalTensor().address_);
            DataCopy(l1, bGlobal, intriParams);
        }

        tscm.EnQue(l1);
        tscm.DeQue<SRC_T>();
        return l1;
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TRANS_T> &aMatrix = LocalTensor<TRANS_T>{}, int32_t curRow = 0,
                                         int32_t curCol = 0)
    {
        if constexpr (!PhyPosIsUB(INPUT_TYPE::pos) && !PhyPosIsL1(INPUT_TYPE::pos)) {
            int posL1 = MATMUL_MODULE(CubeInBuffer)->GetIterIndex(curRow, curCol);
            MATMUL_MODULE(CubeInBuffer)->FreeTensor(posL1, aMatrix);
        }
    }

    // Destroy
    __aicore__ inline void Destroy()
    {
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            MATMUL_MODULE(CubeInBuffer)->Destroy();
        }
    }

public:
    bool iskRowDirec_;
    bool isTranspose_{false};

private:
    int32_t totalRow_;
    int32_t totalCol_;
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC

#endif