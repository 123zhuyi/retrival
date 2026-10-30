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
 * \file cumsum_oneway_ss.h
 * \brief calculate the prefix accumulation sum of tensor specified dimension
 */
#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_ONEWAY_SS_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_ONEWAY_SS_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "cumsum_base/cumsum_oneway_sklansky.h"

namespace Cumsum {
using namespace AscendC;

template <typename DataType, typename PromoteDataType>
class CumsumOnewaySs : public CumsumOnewaySklansky<DataType, PromoteDataType> {
public:
    __aicore__ inline CumsumOnewaySs(TPipe& pipe) : CumsumOnewaySklansky<DataType, PromoteDataType>(pipe)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    const CumsumSklanskyTilingData* tiling_;
    constexpr static int32_t BLOCK_SIZE = 32;

private:
    __aicore__ inline int64_t GetMBlockOffset(int32_t blockIdx);
    __aicore__ inline int64_t GetNBlockOffset(int32_t blockIdx);
    __aicore__ inline void GetOffset(
        int32_t mIdx, int32_t mUbIdx, int32_t nUbIdx, int32_t nUbFactor, OnewaySklanskyProcessData& processPreParam);
    __aicore__ inline void CallOneWaySklansky(int32_t mUbIdx, int32_t nUbIdx);
    __aicore__ inline void OneWaySklanskyProcess(int32_t mUbIdx, int32_t nUbIdx, int32_t mFactor, int32_t nFactor);
};

template <typename DataType, typename PromoteDataType>
__aicore__ inline int64_t CumsumOnewaySs<DataType, PromoteDataType>::GetMBlockOffset(int32_t blockIdx)
{
    if (tiling_->nBlockPara.blockCount == 1) {
        // 核切分在m轴上
        return blockIdx * tiling_->mBlockPara.blockFactor;
    } else {
        // 核切分在n轴上
        return blockIdx / tiling_->nBlockPara.blockCount;
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline int64_t CumsumOnewaySs<DataType, PromoteDataType>::GetNBlockOffset(int32_t blockIdx)
{
    if (tiling_->nBlockPara.blockCount == 1) {
        // 核切分在m轴上
        return 0;
    } else {
        return blockIdx * tiling_->nBlockPara.blockFactor;
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySs<DataType, PromoteDataType>::GetOffset(
    int32_t mIdx, int32_t mUbIdx, int32_t nUbIdx, int32_t nUbFactor, OnewaySklanskyProcessData& processPreParam)
{
    processPreParam.rOffset = 0;
    processPreParam.nFactor = nUbFactor;
    processPreParam.rFactor = tiling_->lenR;
    processPreParam.exclusive = tiling_->exclusive;
    processPreParam.isLenChange = true;
    if (tiling_->nBlockPara.blockCount == 1) {
        int64_t mOffset = GetMBlockOffset(this->blockIdx_);
        if (this->blockIdx_ == (tiling_->mBlockPara.blockCount - 1)) {
            processPreParam.mOffset = mOffset + mUbIdx * tiling_->mUbPara.tailCoreUbPara.ubFactor + mIdx;
        } else {
            processPreParam.mOffset = mOffset + mUbIdx * tiling_->mUbPara.mainCoreUbPara.ubFactor + mIdx;
        }
        processPreParam.nOffset = nUbIdx * tiling_->nUbPara.tailCoreUbPara.ubFactor;
    } else {
        int64_t mOffset = GetMBlockOffset(this->blockIdx_);
        int64_t nOffset = GetNBlockOffset(this->blockIdx_ % tiling_->nBlockPara.blockCount);
        if (this->blockIdx_ % tiling_->nBlockPara.blockCount == (tiling_->nBlockPara.blockCount - 1)) {
            processPreParam.nOffset = nOffset + nUbIdx * tiling_->nUbPara.tailCoreUbPara.ubFactor;
        } else {
            processPreParam.nOffset = nOffset + nUbIdx * tiling_->nUbPara.mainCoreUbPara.ubFactor;
        }
        processPreParam.mOffset = mOffset;
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySs<DataType, PromoteDataType>::OneWaySklanskyProcess(
    int32_t mUbIdx, int32_t nUbIdx, int32_t mFactor, int32_t nFactor)
{
    OnewaySklanskyProcessData processPreParam;
    if (tiling_->mBorrow > 1) {
        int32_t mUbFactor = this->blockIdx_ == tiling_->mBlockPara.blockCount - 1 ?
                                tiling_->mUbPara.tailCoreUbPara.ubFactor :
                                tiling_->mUbPara.mainCoreUbPara.ubFactor;
        processPreParam.mOffset = this->blockIdx_ * tiling_->mBlockPara.blockFactor + mUbIdx * mUbFactor;
        processPreParam.mFactor = mFactor;
        processPreParam.rOffset = 0;
        processPreParam.rFactor = tiling_->lenR;
        processPreParam.nOffset = nUbIdx * tiling_->nUbPara.tailCoreUbPara.ubFactor;
        processPreParam.nFactor = nFactor;
        processPreParam.exclusive = tiling_->exclusive;
        processPreParam.isLenChange = true;
        this->BaseProcessPre(processPreParam);
        this->BaseProcess();
        this->BaseCopyOut();
    } else {
        processPreParam.mFactor = mFactor;
        GetOffset(0, mUbIdx, nUbIdx, nFactor, processPreParam);
        this->BaseProcessPre(processPreParam);
        this->BaseProcess();
        this->BaseCopyOut();
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySs<DataType, PromoteDataType>::CallOneWaySklansky(int32_t mUbIdx, int32_t nUbIdx)
{
    int32_t mFactor = 0;
    int32_t nFactor = 0;
    // 核切分在m轴上或n轴上的通用逻辑
    if (tiling_->nBlockPara.blockCount == 1) {
        // 核切分在m轴上
        if (this->blockIdx_ == (tiling_->mBlockPara.blockCount - 1)) {
            // 处理m轴最后一个block
            mFactor = (mUbIdx == tiling_->mUbPara.tailCoreUbPara.ubCount - 1) ?
                          tiling_->mUbPara.tailCoreUbPara.ubTailFactor :
                          tiling_->mUbPara.tailCoreUbPara.ubFactor;
            nFactor = (nUbIdx == tiling_->nUbPara.tailCoreUbPara.ubCount - 1) ?
                          tiling_->nUbPara.tailCoreUbPara.ubTailFactor :
                          tiling_->nUbPara.tailCoreUbPara.ubFactor;
        } else {
            // 处理m轴非最后一个block
            mFactor = (mUbIdx == tiling_->mUbPara.mainCoreUbPara.ubCount - 1) ?
                          tiling_->mUbPara.mainCoreUbPara.ubTailFactor :
                          tiling_->mUbPara.mainCoreUbPara.ubFactor;
            nFactor = (nUbIdx == tiling_->nUbPara.tailCoreUbPara.ubCount - 1) ?
                          tiling_->nUbPara.tailCoreUbPara.ubTailFactor :
                          tiling_->nUbPara.tailCoreUbPara.ubFactor;
        }
        OneWaySklanskyProcess(mUbIdx, nUbIdx, mFactor, nFactor);
    } else {
        // 核切分在n轴上
        if (this->blockIdx_ % tiling_->nBlockPara.blockCount == (tiling_->nBlockPara.blockCount - 1)) {
            // 处理n轴最后一个block
            nFactor = (nUbIdx == tiling_->nUbPara.tailCoreUbPara.ubCount - 1) ?
                          tiling_->nUbPara.tailCoreUbPara.ubTailFactor :
                          tiling_->nUbPara.tailCoreUbPara.ubFactor;
        } else {
            // 处理n轴非最后一个block
            nFactor = (nUbIdx == tiling_->nUbPara.mainCoreUbPara.ubCount - 1) ?
                          tiling_->nUbPara.mainCoreUbPara.ubTailFactor :
                          tiling_->nUbPara.mainCoreUbPara.ubFactor;
        }
        OneWaySklanskyProcess(0, nUbIdx, 1, nFactor);
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySs<DataType, PromoteDataType>::Init(
    GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace)
{
    tiling_ = tilingData;
    OnewaySklanskyInitData initData;
    initData.mLen = tilingData->lenM;
    initData.nLen = tilingData->lenN;
    initData.rLen = tilingData->lenR;
    initData.mFold = tilingData->mBorrow;
    initData.reverse = tilingData->reverse;
    initData.ubBufferSize = tilingData->xBufSize;
    this->BaseInit(x, y, workspace, initData);
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySs<DataType, PromoteDataType>::Process()
{
    int32_t mUbCount = 0;
    int32_t nUbCount = 0;
    // 核切分在m轴上或n轴上的通用逻辑
    if (tiling_->nBlockPara.blockCount == 1) {
        // 核切分在m轴上
        mUbCount = (this->blockIdx_ == (tiling_->mBlockPara.blockCount - 1)) ? tiling_->mUbPara.tailCoreUbPara.ubCount :
                                                                               tiling_->mUbPara.mainCoreUbPara.ubCount;
        nUbCount = tiling_->nUbPara.tailCoreUbPara.ubCount;
    } else {
        // 核切分在n轴上
        mUbCount = 1;
        nUbCount = (this->blockIdx_ % tiling_->nBlockPara.blockCount == (tiling_->nBlockPara.blockCount - 1)) ?
                       tiling_->nUbPara.tailCoreUbPara.ubCount :
                       tiling_->nUbPara.mainCoreUbPara.ubCount;
    }
    for (int32_t i = 0; i < mUbCount; i++) {
        for (int32_t j = 0; j < nUbCount; j++) {
            CallOneWaySklansky(i, j);
        }
    }
}
} // namespace Cumsum

#endif