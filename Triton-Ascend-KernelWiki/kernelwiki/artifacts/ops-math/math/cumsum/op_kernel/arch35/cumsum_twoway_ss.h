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
 * \file cumsum_twoway_ss.h
 * \brief cumsum twoway sklansky
 */
#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_TWOWAY_SS_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_TWOWAY_SS_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "cumsum_base/cumsum_base.h"
#include "cumsum_base/cumsum_twoway_sklansky.h"

namespace Cumsum {
using namespace AscendC;

template <typename DataType, typename PromoteDataType>
class CumsumTwowaySs : public CumsumTwowaySklansky<DataType, PromoteDataType> {
public:
    __aicore__ inline CumsumTwowaySs(TPipe& pipe) : CumsumTwowaySklansky<DataType, PromoteDataType>(pipe)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    const CumsumSklanskyTilingData* tiling_;
}; // class CumsumTwowaySs

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySs<DataType, PromoteDataType>::Init(
    GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace)
{
    tiling_ = tilingData;
    TwowaySklanskyInitData initData;
    initData.lenM = tiling_->lenM;
    initData.lenR = tiling_->lenR;
    initData.lenN = tiling_->lenN;
    initData.xBufferSize = tiling_->xBufSize;
    initData.xUnfoldBufferSize = tiling_->xUnfoldBufSize;
    CumsumTwowaySklansky<DataType, PromoteDataType>::BaseInit(x, y, workspace, initData);
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySs<DataType, PromoteDataType>::Process()
{
    // fullCore + perCoreMCount
    if (this->blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    int32_t fullCoreNum = tiling_->mBlockPara.blockCount - 1;
    TwowaySklanskyProcessData processData;
    processData.offsetR = 0;
    processData.offsetN = 0;
    processData.ubFactorR = tiling_->rUbPara.tailCoreUbPara.ubTailFactor;
    processData.ubFactorN = tiling_->nUbPara.tailCoreUbPara.ubTailFactor;
    processData.rFoldAfterSize = tiling_->tailCoreTailUbFoldPara.foldLen;
    processData.rFoldAfterCount = tiling_->tailCoreTailUbFoldPara.sklanskyIter;
    processData.rFoldCount = tiling_->tailCoreTailUbFoldPara.foldCount;
    processData.isExclusive = tiling_->exclusive;
    processData.isReverse = tiling_->reverse;
    processData.isLenChange = tiling_->exclusive;
    if (this->blockIdx_ < fullCoreNum) { // 主核
        for (int32_t i = 0; i < tiling_->mBlockPara.blockFactor; i++) {
            processData.offsetM = this->blockIdx_ * tiling_->mBlockPara.blockFactor + i; // R和N全载，M分核
            CumsumTwowaySklansky<DataType, PromoteDataType>::BaseProcessPre(processData);
            CumsumTwowaySklansky<DataType, PromoteDataType>::BaseProcess();
            CumsumTwowaySklansky<DataType, PromoteDataType>::BaseCopyOut();
        }
    } else {
        for (int32_t i = 0; i < tiling_->mBlockPara.blockTailFactor; i++) {
            processData.offsetM = fullCoreNum * tiling_->mBlockPara.blockFactor + i;
            CumsumTwowaySklansky<DataType, PromoteDataType>::BaseProcessPre(processData);
            CumsumTwowaySklansky<DataType, PromoteDataType>::BaseProcess();
            CumsumTwowaySklansky<DataType, PromoteDataType>::BaseCopyOut();
        }
    }
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_TWOWAY_SS_H