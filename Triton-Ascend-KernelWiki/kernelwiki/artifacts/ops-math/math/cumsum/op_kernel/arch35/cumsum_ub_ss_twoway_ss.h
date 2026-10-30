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
 * \file cumsum_ub_ss_twoway_ss.h
 * \brief template class: ub sklansky + two way sklansky
 */
#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_UB_SS_TWOWAY_SS_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_UB_SS_TWOWAY_SS_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "cumsum_base/cumsum_base.h"
#include "cumsum_base/cumsum_twoway_sklansky.h"
#include "cumsum_base/cumsum_ub_sklansky.h"

namespace Cumsum {
using namespace AscendC;

template <typename DataType, typename PromoteDataType>
class CumsumUbSsTwowaySs
    : public CumsumUbSklansky<DataType, PromoteDataType, CumsumTwowaySklansky<DataType, PromoteDataType>> {
public:
    __aicore__ inline CumsumUbSsTwowaySs(TPipe& pipe)
        : CumsumUbSklansky<DataType, PromoteDataType, CumsumTwowaySklansky<DataType, PromoteDataType>>(pipe)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace);
    __aicore__ inline void Process();

    constexpr static int32_t BLK_SIZE = Ops::Base::GetUbBlockSize();

private:
    const CumsumSklanskyTilingData* tiling_;
};

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumUbSsTwowaySs<DataType, PromoteDataType>::Init(
    GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace)
{
    tiling_ = tilingData;
    UbSklanskyInitData initData;
    initData.lenM = tiling_->lenM;
    initData.lenR = tiling_->lenR;
    initData.lenN = tiling_->lenN;
    initData.reverse = tiling_->reverse;

    initData.xBufferSize = tiling_->xBufSize;
    initData.xUnfoldBufferSize = tiling_->xUnfoldBufSize;
    initData.rFoldAfterSize = tiling_->tailCoreMainUbFoldPara.foldLen;
    initData.rFoldAfterCount = tiling_->tailCoreMainUbFoldPara.sklanskyIter;
    initData.rFoldCount = tiling_->tailCoreMainUbFoldPara.foldCount;
    initData.ubFactorR = tiling_->rUbPara.tailCoreUbPara.ubFactor;
    initData.ubTailFactorR = tiling_->rUbPara.tailCoreUbPara.ubTailFactor;
    initData.ubCountR = tiling_->rUbPara.tailCoreUbPara.ubCount;
    initData.rTailFoldAfterSize = tiling_->tailCoreTailUbFoldPara.foldLen;
    initData.rTailFoldAfterCount = tiling_->tailCoreTailUbFoldPara.sklanskyIter;
    initData.rTailFoldCount = tiling_->tailCoreTailUbFoldPara.foldCount;
    initData.ubFactorN = tiling_->nUbPara.tailCoreUbPara.ubTailFactor;

    initData.memoLen = __CumsumUtil::CalLog2(tiling_->rUbPara.tailCoreUbPara.ubCount);
    initData.ubSklanskyBufSize = tiling_->ubSklanskyBufSize;
    // ub间双向不会切N轴
    initData.perMemoSize = Ops::Base::CeilAlign(initData.ubFactorN, static_cast<int32_t>(BLK_SIZE / sizeof(DataType)));
    CumsumUbSklansky<DataType, PromoteDataType, CumsumTwowaySklansky<DataType, PromoteDataType>>::BaseInit(
        x, y, workspace, initData);
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumUbSsTwowaySs<DataType, PromoteDataType>::Process()
{
    // fullCore + perCoreMCount
    if (this->blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    UbSklanskyProcessData processData;
    processData.offsetR = 0;
    processData.offsetN = 0;
    processData.isExclusive = tiling_->exclusive;

    int32_t fullCoreNum = tiling_->mBlockPara.blockCount - 1;
    if (this->blockIdx_ < fullCoreNum) {                                // 主核
        for (int32_t i = 0; i < tiling_->mBlockPara.blockFactor; i++) { // mBlockFactor表示ub间的M的循环
            processData.offsetM = this->blockIdx_ * tiling_->mBlockPara.blockFactor + i; // R和N全载，M分核
            CumsumUbSklansky<DataType, PromoteDataType, CumsumTwowaySklansky<DataType, PromoteDataType>>::
                BaseProcessPre(processData);
            CumsumUbSklansky<DataType, PromoteDataType, CumsumTwowaySklansky<DataType, PromoteDataType>>::BaseProcess();
        }
    } else {
        for (int32_t i = 0; i < tiling_->mBlockPara.blockTailFactor; i++) {
            processData.offsetM = fullCoreNum * tiling_->mBlockPara.blockFactor + i;
            CumsumUbSklansky<DataType, PromoteDataType, CumsumTwowaySklansky<DataType, PromoteDataType>>::
                BaseProcessPre(processData);
            CumsumUbSklansky<DataType, PromoteDataType, CumsumTwowaySklansky<DataType, PromoteDataType>>::BaseProcess();
        }
    }
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_UB_SS_TWOWAY_SS_H