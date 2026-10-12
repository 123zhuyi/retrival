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
 * \file cumsum_ub_ss_oneway_ss.h
 * \brief template class: ub sklansky + one way sklansky
 */
#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_UB_SS_ONEWAY_SS_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_UB_SS_ONEWAY_SS_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "cumsum_base/cumsum_base.h"
#include "cumsum_base/cumsum_oneway_sklansky.h"
#include "cumsum_base/cumsum_ub_sklansky.h"

namespace Cumsum {
using namespace AscendC;

template <typename DataType, typename PromoteDataType>
class CumsumUbSsOnewaySs
    : public CumsumUbSklansky<DataType, PromoteDataType, CumsumOnewaySklansky<DataType, PromoteDataType>> {
public:
    __aicore__ inline CumsumUbSsOnewaySs(TPipe& pipe)
        : CumsumUbSklansky<DataType, PromoteDataType, CumsumOnewaySklansky<DataType, PromoteDataType>>(pipe)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace);
    __aicore__ inline void Process();

    constexpr static int32_t BLK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static int32_t VL_SIZE = Ops::Base::GetVRegSize();

private:
    __aicore__ inline void ProcessBorrowN(UbSklanskyProcessData& processData);
    __aicore__ inline void ProcessNotBorrowN(UbSklanskyProcessData& processData);

    const CumsumSklanskyTilingData* tiling_;
};

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumUbSsOnewaySs<DataType, PromoteDataType>::Init(
    GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace)
{
    tiling_ = tilingData;
    UbSklanskyInitData initData;
    initData.lenM = tiling_->lenM;
    initData.lenR = tiling_->lenR;
    initData.lenN = tiling_->lenN;
    initData.mFold = tiling_->mBorrow;
    initData.ubFactorR = tiling_->rUbPara.tailCoreUbPara.ubFactor;
    initData.ubTailFactorR = tiling_->rUbPara.tailCoreUbPara.ubTailFactor;
    initData.ubCountR = tiling_->rUbPara.tailCoreUbPara.ubCount;
    initData.reverse = tiling_->reverse;
    initData.memoLen = __CumsumUtil::CalLog2(tiling_->rUbPara.tailCoreUbPara.ubCount);
    initData.xBufferSize = tiling_->xBufSize;
    initData.ubSklanskyBufSize = tiling_->ubSklanskyBufSize;

    initData.perMemoSize = VL_SIZE / sizeof(PromoteDataType); // 按照VL len申请
    CumsumUbSklansky<DataType, PromoteDataType, CumsumOnewaySklansky<DataType, PromoteDataType>>::BaseInit(
        x, y, workspace, initData);
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumUbSsOnewaySs<DataType, PromoteDataType>::ProcessBorrowN(UbSklanskyProcessData& processData)
{
    // 借轴之后，M维度上必定是均分的, mBlockFactor必然为1
    int32_t nIndex = this->blockIdx_ % tiling_->nBlockPara.blockCount;
    processData.offsetM = this->blockIdx_ / tiling_->nBlockPara.blockCount;
    processData.ubFactorM = 1;
    if (nIndex != tiling_->nBlockPara.blockCount - 1) { // 整块
        for (int32_t j = 0; j < tiling_->nUbPara.mainCoreUbPara.ubCount; j++) {
            if (j == tiling_->nUbPara.mainCoreUbPara.ubCount - 1) {
                processData.ubFactorN = tiling_->nUbPara.mainCoreUbPara.ubTailFactor;
            } else {
                processData.ubFactorN = tiling_->nUbPara.mainCoreUbPara.ubFactor;
            }
            processData.offsetN =
                nIndex * tiling_->nBlockPara.blockFactor + j * tiling_->nUbPara.mainCoreUbPara.ubFactor;
            CumsumUbSklansky<DataType, PromoteDataType, CumsumOnewaySklansky<DataType, PromoteDataType>>::
                BaseProcessPre(processData);
            CumsumUbSklansky<DataType, PromoteDataType, CumsumOnewaySklansky<DataType, PromoteDataType>>::BaseProcess();
        }
    } else {
        for (int32_t j = 0; j < tiling_->nUbPara.tailCoreUbPara.ubCount; j++) {
            if (j == tiling_->nUbPara.tailCoreUbPara.ubCount - 1) {
                processData.ubFactorN = tiling_->nUbPara.tailCoreUbPara.ubTailFactor;
            } else {
                processData.ubFactorN = tiling_->nUbPara.tailCoreUbPara.ubFactor;
            }
            processData.offsetN = (tiling_->nBlockPara.blockCount - 1) * tiling_->nBlockPara.blockFactor +
                                  j * tiling_->nUbPara.tailCoreUbPara.ubFactor;
            CumsumUbSklansky<DataType, PromoteDataType, CumsumOnewaySklansky<DataType, PromoteDataType>>::
                BaseProcessPre(processData);
            CumsumUbSklansky<DataType, PromoteDataType, CumsumOnewaySklansky<DataType, PromoteDataType>>::BaseProcess();
        }
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumUbSsOnewaySs<DataType, PromoteDataType>::ProcessNotBorrowN(
    UbSklanskyProcessData& processData)
{
    UbParaUnit mUbPara = this->blockIdx_ == tiling_->mBlockPara.blockCount - 1 ? tiling_->mUbPara.tailCoreUbPara :
                                                                                 tiling_->mUbPara.mainCoreUbPara;
    for (int32_t i = 0; i < mUbPara.ubCount; i++) {
        processData.offsetM = this->blockIdx_ * tiling_->mBlockPara.blockFactor + i * mUbPara.ubFactor;
        processData.ubFactorM = i == mUbPara.ubCount - 1 ? mUbPara.ubTailFactor : mUbPara.ubFactor;
        for (int32_t j = 0; j < tiling_->nUbPara.tailCoreUbPara.ubCount; j++) {
            if (j == tiling_->nUbPara.tailCoreUbPara.ubCount - 1) {
                processData.ubFactorN = tiling_->nUbPara.tailCoreUbPara.ubTailFactor;
            } else {
                processData.ubFactorN = tiling_->nUbPara.tailCoreUbPara.ubFactor;
            }
            processData.offsetN = j * tiling_->nUbPara.tailCoreUbPara.ubFactor;
            CumsumUbSklansky<DataType, PromoteDataType, CumsumOnewaySklansky<DataType, PromoteDataType>>::
                BaseProcessPre(processData);
            CumsumUbSklansky<DataType, PromoteDataType, CumsumOnewaySklansky<DataType, PromoteDataType>>::BaseProcess();
        }
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumUbSsOnewaySs<DataType, PromoteDataType>::Process()
{
    // fullCore + perCoreMCount
    if (this->blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    UbSklanskyProcessData processData;
    processData.offsetR = 0;
    processData.isExclusive = tiling_->exclusive;

    if (tiling_->nBlockPara.blockCount != 1) {
        ProcessBorrowN(processData);
    } else {
        ProcessNotBorrowN(processData);
    }
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_UB_SS_ONEWAY_SS_H