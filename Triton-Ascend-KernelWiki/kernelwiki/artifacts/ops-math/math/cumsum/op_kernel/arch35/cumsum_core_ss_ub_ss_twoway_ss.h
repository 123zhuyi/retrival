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
 * \file cumsum_core_ss_ub_ss_twoway_ss.h
 * \brief core sklansky + ub sklansky + twoway sklansky.
 */

#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_H

#include "cumsum_base/cumsum_core_sklansky.h"
#include "cumsum_base/cumsum_ub_sklansky.h"

namespace Cumsum {

using namespace AscendC;

template <typename T, typename PromtT, typename CoreInnerHandler>
class CumsumCoreSsUbSsTwowaySs : public CumsumCoreSklansky<T, PromtT> {
public:
    __aicore__ inline CumsumCoreSsUbSsTwowaySs(TPipe& pipe)
        : CumsumCoreSklansky<T, PromtT>(pipe), coreInnerHandler_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* td, GM_ADDR workspace);

    __aicore__ inline void Process();

private:
    CoreInnerHandler coreInnerHandler_;
    const CumsumSklanskyTilingData* td_;
}; // class CumsumCoreSsUbSsTwowaySs

template <typename T, typename PromtT, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsUbSsTwowaySs<T, PromtT, CoreInnerHandler>::Init(
    GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* td, GM_ADDR workspace)
{
    // 核间初始化
    CumsumCoreSklansky<T, PromtT>::Init(x, y, td);
    td_ = td;

    // 核内初始化
    if (this->blockIdx_ < td->realCoreNum) {
        UbSklanskyInitData initData;
        initData.lenM = td->lenM;
        initData.lenR = td->lenR;
        initData.lenN = td->lenN;
        initData.reverse = td->reverse == 1 ? true : false;
        initData.xBufferSize = td->xBufSize;
        initData.xUnfoldBufferSize = td->xUnfoldBufSize;
        initData.ubSklanskyBufSize = td->ubSklanskyBufSize;
        if (this->blockIdx_ / td->nBorrow % td->rBorrow == td->rBlockPara.blockCount - 1) {
            initData.rFoldAfterSize = td->tailCoreMainUbFoldPara.foldLen;
            initData.rFoldAfterCount = td->tailCoreMainUbFoldPara.sklanskyIter;
            initData.rFoldCount = td->tailCoreMainUbFoldPara.foldCount;
            initData.rTailFoldAfterSize = td_->tailCoreTailUbFoldPara.foldLen;
            initData.rTailFoldAfterCount = td_->tailCoreTailUbFoldPara.sklanskyIter;
            initData.rTailFoldCount = td_->tailCoreTailUbFoldPara.foldCount;
            initData.ubFactorR = td_->rUbPara.tailCoreUbPara.ubFactor;
            initData.ubTailFactorR = td_->rUbPara.tailCoreUbPara.ubTailFactor;
            initData.ubCountR = td_->rUbPara.tailCoreUbPara.ubCount;
            initData.memoLen = __CumsumUtil::CalLog2(td->rUbPara.tailCoreUbPara.ubCount);
        } else {
            initData.rFoldAfterSize = td->mainCoreMainUbFoldPara.foldLen;
            initData.rFoldAfterCount = td->mainCoreMainUbFoldPara.sklanskyIter;
            initData.rFoldCount = td->mainCoreMainUbFoldPara.foldCount;
            initData.rTailFoldAfterSize = td_->mainCoreTailUbFoldPara.foldLen;
            initData.rTailFoldAfterCount = td_->mainCoreTailUbFoldPara.sklanskyIter;
            initData.rTailFoldCount = td_->mainCoreTailUbFoldPara.foldCount;
            initData.ubFactorR = td_->rUbPara.mainCoreUbPara.ubFactor;
            initData.ubTailFactorR = td_->rUbPara.mainCoreUbPara.ubTailFactor;
            initData.ubCountR = td_->rUbPara.mainCoreUbPara.ubCount;
            initData.memoLen = __CumsumUtil::CalLog2(td->rUbPara.mainCoreUbPara.ubCount);
        }
        if (this->blockIdx_ % td->nBorrow == td->nBlockPara.blockCount - 1) {
            initData.ubFactorN = td->nBlockPara.blockTailFactor;
        } else {
            initData.ubFactorN = td->nBlockPara.blockFactor;
        }
        initData.perMemoSize = Ops::Base::CeilAlign(
            initData.ubFactorN, static_cast<int32_t>(Ops::Base::GetUbBlockSize() / sizeof(PromtT)));
        coreInnerHandler_.BaseInit(x, y, workspace, initData);
    }
}

template <typename T, typename PromtT, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsUbSsTwowaySs<T, PromtT, CoreInnerHandler>::Process()
{
    if (this->blockIdx_ < td_->realCoreNum) {
        UbSklanskyProcessData processData;
        if (td_->reverse == 1) {
            int64_t offsetRLen = (this->blockIdx_ / td_->nBorrow % td_->rBorrow + 1) * td_->rBlockPara.blockFactor;
            if (td_->lenR > offsetRLen) {
                processData.offsetR = td_->lenR - offsetRLen;
            } else {
                processData.offsetR = 0;
            }
        } else {
            processData.offsetR = this->blockIdx_ / td_->nBorrow % td_->rBorrow * td_->rBlockPara.blockFactor;
        }
        processData.offsetM = this->blockIdx_ / (td_->rBorrow * td_->nBorrow);
        processData.offsetN = this->blockIdx_ % td_->nBorrow * td_->nBlockPara.blockFactor;
        processData.isExclusive = td_->exclusive == 1 ? true : false;
        if (td_->exclusive == 1 && this->blockIdx_ / td_->nBorrow % td_->rBorrow != 0) {
            processData.isLenChange = true;
        }

        coreInnerHandler_.BaseProcessPre(processData);
        coreInnerHandler_.BaseProcess();
    }

    CumsumCoreSklansky<T, PromtT>::Process();
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_H