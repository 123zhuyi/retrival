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
 * \file cumsum_core_ss_twoway_ss.h
 * \brief core sklansky + twoway sklansky.
 */

#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_TWOWAY_SS_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_TWOWAY_SS_H

#include "cumsum_base/cumsum_core_sklansky.h"
#include "cumsum_base/cumsum_twoway_sklansky.h"

namespace Cumsum {

using namespace AscendC;

template <typename T, typename PromtT, typename CoreInnerHandler>
class CumsumCoreSsTwowaySs : public CumsumCoreSklansky<T, PromtT> {
public:
    __aicore__ inline CumsumCoreSsTwowaySs(TPipe& pipe) : CumsumCoreSklansky<T, PromtT>(pipe), coreInnerHandler_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* td, GM_ADDR workspace);

    __aicore__ inline void Process();

private:
    CoreInnerHandler coreInnerHandler_;
    const CumsumSklanskyTilingData* td_;
}; // class CumsumCoreSsTwowaySs

template <typename T, typename PromtT, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsTwowaySs<T, PromtT, CoreInnerHandler>::Init(
    GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* td, GM_ADDR workspace)
{
    // 核间初始化
    CumsumCoreSklansky<T, PromtT>::Init(x, y, td);
    td_ = td;

    // 核内初始化
    if (this->blockIdx_ < td->realCoreNum) {
        TwowaySklanskyInitData initData;
        initData.lenM = td->lenM;
        initData.lenR = td->lenR;
        initData.lenN = td->lenN;
        initData.xBufferSize = td->xBufSize;
        initData.xUnfoldBufferSize = td->xUnfoldBufSize;
        coreInnerHandler_.BaseInit(x, y, workspace, initData);
    }
}

template <typename T, typename PromtT, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsTwowaySs<T, PromtT, CoreInnerHandler>::Process()
{
    if (this->blockIdx_ < td_->realCoreNum) {
        TwowaySklanskyProcessData processData;

        if (td_->reverse == 1) {
            int64_t offsetR = (this->blockIdx_ / td_->nBorrow % td_->rBorrow + 1) * td_->rBlockPara.blockFactor;
            if (td_->lenR > offsetR) {
                processData.offsetR = td_->lenR - offsetR;
            } else {
                processData.offsetR = 0;
            }
        } else {
            processData.offsetR = this->blockIdx_ / td_->nBorrow % td_->rBorrow * td_->rBlockPara.blockFactor;
        }
        processData.offsetM = this->blockIdx_ / (td_->rBorrow * td_->nBorrow);
        processData.offsetN = this->blockIdx_ % td_->nBorrow * td_->nBlockPara.blockFactor;

        if (this->blockIdx_ % td_->rBorrow < td_->rBorrow - 1) { // 主核
            processData.ubFactorR = td_->rUbPara.mainCoreUbPara.ubTailFactor;
            processData.rFoldAfterSize = td_->mainCoreTailUbFoldPara.foldLen;
            processData.rFoldAfterCount = td_->mainCoreTailUbFoldPara.sklanskyIter;
            processData.rFoldCount = td_->mainCoreTailUbFoldPara.foldCount;

        } else {
            processData.ubFactorR = td_->rUbPara.tailCoreUbPara.ubTailFactor;
            processData.rFoldAfterSize = td_->tailCoreTailUbFoldPara.foldLen;
            processData.rFoldAfterCount = td_->tailCoreTailUbFoldPara.sklanskyIter;
            processData.rFoldCount = td_->tailCoreTailUbFoldPara.foldCount;
        }
        processData.ubFactorN = td_->nUbPara.tailCoreUbPara.ubTailFactor;
        processData.isReverse = td_->reverse;
        processData.isExclusive = td_->exclusive;
        if (td_->exclusive == 1 && this->blockIdx_ % td_->rBorrow == 0) {
            processData.isLenChange = true;
        }
        coreInnerHandler_.BaseProcessPre(processData);
        coreInnerHandler_.BaseProcess();
        coreInnerHandler_.BaseCopyOut();
    }

    CumsumCoreSklansky<T, PromtT>::Process();
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_TWOWAY_SS_H