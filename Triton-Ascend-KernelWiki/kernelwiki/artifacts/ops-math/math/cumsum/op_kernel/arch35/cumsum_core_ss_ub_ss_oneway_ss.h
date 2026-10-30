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
 * \file cumsum_core_ss_ub_ss_oneway_ss.h
 * \brief core sklansky + ub sklansky + twoway sklansky.
 */

#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_UB_SS_ONEWAY_SS_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_UB_SS_ONEWAY_SS_H

#include "cumsum_base/cumsum_core_sklansky.h"
#include "cumsum_base/cumsum_ub_sklansky.h"

namespace Cumsum {

using namespace AscendC;

template <typename T, typename PromtT, typename CoreInnerHandler>
class CumsumCoreSsUbSsOnewaySs : public CumsumCoreSklansky<T, PromtT> {
public:
    __aicore__ inline CumsumCoreSsUbSsOnewaySs(TPipe& pipe)
        : CumsumCoreSklansky<T, PromtT>(pipe), coreInnerHandler_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* td, GM_ADDR workspace);

    __aicore__ inline void Process();

    constexpr static int32_t BLK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static int32_t VL_SIZE = Ops::Base::GetVRegSize();

private:
    __aicore__ inline void ProcessNotBorrowN(UbSklanskyProcessData& processData);
    __aicore__ inline void ProcessBorrowN(UbSklanskyProcessData& processData);

    CoreInnerHandler coreInnerHandler_;
    const CumsumSklanskyTilingData* td_;
}; // class CumsumCoreSsUbSsOnewaySs

template <typename T, typename PromtT, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsUbSsOnewaySs<T, PromtT, CoreInnerHandler>::Init(
    GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* td, GM_ADDR workspace)
{
    // 核间初始化
    CumsumCoreSklansky<T, PromtT>::Init(x, y, td);
    td_ = td;

    if (this->blockIdx_ >= td->realCoreNum) {
        return;
    }

    // 核内初始化
    UbSklanskyInitData initData;
    initData.lenM = td->lenM;
    initData.lenR = td->lenR;
    initData.lenN = td->lenN;
    initData.xBufferSize = td->xBufSize;
    initData.ubSklanskyBufSize = td->ubSklanskyBufSize;
    initData.reverse = td->reverse == 1 ? true : false;
    initData.perMemoSize = VL_SIZE / sizeof(PromtT);                                    // 按照VL len申请
    if (this->blockIdx_ / td->nBorrow % td->rBorrow == td->rBlockPara.blockCount - 1) { // 是否是非reverse的尾核
        initData.ubFactorR = td_->rUbPara.tailCoreUbPara.ubFactor;
        initData.ubTailFactorR = td_->rUbPara.tailCoreUbPara.ubTailFactor;
        initData.ubCountR = td_->rUbPara.tailCoreUbPara.ubCount;
        initData.memoLen = __CumsumUtil::CalLog2(td->rUbPara.tailCoreUbPara.ubCount);
    } else {
        initData.ubFactorR = td_->rUbPara.mainCoreUbPara.ubFactor;
        initData.ubTailFactorR = td_->rUbPara.mainCoreUbPara.ubTailFactor;
        initData.ubCountR = td_->rUbPara.mainCoreUbPara.ubCount;
        initData.memoLen = __CumsumUtil::CalLog2(td->rUbPara.mainCoreUbPara.ubCount);
    }

    coreInnerHandler_.BaseInit(x, y, workspace, initData);
}

template <typename T, typename PromtT, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsUbSsOnewaySs<T, PromtT, CoreInnerHandler>::ProcessBorrowN(
    UbSklanskyProcessData& processData)
{
    // N借轴之后，M维度上必定是均分的
    int32_t nIndex = this->blockIdx_ % td_->nBlockPara.blockCount;
    if (nIndex != td_->nBorrow - 1) { // 整块
        for (int32_t j = 0; j < td_->nUbPara.mainCoreUbPara.ubCount; j++) {
            if (j == td_->nUbPara.mainCoreUbPara.ubCount - 1) {
                processData.ubFactorN = td_->nUbPara.mainCoreUbPara.ubTailFactor;
            } else {
                processData.ubFactorN = td_->nUbPara.mainCoreUbPara.ubFactor;
            }
            processData.offsetN = nIndex * td_->nBlockPara.blockFactor + j * td_->nUbPara.mainCoreUbPara.ubFactor;
            coreInnerHandler_.BaseProcessPre(processData);
            coreInnerHandler_.BaseProcess();
        }
    } else {
        for (int32_t j = 0; j < td_->nUbPara.tailCoreUbPara.ubCount; j++) {
            if (j == td_->nUbPara.tailCoreUbPara.ubCount - 1) {
                processData.ubFactorN = td_->nUbPara.tailCoreUbPara.ubTailFactor;
            } else {
                processData.ubFactorN = td_->nUbPara.tailCoreUbPara.ubFactor;
            }
            processData.offsetN =
                (td_->nBorrow - 1) * td_->nBlockPara.blockFactor + j * td_->nUbPara.tailCoreUbPara.ubFactor;
            coreInnerHandler_.BaseProcessPre(processData);
            coreInnerHandler_.BaseProcess();
        }
    }
}

template <typename T, typename PromtT, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsUbSsOnewaySs<T, PromtT, CoreInnerHandler>::ProcessNotBorrowN(
    UbSklanskyProcessData& processData)
{
    // N不借轴的场景, 核间借R轴，mBlockFactor必然为1
    for (int32_t j = 0; j < td_->nUbPara.tailCoreUbPara.ubCount; j++) {
        if (j == td_->nUbPara.tailCoreUbPara.ubCount - 1) {
            processData.ubFactorN = td_->nUbPara.tailCoreUbPara.ubTailFactor;
        } else {
            processData.ubFactorN = td_->nUbPara.tailCoreUbPara.ubFactor;
        }
        processData.offsetN = j * td_->nUbPara.tailCoreUbPara.ubFactor;
        coreInnerHandler_.BaseProcessPre(processData);
        coreInnerHandler_.BaseProcess();
    }
}

template <typename T, typename PromtT, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsUbSsOnewaySs<T, PromtT, CoreInnerHandler>::Process()
{
    if (this->blockIdx_ >= td_->realCoreNum) {
        return;
    }

    UbSklanskyProcessData processData;
    if (td_->reverse) {
        int64_t rOffset = (this->blockIdx_ / td_->nBorrow % td_->rBorrow + 1) * td_->rBlockPara.blockFactor;
        if (td_->lenR > rOffset) {
            processData.offsetR = td_->lenR - rOffset;
        } else {
            processData.offsetR = 0;
        }
    } else {
        processData.offsetR = this->blockIdx_ / td_->nBorrow % td_->rBorrow * td_->rBlockPara.blockFactor;
    }
    processData.offsetM = this->blockIdx_ / (td_->rBorrow * td_->nBorrow);
    processData.offsetN = this->blockIdx_ % td_->nBorrow * td_->nBlockPara.blockFactor;
    processData.isExclusive = td_->exclusive == 1 ? true : false;
    if (td_->exclusive && this->blockIdx_ / td_->nBorrow % td_->rBorrow != 0) {
        processData.isLenChange = true;
    }
    if (td_->nBlockPara.blockCount != 1) {
        ProcessBorrowN(processData);
    } else {
        ProcessNotBorrowN(processData);
    }

    CumsumCoreSklansky<T, PromtT>::Process();
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_H