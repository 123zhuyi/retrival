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
 * \file cumsum_core_ss_oneway_ss.h
 * \brief calculate the prefix accumulation sum of tensor specified dimension
 */
#ifndef CANN_OPS_BUILT_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_CUMSUM_ONEWAY_SS_H
#define CANN_OPS_BUILT_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_CUMSUM_ONEWAY_SS_H

#include "cumsum_base/cumsum_core_sklansky.h"
#include "cumsum_base/cumsum_oneway_sklansky.h"

namespace Cumsum {
using namespace AscendC;

template <typename DataType, typename PromoteDataType, typename CoreInnerHandler>
class CumsumCoreSsOnewaySs : public CumsumCoreSklansky<DataType, PromoteDataType> {
public:
    __aicore__ inline CumsumCoreSsOnewaySs(TPipe& pipe)
        : CumsumCoreSklansky<DataType, PromoteDataType>(pipe), coreInnerHandler_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace);

    __aicore__ inline void Process();

private:
    CoreInnerHandler coreInnerHandler_;
    const CumsumSklanskyTilingData* td_;
}; // class CumsumCoreSsOnewaySs

template <typename DataType, typename PromoteDataType, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsOnewaySs<DataType, PromoteDataType, CoreInnerHandler>::Init(
    GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* tilingData, GM_ADDR workspace)
{
    CumsumCoreSklansky<DataType, PromoteDataType>::Init(x, y, tilingData);
    td_ = tilingData;

    OnewaySklanskyInitData initData;
    initData.mLen = td_->lenM;
    initData.rLen = td_->lenR;
    initData.nLen = td_->lenN;
    initData.reverse = td_->reverse;
    initData.ubBufferSize = td_->xBufSize;
    coreInnerHandler_.BaseInit(x, y, workspace, initData);
}

template <typename DataType, typename PromoteDataType, typename CoreInnerHandler>
__aicore__ inline void CumsumCoreSsOnewaySs<DataType, PromoteDataType, CoreInnerHandler>::Process()
{
    if (this->blockIdx_ < td_->realCoreNum) {
        OnewaySklanskyProcessData processData;
        if (td_->exclusive == 1 && this->blockIdx_ / td_->nBorrow % td_->rBorrow == 0) {
            processData.isLenChange = true;
        }
        if (td_->reverse) {
            int64_t offsetR = (this->blockIdx_ / td_->nBorrow % td_->rBorrow + 1) * td_->rBlockPara.blockFactor;
            if (td_->lenR > offsetR) {
                processData.rOffset = td_->lenR - offsetR;
            } else {
                processData.rOffset = 0;
            }
        } else {
            processData.rOffset = this->blockIdx_ / td_->nBorrow % td_->rBorrow * td_->rBlockPara.blockFactor;
        }
        processData.mOffset = this->blockIdx_ / (td_->rBorrow * td_->nBorrow);
        processData.nOffset = this->blockIdx_ % td_->nBorrow * td_->nBlockPara.blockFactor;
        processData.mFactor = 1;

        processData.exclusive = td_->exclusive;
        if (this->blockIdx_ % td_->nBorrow == td_->nBlockPara.blockCount - 1) { // 尾核
            processData.nFactor = td_->nBlockPara.blockTailFactor;
        } else {
            processData.nFactor = td_->nBlockPara.blockFactor;
        }

        if (this->blockIdx_ / td_->nBorrow % td_->rBorrow == td_->rBlockPara.blockCount - 1) { // 尾核
            processData.rFactor = td_->rUbPara.tailCoreUbPara.ubTailFactor;
        } else {
            processData.rFactor = td_->rUbPara.mainCoreUbPara.ubTailFactor;
        }

        coreInnerHandler_.BaseProcessPre(processData);
        coreInnerHandler_.BaseProcess();
        coreInnerHandler_.BaseCopyOut();
    }

    CumsumCoreSklansky<DataType, PromoteDataType>::Process();
}

} // namespace Cumsum

#endif
