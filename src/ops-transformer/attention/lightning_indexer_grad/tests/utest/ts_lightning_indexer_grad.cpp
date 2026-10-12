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
 * \file ts_lightning_indexer_grad.h
 * \brief LightningIndexerGrad UTest 相关基类定义.
 */
#include "ts_lightning_indexer_grad.h"
#include "tiling/lig/tiling_data.h"
#include "tiling/lig/tiling_stub.h"
#include "../../op_kernel/lightning_indexer_grad.cpp"

using LayoutType = ops::adv::tests::LIG::LightningIndexerGradParam::LayoutType;
using namespace ops::adv::tests::LIG;

TEST_F(Ts_LightningIndexerGrad, lightning_indexer_grad_normal_case)
{
    LightningIndexerGradCase cs([](LIGHTNING_INDEXER_GRAD_PARAM_){::lightning_indexer_grad<1, 0>(LIG_INPUT_PARAMS);});
    cs.mParam.batch_ = 3;
    cs.mParam.seqlenQ_ = 8;
    cs.mParam.seqlenK_ = 2048;
    cs.mParam.topK_ = 2048;
    cs.mParam.headNumQ_ = 64;
    cs.mParam.headNumK_ = 1;
    cs.mParam.groupNum_ = 64;
    cs.mParam.headDim_ = 128;
    cs.mParam.layoutType_ = LayoutType::BSND_SHAPE;
    cs.mParam.actualSeqLenQuery_ = {};
    cs.mParam.actualSeqLenKey_ = {};
    cs.mParam.sparseMode_ = 3;
    cs.mParam.preTokens_ = 65536;
    cs.mParam.nextTokens_ = 65536;

    cs.mOpInfo.mExp.mTilingKeys[0] = ExpectInfo::kInvalidTilingKey;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    cs.Init(static_cast<int32_t>(this->socVersion_));
    ASSERT_TRUE(cs.Run());
}