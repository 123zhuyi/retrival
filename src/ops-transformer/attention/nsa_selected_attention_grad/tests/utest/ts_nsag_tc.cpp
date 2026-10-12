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
 * \file ts_nsag_param_tc.cpp
 * \brief nsaGrad 用例.
 */

#include "ts_nsag.h"

using Tensor = ops::adv::tests::utils::Tensor;

TEST_F(Ts_nsaGrad, case_nsa_selcted_attention_grad_0)
{
    nsaGradCase cs;
    cs.mParam.B = 64;
    cs.mParam.S1 = 1;
    cs.mParam.S2 = 1024;
    cs.mParam.N1 = 4;
    cs.mParam.N2 = 1;
    cs.mParam.D = 192;
    cs.mParam.D2 = 192;
    cs.mParam.SelectedBlockCount = 16;
    cs.mParam.SelectedBlockSize = 64;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.inputLayout = "TND";
    cs.mParam.sparseMode = 0;
    cs.mOpInfo.mExp.mTilingBlockDim = 24; // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_nsaGrad, case_nsa_selcted_attention_grad_1)
{
    nsaGradCase cs;
    cs.mParam.B = 64;
    cs.mParam.S1 = 1;
    cs.mParam.S2 = 1024;
    cs.mParam.N1 = 4;
    cs.mParam.N2 = 1;
    cs.mParam.D = 192;
    cs.mParam.D2 = 192;
    cs.mParam.SelectedBlockCount = 16;
    cs.mParam.SelectedBlockSize = 64;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.inputLayout = "TND";
    cs.mParam.sparseMode = 2;
    cs.mOpInfo.mExp.mTilingBlockDim = 24; // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_nsaGrad, case_nsa_selcted_attention_grad_diff_headdim_0)
{
    nsaGradCase cs;
    cs.mParam.B = 64;
    cs.mParam.S1 = 1;
    cs.mParam.S2 = 1024;
    cs.mParam.N1 = 4;
    cs.mParam.N2 = 1;
    cs.mParam.D = 192;
    cs.mParam.D2 = 128;
    cs.mParam.SelectedBlockCount = 16;
    cs.mParam.SelectedBlockSize = 64;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.inputLayout = "TND";
    cs.mParam.sparseMode = 0;
    cs.mOpInfo.mExp.mTilingBlockDim = 24; // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_nsaGrad, case_nsa_selcted_attention_grad_diff_headdim_1)
{
    nsaGradCase cs;
    cs.mParam.B = 64;
    cs.mParam.S1 = 1;
    cs.mParam.S2 = 1024;
    cs.mParam.N1 = 4;
    cs.mParam.N2 = 1;
    cs.mParam.D = 192;
    cs.mParam.D2 = 128;
    cs.mParam.SelectedBlockCount = 16;
    cs.mParam.SelectedBlockSize = 64;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.inputLayout = "TND";
    cs.mParam.sparseMode = 2;
    cs.mOpInfo.mExp.mTilingBlockDim = 24; // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_nsaGrad, case_nsa_selcted_attention_grad_diff_headdim_1_deterministic)
{
    nsaGradCase cs;
    cs.isDeterministic = true;
    cs.mParam.B = 64;
    cs.mParam.S1 = 1;
    cs.mParam.S2 = 1024;
    cs.mParam.N1 = 4;
    cs.mParam.N2 = 1;
    cs.mParam.D = 192;
    cs.mParam.D2 = 128;
    cs.mParam.SelectedBlockCount = 16;
    cs.mParam.SelectedBlockSize = 64;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.inputLayout = "TND";
    cs.mParam.sparseMode = 2;
    cs.mOpInfo.mExp.mTilingBlockDim = 24; // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_nsaGrad, case_nsa_selcted_attention_grad_diff_headdim_0_deterministic)
{
    nsaGradCase cs;
    cs.isDeterministic = true;
    cs.mParam.B = 64;
    cs.mParam.S1 = 1;
    cs.mParam.S2 = 1024;
    cs.mParam.N1 = 4;
    cs.mParam.N2 = 1;
    cs.mParam.D = 192;
    cs.mParam.D2 = 128;
    cs.mParam.SelectedBlockCount = 16;
    cs.mParam.SelectedBlockSize = 64;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.inputLayout = "TND";
    cs.mParam.sparseMode = 0;
    cs.mOpInfo.mExp.mTilingBlockDim = 24; // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}