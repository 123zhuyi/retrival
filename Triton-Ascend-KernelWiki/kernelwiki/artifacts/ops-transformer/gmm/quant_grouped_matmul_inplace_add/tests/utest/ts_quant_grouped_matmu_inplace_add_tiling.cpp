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
 * \file ts_quant_grouped_matmul_inplace_add_tiling.cpp
 * \brief QuantGroupedMatmulInplaceAdd tiling用例.
 */

#include "ts_quant_grouped_matmul_inplace_add.h"

using qgmmiaTestParam::Ts_QuantGroupedMatmulInplaceAdd_WithParam_Ascend910_9591;
namespace {
const auto Tc_QuantGroupedMatmulInplaceAdd_Tiling_Case = ::testing::Values(QuantGroupedMatmulInplaceAddCase(
    "QuantGroupedQuantInplaceAddMM_mxfp8_Case0", true, "", /* CaseName, Enable, DebugInfo */
    OpInfo(ControlInfo(true, false),
           ExpectInfo(true, 4, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
    Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_FLOAT8_E5M2),
           GenTensorList("x2", {{512, 128}}, ge::DataType::DT_FLOAT8_E5M2),
           GenTensorList("scale2", {{12, 128, 2}}, ge::DataType::DT_FLOAT8_E8M0),
           GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT)},
          GenTensor("scale1", {12, 96, 2}, ge::DataType::DT_FLOAT8_E8M0),
          GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 1, 1, 1}, 0, 0),
    0));

INSTANTIATE_TEST_SUITE_P(QuantGroupedMatmulInplaceAdd_David, Ts_QuantGroupedMatmulInplaceAdd_WithParam_Ascend910_9591,
                         Tc_QuantGroupedMatmulInplaceAdd_Tiling_Case);
TEST_P(Ts_QuantGroupedMatmulInplaceAdd_WithParam_Ascend910_9591, Tc_Tiling_QGmmInplaceAdd)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

} // namespace