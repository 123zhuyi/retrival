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
 * \file ts_aclnn_nsa_compress.cpp
 * \brief NsaCompress ACLNN 测试用例.
 */

#include "ts_aclnn_nsa_compress.h"

namespace {
TEST_P(Ts_Aclnn_NsaCompress_WithParam_Ascend910B2, Tc_Aclnn_NsaCompress)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_NsaCompress_Aclnn_Case = ::testing::Values(

    AclnnNsaCompressCase("NsaCompress_Aclnn_Case_0", true, "",                   /* CaseName, Enable, DebugInfo */
                         OpInfo(ControlInfo(false, false),                         /* RunTiling, RunKernel */
                                ExpectInfo(true,                                 /* ExpectSuccess */
                                           ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                           ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                         AclnnNsaCompressParam(48, 32, 16,                       /* T, N, D */
                                               ge::DataType::DT_FLOAT16,         /* Dtype */
                                               {16, 32, 48},                     /* ActualSeqLenList */
                                               LayoutType::TND,                  /* Layout */
                                               16,                               /*CompressBlockSize*/
                                               16,                               /*CompressStride*/
                                               0                                 /*ActSeqLenType*/
                                               )),
    AclnnNsaCompressCase(
        "NsaCompress_Aclnn_Case_1", true, "",                   /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(false, false),                         /* RunTiling, RunKernel */
               ExpectInfo(true,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnNsaCompressParam(
            2320, 4, 192,             /* T, N, D */
            ge::DataType::DT_FLOAT16, /* Dtype */
            {93,   144,  150,  180,  204,  282,  330,  377,  417,  476,  503,  562,  643,  709,  765,  798,  887,
             975,  988,  1073, 1092, 1152, 1174, 1223, 1305, 1357, 1384, 1389, 1399, 1403, 1442, 1454, 1510, 1606,
             1696, 1767, 1810, 1856, 1917, 1991, 2027, 2067, 2073, 2087, 2172, 2223, 2225, 2320}, /* ActualSeqLenList */
            LayoutType::TND,                                                                      /* Layout */
            16,                                                                                   /*CompressBlockSize*/
            16,                                                                                   /*CompressStride*/
            0                                                                                     /*ActSeqLenType*/
            ))

);


INSTANTIATE_TEST_SUITE_P(NsaCompress, Ts_Aclnn_NsaCompress_WithParam_Ascend910B2, Tc_NsaCompress_Aclnn_Case);
} // namespace