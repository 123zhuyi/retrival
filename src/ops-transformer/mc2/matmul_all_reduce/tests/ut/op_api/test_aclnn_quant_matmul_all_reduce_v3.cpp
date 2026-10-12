/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "matmul_all_reduce_api_ut_param.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../op_api/aclnn_quant_matmul_all_reduce_v3.h"

namespace MatmulAllReduceUT {

class AclnnQuantMatmulAllReduceV3Test : public testing::TestWithParam<MatmulAllReduceApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV3Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce AclnnQuantMatmulAllReduceV3Test TearDown" << std::endl;
    }
};

TEST_P(AclnnQuantMatmulAllReduceV3Test, param)
{
    auto param = GetParam();
    auto ut = OP_API_UT(
        aclnnQuantMatmulAllReduceV3,
        INPUT(param.x1, param.x2, param.bias, param.x3, param.dequantScale, param.pertokenScale, param.commQuantScale1,
              param.commQuantScale2, param.group.c_str(), param.reduceOp.c_str(), param.commTurn, param.streamMode),
        OUTPUT(param.output)
    );
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    AclnnQuantMatmulAllReduceV3Test,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceApiUtParam>
);

} // namespace MatmulAllReduceUT
