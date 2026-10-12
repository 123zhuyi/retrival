/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

class ViewCopyInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ViewCopyInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ViewCopyInfershape TearDown" << std::endl; }
};

TEST_F(ViewCopyInfershape, copy_dst_storage_shape)
{
    gert::InfershapeContextPara infershapeContextPara("ViewCopy",
                                                      {
                                                          {{{12}, {12}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{20}, {20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {12},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ViewCopyInfershape, dynamic_dst_storage_shape)
{
    gert::InfershapeContextPara infershapeContextPara("ViewCopy",
                                                      {
                                                          {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
