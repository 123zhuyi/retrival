/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MirrorPadInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MirrorPadInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MirrorPadInfershapeTest TearDown" << std::endl; }
};

TEST_F(MirrorPadInfershapeTest, mirror_pad_infershape_case_0)
{
    gert::StorageShape xShape = {{
                                     6,
                                 },
                                 {
                                     6,
                                 }};
    // padShape和pad_value的形状要相同
    gert::StorageShape padShape = {{1, 2}, {1, 2}};
    int pad_value[2] = {2, 2};

    gert::InfershapeContextPara infershapeContextPara(
        "MirrorPad", {{xShape, ge::DT_INT32, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{
        10,
    }};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MirrorPadInfershapeTest, mirror_pad_infershape_case_1)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    int pad_value[2][2] = {{1, 2}, {3, 4}};

    gert::InfershapeContextPara infershapeContextPara(
        "MirrorPad", {{xShape, ge::DT_INT32, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MirrorPadInfershapeTest, mirror_pad_infershape_case_2)
{
    gert::StorageShape xShape = {{8, 16, 32}, {8, 16, 32}};
    gert::StorageShape padShape = {{3, 2}, {3, 2}};
    int pad_value[3][2] = {{1, 2}, {3, 4}, {5, 6}};

    gert::InfershapeContextPara infershapeContextPara(
        "MirrorPad", {{xShape, ge::DT_INT32, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {11, 23, 43},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MirrorPadInfershapeTest, mirror_pad_infershape_case_3_int64)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    int64_t pad_value[2][2] = {{1, 2}, {3, 4}};

    gert::InfershapeContextPara infershapeContextPara(
        "MirrorPad", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT64, ge::FORMAT_ND, true, pad_value}},
        {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MirrorPadInfershapeTest, mirror_pad_infershape_case_4_error_paddings_num)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape padShape = {{3, 2}, {3, 2}};
    int pad_value[3][2] = {{1, 2}, {3, 4}, {5, 6}};

    gert::InfershapeContextPara infershapeContextPara(
        "MirrorPad", {{xShape, ge::DT_INT32, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MirrorPadInfershapeTest, mirror_pad_infershape_case_5_error_negative_output)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    int pad_value[2][2] = {{-5, -1}, {-2, -1}};

    gert::InfershapeContextPara infershapeContextPara(
        "MirrorPad", {{xShape, ge::DT_INT32, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MirrorPadInfershapeTest, mirror_pad_infershape_case_6_unknown_rank)
{
    gert::StorageShape xShape = {{-2}, {-2}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    int pad_value[2][2] = {{1, 2}, {3, 4}};

    gert::InfershapeContextPara infershapeContextPara(
        "MirrorPad", {{xShape, ge::DT_INT32, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, pad_value}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MirrorPadInfershapeTest, mirror_pad_infershape_case_7_dynamic_paddings)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "MirrorPad", {{xShape, ge::DT_INT32, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, false}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: mirror_pad with const paddings of unsupported dtype (DT_FLOAT), expect infershape to fail
TEST_F(MirrorPadInfershapeTest, mirror_pad_infershape_case_8_unsupported_paddings_dtype)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    float pad_value[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};

    gert::InfershapeContextPara infershapeContextPara(
        "MirrorPad", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_FLOAT, ge::FORMAT_ND, true, pad_value}},
        {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
