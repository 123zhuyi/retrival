/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_stateless_normal_infershape.cpp
 * \brief StatelessNormal InferShape UT
 *
 * 被测实现：op_host/stateless_normal_infershape.cpp
 *   → ops::randomCommon::CommonInferShape(context, {{"shape",0}}, {{"y",0}}, MODE_DEPENDENCY)
 *
 * 关键语义（random_infershape_base.cpp）：
 *   xShapeSize = inShape->GetShapeSize()   ← 输入 shape tensor 的元素个数 = 输出 rank
 *   DependencyMode() 按 shape tensor 的 dtype 分派：
 *     DT_INT64 → HandleShapeTensor<int64_t>，逐维写入 const 值
 *     DT_INT32 → HandleShapeTensor<int32_t>
 *     其它     → return false → GRAPH_FAILED
 *   const data 为 nullptr 时 → SetUnknownShape(xShapeSize) → 各维 -1
 *
 * 输入布局（与 op_def / tiling UT 一致）：
 *   [0] shape:  DT_INT64, 1D, const（值依赖，InputsDataDependency({0})）
 *   [1] seed:   DT_INT64 scalar
 *   [2] offset: DT_INT64 scalar
 *   [3] mean:   DT_FLOAT/DT_FLOAT16/DT_BF16 tensor（与输出同 shape）
 *   [4] stdev:  DT_FLOAT/DT_FLOAT16/DT_BF16 tensor（与输出同 shape）
 * 属性：dtype（OPTIONAL Int，0=float32 / 2=float16 / 3=bfloat16）
 *
 * 覆盖矩阵：
 *   1D 输出                     : case_1d
 *   2D 输出                     : case_2d
 *   4D 输出                     : case_4d
 *   含 1 的维度                  : case_dim_with_one
 *   float16 / bfloat16 输出 dtype: case_fp16 / case_bf16
 *   INT32 shape tensor 分支      : case_int32_shape
 *   const data 缺失 → unknown    : case_null_const_1d / case_null_const_2d
 *   非法 shape dtype → FAILED    : case_invalid_shape_dtype
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace std;

class StatelessNormalInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessNormalInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessNormalInferShapeTest TearDown" << std::endl; }
};

// case 1: 1D 输出。shape tensor = [16384]，1 个元素 → 输出 rank 1
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_1d)
{
    vector<int64_t> shapeValue = {16384};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16384}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 2: 2D 输出。shape tensor 含 2 个元素 → 输出 rank 2
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_2d)
{
    vector<int64_t> shapeValue = {32, 512};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 3: 4D 输出，验证高维逐维写入正确
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_4d)
{
    vector<int64_t> shapeValue = {2, 8, 32, 32};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 8, 32, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 4: 含 1 的维度，确认不会被误当作标量或被压缩掉
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_dim_with_one)
{
    vector<int64_t> shapeValue = {1, 4097, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4097}, {4097}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4097}, {4097}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 4097, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 5: float16 输出（dtype attr = 2）。infershape 不依赖 dtype，但保证该组合能通
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_fp16)
{
    vector<int64_t> shapeValue = {64, 256};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 6: bfloat16 输出（dtype attr = 3）
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_bf16)
{
    vector<int64_t> shapeValue = {32, 512};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 7: INT32 shape tensor → DependencyMode 的 HandleShapeTensor<int32_t> 分支
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_int32_shape)
{
    vector<int32_t> shapeValue = {16, 64};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 8: const data 缺失（1 元素）→ GetData 返回 nullptr → SetUnknownShape(1) → {-1}
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_null_const_1d)
{
    vector<int64_t> shapeValue = {};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 9: const data 缺失，shape tensor 有 2 个元素 → SetUnknownShape(2) → {-1,-1}
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_null_const_2d)
{
    vector<int64_t> shapeValue = {};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 10: 非法 shape dtype（DT_FLOAT）→ DependencyMode 两个分支都不命中 → GRAPH_FAILED
TEST_F(StatelessNormalInferShapeTest, stateless_normal_infershape_case_invalid_shape_dtype)
{
    vector<float> shapeValue = {32.0f, 512.0f};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessNormal",
        {
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
