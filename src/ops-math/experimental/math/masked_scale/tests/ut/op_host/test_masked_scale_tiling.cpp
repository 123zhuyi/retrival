// ----------------------------------------------------------------------------
// Copyright (c) Huawei Device Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/arch35/masked_scale_tiling_data.h"
#include "../../../op_kernel/arch35/masked_scale_tiling_key.h"

using namespace ge;

namespace optiling {
struct MaskedScaleCompileInfo {};
} // namespace optiling

class MaskedScaleTilingTest : public testing::Test {};

namespace {
constexpr size_t WORKSPACE_SIZE = 16U * 1024U * 1024U;

void RunSuccessCase(ge::DataType selfDtype, ge::DataType maskDtype, int64_t expectTilingKey,
                    std::initializer_list<int64_t> shape)
{
    optiling::MaskedScaleCompileInfo compileInfo;
    gert::StorageShape storageShape{shape, shape};
    std::vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        {storageShape, selfDtype, FORMAT_ND},
        {storageShape, maskDtype, FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        {storageShape, selfDtype, FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"scale", Ops::Math::AnyValue::CreateFrom<float>(0.5f)},
    };
    gert::TilingContextPara para("MaskedScale", inputDescs, outputDescs, attrs, &compileInfo);

    std::vector<size_t> expectWorkspace = {WORKSPACE_SIZE};
    ExecuteTestCase(para, GRAPH_SUCCESS, expectTilingKey, expectWorkspace);
}
} // namespace

TEST_F(MaskedScaleTilingTest, fp16_uint8_success)
{
    optiling::MaskedScaleCompileInfo compileInfo;
    gert::TilingContextPara para(
        "MaskedScale",
        {
            {{{1024}, {1024}}, DT_FLOAT16, FORMAT_ND},
            {{{1024}, {1024}}, DT_UINT8, FORMAT_ND},
        },
        {
            {{{1024}, {1024}}, DT_FLOAT16, FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(0.5f)),
        },
        &compileInfo);

    std::vector<size_t> expectWorkspace = {WORKSPACE_SIZE};
    ExecuteTestCase(para, GRAPH_SUCCESS, MASKED_SCALE_KEY_FP16_UINT8, expectWorkspace);
}

TEST_F(MaskedScaleTilingTest, fp32_float_success)
{
    optiling::MaskedScaleCompileInfo compileInfo;
    gert::TilingContextPara para(
        "MaskedScale",
        {
            {{{4097}, {4097}}, DT_FLOAT, FORMAT_ND},
            {{{4097}, {4097}}, DT_FLOAT, FORMAT_ND},
        },
        {
            {{{4097}, {4097}}, DT_FLOAT, FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(2.0f)),
        },
        &compileInfo);
    std::vector<size_t> expectWorkspace = {WORKSPACE_SIZE};
    ExecuteTestCase(para, GRAPH_SUCCESS, MASKED_SCALE_KEY_FP32_FP32, expectWorkspace);
}

TEST_F(MaskedScaleTilingTest, bf16_int8_success)
{
    optiling::MaskedScaleCompileInfo compileInfo;
    gert::TilingContextPara para(
        "MaskedScale",
        {
            {{{2, 8}, {2, 8}}, DT_BF16, FORMAT_ND},
            {{{2, 8}, {2, 8}}, DT_INT8, FORMAT_ND},
        },
        {
            {{{2, 8}, {2, 8}}, DT_BF16, FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(-1.0f)),
        },
        &compileInfo);
    std::vector<size_t> expectWorkspace = {WORKSPACE_SIZE};
    ExecuteTestCase(para, GRAPH_SUCCESS, MASKED_SCALE_KEY_BF16_INT8, expectWorkspace);
}

TEST_F(MaskedScaleTilingTest, all_dtype_combinations_success)
{
    RunSuccessCase(DT_FLOAT16, DT_INT8, MASKED_SCALE_KEY_FP16_INT8, {32});
    RunSuccessCase(DT_FLOAT16, DT_FLOAT16, MASKED_SCALE_KEY_FP16_FP16, {32});
    RunSuccessCase(DT_FLOAT16, DT_FLOAT, MASKED_SCALE_KEY_FP16_FP32, {32});
    RunSuccessCase(DT_FLOAT, DT_UINT8, MASKED_SCALE_KEY_FP32_UINT8, {32});
    RunSuccessCase(DT_FLOAT, DT_INT8, MASKED_SCALE_KEY_FP32_INT8, {32});
    RunSuccessCase(DT_FLOAT, DT_FLOAT16, MASKED_SCALE_KEY_FP32_FP16, {32});
    RunSuccessCase(DT_BF16, DT_UINT8, MASKED_SCALE_KEY_BF16_UINT8, {32});
    RunSuccessCase(DT_BF16, DT_FLOAT16, MASKED_SCALE_KEY_BF16_FP16, {32});
    RunSuccessCase(DT_BF16, DT_FLOAT, MASKED_SCALE_KEY_BF16_FP32, {32});
}

TEST_F(MaskedScaleTilingTest, empty_tensor_success)
{
    RunSuccessCase(DT_FLOAT, DT_UINT8, MASKED_SCALE_KEY_FP32_UINT8, {0});
}

TEST_F(MaskedScaleTilingTest, high_dim_shape_success)
{
    RunSuccessCase(DT_FLOAT16, DT_FLOAT, MASKED_SCALE_KEY_FP16_FP32, {2, 3, 4, 5});
}

TEST_F(MaskedScaleTilingTest, unsupported_self_dtype_failed)
{
    optiling::MaskedScaleCompileInfo compileInfo;
    gert::TilingContextPara para(
        "MaskedScale",
        {
            {{{128}, {128}}, DT_INT32, FORMAT_ND},
            {{{128}, {128}}, DT_UINT8, FORMAT_ND},
        },
        {
            {{{128}, {128}}, DT_INT32, FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
        },
        &compileInfo);
    ExecuteTestCase(para, GRAPH_FAILED, 0, "", {});
}

TEST_F(MaskedScaleTilingTest, output_dtype_mismatch_failed)
{
    optiling::MaskedScaleCompileInfo compileInfo;
    gert::TilingContextPara para(
        "MaskedScale",
        {
            {{{128}, {128}}, DT_FLOAT16, FORMAT_ND},
            {{{128}, {128}}, DT_FLOAT16, FORMAT_ND},
        },
        {
            {{{128}, {128}}, DT_FLOAT, FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
        },
        &compileInfo);
    ExecuteTestCase(para, GRAPH_FAILED, 0, "", {});
}

TEST_F(MaskedScaleTilingTest, unsupported_mask_dtype_failed)
{
    optiling::MaskedScaleCompileInfo compileInfo;
    gert::TilingContextPara para(
        "MaskedScale",
        {
            {{{128}, {128}}, DT_FLOAT, FORMAT_ND},
            {{{128}, {128}}, DT_INT32, FORMAT_ND},
        },
        {
            {{{128}, {128}}, DT_FLOAT, FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(1.0f)),
        },
        &compileInfo);
    ExecuteTestCase(para, GRAPH_FAILED, 0, "", {});
}
