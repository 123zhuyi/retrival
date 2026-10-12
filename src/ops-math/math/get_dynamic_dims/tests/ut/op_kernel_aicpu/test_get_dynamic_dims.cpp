/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/aicpu_test_utils.h"
#undef private
#undef protected

#include <vector>

using namespace aicpu;
using std::vector;

namespace {
const vector<int64_t> kShapeInfo = {4, 3, 2, -1, 1, 3, 1, 2, 1, 5, 16, -1, -1, 3, 4};

template <typename T>
void AddInput(NodeDef* node_def, const vector<T>& data, DataType dtype)
{
    auto input_tensor = node_def->AddInputs();
    ASSERT_NE(input_tensor, nullptr);
    input_tensor->GetTensorShape()->SetDimSizes({static_cast<int64_t>(data.size())});
    input_tensor->SetDataType(dtype);
    input_tensor->SetData(const_cast<T*>(data.data()));
    input_tensor->SetDataSize(data.size() * sizeof(T));
}

template <typename T>
uint32_t RunGetDynamicDims(const vector<vector<T>>& inputs, vector<T>& dims, DataType dtype,
                           const vector<int64_t>& shape_info = kShapeInfo, int64_t n = 3,
                           DataType output_dtype = DT_UNDEFINED)
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    node_def->SetOpType("GetDynamicDims");
    for (const auto& input : inputs) {
        AddInput(node_def.get(), input, dtype);
    }

    auto n_attr = CpuKernelUtils::CreateAttrValue();
    n_attr->SetInt(n);
    node_def->AddAttrs("N", n_attr.get());

    auto shape_attr = CpuKernelUtils::CreateAttrValue();
    shape_attr->SetListInt(shape_info);
    node_def->AddAttrs("shape_info", shape_attr.get());

    auto output_tensor = node_def->AddOutputs();
    // ASSERT_* 宏只能用于 void 函数, 此函数返回 uint32_t, 使用 EXPECT_NE 并显式返回
    EXPECT_NE(output_tensor, nullptr);
    if (output_tensor == nullptr) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    output_tensor->GetTensorShape()->SetDimSizes({static_cast<int64_t>(dims.size())});
    output_tensor->SetDataType(output_dtype == DT_UNDEFINED ? dtype : output_dtype);
    output_tensor->SetData(dims.data());
    output_tensor->SetDataSize(dims.size() * sizeof(T));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(node_def.get()), KERNEL_STATUS_OK);
    return CpuKernelRegister::Instance().RunCpuKernel(ctx);
}
} // namespace

class GetDynamicDimsAicpuTest : public testing::Test {};

TEST_F(GetDynamicDimsAicpuTest, Int32Success)
{
    vector<vector<int32_t>> inputs = {{3, 2, 4, 1}, {1, 2, 1}, {16, 112, 112, 3, 4}};
    vector<int32_t> dims(3);

    EXPECT_EQ(RunGetDynamicDims(inputs, dims, DT_INT32), KERNEL_STATUS_OK);
    EXPECT_EQ(dims, vector<int32_t>({4, 112, 112}));
}

TEST_F(GetDynamicDimsAicpuTest, Int64Success)
{
    vector<vector<int64_t>> inputs = {{3, 2, 4, 1}, {1, 2, 1}, {16, 112, 112, 3, 4}};
    vector<int64_t> dims(3);

    EXPECT_EQ(RunGetDynamicDims(inputs, dims, DT_INT64), KERNEL_STATUS_OK);
    EXPECT_EQ(dims, vector<int64_t>({4, 112, 112}));
}

TEST_F(GetDynamicDimsAicpuTest, OutputNumFail)
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    node_def->SetOpType("GetDynamicDims");
    vector<int32_t> x1 = {3, 2, 4, 1};
    vector<int32_t> x2 = {1, 2, 1};
    vector<int32_t> x3 = {16, 112, 112, 3, 4};
    AddInput(node_def.get(), x1, DT_INT32);
    AddInput(node_def.get(), x2, DT_INT32);
    AddInput(node_def.get(), x3, DT_INT32);

    auto n_attr = CpuKernelUtils::CreateAttrValue();
    n_attr->SetInt(3);
    node_def->AddAttrs("N", n_attr.get());
    auto shape_attr = CpuKernelUtils::CreateAttrValue();
    shape_attr->SetListInt(kShapeInfo);
    node_def->AddAttrs("shape_info", shape_attr.get());

    vector<int32_t> dims(3);
    auto output_tensor = node_def->AddOutputs();
    ASSERT_NE(output_tensor, nullptr);
    output_tensor->GetTensorShape()->SetDimSizes({3});
    output_tensor->SetDataType(DT_INT32);
    output_tensor->SetData(dims.data());
    output_tensor->SetDataSize(dims.size() * sizeof(int32_t));
    auto extra_output_tensor = node_def->AddOutputs();
    ASSERT_NE(extra_output_tensor, nullptr);
    extra_output_tensor->GetTensorShape()->SetDimSizes({3});
    extra_output_tensor->SetDataType(DT_INT32);
    extra_output_tensor->SetData(dims.data());
    extra_output_tensor->SetDataSize(dims.size() * sizeof(int32_t));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(node_def.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(GetDynamicDimsAicpuTest, OutputDtypeFail)
{
    vector<vector<int32_t>> inputs = {{3, 2, 4, 1}, {1, 2, 1}, {16, 112, 112, 3, 4}};
    vector<int32_t> dims(3);

    EXPECT_EQ(RunGetDynamicDims(inputs, dims, DT_INT32, kShapeInfo, 3, DT_FLOAT), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(GetDynamicDimsAicpuTest, InputNumNotMatchNFail)
{
    vector<vector<int32_t>> inputs = {{3, 2, 4, 1}, {1, 2, 1}, {16, 112, 112, 3, 4}};
    vector<int32_t> dims(3);

    EXPECT_EQ(RunGetDynamicDims(inputs, dims, DT_INT32, kShapeInfo, 2), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(GetDynamicDimsAicpuTest, InputRankNotMatchShapeInfoFail)
{
    vector<vector<int32_t>> inputs = {{3, 2, 4, 1}, {1, 2}, {16, 112, 112, 3, 4}};
    vector<int32_t> dims(3);

    EXPECT_EQ(RunGetDynamicDims(inputs, dims, DT_INT32), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(GetDynamicDimsAicpuTest, IncompleteShapeInfoFail)
{
    vector<vector<int32_t>> inputs = {{3, 2, 4, 1}};
    vector<int32_t> dims(1);
    vector<int64_t> invalid_shape_info = {4, 3, 2};

    EXPECT_EQ(RunGetDynamicDims(inputs, dims, DT_INT32, invalid_shape_info, 1), KERNEL_STATUS_PARAM_INVALID);
}
