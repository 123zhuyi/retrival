/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/reduce/reduce_tiling.h"

using namespace std;
using namespace ge;

class ReduceMaxDavidTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceMaxDavidTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReduceMaxDavidTiling TearDown" << std::endl;
    }
};

template <typename T>
static void SetConstInput(size_t const_index, ge::DataType dtype, T* const_data,
                          std::initializer_list<int64_t>& axesShape,
                          std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    int64_t dataSize = *(axesShape.begin());
    std::unique_ptr<uint8_t[]> input_tensor_holder =
        std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * dataSize]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor({axesShape, axesShape},              // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                        gert::kFollowing,                    // placement
                        dtype,                               // dt
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < dataSize; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

TEST_F(ReduceMaxDavidTiling, reduce_max_david_tiling1)
{   

    Ops::Base::ReduceOpCompileInfo compileInfo;
 	gert::StorageShape inputShape = {{2048, 0, 48}, {2048, 0, 48}};
 	gert::StorageShape axesShape = {{1}, {1}};
 	std::vector<int64_t> axesValue = {1};
 	gert::StorageShape yShape = {{2048, 48}, {2048, 48}};
 	gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
 	gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara tilingContextPara(
 	    "ReduceMax",
 	    {input, axes},
 	    {y},
 	    &compileInfo);
 	uint64_t expectedTilingKey = 1;
 	std::vector<size_t> expectedWorkspaces = { 16777216 };
 	ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(ReduceMaxDavidTiling, reduce_max_david_tiling2)
{
    // AR case
    Ops::Base::ReduceOpCompileInfo compileInfo;
 	gert::StorageShape inputShape = {{2048, 257}, {2048, 257}};
 	gert::StorageShape axesShape = {{1}, {1}};
 	std::vector<int64_t> axesValue = {1};
 	gert::StorageShape yShape = {{2048}, {2048}};
 	gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
 	gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara tilingContextPara(
 	    "ReduceMax",
 	    {input, axes},
 	    {y},
 	    &compileInfo);
 	uint64_t expectedTilingKey = 5143;
 	std::vector<size_t> expectedWorkspaces = { 16777216 };
 	ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(ReduceMaxDavidTiling, reduce_max_david_tiling3)
{
    // AR case, R < 256
    Ops::Base::ReduceOpCompileInfo compileInfo;
 	gert::StorageShape inputShape = {{2048, 127}, {2048, 127}};
 	gert::StorageShape axesShape = {{1}, {1}};
 	std::vector<int64_t> axesValue = {1};
 	gert::StorageShape yShape = {{2048}, {2048}};
 	gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
 	gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara tilingContextPara(
 	    "ReduceMax",
 	    {input, axes},
 	    {y},
 	    &compileInfo);
 	uint64_t expectedTilingKey = 5143;
 	std::vector<size_t> expectedWorkspaces = { 16777216 };
 	ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(ReduceMaxDavidTiling, reduce_max_david_tiling4)
{
    // RA case
    Ops::Base::ReduceOpCompileInfo compileInfo;
 	gert::StorageShape inputShape = {{2048, 257}, {2048, 257}};
 	gert::StorageShape axesShape = {{1}, {1}};
 	std::vector<int64_t> axesValue = {0};
 	gert::StorageShape yShape = {{2048}, {2048}};
 	gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
 	gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara tilingContextPara(
 	    "ReduceMax",
 	    {input, axes},
 	    {y},
 	    &compileInfo);
 	uint64_t expectedTilingKey = 11817;
 	std::vector<size_t> expectedWorkspaces = { 16859136 };
 	ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(ReduceMaxDavidTiling, reduce_max_david_tiling5)
{
    // RA case, A < 256
    Ops::Base::ReduceOpCompileInfo compileInfo;
 	gert::StorageShape inputShape = {{2048, 127}, {2048, 127}};
 	gert::StorageShape axesShape = {{1}, {1}};
 	std::vector<int64_t> axesValue = {1};
 	gert::StorageShape yShape = {{2048}, {2048}};
 	gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
 	gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara tilingContextPara(
 	    "ReduceMax",
 	    {input, axes},
 	    {y},
 	    &compileInfo);
 	uint64_t expectedTilingKey = 5143;
 	std::vector<size_t> expectedWorkspaces = { 16777216 };
 	ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(ReduceMaxDavidTiling, reduce_max_david_tiling6)
{
    // RA case, A < 256, int64
    Ops::Base::ReduceOpCompileInfo compileInfo;
 	gert::StorageShape inputShape = {{2048, 127}, {2048, 127}};
 	gert::StorageShape axesShape = {{1}, {1}};
 	std::vector<int64_t> axesValue = {0};
 	gert::StorageShape yShape = {{2048}, {2048}};
 	gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_INT64, ge::FORMAT_ND);
 	gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
 	gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara tilingContextPara(
 	    "ReduceMax",
 	    {input, axes},
 	    {y},
 	    &compileInfo);
 	uint64_t expectedTilingKey = 11817;
 	std::vector<size_t> expectedWorkspaces = { 16842752 };
 	ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(ReduceMaxDavidTiling, reduce_max_david_tiling7)
{
    // RA case, A < 256, float16
    Ops::Base::ReduceOpCompileInfo compileInfo;
 	gert::StorageShape inputShape = {{2048, 127}, {2048, 127}};
 	gert::StorageShape axesShape = {{1}, {1}};
 	std::vector<int64_t> axesValue = {0};
 	gert::StorageShape yShape = {{2048}, {2048}};
 	gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_FLOAT16, ge::FORMAT_ND);
 	gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
 	gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara tilingContextPara(
 	    "ReduceMax",
 	    {input, axes},
 	    {y},
 	    &compileInfo);
 	uint64_t expectedTilingKey = 6185;
 	std::vector<size_t> expectedWorkspaces = { 16793600 };
 	ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(ReduceMaxDavidTiling, reduce_max_david_tiling8)
{
    // RA case, A < 256, int8
    Ops::Base::ReduceOpCompileInfo compileInfo;
 	gert::StorageShape inputShape = {{2048, 127}, {2048, 127}};
 	gert::StorageShape axesShape = {{1}, {1}};
 	std::vector<int64_t> axesValue = {0};
 	gert::StorageShape yShape = {{2048}, {2048}};
 	gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_INT8, ge::FORMAT_ND);
 	gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
 	gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
 	gert::TilingContextPara tilingContextPara(
 	    "ReduceMax",
 	    {input, axes},
 	    {y},
 	    &compileInfo);
 	uint64_t expectedTilingKey = 6185;
 	std::vector<size_t> expectedWorkspaces = { 16793600 };
 	ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}