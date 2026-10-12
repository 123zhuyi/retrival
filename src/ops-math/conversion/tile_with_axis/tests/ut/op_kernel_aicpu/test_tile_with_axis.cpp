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

#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

#undef private
#undef protected

#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_TILEWITHAXIS_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, axis, tiles)     \
    auto node_def = CpuKernelUtils::CreateNodeDef();               \
    NodeDefBuilder(node_def.get(), "TileWithAxis", "TileWithAxis") \
        .Input({"x", data_types[0], shapes[0], datas[0]})          \
        .Output({"y", data_types[1], shapes[1], datas[1]})         \
        .Attr("axis", axis)                                        \
        .Attr("tiles", tiles);

// Helper: compute expected output for TileWithAxis using modular indexing
template <typename T>
void ComputeTileExpected(const T* input, const vector<int64_t>& in_shape, T* expected, const vector<int64_t>& out_shape,
                         int64_t axis)
{
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (axis < 0) {
        axis += ndim;
    }

    int64_t out_total = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        out_total *= out_shape[i];
    }

    for (int64_t out_idx = 0; out_idx < out_total; ++out_idx) {
        int64_t tmp = out_idx;
        vector<int64_t> out_indices(ndim);
        for (int64_t i = ndim - 1; i >= 0; --i) {
            out_indices[i] = tmp % out_shape[i];
            tmp /= out_shape[i];
        }

        vector<int64_t> in_indices = out_indices;
        in_indices[axis] = out_indices[axis] % in_shape[axis];

        int64_t in_idx = 0;
        int64_t stride = 1;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            in_idx += in_indices[i] * stride;
            stride *= in_shape[i];
        }

        expected[out_idx] = input[in_idx];
    }
}

// Test 1: Float 3D, axis=1, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Float3D_Axis1_Tiles2)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {2, 3, 2};
    vector<int64_t> out_shape = {2, 6, 2};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    float input[12] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float output[24] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 1, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float expect[24] = {0};
    ComputeTileExpected<float>(input, in_shape, expect, out_shape, 1);
    CompareResult<float>(output, expect, 24);
}

// Test 2: Int32 3D, axis=1, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Int32_3D_Axis1_Tiles2)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32};
    vector<int64_t> in_shape = {3, 2, 1};
    vector<int64_t> out_shape = {3, 4, 1};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    int32_t input[6] = {1, 2, 3, 4, 5, 6};
    int32_t output[12] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 1, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t expect[12] = {0};
    ComputeTileExpected<int32_t>(input, in_shape, expect, out_shape, 1);
    CompareResult<int32_t>(output, expect, 12);
}

// Test 3: Int64 4D, axis=1, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Int64_4D_Axis1_Tiles2)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64};
    vector<int64_t> in_shape = {1, 2, 3, 4};
    vector<int64_t> out_shape = {1, 4, 3, 4};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    int64_t input[24];
    for (int i = 0; i < 24; ++i) {
        input[i] = i + 1;
    }
    int64_t output[48] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 1, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int64_t expect[48] = {0};
    ComputeTileExpected<int64_t>(input, in_shape, expect, out_shape, 1);
    CompareResult<int64_t>(output, expect, 48);
}

// Test 4: Negative axis (-3 on 4D = axis 1)
TEST_F(TEST_TILEWITHAXIS_UT, Int64_4D_NegAxis)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64};
    vector<int64_t> in_shape = {1, 2, 3, 4};
    vector<int64_t> out_shape = {1, 4, 3, 4};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    int64_t input[24];
    for (int i = 0; i < 24; ++i) {
        input[i] = i + 1;
    }
    int64_t output[48] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, -3, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int64_t expect[48] = {0};
    ComputeTileExpected<int64_t>(input, in_shape, expect, out_shape, -3);
    CompareResult<int64_t>(output, expect, 48);
}

// Test 5: tiles=1, identity copy
TEST_F(TEST_TILEWITHAXIS_UT, Int32_Tiles1_Identity)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32};
    vector<int64_t> in_shape = {2, 3};
    vector<int64_t> out_shape = {2, 3};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    int32_t input[6] = {1, 2, 3, 4, 5, 6};
    int32_t output[6] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 0, 1);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t expect[6] = {0};
    ComputeTileExpected<int32_t>(input, in_shape, expect, out_shape, 0);
    CompareResult<int32_t>(output, expect, 6);
}

// Test 6: Int8 3D, axis=2, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Int8_3D_Axis2_Tiles2)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8};
    vector<int64_t> in_shape = {2, 3, 1};
    vector<int64_t> out_shape = {2, 3, 2};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    int8_t input[6] = {1, 2, 3, 4, 5, 6};
    int8_t output[12] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 2, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int8_t expect[12] = {0};
    ComputeTileExpected<int8_t>(input, in_shape, expect, out_shape, 2);
    CompareResult<int8_t>(output, expect, 12);
}

// Test 7: Uint8 3D, axis=2, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Uint8_3D_Axis2_Tiles2)
{
    vector<DataType> data_types = {DT_UINT8, DT_UINT8};
    vector<int64_t> in_shape = {2, 3, 1};
    vector<int64_t> out_shape = {2, 3, 2};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    uint8_t input[6] = {10, 20, 30, 40, 50, 60};
    uint8_t output[12] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 2, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t expect[12] = {0};
    ComputeTileExpected<uint8_t>(input, in_shape, expect, out_shape, 2);
    CompareResult<uint8_t>(output, expect, 12);
}

// Test 8: Int16 3D, axis=0, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Int16_3D_Axis0_Tiles2)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16};
    vector<int64_t> in_shape = {2, 3, 2};
    vector<int64_t> out_shape = {4, 3, 2};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    int16_t input[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int16_t output[24] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 0, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int16_t expect[24] = {0};
    ComputeTileExpected<int16_t>(input, in_shape, expect, out_shape, 0);
    CompareResult<int16_t>(output, expect, 24);
}

// Test 9: Uint16 3D, axis=0, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Uint16_3D_Axis0_Tiles2)
{
    vector<DataType> data_types = {DT_UINT16, DT_UINT16};
    vector<int64_t> in_shape = {2, 3, 2};
    vector<int64_t> out_shape = {4, 3, 2};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    uint16_t input[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    uint16_t output[24] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 0, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint16_t expect[24] = {0};
    ComputeTileExpected<uint16_t>(input, in_shape, expect, out_shape, 0);
    CompareResult<uint16_t>(output, expect, 24);
}

// Test 10: Uint32 3D, axis=1, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Uint32_3D_Axis1_Tiles2)
{
    vector<DataType> data_types = {DT_UINT32, DT_UINT32};
    vector<int64_t> in_shape = {3, 2, 1};
    vector<int64_t> out_shape = {3, 4, 1};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    uint32_t input[6] = {100, 200, 300, 400, 500, 600};
    uint32_t output[12] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 1, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint32_t expect[12] = {0};
    ComputeTileExpected<uint32_t>(input, in_shape, expect, out_shape, 1);
    CompareResult<uint32_t>(output, expect, 12);
}

// Test 11: Uint64 3D, axis=1, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Uint64_3D_Axis1_Tiles2)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64};
    vector<int64_t> in_shape = {1, 2, 3};
    vector<int64_t> out_shape = {1, 4, 3};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    uint64_t input[6] = {1, 2, 3, 4, 5, 6};
    uint64_t output[12] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 1, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint64_t expect[12] = {0};
    ComputeTileExpected<uint64_t>(input, in_shape, expect, out_shape, 1);
    CompareResult<uint64_t>(output, expect, 12);
}

// Test 12: 1D input, axis=0, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Int32_1D_Axis0_Tiles2)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32};
    vector<int64_t> in_shape = {2};
    vector<int64_t> out_shape = {4};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    int32_t input[2] = {42, 99};
    int32_t output[4] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 0, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t expect[4] = {0};
    ComputeTileExpected<int32_t>(input, in_shape, expect, out_shape, 0);
    CompareResult<int32_t>(output, expect, 4);
}

// Test 13: 2D input, axis=0, tiles=2
TEST_F(TEST_TILEWITHAXIS_UT, Int32_2D_Axis0_Tiles2)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32};
    vector<int64_t> in_shape = {1, 2};
    vector<int64_t> out_shape = {2, 2};
    vector<vector<int64_t>> shapes = {in_shape, out_shape};

    int32_t input[2] = {7, 8};
    int32_t output[4] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 0, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t expect[4] = {0};
    ComputeTileExpected<int32_t>(input, in_shape, expect, out_shape, 0);
    CompareResult<int32_t>(output, expect, 4);
}

// Test 14: FAIL - axis out of range (axis=4, 2D input)
TEST_F(TEST_TILEWITHAXIS_UT, Fail_AxisOutOfRange)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{1, 2}, {2, 2}};
    int64_t input[2] = {0};
    int64_t output[4] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 4, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// Test 15: FAIL - negative axis out of range (axis=-4, 2D input)
TEST_F(TEST_TILEWITHAXIS_UT, Fail_NegAxisOutOfRange)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{1, 2}, {2, 2}};
    int64_t input[2] = {0};
    int64_t output[4] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, -4, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// Test 16: FAIL - negative tiles (tiles=-2)
TEST_F(TEST_TILEWITHAXIS_UT, Fail_NegativeTiles)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{1, 2}, {2, 2}};
    int64_t input[2] = {0};
    int64_t output[4] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 0, -2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// Test 17: FAIL - shape mismatch (input[axis]*tiles != output[axis])
TEST_F(TEST_TILEWITHAXIS_UT, Fail_ShapeMismatch)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{1, 2}, {4, 2}};
    int64_t input[2] = {0};
    int64_t output[8] = {0};
    vector<void*> datas = {(void*)input, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, 1, 2);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
