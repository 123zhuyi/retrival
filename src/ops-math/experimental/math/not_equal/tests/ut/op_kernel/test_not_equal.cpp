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
#include "data_utils.h"
#include "tikicpulib.h"
#include "../../../op_kernel/not_equal_tiling_data.h"

using TestUtDefaultTilingStruct = NotEqualTilingData;

constexpr int min(int in1, int in2) { return std::min(in1, in2); }

#include "../../../op_kernel/not_equal.cpp"

void* GmAllocAlign(size_t size) { return GmAlloc(size + 31 >> 5 << 5); }

template <typename T>
constexpr const char* GetTorchType()
{
    if constexpr (std::is_same_v<T, half>)
        return "float16";
    else if constexpr (std::is_same_v<T, float>)
        return "float";
    else if constexpr (std::is_same_v<T, int>)
        return "int32";
    else if constexpr (std::is_same_v<T, int8_t>)
        return "int8";
    else if constexpr (std::is_same_v<T, uint8_t>)
        return "uint8";
    else if constexpr (std::is_same_v<T, bool>)
        return "bool";
    else if constexpr (std::is_same_v<T, bfloat16_t>)
        return "bfloat16";
    else
        static_assert(!std::is_same_v<T, T>);
}

template <typename... Args>
int ExecuteCommand(const char* path, Args... args)
{
    std::string command = path;
    ((command = command + " '" + args + "'"), ...);
    return std::system(command.c_str());
}

class not_equal_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "not_equal_test SetUp\n" << std::endl; }

    static void TearDownTestCase() { std::cout << "not_equal_test TearDown\n" << std::endl; }
};

TEST_F(not_equal_test, test_case_0)
{
    size_t x1_size = 94000 * sizeof(DTYPE_X1);
    size_t x2_size = 94000 * sizeof(DTYPE_X2);
    size_t y_size = 94000 * sizeof(DTYPE_Y);
    size_t workspace_size = 16 << 20;
    size_t tiling_size = sizeof(NotEqualTilingData);
    uint32_t block_dim = 20;

    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(x1_size));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(x2_size));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(y_size));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(workspace_size));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(tiling_size));

    auto tiling_data = reinterpret_cast<NotEqualTilingData*>(tiling);
    tiling_data->size = 94000;

    ICPU_SET_TILING_KEY(0);
    SetKernelMode(KernelMode::AIV_MODE);

    auto x1_type = GetTorchType<DTYPE_X1>();
    auto x1_shape = "(20, 50, 94)";
    auto x1_range = "(-100.0, 100.0)";
    auto x2_type = GetTorchType<DTYPE_X2>();
    auto x2_shape = "(20, 50, 94)";
    auto x2_range = "(-100.0, 100.0)";
    ExecuteCommand("python", "gen_data.py", x1_type, x1_shape, x1_range, x2_type, x2_shape, x2_range);

    ReadFile("x1.bin", x1_size, x1, x1_size);
    ReadFile("x2.bin", x2_size, x2, x2_size);

    ICPU_RUN_KF(not_equal<0>, block_dim, x1, x2, y, workspace, tiling);

    WriteFile("y.bin", y, y_size);

    auto y_type = GetTorchType<DTYPE_Y>();
    auto y_loss = "0";
    EXPECT_EQ(ExecuteCommand("python", "compare_data.py", y_type, y_loss), 0);

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}
