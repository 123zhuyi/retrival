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
 * \file test_tanh_grad.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/tanh_grad.cpp"

using namespace std;

extern "C" __global__ __aicore__ void tanh_grad(GM_ADDR dy, GM_ADDR y, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling);

class TanhGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "tanh_grad_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./tanh_grad_data/");
    }

    static void TearDownTestCase() { std::cout << "tanh_grad_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string TanhGradTest::rootPath = "../../../../";
const std::string TanhGradTest::dataPath = rootPath + "math/tanh_grad/tests/ut/op_kernel/tanh_grad_data";

// 向上对齐函数
template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

TEST_F(TanhGradTest, test_case_float32)
{
    optiling::TanhGradCompileInfo compileInfo = {64, 262144, false};

    gert::TilingContextPara tilingContextPara(
        "TanhGrad",
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    // 生成测试数据
    system("cd ./tanh_grad_data/ && python3 gen_data.py '(256, 33)' 'float32'");

    uint32_t dataCount = 256 * 33;
    size_t byteSize = dataCount * sizeof(float);

    // 分配和读取输入 dy
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ReadFile("./tanh_grad_data/float32_input_dy_tanh_grad.bin", byteSize, dy, byteSize);

    // 分配和读取输入 y
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ReadFile("./tanh_grad_data/float32_input_y_tanh_grad.bin", byteSize, y, byteSize);

    // 分配输出 dx
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));

    // 分配 workspace 和 tiling 数据
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    // 设置 tiling key 并运行核函数
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 注意参数顺序: dy, y, dx, workspace, tiling
    ICPU_RUN_KF(tanh_grad, tilingInfo.blockNum, dy, y, dx, workspace, tiling);

    // 保存输出结果
    WriteFile("./tanh_grad_data/float32_output_tanh_grad.bin", dx, byteSize);

    // 释放内存
    AscendC::GmFree((void*)(dy));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)(dx));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    // 比较结果
    system("cd ./tanh_grad_data/ && python3 compare_data.py 'float32'");
}
