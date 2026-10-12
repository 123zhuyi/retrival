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
 * \file test_expandv.cpp
 * \brief
 */
#define DTYPE_X1 float // 修改数据类型
#include "../../../op_kernel/expandv.cpp"
#include "expandv_tiling.h"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

class ExpandvTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ExpandvTest SetUp\n" << endl;
        std::cout << "ExpandvTest SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./expandv_data/");
    }
    static void TearDownTestCase() { cout << "ExpandvTest TearDown\n" << endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string ExpandvTest::rootPath = "../../../../";
const std::string ExpandvTest::dataPath = rootPath + "experimental/math/expandv/tests/ut/op_kernel/expandv_data";

// 测试用例: 3维形状 (4, 1, 3)，数据类型由DTYPE_X1决定
TEST_F(ExpandvTest, test_case_3d_shape)
{
    // 1. 配置测试 parameters - 3维形状 (4, 1, 3)
    const int shapeM = 4;
    const int shapeN = 1;
    const int shapeO = 3;

    const int ExpectShapeM = 4;
    const int ExpectShapeN = 5;
    const int ExpectShapeO = 3;
    size_t InputElements = shapeM * shapeN * shapeO;
    size_t OutputElements = ExpectShapeM * ExpectShapeN * ExpectShapeO;

    size_t xByteSize = InputElements * sizeof(DTYPE_X1);
    size_t zByteSize = OutputElements * sizeof(DTYPE_X1);

    size_t tiling_data_size = sizeof(ExpandvTilingData);
    uint32_t blockDim = 4;

    // 2. 确定数据类型字符串
    std::string dtype_str;
    if constexpr (std::is_same<DTYPE_X1, float>::value) {
        dtype_str = "float32";
    } else if constexpr (std::is_same<DTYPE_X1, half>::value) {
        dtype_str = "float16";
    } else {
        FAIL() << "Unsupported data type";
    }

    // 3. 生成测试数据
    std::string gen_cmd = "cd ./expandv_data/ && python3 gen_data.py '(4, 1, 3)' '" + dtype_str + "'" + "'(4, 5, 3)' '";
    system(gen_cmd.c_str());

    // 4. 分配内存
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    // 5. 读取输入数据
    std::string fileNameX = "./expandv_data/" + dtype_str + "_input_x_expandv.bin";
    ReadFile(fileNameX, xByteSize, x, xByteSize);

    // 6. 配置tiling数据
    // 对于1024个元素(32*32)，4个block的合理配置：
    // - 每个block处理256个元素
    // - 每个block使用8个tile，每个tile处理32个元素（满足32字节对齐要求）
    ExpandvTilingData* tilingDatafromBin = reinterpret_cast<ExpandvTilingData*>(tiling);
    if constexpr (std::is_same<DTYPE_X1, float>::value) {
        // DT_FLOAT（float32）：对应计算的正确TilingData值 64 72 1 1 8184 64 72 0
        tilingDatafromBin->smallCoreDataNum = 64;
        tilingDatafromBin->bigCoreDataNum = 72;
        tilingDatafromBin->finalBigTileNum = 1;
        tilingDatafromBin->finalSmallTileNum = 1;
        tilingDatafromBin->tileDataNum = 8184;
        tilingDatafromBin->smallTailDataNum = 64;
        tilingDatafromBin->bigTailDataNum = 72;
        tilingDatafromBin->tailBlockNum = 0;
    } else if constexpr (std::is_same<DTYPE_X1, half>::value) {
        // DT_FLOAT16（float16）：对应计算的正确TilingData值 64 80 1 1 16368 64 80 0
        tilingDatafromBin->smallCoreDataNum = 64;
        tilingDatafromBin->bigCoreDataNum = 80;
        tilingDatafromBin->finalBigTileNum = 1;
        tilingDatafromBin->finalSmallTileNum = 1;
        tilingDatafromBin->tileDataNum = 16368;
        tilingDatafromBin->smallTailDataNum = 64;
        tilingDatafromBin->bigTailDataNum = 80;
        tilingDatafromBin->tailBlockNum = 0;
    }
    // 形状/步长字段：固定为(4,1,3)→(4,5,3)的计算值，与数据类型无关
    tilingDatafromBin->in_rank = 3;
    tilingDatafromBin->out_rank = 3;
    // 输入形状数组：inShapeArr = [4,1,3,0,...0]
    tilingDatafromBin->inShapeArr[0] = 4;
    tilingDatafromBin->inShapeArr[1] = 1;
    tilingDatafromBin->inShapeArr[2] = 3;
    // 输出形状数组：outShapeArr = [4,5,3,0,...0]
    tilingDatafromBin->outShapeArr[0] = 4;
    tilingDatafromBin->outShapeArr[1] = 5;
    tilingDatafromBin->outShapeArr[2] = 3;
    // 输入步长数组：inStrideArr = [3,3,1,0,...0]（行优先存储）
    tilingDatafromBin->inStrideArr[0] = 3;
    tilingDatafromBin->inStrideArr[1] = 3;
    tilingDatafromBin->inStrideArr[2] = 1;
    // 输出步长数组：outStrideArr = [15,3,1,0,...0]（行优先存储）
    tilingDatafromBin->outStrideArr[0] = 15;
    tilingDatafromBin->outStrideArr[1] = 3;
    tilingDatafromBin->outStrideArr[2] = 1;
    // 7. 设置运行环境
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // 8. 执行kernel
    ICPU_RUN_KF(expandv<0>, blockDim, x, z, workspace, (uint8_t*)(tilingDatafromBin));

    // 9. 保存输出数据
    std::string fileNameZ = "./expandv_data/" + dtype_str + "_output_expandv.bin";
    WriteFile(fileNameZ, z, zByteSize);

    // 10. 释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    // 11. 比较结果
    std::string compare_cmd = "cd ./expandv_data/ && python3 compare_data.py '" + dtype_str + "'";
    system(compare_cmd.c_str());
}