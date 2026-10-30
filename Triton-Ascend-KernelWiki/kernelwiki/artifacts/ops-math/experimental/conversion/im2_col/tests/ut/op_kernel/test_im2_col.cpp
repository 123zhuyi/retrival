/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

/*!
 * \file test_im2col.cpp
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
#include "tiling_case_executor.h"
#include "../op_host/im2col_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void im2col(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class Im2ColTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "im2col_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./im2col_data/");
    }

    static void TearDownTestCase() { std::cout << "im2col_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string Im2ColTest::rootPath = "../../../../";
const std::string Im2ColTest::dataPath = rootPath + "nn/im2col/tests/ut/op_kernel/im2col_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(Im2ColTest, test_case_float16_1)
{
    /* ---------------- tiling ---------------- */
    optiling::Im2ColCompileInfo compileInfo = {
        64,     // blockDim
        262144, // ubSize
        false};

    gert::TilingContextPara tilingContextPara(
        "Im2Col",
        {
            {{{1, 3, 32, 32}, {1, 3, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 27, 1024}, {1, 27, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    /* ---------------- data gen ---------------- */
    system("cd ./im2col_data/ && python3 gen_im2col.py '(1,3,32,32)' 'float16'");

    uint32_t inputElem = 1 * 3 * 32 * 32;
    uint32_t outputElem = 1 * 27 * 1024;

    size_t inputByteSize = inputElem * sizeof(half);
    size_t outputByteSize = outputElem * sizeof(half);

    /* ---------------- GM alloc ---------------- */
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    ReadFile("./im2col_data/float16_input_im2col.bin", inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);

    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    /* ---------------- run kernel ---------------- */
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(im2col, tilingInfo.blockNum, x, y, workspace, tiling);

    /* ---------------- dump & check ---------------- */
    WriteFile("./im2col_data/float16_output_im2col.bin", y, outputByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./im2col_data/ && python3 compare_im2col.py 'float16'");
}

TEST_F(Im2ColTest, test_case_float32_1)
{
    optiling::Im2ColCompileInfo compileInfo = {64, 262144, false};

    gert::TilingContextPara tilingContextPara(
        "Im2Col",
        {
            {{{1, 3, 16, 16}, {1, 3, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{1, 27, 256}, {1, 27, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./im2col_data/ && python3 gen_im2col.py '(1,3,16,16)' 'float32'");

    uint32_t inputElem = 1 * 3 * 16 * 16;
    uint32_t outputElem = 1 * 27 * 256;

    size_t inputByteSize = inputElem * sizeof(float);
    size_t outputByteSize = outputElem * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    ReadFile("./im2col_data/float32_input_im2col.bin", inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);

    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(im2col, tilingInfo.blockNum, x, y, workspace, tiling);

    WriteFile("./im2col_data/float32_output_im2col.bin", y, outputByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./im2col_data/ && python3 compare_im2col.py 'float32'");
}
