/**
 * \file test_col2im_custom.cpp
 * \brief Col2Im Custom算子单元测试
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
#include "../op_host/col2_im_custom_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void col2_im_custom(GM_ADDR col, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling);

class Col2ImCustomTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "col2im_custom_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./col2_im_data/");
    }

    static void TearDownTestCase() { std::cout << "col2im_custom_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string Col2ImCustomTest::rootPath = "../../../../";
const std::string Col2ImCustomTest::dataPath = rootPath + "custom/col2im_custom/tests/ut/op_kernel/col2_im_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// 测试用例1: float16, 简单配置
TEST_F(Col2ImCustomTest, test_case_float16_basic)
{
    // 输入: [1, 4, 9] -> 输出: [1, 1, 4, 4]
    // kernel=2x2, stride=1, padding=0, dilation=1
    optiling::Col2ImCustomCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "Col2ImCustom",
        {
            // 输入: [N=1, C*kH*kW=4, L=9]
            {{{1, 4, 9}, {1, 4, 9}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出: [N=1, C=1, H=4, W=4]
            {{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    // 设置属性: output_h, output_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
    tilingContextPara.AddAttrs({4, 4, 2, 2, 1, 1, 0, 0, 1, 1});

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    // 生成测试数据
    system("cd ./col2_im_data/ && python3 gen_data.py '(1, 4, 9)' 'float16' 4 4 2 2 1 1 0 0 1 1");

    uint32_t inputDataCount = 1 * 4 * 9;      // N * (C*kH*kW) * L
    uint32_t outputDataCount = 1 * 1 * 4 * 4; // N * C * H * W

    size_t inputByteSize = inputDataCount * sizeof(half);
    size_t outputByteSize = outputDataCount * sizeof(half);

    std::string fileName = "./col2_im_data/float16_input_col.bin";
    uint8_t* col = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, col, inputByteSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);

    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(col2_im_custom, tilingInfo.blockNum, col, x, workspace, tiling);

    fileName = "./col2_im_data/float16_output_x.bin";
    WriteFile(fileName, x, outputByteSize);

    AscendC::GmFree((void*)col);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./col2_im_data/ && python3 compare_data.py 'float16'");
}

// 测试用例2: float32, 带stride
TEST_F(Col2ImCustomTest, test_case_float32_stride)
{
    // 输入: [1, 12, 4] -> 输出: [1, 3, 5, 5]
    // kernel=2x2, stride=2, padding=0, dilation=1
    optiling::Col2ImCustomCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "Col2ImCustom",
        {
            // 输入: [N=1, C*kH*kW=12, L=4]
            {{{1, 12, 4}, {1, 12, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出: [N=1, C=3, H=5, W=5]
            {{{1, 3, 5, 5}, {1, 3, 5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    // 设置属性: stride=2
    tilingContextPara.AddAttrs({5, 5, 2, 2, 2, 2, 0, 0, 1, 1});

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./col2_im_data/ && python3 gen_data.py '(1, 12, 4)' 'float32' 5 5 2 2 2 2 0 0 1 1");

    uint32_t inputDataCount = 1 * 12 * 4;
    uint32_t outputDataCount = 1 * 3 * 5 * 5;

    size_t inputByteSize = inputDataCount * sizeof(float);
    size_t outputByteSize = outputDataCount * sizeof(float);

    std::string fileName = "./col2_im_data/float32_input_col.bin";
    uint8_t* col = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, col, inputByteSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);

    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(col2_im_custom, tilingInfo.blockNum, col, x, workspace, tiling);

    fileName = "./col2_im_data/float32_output_x.bin";
    WriteFile(fileName, x, outputByteSize);

    AscendC::GmFree((void*)col);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./col2_im_data/ && python3 compare_data.py 'float32'");
}

// 测试用例3: float16, 带padding
TEST_F(Col2ImCustomTest, test_case_float16_padding)
{
    // 输入: [2, 8, 16] -> 输出: [2, 2, 6, 6]
    // kernel=2x2, stride=1, padding=1, dilation=1
    optiling::Col2ImCustomCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "Col2ImCustom",
        {
            // 输入: [N=2, C*kH*kW=8, L=16]
            {{{2, 8, 16}, {2, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出: [N=2, C=2, H=6, W=6]
            {{{2, 2, 6, 6}, {2, 2, 6, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    // 设置属性: padding=1
    tilingContextPara.AddAttrs({6, 6, 2, 2, 1, 1, 1, 1, 1, 1});

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./col2_im_data/ && python3 gen_data.py '(2, 8, 16)' 'float16' 6 6 2 2 1 1 1 1 1 1");

    uint32_t inputDataCount = 2 * 8 * 16;
    uint32_t outputDataCount = 2 * 2 * 6 * 6;

    size_t inputByteSize = inputDataCount * sizeof(half);
    size_t outputByteSize = outputDataCount * sizeof(half);

    std::string fileName = "./col2_im_data/float16_input_col.bin";
    uint8_t* col = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, col, inputByteSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);

    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(col2_im_custom, tilingInfo.blockNum, col, x, workspace, tiling);

    fileName = "./col2_im_data/float16_output_x.bin";
    WriteFile(fileName, x, outputByteSize);

    AscendC::GmFree((void*)col);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./col2_im_data/ && python3 compare_data.py 'float16'");
}

// 测试用例4: float32, 带dilation
TEST_F(Col2ImCustomTest, test_case_float32_dilation)
{
    // 输入: [1, 4, 4] -> 输出: [1, 1, 6, 6]
    // kernel=2x2, stride=1, padding=0, dilation=2
    optiling::Col2ImCustomCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "Col2ImCustom",
        {
            // 输入: [N=1, C*kH*kW=4, L=4]
            {{{1, 4, 4}, {1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出: [N=1, C=1, H=6, W=6]
            {{{1, 1, 6, 6}, {1, 1, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    // 设置属性: dilation=2
    tilingContextPara.AddAttrs({6, 6, 2, 2, 1, 1, 0, 0, 2, 2});

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./col2_im_data/ && python3 gen_data.py '(1, 4, 4)' 'float32' 6 6 2 2 1 1 0 0 2 2");

    uint32_t inputDataCount = 1 * 4 * 4;
    uint32_t outputDataCount = 1 * 1 * 6 * 6;

    size_t inputByteSize = inputDataCount * sizeof(float);
    size_t outputByteSize = outputDataCount * sizeof(float);

    std::string fileName = "./col2_im_data/float32_input_col.bin";
    uint8_t* col = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, col, inputByteSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);

    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(col2_im_custom, tilingInfo.blockNum, col, x, workspace, tiling);

    fileName = "./col2_im_data/float32_output_x.bin";
    WriteFile(fileName, x, outputByteSize);

    AscendC::GmFree((void*)col);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./col2_im_data/ && python3 compare_data.py 'float32'");
}

// 测试用例5: float16, 大尺寸
TEST_F(Col2ImCustomTest, test_case_float16_large)
{
    // 输入: [4, 36, 225] -> 输出: [4, 9, 32, 32]
    // kernel=2x2, stride=2, padding=1, dilation=1
    optiling::Col2ImCustomCompileInfo compileInfo = {};

    gert::TilingContextPara tilingContextPara(
        "Col2ImCustom",
        {
            // 输入: [N=4, C*kH*kW=36, L=225]
            {{{4, 36, 225}, {4, 36, 225}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出: [N=4, C=9, H=32, W=32]
            {{{4, 9, 32, 32}, {4, 9, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    tilingContextPara.AddAttrs({32, 32, 2, 2, 2, 2, 1, 1, 1, 1});

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./col2_im_data/ && python3 gen_data.py '(4, 36, 225)' 'float16' 32 32 2 2 2 2 1 1 1 1");

    uint32_t inputDataCount = 4 * 36 * 225;
    uint32_t outputDataCount = 4 * 9 * 32 * 32;

    size_t inputByteSize = inputDataCount * sizeof(half);
    size_t outputByteSize = outputDataCount * sizeof(half);

    std::string fileName = "./col2_im_data/float16_input_col.bin";
    uint8_t* col = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, col, inputByteSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);

    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(col2_im_custom, tilingInfo.blockNum, col, x, workspace, tiling);

    fileName = "./col2_im_data/float16_output_x.bin";
    WriteFile(fileName, x, outputByteSize);

    AscendC::GmFree((void*)col);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./col2_im_data/ && python3 compare_data.py 'float16'");
}