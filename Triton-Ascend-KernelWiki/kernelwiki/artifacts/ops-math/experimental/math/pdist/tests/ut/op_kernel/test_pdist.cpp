/*!
 * \file test_pdist.cpp
 * \brief Pdist kernel UT: generates input via gen_data.py, runs kernel on CPU simulator,
 *        writes output, and compares via compare_data.py.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/pdist.cpp"

using namespace std;

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class PdistKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PdistKernelTest SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./pdist_data/");
    }
    static void TearDownTestCase()
    {
        std::cout << "PdistKernelTest TearDown" << std::endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string PdistKernelTest::rootPath = "../../../../";
const std::string PdistKernelTest::dataPath = rootPath + "math/pdist/tests/ut/op_kernel/pdist_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(PdistKernelTest, test_fp32_p2_4x3)
{
    uint32_t N = 4, M = 3;
    float p = 2.0f;
    uint32_t blockDim = 1;

    std::string genCmd = "cd ./pdist_data/ && python3 gen_data.py '(" +
        std::to_string(N) + "," + std::to_string(M) + ")' '" +
        std::to_string(p) + "' 'float32'";
    system(genCmd.c_str());

    uint32_t inputCount = N * M;
    uint32_t outputCount = N * (N - 1) / 2;
    size_t inputByteSize = inputCount * sizeof(float);
    size_t outputByteSize = outputCount * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(PdistTilingData));

    ReadFile("./pdist_data/float32_input_x.bin", inputByteSize, x, inputByteSize);

    PdistTilingData* tilingData = reinterpret_cast<PdistTilingData*>(tiling);
    tilingData->rows = N;
    tilingData->cols = M;
    tilingData->pValue = p;
    tilingData->computeNum = outputCount;
    tilingData->ubTensorEachLoop = 4096;
    tilingData->coreNumVar = blockDim;
    tilingData->tilingKey = 1;
    tilingData->reduceBufSize = 1024;
    tilingData->numBlockEachCore = outputCount / blockDim / 8;
    uint64_t lastNums = outputCount % (8 * blockDim);
    tilingData->lastNumsBlocks = lastNums / 8;
    tilingData->lastNumsNoneFullBlock = lastNums % 8;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(pdist, blockDim, x, y, workspace, tiling);

    WriteFile("./pdist_data/float32_output_pdist.bin", y, outputByteSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./pdist_data/ && python3 compare_data.py 'float32'");
}
