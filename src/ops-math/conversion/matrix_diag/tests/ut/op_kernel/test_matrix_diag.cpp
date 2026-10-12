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
 * \file test_matrix_diag.cpp
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
#include "../../../op_host/matrix_diag_tiling_arch35.h"

using namespace std;

extern "C" __global__ __aicore__ void matrix_diag(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class MatrixDiagTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "matrix_diag_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./matrix_diag_data/");
    }
    static void TearDownTestCase() {
        std::cout << "matrix_diag_test TearDown" << std::endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string MatrixDiagTest::rootPath = "../../../../";
const std::string MatrixDiagTest::dataPath = rootPath + "conversion/matrix_diag/tests/ut/op_kernel/matrix_diag_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b) {
    return (a + b - 1) / b * b;
}

TEST_F(MatrixDiagTest, test_case_matrixdiag_uint8_batch_600x129) {
    MatrixDiagAsc::MatrixDiagCompileInfo compileInfo = {64, 253952, 128, 32, true};
    gert::TilingContextPara tilingContextPara("MatrixDiag",
                                              {{{{600, 129}, {600, 129}}, ge::DT_INT8, ge::FORMAT_ND},},
                                              {{{{600, 129, 129}, {600, 129, 129}}, ge::DT_INT8, ge::FORMAT_ND},},
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./matrix_diag_data/ && python3 gen_data.py '(600, 129)' 'uint8'");
    uint32_t dataCount = 600 * 129;
    size_t inputByteSize = dataCount * sizeof(uint8_t);
    std::string fileName = "./matrix_diag_data/uint8_input_t_matrix_diag.bin";;
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(uint8_t);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(matrix_diag, tilingInfo.blockNum, x, y, workspace, tiling);

    fileName = "./matrix_diag_data/uint8_output_t_matrix_diag.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./matrix_diag_data/ && python3 compare_data.py 'uint8'");
}
