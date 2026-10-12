/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/angle_v2_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void angle_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class angle_v2_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "angle_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "angle_v2_test TearDown\n" << endl;
    }
};

TEST_F(angle_v2_test, test_case_fp32) {
    uint32_t totalLength = 8;

    // inputs
    size_t x_size = totalLength * sizeof(float);
    size_t y_size = totalLength * sizeof(float);
    size_t tiling_data_size = sizeof(AngleV2TilingData);

    uint8_t *x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;
    system("cp -r ../../../../math/angle_v2/tests/ut/op_kernel/angle_v2_data ./");
    system("chmod -R 755 ./angle_v2_data/");
    system("cd ./angle_v2_data/ && rm -rf ./*bin");
    system("cd ./angle_v2_data/ && python3 gen_data.py 8 float32");
    system("cd ./angle_v2_data/ && python3 gen_tiling.py 8 float32");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/angle_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/angle_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    AngleV2TilingData* tilingDatafromBin = reinterpret_cast<AngleV2TilingData*>(tiling);
    tilingDatafromBin->totalLength = 8;
    tilingDatafromBin->formerNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->formerLength = 8;
    tilingDatafromBin->tailLength = 8;
    tilingDatafromBin->alignNum = 8;
    tilingDatafromBin->totalLengthAligned = 8;
    tilingDatafromBin->tileLength = 6464;
    tilingDatafromBin->dataPerRepeat = 64;

    auto angle_v2_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
            ::angle_v2(x, y, workspace, tiling);
        };

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(angle_v2_kernel, numBlocks, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
