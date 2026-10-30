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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/transform_bias_rescale_qkv_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void transform_bias_rescale_qkv(
    GM_ADDR qkv, GM_ADDR qkv_bias, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR workspace, GM_ADDR tiling);

class transform_bias_rescale_qkv_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "transform_bias_rescale_qkv SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "transform_bias_rescale_qkv TearDown\n" << endl;
    }
};

TEST_F(transform_bias_rescale_qkv_test, test_transform_bias_rescale_qkv_float_0)
{
    system(
        "cp -rf "
        "../../../../math/transform_bias_rescale_qkv/tests/ut/op_kernel/transform_bias_rescale_qkv_data ./");
    system("chmod -R 755 ./transform_bias_rescale_qkv_data/");
    system(
        "cd ./transform_bias_rescale_qkv_data/ && python3 gen_data.py '(3, 4, 144)' '(144)' '(3, 3, 4, 16)' 'float32'");
    size_t B = 3;
    size_t T = 4;
    size_t N = 3;
    size_t D = 16;

    size_t qkvFileSize = B * T * 3 * N * D * sizeof(float);
    size_t qkvBiasFileSize = 3 * N * D * sizeof(float);

    size_t qFileSize = B * T * N * D * sizeof(float);

    uint8_t* qkv = (uint8_t*)AscendC::GmAlloc(qkvFileSize);
    uint8_t* qkvBias = (uint8_t*)AscendC::GmAlloc(qkvBiasFileSize);

    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qFileSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(qFileSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(qFileSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(TransformBiasRescaleQkvTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string qkvFileName = "./transform_bias_rescale_qkv_data/float32_qkv_transform_bias_rescale_qkv.bin";
    std::string qkvBiasFileName = "./transform_bias_rescale_qkv_data/float32_qkv_bias_transform_bias_rescale_qkv.bin";

    ReadFile(qkvFileName, qkvFileSize, qkv, qkvFileSize);
    ReadFile(qkvBiasFileName, qkvBiasFileSize, qkvBias, qkvBiasFileSize);

    TransformBiasRescaleQkvTilingData* tilingDatafromBin = reinterpret_cast<TransformBiasRescaleQkvTilingData*>(tiling);
    tilingDatafromBin->qkvShapeSize = B * T * 3 * N * D;
    tilingDatafromBin->needCoreNum = 1;
    tilingDatafromBin->batch = B;
    tilingDatafromBin->token = T;
    tilingDatafromBin->dimension = N * D;
    tilingDatafromBin->numHeads = N;
    tilingDatafromBin->dimPerHead = D;
    tilingDatafromBin->maxEleNumUB = 12 * 1024;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(transform_bias_rescale_qkv, blockDim, qkv, qkvBias, q, k, v, workspace, (uint8_t*)tilingDatafromBin);
    std::string qFileName = "./transform_bias_rescale_qkv_data/float32_q_output_transform_bias_rescale_qkv.bin";
    std::string kFileName = "./transform_bias_rescale_qkv_data/float32_k_output_transform_bias_rescale_qkv.bin";
    std::string vFileName = "./transform_bias_rescale_qkv_data/float32_v_output_transform_bias_rescale_qkv.bin";
    WriteFile(qFileName, q, qFileSize);
    WriteFile(kFileName, k, qFileSize);
    WriteFile(vFileName, v, qFileSize);

    AscendC::GmFree((void*)qkv);
    AscendC::GmFree((void*)qkvBias);
    AscendC::GmFree((void*)q);
    AscendC::GmFree((void*)k);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./transform_bias_rescale_qkv_data/ && python3 compare_data.py 'float32'");
}