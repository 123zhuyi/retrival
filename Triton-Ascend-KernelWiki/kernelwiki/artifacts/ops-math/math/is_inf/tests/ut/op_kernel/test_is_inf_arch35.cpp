// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// #include <array>
// #include <vector>
// #include <iostream>
// #include <string>
// #include <cstdint>
// #include "gtest/gtest.h"
// #include "tikicpulib.h"
// #include "test_is_inf_arch35.h"
// #include "data_utils.h"

// #include <cstdint>
// using namespace std;

// KERNEL_API void is_inf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

// class is_inf_test : public testing::Test {
//  protected:
//   static void SetUpTestCase() {
//     cout << "is_inf_test Regbase SetUp\n" << endl;
//   }
//   static void TearDownTestCase() {
//     cout << "is_inf_test Regbase TearDown\n" << endl;
//   }
// };

// TEST_F(is_inf_test, test_case_fp16_001) {
//   size_t xByteSize = 64 * sizeof(half);
//   size_t tiling_data_size = sizeof(IsInfRegbaseTilingData);
//   uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
//   uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
//   uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
//   uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//   uint32_t numBlocks = 1;

//   char* path_ = get_current_dir_name();
//   string path(path_);

//   IsInfRegbaseTilingData* tilingDatafromBin = reinterpret_cast<IsInfRegbaseTilingData*>(tiling);

//   tilingDatafromBin->baseTiling.scheMode = 1;
//   tilingDatafromBin->baseTiling.dim0 = 64;
//   tilingDatafromBin->baseTiling.blockFormer = 64;
//   tilingDatafromBin->baseTiling.blockNum = 1;
//   tilingDatafromBin->baseTiling.ubFormer = 6400;
//   tilingDatafromBin->baseTiling.ubLoopOfFormerBlock = 1;
//   tilingDatafromBin->baseTiling.ubLoopOfTailBlock = 1;
//   tilingDatafromBin->baseTiling.ubTailOfFormerBlock = 64;
//   tilingDatafromBin->baseTiling.ubTailOfTailBlock = 64;
//   tilingDatafromBin->baseTiling.elemNum = 6400;

//   ICPU_SET_TILING_KEY(101UL);
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);

//   system("cp -r ../../../../../../../ops/math/is_inf/tests/ut/op_kernel/is_inf_data ./");
//   system("chmod -R 755 ./is_inf_data/");
//   system("cd ./is_inf_data/ && rm -rf ./*bin && python3 gen_data.py '[64]' 'float16'");
//   ReadFile(path + "/is_inf_data/float16_input.bin", xByteSize, x, xByteSize);

//   ICPU_RUN_KF(is_inf, numBlocks, x, y, workspace, tiling);

//   WriteFile(path + "/is_inf_data/float16_output_rel.bin", y, xByteSize);

//   AscendC::GmFree(x);
//   AscendC::GmFree(y);
//   AscendC::GmFree(workspace);
//   AscendC::GmFree(tiling);
//   free(path_);

//   int res = system("cd ./is_inf_data/ && python3 compare_data.py 'float16'");
//   system("rm -rf is_inf_data");
//   ASSERT_EQ(res, 0);
// }

// TEST_F(is_inf_test, test_case_bf16_002) {
//   size_t xByteSize = 64 * sizeof(half);
//   size_t tiling_data_size = sizeof(IsInfRegbaseTilingData);
//   uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
//   uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
//   uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
//   uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//   uint32_t numBlocks = 1;

//   char* path_ = get_current_dir_name();
//   string path(path_);

//   IsInfRegbaseTilingData* tilingDatafromBin = reinterpret_cast<IsInfRegbaseTilingData*>(tiling);

//   tilingDatafromBin->baseTiling.scheMode = 1;
//   tilingDatafromBin->baseTiling.dim0 = 64;
//   tilingDatafromBin->baseTiling.blockFormer = 64;
//   tilingDatafromBin->baseTiling.blockNum = 1;
//   tilingDatafromBin->baseTiling.ubFormer = 6400;
//   tilingDatafromBin->baseTiling.ubLoopOfFormerBlock = 1;
//   tilingDatafromBin->baseTiling.ubLoopOfTailBlock = 1;
//   tilingDatafromBin->baseTiling.ubTailOfFormerBlock = 64;
//   tilingDatafromBin->baseTiling.ubTailOfTailBlock = 64;
//   tilingDatafromBin->baseTiling.elemNum = 6400;

//   ICPU_SET_TILING_KEY(102UL);
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);

//   system("cp -r ../../../../../../../ops/math/is_inf/tests/ut/op_kernel/is_inf_data ./");
//   system("chmod -R 755 ./is_inf_data/");
//   system("cd ./is_inf_data/ && rm -rf ./*bin && python3 gen_data.py '[64]' 'bfloat16'");
//   ReadFile(path + "/is_inf_data/bfloat16_input.bin", xByteSize, x, xByteSize);

//   ICPU_RUN_KF(is_inf, numBlocks, x, y, workspace, tiling);

//   WriteFile(path + "/is_inf_data/bfloat16_output_rel.bin", y, xByteSize);

//   AscendC::GmFree(x);
//   AscendC::GmFree(y);
//   AscendC::GmFree(workspace);
//   AscendC::GmFree(tiling);
//   free(path_);

//   int res = system("cd ./is_inf_data/ && python3 compare_data.py 'bfloat16'");
//   system("rm -rf is_inf_data");
//   ASSERT_EQ(res, 0);
// }

// TEST_F(is_inf_test, test_case_fp32_003) {
//   size_t xByteSize = 64 * sizeof(float);
//   size_t tiling_data_size = sizeof(IsInfRegbaseTilingData);
//   uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
//   uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
//   uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
//   uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//   uint32_t numBlocks = 1;

//   char* path_ = get_current_dir_name();
//   string path(path_);

//   IsInfRegbaseTilingData* tilingDatafromBin = reinterpret_cast<IsInfRegbaseTilingData*>(tiling);

//   tilingDatafromBin->baseTiling.scheMode = 1;
//   tilingDatafromBin->baseTiling.dim0 = 64;
//   tilingDatafromBin->baseTiling.blockFormer = 64;
//   tilingDatafromBin->baseTiling.blockNum = 1;
//   tilingDatafromBin->baseTiling.ubFormer = 6400;
//   tilingDatafromBin->baseTiling.ubLoopOfFormerBlock = 1;
//   tilingDatafromBin->baseTiling.ubLoopOfTailBlock = 1;
//   tilingDatafromBin->baseTiling.ubTailOfFormerBlock = 64;
//   tilingDatafromBin->baseTiling.ubTailOfTailBlock = 64;
//   tilingDatafromBin->baseTiling.elemNum = 6400;

//   ICPU_SET_TILING_KEY(103UL);
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);

//   system("cp -r ../../../../../../../ops/math/is_inf/tests/ut/op_kernel/is_inf_data ./");
//   system("chmod -R 755 ./is_inf_data/");
//   system("cd ./is_inf_data/ && rm -rf ./*bin && python3 gen_data.py '[64]' 'float32'");
//   ReadFile(path + "/is_inf_data/float32_input.bin", xByteSize, x, xByteSize);

//   ICPU_RUN_KF(is_inf, numBlocks, x, y, workspace, tiling);

//   WriteFile(path + "/is_inf_data/float32_output_rel.bin", y, xByteSize);

//   AscendC::GmFree(x);
//   AscendC::GmFree(y);
//   AscendC::GmFree(workspace);
//   AscendC::GmFree(tiling);
//   free(path_);

//   int res = system("cd ./is_inf_data/ && python3 compare_data.py 'float32'");
//   system("rm -rf is_inf_data");
//   ASSERT_EQ(res, 0);
// }
