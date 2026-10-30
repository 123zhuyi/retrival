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
#include "gtest/gtest.h"
#include "../../../op_host/arch32/grouped_bias_add_grad_tiling.h"


#ifdef __CCE_KT_TEST__
#include <cstdint>
#include "tikicpulib.h"
#include "data_utils.h"
#endif

extern "C" __global__ __aicore__ void grouped_bias_add_grad(GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias,
                                                            GM_ADDR workspace, GM_ADDR tiling);

class grouped_bias_add_grad_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "grouped_bias_add_grad_test SetUp\n" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "grouped_bias_add_grad_test TearDown\n" << std::endl;
  }
};

TEST_F(grouped_bias_add_grad_test, test_case_fp16_group_idx) {
  system(
      "cp -rf "
      "../../../../math/grouped_bias_add_grad/tests/ut/op_kernel/grouped_bias_add_grad_data ./");
  system("chmod -R 755 ./grouped_bias_add_grad_data/");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_data.py 3");
  system("cd ./grouped_bias_add_grad_data/ && python3 gen_tiling.py 'case3'");
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  size_t tilingSize = sizeof(GroupedBiasAddGradTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  uint32_t numBlocks = 12;

  size_t gradYByteSize = 1968 * 458 * sizeof(half);
  size_t groupIdxByteSize = 3 * sizeof(int32_t);
  size_t outByteSize = 3 * 458 * sizeof(half);
  size_t workspaceBytesSize = 32 * 1024 * 1024 + 48 * sizeof(int32_t);

  uint8_t* grad_y = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
  uint8_t* group_idx = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
  uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

  uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

  std::string curPath = ".";
  ReadFile(curPath + "/grouped_bias_add_grad_data/grad_y.bin", gradYByteSize, grad_y, gradYByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/group_idx.bin", groupIdxByteSize, group_idx, groupIdxByteSize);
  ReadFile(curPath + "/grouped_bias_add_grad_data/tiling.bin", tilingSize, tiling, tilingSize);

  ICPU_SET_TILING_KEY(1000010);
  ICPU_RUN_KF(grouped_bias_add_grad, numBlocks, grad_y, group_idx, out, workSpace, tiling);

  WriteFile(curPath + "/grouped_bias_add_grad_data/output.bin", out, outByteSize);
  AscendC::GmFree((void*)grad_y);
  AscendC::GmFree((void*)group_idx);
  AscendC::GmFree((void*)out);
  AscendC::GmFree((void*)workSpace);
  AscendC::GmFree((void*)tiling);

  system("cd ./grouped_bias_add_grad_data/ && python3 compare_data.py 'float16'");
}
