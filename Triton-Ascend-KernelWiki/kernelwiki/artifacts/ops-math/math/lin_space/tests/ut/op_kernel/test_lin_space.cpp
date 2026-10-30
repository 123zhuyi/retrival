/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_host/lin_space_tiling.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void lin_space(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output,
                                                GM_ADDR workspace, GM_ADDR tiling);

class lin_space_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "lin_space_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "lin_space_test TearDown\n" << endl;
  }
};

TEST_F(lin_space_test, test_case_101) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 1.0;
  tilingDatafromBin->num = 101;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 13;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 5;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 5;

  ICPU_SET_TILING_KEY(101);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_102) {
  size_t outputByteSize = 640000 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 32;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 0;
  tilingDatafromBin->stop = 8.0;
  tilingDatafromBin->scalar = 0.000013;
  tilingDatafromBin->num = 640000;
  tilingDatafromBin->matrixLen = 256;
  tilingDatafromBin->realCoreNum = 32;
  tilingDatafromBin->numPerCore = 20000;
  tilingDatafromBin->tailNum = 20000;
  tilingDatafromBin->innerLoopNum = 39;
  tilingDatafromBin->innerLoopTail = 3616;
  tilingDatafromBin->innerTailLoopNum = 39;
  tilingDatafromBin->innerTailLoopTail = 3616;
  tilingDatafromBin->outerLoopNum = 5;
  tilingDatafromBin->outerLoopNumTail = 3616;
  tilingDatafromBin->outerTailLoopNum = 5;
  tilingDatafromBin->outerTailLoopNumTail = 3616;

  ICPU_SET_TILING_KEY(102);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_201) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 1.0;
  tilingDatafromBin->num = 101;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 13;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 5;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 5;

  ICPU_SET_TILING_KEY(201);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_202) {
  size_t outputByteSize = 640000 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 32;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 0;
  tilingDatafromBin->stop = 8.0;
  tilingDatafromBin->scalar = 0.000013;
  tilingDatafromBin->num = 640000;
  tilingDatafromBin->matrixLen = 256;
  tilingDatafromBin->realCoreNum = 32;
  tilingDatafromBin->numPerCore = 20000;
  tilingDatafromBin->tailNum = 20000;
  tilingDatafromBin->innerLoopNum = 39;
  tilingDatafromBin->innerLoopTail = 3616;
  tilingDatafromBin->innerTailLoopNum = 39;
  tilingDatafromBin->innerTailLoopTail = 3616;
  tilingDatafromBin->outerLoopNum = 5;
  tilingDatafromBin->outerLoopNumTail = 3616;
  tilingDatafromBin->outerTailLoopNum = 5;
  tilingDatafromBin->outerTailLoopNumTail = 3616;

  ICPU_SET_TILING_KEY(202);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(lin_space_test, test_case_301) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 1.0;
  tilingDatafromBin->num = 101;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 13;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 5;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 5;

  ICPU_SET_TILING_KEY(201);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_302) {
  size_t outputByteSize = 640000 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 32;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 0;
  tilingDatafromBin->stop = 8.0;
  tilingDatafromBin->scalar = 0.000013;
  tilingDatafromBin->num = 640000;
  tilingDatafromBin->matrixLen = 256;
  tilingDatafromBin->realCoreNum = 32;
  tilingDatafromBin->numPerCore = 20000;
  tilingDatafromBin->tailNum = 20000;
  tilingDatafromBin->innerLoopNum = 39;
  tilingDatafromBin->innerLoopTail = 3616;
  tilingDatafromBin->innerTailLoopNum = 39;
  tilingDatafromBin->innerTailLoopTail = 3616;
  tilingDatafromBin->outerLoopNum = 5;
  tilingDatafromBin->outerLoopNumTail = 3616;
  tilingDatafromBin->outerTailLoopNum = 5;
  tilingDatafromBin->outerTailLoopNumTail = 3616;

  ICPU_SET_TILING_KEY(102);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_402) {
  size_t outputByteSize = 640000 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 32;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 0;
  tilingDatafromBin->stop = 8.0;
  tilingDatafromBin->scalar = 0.000013;
  tilingDatafromBin->num = 640000;
  tilingDatafromBin->matrixLen = 256;
  tilingDatafromBin->realCoreNum = 32;
  tilingDatafromBin->numPerCore = 20000;
  tilingDatafromBin->tailNum = 20000;
  tilingDatafromBin->innerLoopNum = 39;
  tilingDatafromBin->innerLoopTail = 3616;
  tilingDatafromBin->innerTailLoopNum = 39;
  tilingDatafromBin->innerTailLoopTail = 3616;
  tilingDatafromBin->outerLoopNum = 5;
  tilingDatafromBin->outerLoopNumTail = 3616;
  tilingDatafromBin->outerTailLoopNum = 5;
  tilingDatafromBin->outerTailLoopNumTail = 3616;

  ICPU_SET_TILING_KEY(402);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_502) {
  size_t outputByteSize = 640000 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 32;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 0;
  tilingDatafromBin->stop = 8.0;
  tilingDatafromBin->scalar = 0.000013;
  tilingDatafromBin->num = 640000;
  tilingDatafromBin->matrixLen = 256;
  tilingDatafromBin->realCoreNum = 32;
  tilingDatafromBin->numPerCore = 20000;
  tilingDatafromBin->tailNum = 20000;
  tilingDatafromBin->innerLoopNum = 39;
  tilingDatafromBin->innerLoopTail = 3616;
  tilingDatafromBin->innerTailLoopNum = 39;
  tilingDatafromBin->innerTailLoopTail = 3616;
  tilingDatafromBin->outerLoopNum = 5;
  tilingDatafromBin->outerLoopNumTail = 3616;
  tilingDatafromBin->outerTailLoopNum = 5;
  tilingDatafromBin->outerTailLoopNumTail = 3616;

  ICPU_SET_TILING_KEY(502);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_601) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 1.0;
  tilingDatafromBin->num = 101;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 13;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 5;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 5;

  ICPU_SET_TILING_KEY(601);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_602) {
  size_t outputByteSize = 640000 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 32;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 0;
  tilingDatafromBin->stop = 8.0;
  tilingDatafromBin->scalar = 0.000013;
  tilingDatafromBin->num = 640000;
  tilingDatafromBin->matrixLen = 256;
  tilingDatafromBin->realCoreNum = 32;
  tilingDatafromBin->numPerCore = 20000;
  tilingDatafromBin->tailNum = 20000;
  tilingDatafromBin->innerLoopNum = 39;
  tilingDatafromBin->innerLoopTail = 3616;
  tilingDatafromBin->innerTailLoopNum = 39;
  tilingDatafromBin->innerTailLoopTail = 3616;
  tilingDatafromBin->outerLoopNum = 5;
  tilingDatafromBin->outerLoopNumTail = 3616;
  tilingDatafromBin->outerTailLoopNum = 5;
  tilingDatafromBin->outerTailLoopNumTail = 3616;

  ICPU_SET_TILING_KEY(602);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_701) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 1.0;
  tilingDatafromBin->num = 101;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 13;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 5;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 5;

  ICPU_SET_TILING_KEY(701);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_103) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 0.0;
  tilingDatafromBin->num = 1;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 1;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 1;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 1;

  ICPU_SET_TILING_KEY(103);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_203) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 0.0;
  tilingDatafromBin->num = 1;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 1;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 1;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 1;

  ICPU_SET_TILING_KEY(203);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_303) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 0.0;
  tilingDatafromBin->num = 1;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 1;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 1;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 1;

  ICPU_SET_TILING_KEY(303);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_403) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 0.0;
  tilingDatafromBin->num = 1;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 1;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 1;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 1;

  ICPU_SET_TILING_KEY(403);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_503) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 0.0;
  tilingDatafromBin->num = 1;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 1;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 1;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 1;

  ICPU_SET_TILING_KEY(503);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_603) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 0.0;
  tilingDatafromBin->num = 1;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 1;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 1;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 1;

  ICPU_SET_TILING_KEY(603);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(lin_space_test, test_case_703) {
  size_t outputByteSize = 101 * sizeof(float);
  size_t tiling_data_size = sizeof(LinSpaceTilingData);

  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t numBlocks = 1;
  system("cp -r ../../../../math/lin_space/tests/ut/op_kernel/lin_space_data ./");
  system("chmod -R 755 ./lin_space_data/");
  system("cd ./lin_space_data/ && rm -rf ./*bin");
  system("cd ./lin_space_data/ && python3 gen_data.py 1 100 100 float32");
  system("cd ./lin_space_data/ && python3 gen_tiling.py case0");

  char* path_ = get_current_dir_name();
  string path(path_);

  LinSpaceTilingData *tilingDatafromBin = reinterpret_cast<LinSpaceTilingData *>(tiling);

  tilingDatafromBin->start = 1.0;
  tilingDatafromBin->stop = 101.0;
  tilingDatafromBin->scalar = 0.0;
  tilingDatafromBin->num = 1;
  tilingDatafromBin->matrixLen = 8;
  tilingDatafromBin->realCoreNum = 1;
  tilingDatafromBin->numPerCore = 8;
  tilingDatafromBin->tailNum = 1;
  tilingDatafromBin->innerLoopNum = 0;
  tilingDatafromBin->innerLoopTail = 0;
  tilingDatafromBin->innerTailLoopNum = 0;
  tilingDatafromBin->innerTailLoopTail = 0;
  tilingDatafromBin->outerLoopNum = 1;
  tilingDatafromBin->outerLoopNumTail = 8;
  tilingDatafromBin->outerTailLoopNum = 1;
  tilingDatafromBin->outerTailLoopNumTail = 1;

  ICPU_SET_TILING_KEY(703);
  ICPU_RUN_KF(lin_space, numBlocks, nullptr, nullptr, nullptr, output, workspace, (uint8_t *)(tilingDatafromBin));

  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}
