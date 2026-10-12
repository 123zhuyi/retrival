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
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <cstdint>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "attention_update_tiling.h"

using namespace std;

namespace {
template <typename T1, typename T2>
inline T1 CeilA2B(T1 a, T2 b) {
    return (a + b - 1) / b;
}

template <typename T>
uint8_t* CreateTensorList(const std::vector<std::vector<uint64_t>>& shapeInfos, char* d_type) {
    uint64_t tensorListDescCount = 1 + shapeInfos.size() * 2;
    for (auto s : shapeInfos) {
        tensorListDescCount += s.size();
    }
    std::vector<uint64_t> shapeSizeList;
    uint64_t* tensorListDesc = (uint64_t*)AscendC::GmAlloc(tensorListDescCount * sizeof(uint64_t));
    *tensorListDesc = (tensorListDescCount - shapeInfos.size()) * sizeof(uint64_t);
    uint64_t addrIndex = 0;
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        addrIndex++;
        uint16_t dimCount = shapeInfos[i].size();
        *(tensorListDesc + addrIndex) = ((uint64_t)(i) << 32) + dimCount;
        uint64_t shapeSize = 1;
        for (size_t j = 0; j < dimCount; j++) {
            addrIndex++;
            *(tensorListDesc + addrIndex) = shapeInfos[i][j];
            shapeSize *= shapeInfos[i][j];
        }
        shapeSizeList.push_back(shapeSize);
    }
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        addrIndex++;
        uint64_t dataSize = shapeSizeList[i] * sizeof(T);
        uint8_t* dataPtr = (uint8_t*)AscendC::GmAlloc(CeilA2B(dataSize, 32) * 32);
        *(tensorListDesc + addrIndex) = (uint64_t)dataPtr;
    }
    return (uint8_t*)tensorListDesc;
}

template <typename T>
void FreeTensorList(uint8_t* addr, const std::vector<std::vector<uint64_t>>& shapeInfos, char* d_type) {
    uint64_t dataPtrOffset = *((uint64_t*)addr);
    uint8_t* dataAddr = addr + dataPtrOffset;
    for (size_t i = 0; i < shapeInfos.size(); i++) {
        uint64_t shapeSize = 1;
        for (size_t j = 0; j < shapeInfos[i].size(); j++) {
            shapeSize *= shapeInfos[i][j];
        }
        uint8_t* tensorAddr = (uint8_t*)(*((uint64_t*)(dataAddr) + i));
        AscendC::GmFree((void*)(tensorAddr));
    }

    AscendC::GmFree((void*)addr);
}
} // namespace

extern "C" void attention_update(GM_ADDR les, GM_ADDR in, GM_ADDR out, GM_ADDR lseout, GM_ADDR workspace, GM_ADDR tiling);

struct AttentionUpdateTestParam {
  string caseName;

  size_t dataTypeSize;

  uint32_t tilingKey;
  DecodeUpdateTilingData tiling;
};

class AttentionUpdateTest : public testing::TestWithParam<AttentionUpdateTestParam> {
protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool3DTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool3DTest TearDown" << std::endl;
  }
};

TEST_P(AttentionUpdateTest, test_case_attention_update) {
  AttentionUpdateTestParam param = GetParam();

  uint32_t tilingKey = param.tilingKey;
  DecodeUpdateTilingData tilingParam = param.tiling;

  int64_t lseShapeSize = tilingParam.totalLength;
  int64_t goShapeSize = tilingParam.totalLength * tilingParam.hDim;
  int64_t outputShapeSize = tilingParam.totalLength * tilingParam.hDim;
  int64_t lseoutShapeSize = tilingParam.totalLength;
  int64_t lseByteSize = lseShapeSize * param.dataTypeSize;
  int64_t goByteSize = goShapeSize * param.dataTypeSize;
  int64_t outputByteSize = outputShapeSize * param.dataTypeSize;
  int64_t lseoutByteSize = lseoutShapeSize * param.dataTypeSize;

  std::vector<std::vector<uint64_t>> lseShapeInfos = {{128}, {128}};
  std::vector<std::vector<uint64_t>> goShapeInfos = {{128, 512}, {128, 512}};

  int64_t workspaceSize = 16 * 1024 * 1204;
  int64_t tilingSize = sizeof(DecodeUpdateTilingData);

  uint8_t* lse = CreateTensorList<float>(lseShapeInfos, "float32");
  uint8_t* go = CreateTensorList<float>(goShapeInfos, "float32");
  uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
  uint8_t* lseout = tilingParam.updateType == 0 ? nullptr : (uint8_t*)AscendC::GmAlloc(lseoutByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

  DecodeUpdateTilingData* tilingData = reinterpret_cast<DecodeUpdateTilingData*>(tiling);
  tilingData->formerLength = tilingParam.formerLength;
  tilingData->tailLength = tilingParam.tailLength;
  tilingData->formerNum = tilingParam.formerNum;
  tilingData->tailNum = tilingParam.tailNum;
  tilingData->hDim = tilingParam.hDim;
  tilingData->sp = tilingParam.sp;
  tilingData->totalLength = tilingParam.totalLength;

  uint32_t blockDim = tilingParam.formerNum + tilingParam.tailNum;
  ICPU_SET_TILING_KEY(tilingKey);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(attention_update, blockDim, lse, go, output, lseout, workspace, tiling);

  FreeTensorList<float>(lse, lseShapeInfos, "float32");
  FreeTensorList<float>(go, goShapeInfos, "float32");
  AscendC::GmFree((void *)output);
  if (tilingParam.updateType != 0) {
    AscendC::GmFree((void *)lseout);
  }
  AscendC::GmFree((void *)workspace);
  AscendC::GmFree((void *)tiling);
}

static AttentionUpdateTestParam cases[] = {
  {
    "test_case_float32",
    sizeof(float),
    0, 
    {
      4, 3, 8,
      32, 512, 2,
      128
    }
  },
  {
    "test_case_float32_updateType1_case0",
    sizeof(float),
    0, 
    {
      4, 3, 8,
      32, 512, 2,
      128, 1  // 在末尾添加1，设置updateType为1
    }
  },
};

INSTANTIATE_TEST_CASE_P(AttentionUpdate, AttentionUpdateTest, testing::ValuesIn(cases));
