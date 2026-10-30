/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROPE_MATRIX_TILING_H
#define ROPE_MATRIX_TILING_H

#include <iostream>
// must define this before tiling.h, otherwise will compile error
typedef uint8_t char_t; 
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../inc/rope_matrix_extern.h"

namespace npu_ops_transformer_ext {
namespace RopeMatrix {

uint8_t *GenerateTiling(RopeMatrixTiling *ropeTiling, uint32_t* mmTilingData, uint32_t mmSize);

uint32_t CeilingTiling(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b -1) / b;
}

uint8_t *GetTilingBuf(optiling::TCubeTiling *tilingData, uint32_t* mmTilingData, uint32_t mmSize)
{
    uint8_t *buf = nullptr;
    uint32_t tilingSize = tilingData->GetDataSize();
    if (tilingSize > 0) {
        buf = (uint8_t *)malloc(tilingSize);
        tilingData->SaveToBuffer(buf, tilingSize);

        uint32_t num = tilingSize / sizeof(uint32_t);
        num = (num < mmSize) ? num : mmSize;
        uint32_t *ptr = mmTilingData;
        auto mmtiling32 = reinterpret_cast<uint32_t *>(buf);
        for (uint32_t i = 0; i < num; i++) {
            *(ptr + i) = *(mmtiling32 + i);
        }

        free(buf);
        buf = nullptr;
    }
    return reinterpret_cast<uint8_t *>(mmTilingData);
}

uint8_t *GenerateTiling(RopeMatrixTiling *ropeTiling, uint32_t* mmTilingData, uint32_t mmSize)
{
    constexpr uint32_t baseSize = 128;
    uint32_t usedCoreNum = ropeTiling->blockDim;
    uint32_t B = ropeTiling->b;
    uint32_t H = ropeTiling->n;
    uint32_t M = ropeTiling->s;
    uint32_t N = ropeTiling->d;
    uint32_t K = ropeTiling->d;

    matmul_tiling::TPosition leftPosition = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat leftFormat = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType leftDtype = matmul_tiling::DataType::DT_BFLOAT16;
    bool isTransA = false;

    matmul_tiling::TPosition rightPosition = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat rightFormat = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType rightDtype = matmul_tiling::DataType::DT_BFLOAT16;
    bool isTransB = false;

    matmul_tiling::TPosition resultPosition = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat resultFormat = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType resultDtype = matmul_tiling::DataType::DT_BFLOAT16;

    bool isBias = false;
    uint32_t aicRatio = 2;

    uint32_t calSingleCoreM = CeilingTiling(M, usedCoreNum * aicRatio) * aicRatio;
    uint32_t baseM = baseSize;
    uint32_t baseN = baseSize;

    optiling::TCubeTiling tilingData;
    const char *socVersion = "Ascend910B3";
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

    tilingApi.SetOrgShape(M, N, K); // 完成的MNK大小，单位为元素个数
    tilingApi.SetShape(M, N, K); // matmul计算形状的MNK，考虑脏数据
    tilingApi.SetSingleShape(calSingleCoreM, baseSize, baseSize);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    int64_t checkCode = -1;
    if (res == checkCode) {
        std::cout << "gen tiling failed" << B << H << M << N << K << std::endl;
    }
    return GetTilingBuf(&tilingData, mmTilingData, mmSize);
}

} // namespace RopeMatrix
} //namespace npu_ops_transformer_ext

#endif