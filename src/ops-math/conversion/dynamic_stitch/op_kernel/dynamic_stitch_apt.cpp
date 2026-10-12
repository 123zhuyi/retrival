/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file dynamic_stitch_apt.cpp
 * \brief
 */

#include "arch35/dynamic_stitch_indices_deduplicate.h"
#include "arch35/dynamic_stitch_scatter_simd.h"
#include "arch35/dynamic_stitch_scatter_simt.h"

#define TILING_KEY_SIMT_B8  100001
#define TILING_KEY_SIMT_B16 100002
#define TILING_KEY_SIMT_B32 100004
#define TILING_KEY_SIMT_B64 100008
#define TILING_KEY_SIMD_B8  200001
#define TILING_KEY_SIMD_B16 200002
#define TILING_KEY_SIMD_B32 200004
#define TILING_KEY_SIMD_B64 200008

using namespace DynamicStitch;

__global__ __aicore__ void dynamic_stitch(GM_ADDR indices, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(DynamicStitchTilingData);
    GET_TILING_DATA_WITH_STRUCT(DynamicStitchTilingData, tilingData, tiling);
    TPipe pipe;

    SetSysWorkspace(workspace);
    GM_ADDR usrWorkSpace = AscendC::GetUserWorkspace(workspace);
    // 索引去重
    DynamicStitch::DynamicStitchIndicesDeDuplicate<float> deDuplicateOp(&pipe, &tilingData);
    deDuplicateOp.Init(indices, usrWorkSpace);
    deDuplicateOp.Process();
    pipe.Reset();

    // tensor处理, 2个模版，8个tilingKey
    if (TILING_KEY_IS(TILING_KEY_SIMT_B8)) {
        DynamicStitch::DynamicStitchScatterSimt<int8_t> scatterSimtOp(&pipe, &tilingData);
        scatterSimtOp.Init(x, y, usrWorkSpace);
        scatterSimtOp.Process();
        pipe.Reset();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_B16)) {
        DynamicStitch::DynamicStitchScatterSimt<int16_t> scatterSimtOp(&pipe, &tilingData);
        scatterSimtOp.Init(x, y, usrWorkSpace);
        scatterSimtOp.Process();
        pipe.Reset();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_B32)) {
        DynamicStitch::DynamicStitchScatterSimt<int32_t> scatterSimtOp(&pipe, &tilingData);
        scatterSimtOp.Init(x, y, usrWorkSpace);
        scatterSimtOp.Process();
        pipe.Reset();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_B64)) {
        DynamicStitch::DynamicStitchScatterSimt<int64_t> scatterSimtOp(&pipe, &tilingData);
        scatterSimtOp.Init(x, y, usrWorkSpace);
        scatterSimtOp.Process();
        pipe.Reset();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_B8)) {
        DynamicStitch::DynamicStitchScatterSimd<int8_t> scatterSimdOp(&pipe, &tilingData);
        scatterSimdOp.Init(indices, x, y, usrWorkSpace);
        scatterSimdOp.Process();
        pipe.Reset();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_B16)) {
        DynamicStitch::DynamicStitchScatterSimd<int16_t> scatterSimdOp(&pipe, &tilingData);
        scatterSimdOp.Init(indices, x, y, usrWorkSpace);
        scatterSimdOp.Process();
        pipe.Reset();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_B32)) {
        DynamicStitch::DynamicStitchScatterSimd<int32_t> scatterSimdOp(&pipe, &tilingData);
        scatterSimdOp.Init(indices, x, y, usrWorkSpace);
        scatterSimdOp.Process();
        pipe.Reset();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_B64)) {
        DynamicStitch::DynamicStitchScatterSimd<int64_t> scatterSimdOp(&pipe, &tilingData);
        scatterSimdOp.Init(indices, x, y, usrWorkSpace);
        scatterSimdOp.Process();
        pipe.Reset();
    }
}
