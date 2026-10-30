/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/arch35/scale_tiling_struct.h"
#include "../../../op_kernel/arch35/scale_kernel.h"

#ifndef GET_TILING_DATA_WITH_STRUCT
#define REGISTER_TILINGDATA_SIZE(tiling_struct, counter)
#if defined(ASCENDC_CPU_DEBUG)
template <class T>
inline __aicore__ void InitTilingData(const __gm__ uint8_t *p, T *td)
{
    constexpr uint64_t sz = sizeof(T);
    constexpr uint32_t judge = sz > 15 ? sz - 15 : 0;
    uint32_t i = 0;
    if (judge > 0) {
        for (; i < judge; i += 16) {
            (*(uint64_t*)((uint8_t*)td + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *)p + i));
            (*(uint64_t*)((uint8_t*)td + i + 8)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *)p + i + 8));
        }
    }
    if (sz & 0x08) { (*(uint64_t*)((uint8_t*)td + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t*)p + i)); i += 8; }
    if (sz & 0x04) { (*(uint32_t*)((uint8_t*)td + i)) = (*(const __gm__ uint32_t*)((const __gm__ uint8_t*)p + i)); i += 4; }
    if (sz & 0x02) { (*(uint16_t*)((uint8_t*)td + i)) = (*(const __gm__ uint16_t*)((const __gm__ uint8_t*)p + i)); i += 2; }
    if (sz & 0x01) { (*(uint8_t*)((uint8_t*)td + i)) = (*(const __gm__ uint8_t*)((const __gm__ uint8_t*)p + i)); }
}
#endif
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    REGISTER_TILINGDATA_SIZE(tiling_struct, __COUNTER__);                    \
    tiling_struct tiling_data;                                               \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);
#endif

void scale_float_rank4_no_bias(GM_ADDR x, GM_ADDR scale_in, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(ScaleTilingData<4>, td, tiling);
    GM_ADDR ins[3] = {x, scale_in, bias};
    GM_ADDR outs[1] = {y};
    ScaleKernel<float, 4> kernel;
    kernel.Init(ins, outs, &td);
    kernel.Process();
}

void scale_float_rank4_with_bias(GM_ADDR x, GM_ADDR scale_in, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(ScaleTilingData<4>, td, tiling);
    GM_ADDR ins[3] = {x, scale_in, bias};
    GM_ADDR outs[1] = {y};
    ScaleKernel<float, 4> kernel;
    kernel.Init(ins, outs, &td);
    kernel.Process();
}

class ScaleKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScaleKernelTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ScaleKernelTest TearDown" << std::endl;
    }
};

static void FillScaleTilingData4(ScaleTilingData<4>* td, bool hasBias)
{
    memset(td, 0, sizeof(ScaleTilingData<4>));
    td->split = {2, 3, 1, 3};
    td->multicore = {1, 1, 1, 0};
    td->rank = 2;
    td->per_buf_bytes = 15744;
    td->per_buf_elems = 3936;
    td->max_bro_shape[0] = 1;
    td->max_bro_shape[1] = 1;
    td->max_bro_shape[2] = 3;
    td->max_bro_shape[3] = 5;
    td->num_inputs = hasBias ? 3 : 2;
    td->num_outputs = 1;
    td->has_bias = hasBias ? 1 : 0;
    td->input_shapes[0][0] = 1; td->input_shapes[0][1] = 1; td->input_shapes[0][2] = 3; td->input_shapes[0][3] = 5;
    td->input_strides[0][0] = 0; td->input_strides[0][1] = 0; td->input_strides[0][2] = 5; td->input_strides[0][3] = 1;
    td->input_shapes[1][0] = 1; td->input_shapes[1][1] = 1; td->input_shapes[1][2] = 1; td->input_shapes[1][3] = 5;
    td->input_strides[1][0] = 0; td->input_strides[1][1] = 0; td->input_strides[1][2] = 0; td->input_strides[1][3] = 1;
    if (hasBias) {
        td->input_shapes[2][0] = 1; td->input_shapes[2][1] = 1; td->input_shapes[2][2] = 1; td->input_shapes[2][3] = 5;
        td->input_strides[2][0] = 0; td->input_strides[2][1] = 0; td->input_strides[2][2] = 0; td->input_strides[2][3] = 1;
    }
    td->output_shapes[0][0] = 1; td->output_shapes[0][1] = 1; td->output_shapes[0][2] = 3; td->output_shapes[0][3] = 5;
    td->output_strides[0][0] = 0; td->output_strides[0][1] = 0; td->output_strides[0][2] = 5; td->output_strides[0][3] = 1;
}

TEST_F(ScaleKernelTest, test_float_no_bias_rank4)
{
    constexpr int64_t N = 15;
    constexpr size_t ELEM_SIZE = sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* scale_in = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);

    float* xF = reinterpret_cast<float*>(x);
    float* scF = reinterpret_cast<float*>(scale_in);
    float* yF = reinterpret_cast<float*>(y);
    for (int i = 0; i < N; i++) xF[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 5; i++) scF[i] = 2.0f;
    memset(bias, 0, N * ELEM_SIZE);
    memset(y, 0, N * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScaleTilingData<4>));

    ScaleTilingData<4>* td = reinterpret_cast<ScaleTilingData<4>*>(tiling);
    FillScaleTilingData4(td, false);

    ICPU_SET_TILING_KEY(4);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scale_float_rank4_no_bias, 1, x, scale_in, bias, y, workspace, tiling);

    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>(i + 1) * 2.0f;
        EXPECT_FLOAT_EQ(yF[i], expected);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(scale_in);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ScaleKernelTest, test_float_with_bias_rank4)
{
    constexpr int64_t N = 15;
    constexpr size_t ELEM_SIZE = sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* scale_in = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(N * ELEM_SIZE);

    float* xF = reinterpret_cast<float*>(x);
    float* scF = reinterpret_cast<float*>(scale_in);
    float* biF = reinterpret_cast<float*>(bias);
    float* yF = reinterpret_cast<float*>(y);
    for (int i = 0; i < N; i++) xF[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 5; i++) scF[i] = 2.0f;
    for (int i = 0; i < 5; i++) biF[i] = 1.0f;
    memset(y, 0, N * ELEM_SIZE);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScaleTilingData<4>));

    ScaleTilingData<4>* td = reinterpret_cast<ScaleTilingData<4>*>(tiling);
    FillScaleTilingData4(td, true);

    ICPU_SET_TILING_KEY(4);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scale_float_rank4_with_bias, 1, x, scale_in, bias, y, workspace, tiling);

    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>(i + 1) * 2.0f + 1.0f;
        EXPECT_FLOAT_EQ(yF[i], expected);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(scale_in);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
