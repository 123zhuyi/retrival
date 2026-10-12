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
 * \file transpose_apt.cpp
 * \brief Transpose 算子 Kernel 层主入口文件
 *
 * 本文件是 Transpose 算子 Kernel 侧的唯一入口，负责根据 TilingKey 将执行流分发到
 * 9 种具体的转置策略类。所有策略共享同一套 Tiling 数据结构，但使用不同的硬件加速路径。
 *
 * 分发机制：
 *   - transpose() 为全局 Kernel 入口函数，通过 TILING_KEY_IS 宏判断当前 TilingKey
 *   - 每个 TilingKey 对应一个 Process 函数（如 TransposeTensorMoveProcess）
 *   - Process 函数内部通过 sizeof(DTYPE_X) 在编译期分发到具体类型的模板实例
 *
 * sizeof(DTYPE_X) 分发原理：
 *   - DTYPE_X 是编译期宏，表示输入张量的数据类型
 *   - sizeof(DTYPE_X) == 1  → int8_t   (int8/uint8/bool/hifloat8/fp8_e5m2/fp8_e4m3fn)
 *   - sizeof(DTYPE_X) == 2  → int16_t  (fp16/bf16/int16/uint16)
 *   - sizeof(DTYPE_X) == 4  → int32_t  (fp32/int32/uint32/complex64)
 *   - sizeof(DTYPE_X) == 8  → int64_t  (fp64/int64/uint64/complex128)
 *   - else 分支：直接使用 DTYPE_X 作为模板参数（处理 complex64/128 等复合类型）
 *   这种方式避免了对每种具体 dtype 的穷举，只需按字节宽度分类即可
 *
 * TilingKey 常量含义：
 *   10000 TENSOR_MOVE        - 1维输入，纯数据搬运（DataCopyPad直搬）
 *   10001 SMALL_SHAPE         - 小数据量，SIMT模式逐元素地址计算（GM→GM直读直写）
 *   10002 CUT_ONCE            - 2-5维，NDDMA 5维搬运，单轴切分
 *   10003 CUT_TWICE           - 2-5维，NDDMA 5维搬运，双轴切分（4种区间）
 *   10004 N_LAST_TRANSPOSE    - 尾轴不转置且尾轴≥32，双缓冲流水线连续行搬移
 *   10005 BIG_DIM             - 维度>5，压缩到5维NDDMA格式
 *   10006 GATHER_TRANSPOSE    - 尾轴转置+大shape，DataCopyGather硬件指令+RegBase
 *   10007 VCONV_TRANSPOSE     - 2D perm=[1,0]+16bit，TransDataTo5HD硬件指令
 *   10008 VCONV_021_TRANSPOSE - 3D perm=[0,2,1]+8/16/32bit，TransDataTo5HD硬件指令
 */

#include "arch35/transpose_big_dim.h"
#include "arch35/transpose_cut_one_axis.h"
#include "arch35/transpose_cut_two_axis.h"
#include "arch35/transpose_n_last.h"
#include "arch35/transpose_small_shape.h"
#include "arch35/transpose_tensor_move.h"
#include "arch35/transpose_with_gather.h"
#include "arch35/transpose_transdata_5hd.h"
#include "arch35/transpose_transdata_5hd_021.h"

/* TilingKey 常量定义，与 Host 侧 SplitMode 枚举值一一对应 */
#define TENSOR_MOVE 10000      // 融合后仅1维，等价于纯数据搬运
#define SMALL_SHAPE 10001      // 小数据量(<阈值字节)，SIMT模式GM→GM直读直写
#define CUT_ONCE 10002         // NDDMA 5维搬运，切1个轴（输出切分轴≤输入切分轴映射）
#define CUT_TWICE 10003        // NDDMA 5维搬运，切2个轴（4种数据区间：Main/InputTail/OutputTail/Tail）
#define N_LAST_TRANSPOSE 10004 // 尾轴不转置 + 尾轴≥32元素，双缓冲流水线连续行搬移
#define BIG_DIM 10005          // 维度>5，压缩到5维NDDMA格式进行搬运
#define GATHER_TRANSPOSE 10006 // 尾轴转置+大shape，使用DataCopyGather硬件指令(RegBase路径)
#define VCONV_TRANSPOSE 10007  // 2D perm=[1,0]+16bit(DAV_5102)，TransDataTo5HD硬件指令
#define VCONV_021_TRANSPOSE 10008 // 3D perm=[0,2,1]+8/16/32bit(DAV_5102)，TransDataTo5HD硬件指令

using namespace Transpose;

/**
 * @brief TENSOR_MOVE(10000) 策略的 Process 函数
 * 纯数据搬运场景，经 RemoveAxisV2+MergeAxisV2 后 reduced dim==1。
 * 使用 DataCopyPad 直搬，双缓冲流水线加速。
 *
 * @param x         输入张量 GM 地址
 * @param y         输出张量 GM 地址
 * @param tilingData Tiling 数据指针（TransposeOpTilingData 结构）
 * @param pipe      管道对象指针，用于双缓冲初始化
 */
extern "C" __aicore__ inline void TransposeTensorMoveProcess(GM_ADDR x, GM_ADDR y,
                                                             const TransposeOpTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeTensorMove<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeTensorMove<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeTensorMove<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeTensorMove<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeTensorMove<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

/**
 * @brief SMALL_SHAPE(10001) 策略的 Process 函数
 * 小数据量场景，使用 SIMT 模式（2048线程并行）直接 GM→GM 读写。
 * 每个线程独立计算输出地址，无需 UB 中转。
 * 注意：不需要 TPipe 参数（无缓冲队列）。
 *
 * @param x         输入张量 GM 地址
 * @param y         输出张量 GM 地址
 * @param tilingData Tiling 数据指针
 */
extern "C" __aicore__ inline void TransposeSmallShapeProcess(GM_ADDR x, GM_ADDR y,
                                                             const TransposeOpTilingData* tilingData)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeSmallShape<int8_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeSmallShape<int16_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeSmallShape<int32_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeSmallShape<int64_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else {
        Transpose::TransposeSmallShape<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
}

/**
 * @brief CUT_ONCE(10002) 策略的 Process 函数
 * NDDMA 5维搬运，单轴切分。适用于 outCutIndex ≤ FindOutIndex(inCutIndex) 的场景。
 * 通过 DataCopy<T, NDDMA_MAX_DIM_NUM> 5维NDDMA自动转置，硬件完成维度重排。
 *
 * @param x         输入张量 GM 地址
 * @param y         输出张量 GM 地址
 * @param tilingData Tiling 数据指针
 * @param pipe      管道对象指针
 */
extern "C" __aicore__ inline void TransposeCutOneAxisProcess(GM_ADDR x, GM_ADDR y,
                                                             const TransposeOpTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeCutOneAxis<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeCutOneAxis<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeCutOneAxis<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeCutOneAxis<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeCutOneAxis<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

/**
 * @brief CUT_TWICE(10003) 策略的 Process 函数
 * NDDMA 5维搬运，双轴切分。需要同时切输入轴和输出轴，数据被划分为4种区间：
 * Main / InputTail / OutputTail / Tail，每种区间使用不同的 NDDMA 参数。
 *
 * @param x         输入张量 GM 地址
 * @param y         输出张量 GM 地址
 * @param tilingData Tiling 数据指针
 * @param pipe      管道对象指针
 */
extern "C" __aicore__ inline void TransposeCutTwoAxisProcess(GM_ADDR x, GM_ADDR y,
                                                             const TransposeOpTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeCutTwoAxis<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeCutTwoAxis<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeCutTwoAxis<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeCutTwoAxis<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeCutTwoAxis<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

/**
 * @brief BIG_DIM(10005) 策略的 Process 函数
 * 维度>5 的场景，将原始 shape 压缩到5维 NDDMA 格式进行搬运。
 * Host 侧 FlushBaseNumForBigDim() 预计算 nddmaIdx 映射，Kernel 侧据此设置 NDDMA 参数。
 *
 * @param x         输入张量 GM 地址
 * @param y         输出张量 GM 地址
 * @param tilingData Tiling 数据指针
 * @param pipe      管道对象指针
 */
extern "C" __aicore__ inline void TransposeBigDimProcess(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                                         TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeBigDim<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeBigDim<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeBigDim<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeBigDim<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeBigDim<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

/**
 * @brief N_LAST_TRANSPOSE(10004) 策略的 Process 函数
 * 尾轴不转置（perm[dim-1]==dim-1）且尾轴≥32元素的场景。
 * 尾轴在输入输出中保持连续，可按连续行搬移，使用双缓冲流水线加速。
 * 支持8维循环展开（loop4~7）实现高效的 CopyOut。
 *
 * @param x         输入张量 GM 地址
 * @param y         输出张量 GM 地址
 * @param tilingData Tiling 数据指针
 * @param pipe      管道对象指针
 */
extern "C" __aicore__ inline void TransposeNLastProcess(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData,
                                                        TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeNLast<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeNLast<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeNLast<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeNLast<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else {
        Transpose::TransposeNLast<DTYPE_X> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

/**
 * @brief GATHER_TRANSPOSE(10006) 策略的 Process 函数
 * 尾轴转置+大shape场景，使用 Gather 硬件指令（RegBase路径）。
 * 根据预计算的索引数组从 UB 中按任意跨步读取数据，实现尾轴转置。
 * 预生成4组索引覆盖 main/tail 的4种组合。
 *
 * @param x         输入张量 GM 地址
 * @param y         输出张量 GM 地址
 * @param tilingData Tiling 数据指针（GatherTransposeTilingData 结构，非通用结构）
 * @param pipe      管道对象指针
 */
extern "C" __aicore__ inline void TransposeGatherProcess(GM_ADDR x, GM_ADDR y,
                                                         const GatherTransposeTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::TransposeWithGather<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::TransposeWithGather<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::TransposeWithGather<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Transpose::TransposeWithGather<int64_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

/**
 * @brief VCONV_TRANSPOSE(10007) 策略的 Process 函数
 * DAV_5102(Ascend950) 特定路径：2D perm=[1,0] + 16bit数据类型。
 * 使用 TransDataTo5HD 硬件指令实现2D矩阵转置，以16行为单位进行行列互换。
 * 仅实例化 int16_t 模板（对应 fp16/bf16）。
 *
 * @param x         输入张量 GM 地址
 * @param y         输出张量 GM 地址
 * @param tilingData Tiling 数据指针（TransposeVCONVTilingData 结构）
 * @param pipe      管道对象指针
 */
extern "C" __aicore__ inline void TransposeVconvProcess(GM_ADDR x, GM_ADDR y,
                                                        const TransposeVCONVTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::KernelTransDataTo5HD<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

/**
 * @brief VCONV_021_TRANSPOSE(10008) 策略的 Process 函数
 * DAV_5102(Ascend950) 特定路径：3D perm=[0,2,1]（H↔W转置，N维不变）。
 * 使用 TransDataTo5HD 硬件指令，支持 8/16/32bit 数据类型。
 * 8bit 需要高低半分别处理（evenParams/oddParams），32bit dstStride 需加倍。
 *
 * @param x         输入张量 GM 地址
 * @param y         输出张量 GM 地址
 * @param tilingData Tiling 数据指针（Transpose021VCONVTilingData 结构）
 * @param pipe      管道对象指针
 */
extern "C" __aicore__ inline void Transpose021VconvProcess(GM_ADDR x, GM_ADDR y,
                                                           const Transpose021VCONVTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Transpose::KernelTransDataTo5HD021<int8_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Transpose::KernelTransDataTo5HD021<int16_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Transpose::KernelTransDataTo5HD021<int32_t> op;
        op.Init(x, y, tilingData, pipe);
        op.Process();
    }
}

/**
 * @brief Transpose 算子 Kernel 主入口函数
 *
 * 执行流程：
 * 1. 创建 TPipe 管道对象
 * 2. 声明任务类型为 AIV_ONLY（仅使用 Vector 核）
 * 3. 通过 TILING_KEY_IS 判断当前 TilingKey，从 tiling 数据中解析对应的结构体
 * 4. 调用对应策略的 Process 函数
 *
 * Tiling 数据结构说明：
 *   - TilingKey 10000-10005 使用 TransposeTilingData（内含 TransposeOpTilingData）
 *   - TilingKey 10006 使用 GatherTransposeTilingData
 *   - TilingKey 10007 使用 TransposeVCONVTilingData
 *   - TilingKey 10008 使用 Transpose021VCONVTilingData
 *
 * @param x         输入张量 GM 地址
 * @param perm      perm 数组 GM 地址（Kernel 侧不直接使用，通过 TilingData 传递）
 * @param y         输出张量 GM 地址
 * @param workspace 工作空间 GM 地址
 * @param tiling    Tiling 数据 GM 地址
 */
extern "C" __global__ __aicore__ void transpose(GM_ADDR x, GM_ADDR perm, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 声明为纯 Vector 核任务，不使用 Cube 核
    if (TILING_KEY_IS(TENSOR_MOVE)) {               // 10000: 纯数据搬运
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeTensorMoveProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(SMALL_SHAPE)) { // 10001: SIMT 小shape直读直写
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeSmallShapeProcess(x, y, &tilingData.transposeOpTiling);
    } else if (TILING_KEY_IS(CUT_ONCE)) { // 10002: NDDMA单轴切分
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeCutOneAxisProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(CUT_TWICE)) { // 10003: NDDMA双轴切分
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeCutTwoAxisProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(N_LAST_TRANSPOSE)) { // 10004: 尾轴不转置+连续行搬移
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeNLastProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(BIG_DIM)) { // 10005: >5维压缩到5维NDDMA
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        TransposeBigDimProcess(x, y, &tilingData.transposeOpTiling, &pipe);
    } else if (TILING_KEY_IS(GATHER_TRANSPOSE)) { // 10006: DataCopyGather硬件加速
        GET_TILING_DATA_WITH_STRUCT(GatherTransposeTilingData, tilingData, tiling);
        TransposeGatherProcess(x, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(VCONV_TRANSPOSE)) { // 10007: TransDataTo5HD 2D转置
        GET_TILING_DATA_WITH_STRUCT(TransposeVCONVTilingData, tilingData, tiling);
        TransposeVconvProcess(x, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(VCONV_021_TRANSPOSE)) { // 10008: TransDataTo5HD 3D 021转置
        GET_TILING_DATA_WITH_STRUCT(Transpose021VCONVTilingData, tilingData, tiling);
        Transpose021VconvProcess(x, y, &tilingData, &pipe);
    }
}
