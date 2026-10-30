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
 * \file sinkhorn.h
 * \brief
 */
#ifndef SINKHORN_KERNEL_H_
#define SINKHORN_KERNEL_H_

#include "kernel_operator.h"

#ifndef __CCE_KT_TEST__
// 内核函数不支持"%e"
#define FLOAT_FMT "%f"
#else
#define FLOAT_FMT "%.10e"
#endif

#define OP_LOGD_0(fmt, ...)                                \
    do {                                                   \
        printf("[%d] " fmt "\n", blockIdx, ##__VA_ARGS__); \
    } while (0)

#define OP_LOGD_0_0(fmt, ...)          \
    if (blockIdx == 0) {               \
        OP_LOGD_0(fmt, ##__VA_ARGS__); \
    }

#define MAX_HALF_DUMP_NUM 64
#define ONE_DUMP_NUM 8
#define DUMP_LT_0(tensor, dataLen, fmt, ...)                                \
    do {                                                                    \
        printf("[%d] " fmt " " #tensor ": ", blockIdx, ##__VA_ARGS__);      \
        for (int i = 0; i < dataLen; i++) {                                 \
            if (i > 0 && (i % ONE_DUMP_NUM == 0)) {                         \
                printf("\n");                                               \
            }                                                               \
            if (i == MAX_HALF_DUMP_NUM) {                                   \
                i = dataLen - MAX_HALF_DUMP_NUM;                            \
                i = ((i + ONE_DUMP_NUM - 1) / ONE_DUMP_NUM) * ONE_DUMP_NUM; \
                if (i <= MAX_HALF_DUMP_NUM) {                               \
                    i = MAX_HALF_DUMP_NUM;                                  \
                } else {                                                    \
                    printf("...... %d\n", i);                               \
                }                                                           \
            }                                                               \
            printf(FLOAT_FMT " ", tensor.GetValue(i));                      \
        }                                                                   \
        printf("\n");                                                       \
    } while (0)

#define DUMP_LT_0_0(tensor, dataLen, fmt, ...)          \
    if (blockIdx == 0) {                                \
        DUMP_LT_0(tensor, dataLen, fmt, ##__VA_ARGS__); \
    }

constexpr int PRINT_LEVEL = 0;

#if PRINT_LEVEL == 0
#define OP_LOGD_1(fmt, ...)
#define OP_LOGD_2(fmt, ...)
#define OP_LOGD_3(fmt, ...)
#define OP_LOGD_0_1(fmt, ...)
#define OP_LOGD_0_2(fmt, ...)
#define OP_LOGD_0_3(fmt, ...)
#define DUMP_LT_1(tensor, dataLen, fmt, ...)
#define DUMP_LT_2(tensor, dataLen, fmt, ...)
#define DUMP_LT_3(tensor, dataLen, fmt, ...)
#define DUMP_LT_0_1(tensor, dataLen, fmt, ...)
#define DUMP_LT_0_2(tensor, dataLen, fmt, ...)
#define DUMP_LT_0_3(tensor, dataLen, fmt, ...)

#elif PRINT_LEVEL == 1
#define OP_LOGD_1 OP_LOGD_0
#define OP_LOGD_2(fmt, ...)
#define OP_LOGD_3(fmt, ...)
#define OP_LOGD_0_1 OP_LOGD_0_0
#define OP_LOGD_0_2(fmt, ...)
#define OP_LOGD_0_3(fmt, ...)
#define DUMP_LT_1 DUMP_LT_0
#define DUMP_LT_2(tensor, dataLen, fmt, ...)
#define DUMP_LT_3(tensor, dataLen, fmt, ...)
#define DUMP_LT_0_1 DUMP_LT_0_0
#define DUMP_LT_0_2(tensor, dataLen, fmt, ...)
#define DUMP_LT_0_3(tensor, dataLen, fmt, ...)

#elif PRINT_LEVEL == 2
#define OP_LOGD_1 OP_LOGD_0
#define OP_LOGD_2 OP_LOGD_0
#define OP_LOGD_3(fmt, ...)
#define OP_LOGD_0_1 OP_LOGD_0_0
#define OP_LOGD_0_2 OP_LOGD_0_0
#define OP_LOGD_0_3(fmt, ...)
#define DUMP_LT_1 DUMP_LT_0
#define DUMP_LT_2 DUMP_LT_0
#define DUMP_LT_3(tensor, dataLen, fmt, ...)
#define DUMP_LT_0_1 DUMP_LT_0_0
#define DUMP_LT_0_2 DUMP_LT_0_0
#define DUMP_LT_0_3(tensor, dataLen, fmt, ...)

#elif PRINT_LEVEL == 3
#define OP_LOGD_1 OP_LOGD_0
#define OP_LOGD_2 OP_LOGD_0
#define OP_LOGD_3 OP_LOGD_0
#define OP_LOGD_0_1 OP_LOGD_0_0
#define OP_LOGD_0_2 OP_LOGD_0_0
#define OP_LOGD_0_3 OP_LOGD_0_0
#define DUMP_LT_1 DUMP_LT_0
#define DUMP_LT_2 DUMP_LT_0
#define DUMP_LT_3 DUMP_LT_0
#define DUMP_LT_0_1 DUMP_LT_0_0
#define DUMP_LT_0_2 DUMP_LT_0_0
#define DUMP_LT_0_3 DUMP_LT_0_0
#endif

// 多核汇总D1
// 多核汇总可能会稍微快一点，但是测试时发现不稳定，有时极慢，不推荐
#ifdef MULTI_CORE_SUM_FOR_D1
#undef MULTI_CORE_SUM_FOR_D1
#endif

namespace AscendC {

// 为2会导致一个tiling数据量下降，对性能没有提升，不推荐
constexpr uint32_t COST_BUFFER_NUM = 1;

constexpr uint32_t SHAPEOUT_SIZE = 2;
constexpr uint32_t BIT_NUM_PER_BYTE = 8;
constexpr uint32_t HEADER_BLOCK_SIZE = 64;
constexpr uint32_t HEADER_SIZE_IN_INT64 = 8;

constexpr uint32_t OFFSET_SHIFT_BITS = 3;     // offset偏移量移位输，<<3 等价于 *8
constexpr uint32_t INT64_LENGTH_IN_INT32 = 2; // INT64 相当于 2个int32长
constexpr uint32_t GATHER_RESULT_STRIDE = 8;
constexpr uint64_t LOOP_FLAG_INDEX = 0;

// T: 表示运算过程中的数据类型
// IT: 表示输入cost的数据类型
template <typename T, typename IT = T>
class KernelSinkhorn {
public:
    __aicore__ inline KernelSinkhorn() {}
    __aicore__ inline void Init(GM_ADDR cost, GM_ADDR p, GM_ADDR workspace, const SinkhornTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitWS(GM_ADDR workspace, bool isFormer, uint64_t formerNum, uint64_t formerRow,
                                  uint64_t tailRow);
    __aicore__ inline void InitUB();
    __aicore__ inline void InitD0GlobalInWS();
    __aicore__ inline void InitD1GlobalInWS();
    __aicore__ inline void InitD1GlobalInWSNew();
    __aicore__ inline void InitD();
    __aicore__ inline void ExpCost();
    __aicore__ inline void CopyInForExp(uint32_t ind, uint32_t length);
    __aicore__ inline void ComputeForExp(uint32_t length);
    template <typename _IT>
    __aicore__ inline void CopyOutForExp(uint32_t ind, uint32_t length);
    __aicore__ inline void SetLoopFlag(uint64_t loop);
    __aicore__ inline uint64_t GetLoopFlag();
    __aicore__ inline void ComputeResultCore(int t, uint32_t row, LocalTensor<T> d1Local);
    __aicore__ inline void ComputeResult();
    template <typename _IT>
    __aicore__ inline void CopyInFromP(uint16_t row, const GlobalTensor<_IT>& pG);
    template <typename _IT>
    __aicore__ inline void SaveP(uint16_t row, const GlobalTensor<_IT>& pG, const LocalTensor<T>& localTensor);
    __aicore__ inline void ComputeD0(uint32_t row, LocalTensor<T> costSrcLocal, LocalTensor<T> d1InLocal);
    // 计算每个Tile的d1   torch.sum(d0.unsqueeze(1) * cost, 1)
    __aicore__ inline void ComputeD1(uint32_t row, LocalTensor<T> costSrcLocal, LocalTensor<T> d0OutLocal);
    __aicore__ inline void CopyInD1BlockInWS(DataCopyExtParams copyParams, DataCopyPadExtParams<T> padParams);
    __aicore__ inline void SumD1Block(DataCopyExtParams copyParams);
    __aicore__ inline void UpdateD0();
    __aicore__ inline void SumD1(int block);
    __aicore__ inline void UpdateD1();
    __aicore__ inline void DataCacheClean(GlobalTensor<T> global);

private:
    // 输入
    GlobalTensor<IT> costGlobal; // Exp计算前使用，后期切勿再使用
    float tol;

    // 输出
    GlobalTensor<IT> pGlobal; // Exp的输出也存放在这里

    // Workspace空间
    GlobalTensor<uint64_t> headerInWS; // 头，64B对齐，[0]: loopFlag, 控制是否退出循环，其他空间预留
    GlobalTensor<T> d0GlobalInWS;      // Global d0, 大小为totalRow
    GlobalTensor<T> d0BlockInWS;     // Block d0, 这个是前者+offset之后的地址，不额外占用workspace空间
    GlobalTensor<T> d1GlobalInWS;    // Global d1, 大小为totalCol
    GlobalTensor<T> d1GlobalInWSNew; // Global d1 new, 新的d1汇总数据， 大小为totalCol
    GlobalTensor<T> d1BlockInWS;     // Block d1, 每个Block需要一块，每块大小totalCol

    // 用于Vector计算，存放在UB中
    TPipe pipe;
    TQue<QuePosition::VECIN, COST_BUFFER_NUM> costInQueue;   // 存放输入，大小为tileRow*totalCol
    TQue<QuePosition::VECOUT, COST_BUFFER_NUM> costOutQueue; // 存放输出，大小为tileRow*totalCol

    // d0, d1空间
    TQue<QuePosition::VECIN, 1> d0InQueue, d1InQueue; // d0, d1作为输入的空间，大小分别为tileRow, totalCol
    TQue<QuePosition::VECIN, 1>
        d1OldInQueue; // 旧 d1 搬入空间（收敛判断用），大小 totalCol，专用 VECIN 以 EnQue/DeQue 提供 MTE2->V 同步
    TQue<QuePosition::VECOUT, 1> d0OutQueue, d0OutQueue2, d0OutQueue3; // d0临时输出空间，大小分别为tileRow
    TQue<QuePosition::VECOUT, 1> d1OutQueue, d1OutQueue2, d1OutQueue3; // d1临时输出空间，大小分别为totalCol

    uint32_t numBlocks;
    uint32_t blockIdx;

    uint32_t blockRow; // 当前Block的行数

    uint64_t tileNum;        // Tile的数量
    uint64_t lastTileRow;    // last Tile行数
    uint64_t lastTileLength; // last Tile长度

    uint64_t tileRow;    // Tile行数
    uint64_t tileLength; // Tile长度(非last)

    uint64_t totalRow;        // 总行数
    uint64_t totalCol;        // 总列数
    uint64_t totalColAligned; // 对齐后的列数

    uint32_t loopCount = 0; // 循环次数

    uint16_t rowLengthAligned;

    static constexpr float eps = 0.00000001f;
};

} // namespace AscendC

#include "sinkhorn.inc"
#include "sinkhorn_exp.inc"
#include "sinkhorn_update.inc"
#endif // SINKHORN_KERNEL_H_
