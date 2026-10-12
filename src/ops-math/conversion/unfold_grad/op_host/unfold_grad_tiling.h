/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unfold_grad_tiling.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_UNFOLD_GRAD_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_UNFOLD_GRAD_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(UnfoldGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchNum);
TILING_DATA_FIELD_DEF(int64_t, batchNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, batchNumTailCore);
TILING_DATA_FIELD_DEF(int64_t, maxBatchNum4Ub);
TILING_DATA_FIELD_DEF(int64_t, useCoreNum);
TILING_DATA_FIELD_DEF(int64_t, ubSizeT1);
TILING_DATA_FIELD_DEF(int64_t, ubSizeT2);
TILING_DATA_FIELD_DEF(int64_t, outputNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, inputNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, iterationNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, handleNUMOnceIterationPerCore);
TILING_DATA_FIELD_DEF(int64_t, tasksOnceMaxPerCore);
TILING_DATA_FIELD_DEF(int64_t, inputSizeLength);
TILING_DATA_FIELD_DEF(int64_t, rowAvailableLengthSrc);
TILING_DATA_FIELD_DEF(int64_t, lowestCommonMultiple);
TILING_DATA_FIELD_DEF(int64_t, colOnceMaxPerUB);
TILING_DATA_FIELD_DEF(int64_t, tailColLength);

TILING_DATA_FIELD_DEF(int64_t, typeSizeT1);
TILING_DATA_FIELD_DEF(int64_t, typeSizeT2);
TILING_DATA_FIELD_DEF(int64_t, width);
TILING_DATA_FIELD_DEF(int64_t, gradOutSizeDim);
TILING_DATA_FIELD_DEF(int64_t, inputSizeLastDim);
TILING_DATA_FIELD_DEF(int64_t, dim);
TILING_DATA_FIELD_DEF(int64_t, size);
TILING_DATA_FIELD_DEF(int64_t, step);
TILING_DATA_FIELD_DEF(int64_t, loop);
TILING_DATA_FIELD_DEF(int64_t, tail);
TILING_DATA_FIELD_DEF(int64_t, sizeLevel);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(UnfoldGrad, UnfoldGradTilingData)

struct Tiling4UnfoldGradCompileInfo {};

class UnfoldGradTiling {
public:
    explicit UnfoldGradTiling(gert::TilingContext *context_) : context(context_){};
    virtual ~UnfoldGradTiling() = default;
    ge::graphStatus DoTiling();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetInputTensorInfo();
    ge::graphStatus SetAttrParams();
    ge::graphStatus TilingBlock4SizeLevelBig();
    ge::graphStatus Tiling4Block();
    uint64_t GetTilingKey();
    bool GetDataTypeKey(ge::DataType dataType);

    void PrintInfo();
    void SetTilingKey();
    void SetTilingData();

private:
    UnfoldGradTilingData tiling;
    gert::TilingContext *context = nullptr;

    template <typename T>
    inline auto Max(T x, T y) const -> T
    {
        return x >= y ? x : y;
    }

    inline int64_t Gcd(int64_t x, int64_t y)
    {
        while (y != 0) {
            int64_t tmp = x % y;
            x = y;
            y = tmp;
        }

        return x;
    }

    inline int64_t getLowestCommonMultiple(int64_t x, int64_t y)
    {
        int64_t greatestCommonDivisor = Gcd(x, y);

        return greatestCommonDivisor == 0 ? 0 : x / greatestCommonDivisor * y;
    }

    int64_t usrWorkspaceSize = 1;
    int64_t dataTypeTilingKey = 0;
    uint64_t tilingKey_{ 0 };

    int64_t batchNum = 1;         // 总任务批次
    int64_t batchNumPerCore = 1;  // 每个核的任务批次
    int64_t batchNumTailCore = 1; // 尾核的任务批次
    int64_t maxBatchNum4Ub = 0; // 一个UB最多可以处理的任务批次 0 表示一个UB放不下， 大于0表示可以放下
    int64_t useCoreNum = 1;                    // 实际使用的核数
    int64_t ubSizeT1 = 0;                      //  用于T1的UB1、UB2大小
    int64_t ubSizeT2 = 0;                      // TransDataTo5HD要求对齐512B
    int64_t outputNumPerCore = 1;              // 一个核一个批次输出的数据量
    int64_t inputNumPerCore = 1;               // 一个核一个批次输入的数据量
    int64_t iterationNumPerCore = 0;           // 迭代次数   dim == 1不需要该参数
    int64_t handleNUMOnceIterationPerCore = 0; // 当前输入一轮迭代处理的数据量
    int64_t tasksOnceMaxPerCore = 0;           // 每个核一个UB最多处理的数据量
    int64_t inputSizeLength = 0;               // inputSize的维度大小
    int64_t rowAvailableLengthSrc = 0;         // 转置前每行可用的长度
    int64_t lowestCommonMultiple = 1; // 最小公倍数，非尾轴情况填充冗余值算法第一次转置的行长度
    int64_t colOnceMaxPerUB = 0;      // 非尾轴情况填充冗余值算法时一个UB可处理的最大行数量
    int64_t tailColLength = 0;        // 非尾轴情况填充冗余值算法时尾块的行数量

    int64_t typeSizeT1 = 0;       // 输入输出数据类型大小
    int64_t typeSizeT2 = 4;       // T2数据类型大小，默认为fp32
    int64_t width = 0;            // TransDataTo5HD的行数 fp16/bf16为16，fp32为8
    int64_t gradOutSizeDim = 0;   // grad_out.shape[dim]
    int64_t dim = 0;              // 进行展开操作的维度
    int64_t inputSizeLastDim = 0; // inputSize的最后一维
    int64_t size = 0;             // 展开的每个切片的大小
    int64_t step = 0;             // 每个切片之间的步长

    int64_t totalCoreNum = 0;
    int64_t ubSize = 0;
    ge::DataType gardInDType = ge::DT_UNDEFINED;

    int64_t loop = 0; // 一个迭代内的循环次数
    int64_t tail = 0; // 尾块大小
    int64_t sizeLevel = 0; // 0 标记size 在UB Trans可处理数量范围内，1 超出UB Trans限制
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_FUSED_EMA_ADAM_H_
