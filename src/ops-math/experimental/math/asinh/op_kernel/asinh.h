/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file asinh.h
 * \brief
 */
#ifndef ASINH_H
#define ASINH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "asinh_tiling_data.h"
#include "asinh_tiling_key.h"
namespace NsAsinh {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_TO_BIT = 8;
template <typename T>
class Asinh {
public:
    __aicore__ inline Asinh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AsinhTilingData* tilingData);
    __aicore__ inline void Process();

    inline static constexpr T taylorCoefficients[] = {1.0,         -1.0 / 6,     3.0 / 40,      -5.0 / 112,
                                                      35.0 / 1152, -63.0 / 2816, 231.0 / 13312, -143.0 / 10240};
    inline static constexpr T Boudry = 0.70710678118654752440084436210485;
    inline static constexpr int32_t ELEMTENT_ALIGN = 256 / sizeof(T); /* 和BUFFER_ALIGN的大小保持一致 */

private:
    __aicore__ inline void CopyIn(uint64_t progress, uint64_t tileLength);
    __aicore__ inline void CopyOut(uint64_t progress, uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t progress, uint64_t tileLength);
    __aicore__ inline void ComputeArcSinh(LocalTensor<T> yLocal, LocalTensor<T> xLocal, uint64_t tileLength);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;   // 占用BUFFER_NUM个tileBufferLen_内存
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY; // 占用BUFFER_NUM个tileBufferLen_内存
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    TBuf<TPosition::VECCALC> outputTempBuf;   // 占用1个tileBufferLen_内存
    TBuf<TPosition::VECCALC> xPowTempBuf;     // 占用1个tileBufferLen_内存
    TBuf<TPosition::VECCALC> calcTempBuf;     // 占用1个tileBufferLen_内存
    TBuf<TPosition::VECCALC> xBoudryMarkMask; // 占用1/8（转换成bit使用 BYTE_TO_BIT）个tileBufferLen_内存
    TBuf<TPosition::VECCALC> xSignMask;       // 占用1/8（转换成bit使用 BYTE_TO_BIT）个tileBufferLen_内存

    uint64_t loopCount_ = 0;
    uint64_t blockLength_ = 0;
    uint64_t tileBufferLen_ = 0;
    uint64_t tailTileLen_ = 0;
};

template <typename T>
__aicore__ inline void Asinh<T>::Init(GM_ADDR x, GM_ADDR y, const AsinhTilingData* tilingData)
{
    auto blockIdx_ = GetBlockIdx();
    uint64_t offset;

    if (blockIdx_ >= tilingData->formerCoreNum) {
        blockLength_ = tilingData->tailCoreDataNum;
        loopCount_ = tilingData->tailCoreLoopCount;
        tileBufferLen_ = tilingData->tailCoreFormerDataNum;
        tailTileLen_ = tilingData->tailCoreTailDataNum;
        offset = tilingData->formerCoreNum * tilingData->formerCoreDataNum +
                 (blockIdx_ - tilingData->formerCoreNum) * blockLength_;
    } else {
        blockLength_ = tilingData->formerCoreDataNum;
        loopCount_ = tilingData->formerCoreLoopCount;
        tileBufferLen_ = tilingData->formerCoreFormerDataNum;
        tailTileLen_ = tilingData->formerCoreTailDataNum;
        offset = blockLength_ * blockIdx_;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(outputTempBuf, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(xPowTempBuf, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(calcTempBuf, tileBufferLen_ * sizeof(T));
    pipe.InitBuffer(xSignMask, (tileBufferLen_ + BYTE_TO_BIT - 1) / BYTE_TO_BIT);
    pipe.InitBuffer(xBoudryMarkMask, (tileBufferLen_ + BYTE_TO_BIT - 1) / BYTE_TO_BIT);
}

template <typename T>
__aicore__ inline void Asinh<T>::CopyIn(uint64_t progress, uint64_t tileLength)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = tileLength * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * tileBufferLen_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Asinh<T>::CopyOut(uint64_t progress, uint64_t tileLength)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = tileLength * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * tileBufferLen_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Asinh<T>::ComputeArcSinh(LocalTensor<T> yLocal, LocalTensor<T> xLocal, uint64_t tileLength)
{
    /* 按照泰勒公式计算arcsin(x) = C*x + C*x^3 + C*x^5 + C*x^7 + C*x^9 + C*x^11 + C*x^13 + C*x^15 .....
        其中C为泰勒系数，参考常量taylorCoefficients，注意xLocal后续计算还需要所以不能修改 */
    LocalTensor<T> xPowTempTensor = xPowTempBuf.Get<T>(); // 保存x幂的计算结果
    LocalTensor<T> calcTempTensor = calcTempBuf.Get<T>(); // 保存每个泰勒展开项结果

    /* 泰勒公式的第一个C*x */
    DataCopy(xPowTempTensor, xLocal, tileLength);
    DataCopy(yLocal, xLocal, tileLength);

    Mul(xLocal, xLocal, xLocal, tileLength); // x^2，注意xLocal平方后面要用，因此此次修改
    for (auto i = 1; i < sizeof(taylorCoefficients) / sizeof(T); i++) {
        Mul(xPowTempTensor, xPowTempTensor, xLocal, tileLength);
        Muls(calcTempTensor, xPowTempTensor, (T)taylorCoefficients[i], tileLength);
        Add(yLocal, yLocal, calcTempTensor, tileLength);
    }
}

template <typename T>
__aicore__ inline void Asinh<T>::Compute(uint64_t progress, uint64_t tileLength)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    LocalTensor<T> outputTempTensor = outputTempBuf.Get<T>();
    LocalTensor<uint8_t> xSign = xSignMask.Get<uint8_t>();
    LocalTensor<uint8_t> xBoudryMark = xBoudryMarkMask.Get<uint8_t>();

    // 提取正负值标记，小于0标记符号位，然后取绝对值进行计算
    CompareScalar(xSign, xLocal, (T)0.0, AscendC::CMPMODE::LT, tileLength);
    Abs(xLocal, xLocal, tileLength);

    // 计算边界值标记掩码
    CompareScalar(xBoudryMark, xLocal, (T)Boudry, AscendC::CMPMODE::LT, tileLength);

    // 小于Boudry的按照泰勒展开公式直接计算放到yLocal中
    DataCopy(outputTempTensor, xLocal, tileLength); /* 先保存xLocal下来*/
    ComputeArcSinh(yLocal, xLocal, tileLength);     /* 注意，里面已经将X改成X^2 */

    // 大于Boudry按照arcsinh(x) = ln(x + sqrt(x^2+1))计算放到outputTempTensor中
    Adds(xLocal, xLocal, (T)1.0, tileLength);
    Sqrt(xLocal, xLocal, tileLength);
    Add(xLocal, xLocal, outputTempTensor, tileLength);
    Ln(outputTempTensor, xLocal, tileLength);

    // 根据边界值掩码小于Boudry取yLocal计算结果，大于Boudry取outputTempTensor结果
    Select(yLocal, xBoudryMark, yLocal, outputTempTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength);

    // arcsinh(-x) = -arcsinh(x), 根据符号标记合并还原计算负值情况
    Muls(outputTempTensor, yLocal, (T)(-1.0), tileLength);
    Select(yLocal, xSign, outputTempTensor, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tileLength);

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Asinh<T>::Process()
{
    for (auto i = 0; i < loopCount_ - 1; i++) {
        CopyIn(i, tileBufferLen_);
        Compute(i, tileBufferLen_);
        CopyOut(i, tileBufferLen_);
    }
    CopyIn(loopCount_ - 1, tailTileLen_);
    // 计算的时候必须保证tensor大小是256字节整数倍
    Compute(loopCount_ - 1, (tailTileLen_ + ELEMTENT_ALIGN - 1) / ELEMTENT_ALIGN * ELEMTENT_ALIGN);
    CopyOut(loopCount_ - 1, tailTileLen_);
}

} // namespace NsAsinh
#endif // ASINH_H