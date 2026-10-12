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
 * \file memory_copy.h
   GM->L1
   PA
   PARope
 * \brief
 */
#ifndef MEMMORY_COPY_H
#define MEMMORY_COPY_H
#include "fia_public_define.h"

constexpr uint32_t HALF_SIZE_DIVISOR = 2;
constexpr uint32_t ND_MATRIX_STRIDE_LIMIT = 65536; // Mutil ND2NZ搬运时，Nd2NzParams支持的srcNdMatrixStride的取值范围为[0, 65536]，单位为元素
// ----------------------------------------------GmLayout--------------------------------
enum class GmFormat {
    BSNGD = 0,
    BNGSD = 1,
    NGBSD = 2,
    TNGD = 3,
    NGTD = 4,
    BSND = 5,
    BNSD = 6,
    TND = 7,
    NTD = 8,
    NGD = 12, // post_quant
};

template <GmFormat FORMAT>
struct GmLayout {
};

template <>
struct GmLayout<GmFormat::BSNGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, g, s, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        uint64_t sStride = nStride * n;
        uint64_t bStride = sStride * s;
        stride = AscendC::MakeStride(bStride, nStride, gStride, sStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::BNGSD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, g, s, d);
        uint64_t dStride = 1;
        uint64_t sStride = dStride * d;
        uint64_t gStride = sStride * s;
        uint64_t nStride = gStride * g;
        uint64_t bStride = nStride * n;
        stride = AscendC::MakeStride(bStride, nStride, gStride, sStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::BSND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, s, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t sStride = nStride * n;
        uint64_t bStride = sStride * s;
        stride = AscendC::MakeStride(bStride, nStride, sStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::BNSD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, s, d);
        uint64_t dStride = 1;
        uint64_t sStride = dStride * d;
        uint64_t nStride = sStride * s;
        uint64_t bStride = nStride * n;
        stride = AscendC::MakeStride(bStride, nStride, sStride, dStride);
    }
};

// ----------------------------------------------ActualSeqLensParser--------------------------------
enum class ActualSeqLensMode
{
    BY_BATCH = 0,
    ACCUM = 1,
};

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode() {
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}
template <FIA_LAYOUT LAYOUT_T, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode() {
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}


template <ActualSeqLensMode MODE>
class ActualSeqLensParser {
};

template <>
class ActualSeqLensParser<ActualSeqLensMode::ACCUM> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<uint64_t> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal = 0)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
    }

    __aicore__ inline uint64_t GetTBase(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        return actualSeqLengthsGm.GetValue(bIdx - 1);
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return (actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1));
    }

    __aicore__ inline uint64_t GetTSize() const
    {
        return actualSeqLengthsGm.GetValue(actualLenDims - 1);
    }
private:
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    uint32_t actualLenDims;
};

template <>
class ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<uint64_t> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
        this->defaultVal = defaultVal;
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (actualLenDims == 0) {
            return defaultVal;
        }
        if (actualLenDims == 1) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return actualSeqLengthsGm.GetValue(bIdx);
    }

    __aicore__ inline uint32_t GetActualLenDims() const 
    {
        return actualLenDims;
    }
private:
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    uint32_t actualLenDims;
    uint64_t defaultVal;
};

// ----------------------------------------------GmLayoutParams--------------------------------
enum class FormatCategory
{
    GM_Q_OUT_BNGSD = 0,
    GM_Q_OUT_TND = 1,
    GM_KV_BNSD = 2,
    GM_KV_TND = 3,
};

template <GmFormat FORMAT>
struct GmLayoutParams {};

template <>
struct GmLayoutParams<GmFormat::BSNGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <>
struct GmLayoutParams<GmFormat::BNGSD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <>
struct GmLayoutParams<GmFormat::BSND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_BNSD;
};

template <>
struct GmLayoutParams<GmFormat::BNSD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_BNSD;
};

// ----------------------------------------------OffsetCalculator--------------------------------
template <GmFormat FORMAT, FormatCategory CATEGORY>
struct OffsetCalculatorImpl {};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_BNGSD> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> actualSeqLensQParser;
    bool isQPaddingFlag = false;
    uint64_t qPaddingSize = 0;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t d,
                                GlobalTensor<uint64_t> actualSeqLengthsGmQ, uint32_t actualLenQDims,
                                bool isQPaddingFlag = false, uint64_t qPaddingSize = 0)
    {
        this->isQPaddingFlag = isQPaddingFlag;
        this->qPaddingSize = qPaddingSize;
        if(actualLenQDims != 0) {
            actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims, 0);
        }
        gmLayout.MakeLayout(b, n2, g, s1, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t dIdx)
    {
        if (isQPaddingFlag) {
            s1Idx += GetDimS1() - qPaddingSize - actualSeqLensQParser.GetActualSeqLength(bIdx);
        }
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + gIdx * GetStrideG() + s1Idx * GetStrideS1() +
                          dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<4>(gmLayout.stride); // 4:代表第5个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimG()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimS1()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<4>(gmLayout.shape); // 4:代表第5个维度，索引从0开始
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_BNSD> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> actualSeqLensKVParser;
    bool isKvPaddingFlag = false;
    uint64_t kvPaddingSize = 0;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d)
    {
        gmLayout.MakeLayout(b, n2, s2, d);
    }

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d, GlobalTensor<uint64_t> actualSeqLengthsGm,
                                uint32_t actualLenKvDims, bool isKvPaddingFlag = false, uint64_t kvPaddingSize = 0)
    {
        this->isKvPaddingFlag = isKvPaddingFlag;
        this->kvPaddingSize = kvPaddingSize;
        if(actualLenKvDims != 0) {
            actualSeqLensKVParser.Init(actualSeqLengthsGm, actualLenKvDims, 0);
        }
        gmLayout.MakeLayout(b, n2, s2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        if (isKvPaddingFlag) {
            s2Idx += GetDimS2() - kvPaddingSize - actualSeqLensKVParser.GetActualSeqLength(bIdx);
        }
        
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + s2Idx * GetStrideS2() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimS2()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }
};

template <GmFormat FORMAT>
struct OffsetCalculator : public OffsetCalculatorImpl<FORMAT, GmLayoutParams<FORMAT>::CATEGORY> {
};

// ----------------------------------------------CopyQueryGmToL1--------------------------------
template <typename Q_T, GmFormat FORMAT>
struct FaGmTensor {
    GlobalTensor<Q_T> gmTensor;
    OffsetCalculator<FORMAT> offsetCalculator;
};

enum class L1Format
{
    NZ = 0
};

template <typename Q_T, L1Format FORMAT>
struct FaL1Tensor {
    LocalTensor<Q_T> tensor;
    uint32_t rowCount;
};

struct GmCoord {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t dIdx;
    uint32_t gS1DealSize;
    uint32_t dDealSize;
};

template <typename T>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor,
    uint32_t nValue, uint32_t dValue, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = nValue; //nd矩阵的行数
    nd2nzPara.dValue = dValue; //nd矩阵的列数
    nd2nzPara.srcDValue = srcDValue; //同一nd矩阵相邻行起始地址间的偏移
    nd2nzPara.dstNzC0Stride = dstNzC0Stride;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

template <typename T>
__aicore__ inline void CopyMultiMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor,
    uint32_t srcNdMatrixNum, uint32_t srcNdMatrixStride, uint32_t dstNzMatrixStride, uint32_t nValue, uint32_t dValue, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    if (unlikely(srcNdMatrixStride > ND_MATRIX_STRIDE_LIMIT)) {
        uint64_t l1Offset = 0;
        uint64_t gmOffset = 0;
        for (uint32_t i = 0; i < srcNdMatrixNum; i++) {
            CopySingleMatrixNDToNZ(l1Tensor[l1Offset], gmTensor[gmOffset], nValue, dValue, srcDValue, dstNzC0Stride);
            gmOffset += srcNdMatrixStride;
            l1Offset += dstNzMatrixStride;
        }
    } else {
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = srcNdMatrixNum;
        nd2nzPara.nValue = nValue; //nd矩阵的行数
        if constexpr (IsSameType<T, int4b_t>::value) {
            nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
            nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
        } else {
            nd2nzPara.dValue = dValue; //nd矩阵的列数
            nd2nzPara.srcDValue = srcDValue; //同一nd矩阵相邻行起始地址间的偏移
        }
        nd2nzPara.dstNzC0Stride = dstNzC0Stride;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzPara.dstNzMatrixStride = dstNzMatrixStride;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    }
}

template <typename Q_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ>
class CopyQueryGmToL1 {
public:
    __aicore__ inline void operator()(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor,
                                      FaGmTensor<Q_T, GM_FORMAT> &srcTensor,
                                      GmCoord &gmCoord)
    {
        if constexpr ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD)) {
            ProcessS1G(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::BNGSD) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
            if( offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0 ) {
                ProcessGS1(dstTensor, srcTensor, gmCoord);
            } else {
                ProcessContinuous(dstTensor, srcTensor, gmCoord);
            }
        } else if constexpr (GM_FORMAT == GmFormat::NGTD) {
            ProcessGS1(dstTensor, srcTensor, gmCoord);
        }
    }

private:
    __aicore__ inline void ProcessS1G(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor,
                                      GmCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
        uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, gmCoord.dIdx);

        if (offsetCalculator.GetDimG() == 1) {
            CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[queryGmbaseOffset], s1IdxEnd - s1IdxStart, gmCoord.dDealSize,
                                    offsetCalculator.GetStrideS1(), dstTensor.rowCount);
            return;
        }

        // 处理第一个S
        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = offsetCalculator.GetDimG() - gIdxStart;
        }

        uint64_t offset = queryGmbaseOffset + gIdxStart * offsetCalculator.GetDimD();
        CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[offset], headSize, gmCoord.dDealSize,
                               offsetCalculator.GetStrideG(), dstTensor.rowCount);

        if (s1IdxEnd - s1IdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideS1();
            uint64_t l1Offset = headSize * 16U;
            if (s1IdxEnd - s1IdxStart > 1) {
                CopyMultiMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset],
                        s1IdxEnd - s1IdxStart - 1, offsetCalculator.GetStrideS1(), offsetCalculator.GetDimG() * 16U,
                        offsetCalculator.GetDimG(), gmCoord.dDealSize,
                        offsetCalculator.GetStrideG(), dstTensor.rowCount);
                gmOffset += (s1IdxEnd - s1IdxStart - 1) * offsetCalculator.GetStrideS1();
                l1Offset += (s1IdxEnd - s1IdxStart - 1) * offsetCalculator.GetDimG() * 16U;
            }

            // 处理尾块
            if (gIdxEnd > 0) {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], gIdxEnd,
                                       gmCoord.dDealSize, offsetCalculator.GetStrideG(), dstTensor.rowCount);
            }
        }
    }

    __aicore__ inline void ProcessContinuous(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor,
                                             FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        // B*N2*GS1*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t gIdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimS1();
        uint32_t s1IdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimS1();

        uint64_t offset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, s1IdxStart, gmCoord.dIdx);
        CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[offset], gmCoord.gS1DealSize, gmCoord.dDealSize,
                               offsetCalculator.GetDimD(), dstTensor.rowCount);
    }

    __aicore__ inline void ProcessGS1(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor,
                                      GmCoord &gmCoord)
    {
        // N2*G*T(BS1)*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t s1Size = 0;
        if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        } else {
            if( offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0 ) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            } else {
                s1Size = offsetCalculator.GetDimS1();
            }
        }

        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, gmCoord.dIdx);

        // 处理第一个S
        uint32_t headSize = 0;
        if (gIdxStart == gIdxEnd) {
            headSize = s1IdxEnd - s1IdxStart;
        } else {
            headSize = s1Size - s1IdxStart;
        }

        uint64_t offset = queryGmbaseOffset + s1IdxStart * offsetCalculator.GetDimD();
        CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[offset], headSize, gmCoord.dDealSize,
                               offsetCalculator.GetStrideS1(), dstTensor.rowCount);

        if (gIdxEnd - gIdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideG();
            uint64_t l1Offset = headSize * 16U;

            if (gIdxEnd - gIdxStart > 1) {
                CopyMultiMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset],
                        gIdxEnd - gIdxStart - 1, offsetCalculator.GetStrideG(), s1Size * 16U,
                        s1Size, gmCoord.dDealSize, offsetCalculator.GetStrideS1(), dstTensor.rowCount);
                gmOffset += (gIdxEnd - gIdxStart - 1) * offsetCalculator.GetStrideG();
                l1Offset += (gIdxEnd - gIdxStart - 1) * s1Size * 16U;
            }

            // 处理尾块
            if (s1IdxEnd > 0) {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], s1IdxEnd,
                                       gmCoord.dDealSize, offsetCalculator.GetStrideS1(), dstTensor.rowCount);
            }
        }
    }
};

// ----------------------------------------------CopyAttenOutUbToGm--------------------------------
enum class UbFormat
{
    GS1 = 0,
    S1G = 1
};

template <typename OUT_T>
struct FaUbTensor {
    LocalTensor<OUT_T> tensor;
    uint32_t rowCount;
    uint32_t colCount;
};

template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm
{
public:
    __aicore__ inline void SafeStrideCopy(GlobalTensor<OUT_T> gmTensor, const LocalTensor<OUT_T> ubTensor,
                                            uint32_t blockCount, uint32_t blockLen, uint32_t srcStride, uint64_t dstStride)
    {
        DataCopyExtParams dataCopyParams;
        // B*S过大时，跳写参数dataCopyParams.dstStride(uint32_t)计算结果将溢出，使用for循环拷贝代替
        if (dstStride > UINT32_MAX) {
            uint64_t gmSingleStride = (dstStride + blockLen) / sizeof(OUT_T);
            uint64_t ubSingleStride = (srcStride * fa_base_vector::BYTE_BLOCK + blockLen) / sizeof(OUT_T);
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = blockLen;
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0; // 单位为Byte
            for (uint32_t i = 0; i < blockCount; i++) {
                DataCopyPad(gmTensor[i * gmSingleStride], ubTensor[i * ubSingleStride], dataCopyParams);
            }
        } else {
            // dataCopyParams.dstStride(uint32_t)没有溢出时，进行跳写
            dataCopyParams.blockCount = blockCount;
            dataCopyParams.blockLen = blockLen;
            dataCopyParams.srcStride = srcStride;
            dataCopyParams.dstStride = dstStride; // 单位为Byte
            DataCopyPad(gmTensor, ubTensor, dataCopyParams);
        }
    }
    __aicore__ inline void operator()(FaGmTensor<OUT_T, GM_FORMAT> &dstTensor,
                                      FaUbTensor<OUT_T> &srcTensor,
                                      GmCoord &gmCoord)
    {
        if constexpr (UB_FORMAT == UbFormat::GS1) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = dstTensor.offsetCalculator;
            uint32_t s1Size = 0;
            if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            } else {
                if( offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0 ) {
                    s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
                } else {
                    s1Size = offsetCalculator.GetDimS1();
                }
            }
            uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
            uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
            uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
            uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

            uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, 0);

            // 处理第一个S
            uint32_t headS1 = 0;
            if (gIdxStart == gIdxEnd) {
                headS1 = s1IdxEnd - s1IdxStart;
            } else {
                headS1 = s1Size - s1IdxStart;
            }
            uint64_t gmOffset = attenOutGmbaseOffset + s1IdxStart * offsetCalculator.GetStrideS1();
            uint64_t ubOffset = 0;
            uint32_t blockCount = headS1;
            uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
            uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (fa_base_vector::BYTE_BLOCK / sizeof(OUT_T));
            uint64_t dstStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
            SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                            dstStride);

            if (gIdxEnd - gIdxStart >= 1) {
                // 处理中间块
                gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideG();
                ubOffset = headS1 * srcTensor.colCount;
                for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                    blockCount = s1Size;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                    srcStride, dstStride);
                    gmOffset += offsetCalculator.GetStrideG();
                    ubOffset += s1Size * srcTensor.colCount;
                }

                // 处理尾块
                if (s1IdxEnd > 0) {
                    blockCount = s1IdxEnd;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                    srcStride, dstStride);
                }
            }
        } else if constexpr (UB_FORMAT == UbFormat::S1G) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = dstTensor.offsetCalculator;
            uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
            uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
            uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
            uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

            uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, 0);

            // 处理第一个S
            uint32_t headSize = 0;
            if (s1IdxStart == s1IdxEnd) {
                headSize = gIdxEnd - gIdxStart;
            } else {
                headSize = offsetCalculator.GetDimG() - gIdxStart;
            }
            uint64_t gmOffset = attenOutGmbaseOffset + gIdxStart * offsetCalculator.GetStrideG();
            uint64_t ubOffset = 0;
            uint32_t blockCount = headSize;
            uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
            uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (fa_base_vector::BYTE_BLOCK / sizeof(OUT_T));
            uint64_t dstStride = (offsetCalculator.GetStrideG() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
            SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                            dstStride);

            if (s1IdxEnd - s1IdxStart >= 1) {
                // 处理中间块
                gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideS1();
                ubOffset = ((uint64_t)headSize) * ((uint64_t)srcTensor.colCount);
                for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                    blockCount = offsetCalculator.GetDimG();
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                    srcStride, dstStride);
                    gmOffset += offsetCalculator.GetStrideS1();
                    ubOffset += offsetCalculator.GetDimG() * srcTensor.colCount;
                }

                // 处理尾块
                if (gIdxEnd > 0) {
                    blockCount = gIdxEnd;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                    srcStride, dstStride);
                }
            }
        }
    }
};

// ----------------------------------------------CopyKvGmToL1--------------------------------
struct GmKvCoord {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
    uint32_t s2DealSize;
    uint32_t dDealSize;
};

template <typename KV_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ>
class CopyKvGmToL1
{
public:
    __aicore__ inline void operator()(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor,
                                      FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                      GmKvCoord &gmCoord)
    {
        if constexpr (GM_FORMAT == GmFormat::BNSD || GM_FORMAT == GmFormat::BSND ||
                      GM_FORMAT == GmFormat::NTD || GM_FORMAT == GmFormat::TND) {
            ProcessContinuousOrTensorlist(dstTensor, srcTensor, gmCoord);
        }
    }

private:
    __aicore__ inline void ProcessContinuousOrTensorlist(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor,
                                                         FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                                         GmKvCoord &gmCoord)
    {
        // B*N2*GS1*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gmCoord.s2Idx, gmCoord.dIdx);
        CopySingleMatrixNDToNZ(dstTensor.tensor, srcTensor.gmTensor[offset], gmCoord.s2DealSize, gmCoord.dDealSize,
                               offsetCalculator.GetStrideS2(), dstTensor.rowCount);
    }
};

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr GmFormat GetQueryGmFormat() {
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) ||
                  (LAYOUT_T == FIA_LAYOUT::BNSD) ||
                  (LAYOUT_T == FIA_LAYOUT::TND) ||
                  (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    }
}

template <FIA_LAYOUT KV_LAYOUT_T, const bool PAGE_ATTENTION>
__aicore__ inline constexpr GmFormat GetKVFormat() {
    static_assert((KV_LAYOUT_T == FIA_LAYOUT::BSH) ||
                    (KV_LAYOUT_T == FIA_LAYOUT::BNSD) ||
                    (KV_LAYOUT_T == FIA_LAYOUT::TND) ||
                    (KV_LAYOUT_T == FIA_LAYOUT::NTD),
                    "Get Key or Value GmFormat fail, KV_LAYOUT_T is incorrect when KV Continuous or TensorList");
    if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BSH) {
        return GmFormat::BSND;
    } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BNSD) {
        return GmFormat::BNSD;
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr UbFormat GetOutUbFormat() {
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) ||
                  (LAYOUT_T == FIA_LAYOUT::BNSD) ||
                  (LAYOUT_T == FIA_LAYOUT::TND) ||
                  (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH || LAYOUT_T == FIA_LAYOUT::TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        return UbFormat::GS1;
    }
}

// ----------------------------------------------Copy LSE UB To Gm--------------------------------
template <typename T, ActualSeqLensMode Q_MODE>
__aicore__ inline void DataCopySoftmaxLseBSND(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                 uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount, 
                                                 const ConstInfo &constInfo,
                                                 ActualSeqLensParser<Q_MODE> qActSeqLensParser, uint64_t bIdx)
{
    uint32_t startS1Idx = mOffset / constInfo.gSize;
    uint32_t startGIdx = mOffset % constInfo.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;
    uint64_t s1LeftPaddingSize = 0;
    if (constInfo.isQHasLeftPadding) {
        s1LeftPaddingSize = constInfo.qSeqSize - constInfo.qLeftPaddingSize - qActSeqLensParser.GetActualSeqLength(bIdx);
    }

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + startGIdx * constInfo.qSeqSize + s1Idx + s1LeftPaddingSize;
        if (s1Idx != endS1Idx) {
            curDealRowCount =  constInfo.gSize - startGIdx;
        }
        else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.qSeqSize - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, ActualSeqLensMode Q_MODE>
__aicore__ inline void DataCopySoftmaxLseBNSD(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                            uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                            const ConstInfo &constInfo,
                                            ActualSeqLensParser<Q_MODE> qActSeqLensParser, uint64_t bIdx)
{
    uint64_t gOffset = mOffset / qActSeqLensParser.GetActualSeqLength(bIdx) * constInfo.qSeqSize;
    uint64_t seqOffset = mOffset % qActSeqLensParser.GetActualSeqLength(bIdx);
    uint64_t s1LeftPaddingSize = 0;
    if (constInfo.isQHasLeftPadding) {
        s1LeftPaddingSize = constInfo.qSeqSize - constInfo.qLeftPaddingSize - qActSeqLensParser.GetActualSeqLength(bIdx);
    }
    uint64_t outOffset = bN2Offset + gOffset + seqOffset + s1LeftPaddingSize;
    uint64_t ubOffset = 0;
    // dealCount ≤ 当前actQs剩余部分，则直接搬运全部dealCount
    if ((qActSeqLensParser.GetActualSeqLength(bIdx) - seqOffset) >= dealCount) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        return;
    }
    // dealCount > 当前actQs剩余部分，分块搬运dealCount
    // dealCount首块
    uint64_t headActSeq = qActSeqLensParser.GetActualSeqLength(bIdx) - seqOffset;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = headActSeq;
    dataCopyParams.blockLen = sizeof(float);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    outOffset += constInfo.qSeqSize - qActSeqLensParser.GetActualSeqLength(bIdx) + headActSeq;
    ubOffset += headActSeq * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
    // dealCount中间块
    uint64_t pendingCount = dealCount - headActSeq;
    while (pendingCount > qActSeqLensParser.GetActualSeqLength(bIdx)) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = qActSeqLensParser.GetActualSeqLength(bIdx);
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        outOffset += constInfo.qSeqSize;
        ubOffset += qActSeqLensParser.GetActualSeqLength(bIdx) * fa_base_vector::FP32_BLOCK_ELEMENT_NUM;
        pendingCount -= qActSeqLensParser.GetActualSeqLength(bIdx);
    }
    // dealCount尾块
    if (pendingCount > 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = pendingCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    }
}

// ---------------------------------------Set attention Gm To Zero--------------------------------
template <GmFormat FORMAT, typename OUT_T>
__aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, OffsetCalculator<FORMAT> &offsetCalculator,
                                           GlobalTensor<OUT_T>& attentionOutGm)
{  
    if constexpr (FORMAT == GmFormat::BNGSD) {
        uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, 0, 0); 
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
    }  else if constexpr (FORMAT == GmFormat::BSNGD) {
        uint32_t s1Size = offsetCalculator.GetDimS1();
        for (int s1Idx = 0; s1Idx < s1Size; s1Idx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, s1Idx, 0);  
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
        }
    } 
}
#endif
