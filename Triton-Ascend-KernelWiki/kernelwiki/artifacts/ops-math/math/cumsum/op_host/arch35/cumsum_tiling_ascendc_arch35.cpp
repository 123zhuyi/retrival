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
 * \file cumsum_tiling_ascendc.cc
 * \brief cumsum tiling for ascendc float
 */

#include "cumsum_tiling_ascendc_arch35.h"
#include "cumsum_tiling.h"
#include "util/const_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "util/math_util.h"

namespace optiling {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_AXIS_INDEX = 1;
constexpr size_t ATTR_EXCLUSIVE_INDEX = 0;
constexpr size_t ATTR_REVERSE_INDEX = 1;
constexpr size_t WORKSPACE_SIZE = 16777216; // 16 * 1024 * 1024;
constexpr int32_t CAST_MULT = 3;
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int32_t CONST_4 = 4;
constexpr int32_t CONST_8 = 8;
constexpr int32_t CONST_16 = 16;
constexpr int32_t CONST_32 = 32;
constexpr int32_t FP16_FOLD_MAX = 4;
constexpr int32_t SS_ITER_0 = 0;
constexpr int32_t SS_ITER_1 = 1;
constexpr int32_t SS_ITER_2 = 2;
constexpr int32_t SS_ITER_3 = 3;
constexpr int32_t SS_ITER_4 = 4;
constexpr int32_t SS_ITER_5 = 5;
constexpr int32_t SS_ITER_MAX = 6;
constexpr int32_t FOLD_LEN_MIN = 256;
constexpr int32_t UB_SS_BUFFER_SIZE = 8 * 1024;
constexpr int32_t UB_SS_BR_BUFF_SIZE = 512;
constexpr int32_t UB_SS_BR_IDX_BUFF_SIZE = 8 * 1024;
constexpr int32_t TWOWAY_BUFF_SIZE = 2 * 1024;
constexpr int32_t TWOWAY_IDX_BUFF_SIZE = 8 * 1024;
constexpr int32_t ADD_FACTOR_OFFSET_MAX = 64 * 256;
constexpr int32_t CORE_NUM_MAX = 64;

constexpr uint64_t TILING_KEY_ONEWAY = 1001;
constexpr uint64_t TILING_KEY_TWOWAY = 1002;
constexpr uint64_t TILING_KEY_UB_SS_ONEWAY = 1011;
constexpr uint64_t TILING_KEY_UB_SS_TWOWAY = 1012;
constexpr uint64_t TILING_KEY_CORE_SS_ONEWAY = 1101;
constexpr uint64_t TILING_KEY_CORE_SS_TWOWAY = 1102;
constexpr uint64_t TILING_KEY_CORE_SS_UB_SS_ONEWAY = 1111;
constexpr uint64_t TILING_KEY_CORE_SS_UB_SS_TWOWAY = 1112;

struct BlockPara {
    int32_t blockCount = 0;
    int64_t blockFactor = 0;
    int64_t blockTailFactor = 0;
};

struct __UbPara {
    int32_t ubCount = 0;
    int32_t ubFactor = 0;
    int32_t ubTailFactor = 0;
};

struct UbPara {
    __UbPara mainCoreUbPara;
    __UbPara tailCoreUbPara;
};

struct FoldPara {
    int32_t foldCount = 0;
    int64_t foldLen = 0;
    int32_t sklanskyIter = 0;
};

struct CumsumCoreInnerTilingData {
    bool exclusive = false;
    bool reverse = false;
    int32_t realCoreNum = 0;
    int64_t lenM = 0;
    int64_t lenR = 0;
    int64_t lenN = 0;
    int32_t xBufSize = 0;
    int32_t xUnfoldBufSize = 0;
    int32_t ubSklanskyBufSize = 0;
    FoldPara mainCoreMainUbFoldPara;
    FoldPara mainCoreTailUbFoldPara;
    FoldPara tailCoreMainUbFoldPara;
    FoldPara tailCoreTailUbFoldPara;
    BlockPara mBlockPara;
    BlockPara rBlockPara;
    BlockPara nBlockPara;
    UbPara mUbPara;
    UbPara rUbPara;
    UbPara nUbPara;
};

struct IterGroupTiling {
    int64_t iterGroupBlockFactor = 0;           // 每一组的R的核切分factor
    int32_t iterGroupMainBlockCount = 0;        // 每一组的R的核切分主核个数
    int32_t iterGroupTailBlockCount = 0;        // 每一组的R的核切分尾核个数
    int32_t iterGroupMainCoreUbFactorCount = 0; // 主核的UB切分factor个数
    int32_t iterGroupMainCoreUbFactor = 0;      // 主核的UB切分factor
    int32_t iterGroupMainCoreUbTailFactor = 0;  // 主核的UB切分factor尾块
    int32_t iterGroupTailCoreUbFactorCount = 0; // 尾核的UB切分factor个数
    int32_t iterGroupTailCoreUbFactor = 0;      // 尾核的UB切分factor
    int32_t iterGroupTailCoreUbTailFactor = 0;  // 尾核的UB切分factor尾块
};

struct SklanskyItersTiling {
    int32_t iterGroupCount = 0;       // 每一行迭代中，分为几组
    int32_t iterGroupAddCount = 0;    // 每一组需要执行的add次数
    int32_t iterGroupAddOffset = 0;   // 每一组add之间的偏移量
    int32_t iterGroupStartOffset = 0; // 每一行迭代起始偏移量
    int32_t iterGroupCoreNum = 0;     // 每一组使用的核数量

    IterGroupTiling mainIterGroupMainNTiling; // group主块，N主块的tiling参数
    IterGroupTiling tailIterGroupMainNTiling; // group尾块，N主块的tiling参数
    IterGroupTiling mainIterGroupTailNTiling; // group主块，N尾块的tiling参数
    IterGroupTiling tailIterGroupTailNTiling; // group尾块，N尾块的tiling参数
};

struct CumsumCoreOuterTilingData {
    int32_t addFactorRepeatTimesMain = 0;  // 主块N，addFactor重复几次
    int32_t addFactorRepeatTimesTail = 0;  // 尾块N，addFactor重复几次
    int32_t coreGroupCount = 0;            // 按M*nBorrowLen，分组后的数量
    int32_t coreGroupCoreNum = 0;          // 按M*nBorrowLen，分组后每一组使用的核数
    int32_t sklanskyItersCount = 0;        // sklansky迭代行数
    int32_t addFactorBufferSize = 0;       // addFactor，UB缓存的大小
    int32_t addDataBufferSize = 0;         // addData，UB缓存的大小
    int32_t addFactorOffsetBufferSize = 0; // addFactor偏移量，UB缓存的大小

    SklanskyItersTiling sklanskyItersTiling[SS_ITER_MAX]; // sklansky迭代每一行的tiling
};

struct CumsumTilingData {
    CumsumCoreInnerTilingData innerTd;
    CumsumCoreOuterTilingData outerTd;
};

enum class SklanskyPattern : uint32_t
{
    SS_ONEWAY = 0,
    SS_TWOWAY = 1,
};

class CumsumAscendcTilingImpl {
public:
    explicit CumsumAscendcTilingImpl(gert::TilingContext* context) : context_(context)
    {}

    ge::graphStatus Init(const CumsumCompileInfo* compileInfo);

    ge::graphStatus DoTiling();

private:
    void TilingStrategy();

    void NGreaterCl();

    void NLesserCl();

    void NGreaterClRFullLoad();

    void NGreaterClRNotFullLoad();

    void NGreaterClRNotFullLoadBorrowR();

    void RNGreaterCl();

    void RNLesserCl();

    int64_t CalcBorrowM();

    void RNGreaterClBorrowM(int64_t mAfterBorrow);

    void RNGreaterClRFullLoad();

    void RNGreaterClRNotFullLoad();

    void RNGreaterClRNotFullLoadNotBorrowR();

    void RNGreaterClRNotFullLoadBorrowR();

    void RNGreaterClRNotFullLoadBorrowRTwoway();

    void MRNGreaterCl();

    void MRNLesserCl();

    void CalcBufferSize();

    int32_t CalcXUnfoldBufSize();

    int32_t CalcXBufSize(int32_t xBufferMax);

    int64_t CalcUnfoldR(CumsumCoreInnerTilingData& innerTd);

    int32_t CalcUnfoldN(UbPara& ubPara);

    void TilingStrategyOuterTd();

    void TilingStrategyOuterTdSklanskyItersTiling();

    void InitIterGroupTiling(int64_t rLen, int32_t coreNum, IterGroupTiling& mainNTiling, IterGroupTiling& tailNTiling);

    void InitIterGroupUbTiling(int32_t nFactor, IterGroupTiling& iterGroupTiling);

    void DoBlockSplitBalance(
        int64_t splitLen, int32_t coreNum, int32_t& mainBlockNum, int32_t& tailBlockNum, int64_t& blockFactor);

    void JudgeSklanskyPatten(int64_t rLen, int64_t alignN);

    void DoFold(int64_t rLen, int64_t alignN, int32_t& foldCount, int64_t& foldLen);

    void CoreFullLoad(int64_t len, BlockPara& blockPara, __UbPara& ubPara);

    void UbFullLoad(int64_t len, __UbPara& ubPara);

    int32_t MaxForFullUb(int32_t bufferMax, int32_t alignLen);

    void DoBlockSplit(int64_t splitLen, int32_t coreNum, BlockPara& blockPara);

    void DoUbSplit(int64_t splitLen, int64_t splitFactor, __UbPara& ubPara);

    int64_t LastPow2(int64_t n);

    void FillTilingData();

    void FillFoldPara(CumsumFoldPara& dstFoldPara, FoldPara& srcFoldPara);

    void FillBlockPara(CumsumBlockPara& dstBlockPara, BlockPara& srcBlockPara);

    void FillUbPara(CumsumUbPara& dstUbPara, UbPara& srcUbPara);

    void FillSklanskyItersTiling(
        CumsumSklanskyItersTiling& cumsumSklanskyItersTiling, SklanskyItersTiling& sklanskyItersTiling);

    void FillIterGroupTiling(CumsumIterGroupTiling& dstIterGroupTiling, IterGroupTiling& srcIterGroupTiling);

    void PrintTilingData();

    void PrintFoldPara();

    void PrintBlockPara();

    void PrintUbPara();

    void PrintSklanskyItersTiling();

    void PrintIterGroupTiling(IterGroupTiling& iterGroupTiling);

private:
    int32_t clSize_ = 0;
    int32_t blockSize_ = 0;
    int32_t vRegSize_ = 0;
    int32_t ubSize_ = 0;
    int32_t coreNum_ = 0;

    SklanskyPattern ssPatten_ = SklanskyPattern::SS_ONEWAY;
    bool dtCast_ = false;
    int32_t dtSize_ = 0;
    int32_t dtSizeCast_ = 0;
    int64_t alignN_ = 0;
    int32_t foldCount_ = 0;
    int64_t foldLen_ = 0;
    int32_t borrowRCount_ = 1;
    int32_t borrowNCount_ = 1;
    int32_t borrowMCount_ = 1;
    uint64_t tilingKey_ = 0;

    CumsumTilingData td_;
    gert::TilingContext* context_;
}; // class CumsumAscendcTilingImpl

ge::graphStatus CumsumAscendcTilingImpl::Init(const CumsumCompileInfo* compileInfo)
{
    clSize_ = static_cast<int32_t>(compileInfo->clSize);
    blockSize_ = static_cast<int32_t>(compileInfo->blockSize);
    vRegSize_ = static_cast<int32_t>(compileInfo->vRegSize);
    ubSize_ = static_cast<int32_t>(compileInfo->ub_size);
    coreNum_ = static_cast<int32_t>(compileInfo->core_num);
    OP_CHECK_IF(
        coreNum_ > CORE_NUM_MAX,
        OP_LOGW(
            context_->GetNodeName(), "core num greater than 64, not support, set core num = 64. coreNum:%d", coreNum_),
        coreNum_ = CORE_NUM_MAX);

    auto inputXDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    ge::DataType dType = inputXDesc->GetDataType();
    dtSize_ = GetSizeByDataType(dType);
    dtSizeCast_ = GetSizeByDataType(ge::DataType::DT_FLOAT);
    dtCast_ = (dType == ge::DataType::DT_FLOAT16 || dType == ge::DataType::DT_BF16) ? true : false;

    int64_t axis = 0;
    OP_CHECK_IF(
        !Ops::Base::GetConstInt(context_, INPUT_AXIS_INDEX, axis),
        OP_LOGE(context_->GetNodeName(), "axis Get const int fail."), return ge::GRAPH_FAILED);
    auto xStorage = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    const gert::Shape& xShape = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());
    size_t xDimNum = xShape.GetDimNum();
    if (axis < 0) {
        axis += static_cast<int64_t>(xDimNum);
    }
    OP_CHECK_IF(
        axis >= static_cast<int64_t>(xDimNum) || axis < static_cast<int64_t>(-xDimNum),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "axis",
            std::to_string(axis).c_str(),
            ("The value of axis must be within the range [-" + std::to_string(xDimNum) + 
             ", " + std::to_string(xDimNum - 1) + "].").c_str()),
        return ge::GRAPH_FAILED);

    // 合轴
    td_.innerTd.lenR = xShape.GetDim(axis);
    td_.innerTd.lenM = 1;
    td_.innerTd.lenN = 1;
    for (int64_t i = 0; i < axis; i++) {
        td_.innerTd.lenM = td_.innerTd.lenM * xShape[i];
    }
    for (int64_t i = axis + 1; i < static_cast<int64_t>(xDimNum); i++) {
        td_.innerTd.lenN = td_.innerTd.lenN * xShape[i];
    }
    alignN_ = Ops::Base::CeilAlign(td_.innerTd.lenN * dtSize_, static_cast<int64_t>(blockSize_));

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const bool* exclusive = attrs->GetAttrPointer<bool>(ATTR_EXCLUSIVE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, exclusive);
    const bool* reverse = attrs->GetAttrPointer<bool>(ATTR_REVERSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, reverse);
    td_.innerTd.exclusive = *exclusive;
    td_.innerTd.reverse = *reverse;

    td_.innerTd.ubSklanskyBufSize = UB_SS_BUFFER_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CumsumAscendcTilingImpl::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter CumsumAscendcTilingImpl DoTiling");

    TilingStrategy();

    FillTilingData();

    PrintTilingData();

    context_->SetBlockDim(
        td_.innerTd.realCoreNum == 0 ?
            1 :
            std::max(td_.innerTd.realCoreNum, td_.outerTd.coreGroupCount * td_.outerTd.coreGroupCoreNum));
    context_->SetTilingKey(tilingKey_);
    if (tilingKey_ == TILING_KEY_CORE_SS_ONEWAY ||
        tilingKey_ == TILING_KEY_CORE_SS_TWOWAY ||
        tilingKey_ == TILING_KEY_CORE_SS_UB_SS_ONEWAY ||
        tilingKey_ == TILING_KEY_CORE_SS_UB_SS_TWOWAY) {
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Failed to set ScheduleMode!"),
                    return ge::GRAPH_FAILED);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;

    OP_LOGD(context_->GetNodeName(), "Exit CumsumAscendcTilingImpl DoTiling");
    return ge::GRAPH_SUCCESS;
}

void CumsumAscendcTilingImpl::TilingStrategy()
{
    // 空tensor
    if (td_.innerTd.lenM == 0 || td_.innerTd.lenR == 0 || td_.innerTd.lenN == 0) {
        td_.innerTd.realCoreNum = 0;
        return;
    }

    if (td_.innerTd.lenN * dtSizeCast_ >= clSize_) { // N大于cacheLine
        NGreaterCl();
    } else { // N小于cacheLine
        // N切分策略都一样，不切核，不切UB
        CoreFullLoad(td_.innerTd.lenN, td_.innerTd.nBlockPara, td_.innerTd.nUbPara.tailCoreUbPara);
        NLesserCl();
    }

    CalcBufferSize();

    TilingStrategyOuterTd();
}

void CumsumAscendcTilingImpl::NGreaterCl()
{
    int32_t clNSize = dtCast_ ? clSize_ * CAST_MULT : clSize_;
    if (td_.innerTd.lenR * clNSize <= ubSize_) { // R可以全载
        NGreaterClRFullLoad();
    } else { // R不能全载
        NGreaterClRNotFullLoad();
    }
}

void CumsumAscendcTilingImpl::NLesserCl()
{
    if (td_.innerTd.lenR * td_.innerTd.lenN * dtSizeCast_ >= clSize_) { // R*N大于cacheline
        RNGreaterCl();
    } else { // R*N小于cacheline
        // R切分策略都一样，不切核，不切UB
        CoreFullLoad(td_.innerTd.lenR, td_.innerTd.rBlockPara, td_.innerTd.rUbPara.tailCoreUbPara);
        RNLesserCl();
    }
}

void CumsumAscendcTilingImpl::NGreaterClRFullLoad()
{
    // R不分核，不分UB
    CoreFullLoad(td_.innerTd.lenR, td_.innerTd.rBlockPara, td_.innerTd.rUbPara.tailCoreUbPara);

    // M切核
    DoBlockSplit(td_.innerTd.lenM, coreNum_, td_.innerTd.mBlockPara);

    if (td_.innerTd.lenM >= coreNum_) { // M够切核
        // N不切核
        DoBlockSplit(td_.innerTd.lenN, 1, td_.innerTd.nBlockPara);

        int32_t alignN = dtCast_ ? alignN_ * CAST_MULT : alignN_;
        int64_t alignRN = td_.innerTd.lenR * alignN;
        if (alignRN <= ubSize_) { // N可以全载
            UbFullLoad(td_.innerTd.nBlockPara.blockTailFactor, td_.innerTd.nUbPara.tailCoreUbPara);

            int64_t mMaxForFullUb = Ops::Base::FloorDiv(static_cast<int64_t>(ubSize_), alignRN);
            DoUbSplit(td_.innerTd.mBlockPara.blockFactor, mMaxForFullUb, td_.innerTd.mUbPara.mainCoreUbPara);
            DoUbSplit(td_.innerTd.mBlockPara.blockTailFactor, mMaxForFullUb, td_.innerTd.mUbPara.tailCoreUbPara);
        } else { // N不能全载
            // M每次UB载入1
            DoUbSplit(td_.innerTd.mBlockPara.blockFactor, 1, td_.innerTd.mUbPara.mainCoreUbPara);
            DoUbSplit(td_.innerTd.mBlockPara.blockTailFactor, 1, td_.innerTd.mUbPara.tailCoreUbPara);

            int32_t nMaxForFullUb = MaxForFullUb(ubSize_, td_.innerTd.lenR);
            nMaxForFullUb = (nMaxForFullUb % blockSize_ == 0) ?
                                nMaxForFullUb :
                                Ops::Base::CeilAlign(nMaxForFullUb, blockSize_) - blockSize_;
            DoUbSplit(
                td_.innerTd.nBlockPara.blockTailFactor, static_cast<int64_t>(nMaxForFullUb / dtSize_),
                td_.innerTd.nUbPara.tailCoreUbPara);
        }

        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount;
        tilingKey_ = TILING_KEY_ONEWAY;
    } else { // M不够切核
        // 借轴N切核
        borrowNCount_ = Ops::Base::FloorDiv(static_cast<int64_t>(coreNum_), td_.innerTd.lenM);
        int32_t borrowNMax = Ops::Base::CeilDiv(td_.innerTd.lenN * dtSize_, static_cast<int64_t>(clSize_));
        borrowNCount_ = std::min(borrowNCount_, borrowNMax);
        DoBlockSplit(td_.innerTd.lenN, borrowNCount_, td_.innerTd.nBlockPara);
        borrowNCount_ = td_.innerTd.nBlockPara.blockCount;

        UbFullLoad(td_.innerTd.mBlockPara.blockFactor, td_.innerTd.mUbPara.mainCoreUbPara);
        UbFullLoad(td_.innerTd.mBlockPara.blockTailFactor, td_.innerTd.mUbPara.tailCoreUbPara);

        int32_t nMaxForFullUb = MaxForFullUb(ubSize_, td_.innerTd.lenR);
        nMaxForFullUb = (nMaxForFullUb % blockSize_ == 0) ?
                            nMaxForFullUb :
                            Ops::Base::CeilAlign(nMaxForFullUb, blockSize_) - blockSize_;
        DoUbSplit(
            td_.innerTd.nBlockPara.blockFactor, static_cast<int64_t>(nMaxForFullUb / dtSize_),
            td_.innerTd.nUbPara.mainCoreUbPara);
        DoUbSplit(
            td_.innerTd.nBlockPara.blockTailFactor, static_cast<int64_t>(nMaxForFullUb / dtSize_),
            td_.innerTd.nUbPara.tailCoreUbPara);

        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount * borrowNCount_;
        tilingKey_ = TILING_KEY_ONEWAY;
    }
}

void CumsumAscendcTilingImpl::NGreaterClRNotFullLoad()
{
    if (td_.innerTd.lenM > static_cast<int64_t>(coreNum_) / CONST_2) { // M够切核
        DoBlockSplit(td_.innerTd.lenM, coreNum_, td_.innerTd.mBlockPara);
        DoUbSplit(td_.innerTd.mBlockPara.blockFactor, 1, td_.innerTd.mUbPara.mainCoreUbPara);
        DoUbSplit(td_.innerTd.mBlockPara.blockTailFactor, 1, td_.innerTd.mUbPara.tailCoreUbPara);

        // R,N不切核
        DoBlockSplit(td_.innerTd.lenR, 1, td_.innerTd.rBlockPara);
        DoBlockSplit(td_.innerTd.lenN, 1, td_.innerTd.nBlockPara);

        int32_t rMaxForFullUb =
            LastPow2(MaxForFullUb(ubSize_ - UB_SS_BUFFER_SIZE - UB_SS_BR_BUFF_SIZE - UB_SS_BR_IDX_BUFF_SIZE, clSize_));
        DoUbSplit(td_.innerTd.rBlockPara.blockTailFactor, rMaxForFullUb, td_.innerTd.rUbPara.tailCoreUbPara);
        DoUbSplit(
            td_.innerTd.nBlockPara.blockTailFactor, static_cast<int64_t>(clSize_ / dtSizeCast_),
            td_.innerTd.nUbPara.tailCoreUbPara);

        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount;
        tilingKey_ = TILING_KEY_UB_SS_ONEWAY;
    } else { // M不够切核
        DoBlockSplit(td_.innerTd.lenM, coreNum_, td_.innerTd.mBlockPara);
        UbFullLoad(td_.innerTd.mBlockPara.blockFactor, td_.innerTd.mUbPara.mainCoreUbPara);
        UbFullLoad(td_.innerTd.mBlockPara.blockTailFactor, td_.innerTd.mUbPara.tailCoreUbPara);

        int32_t borrowNMax = Ops::Base::CeilDiv(td_.innerTd.lenN * dtSizeCast_, static_cast<int64_t>(clSize_));
        if (td_.innerTd.lenM * borrowNMax > static_cast<int64_t>(coreNum_) / CONST_2) { // 借轴N后够分核
            borrowNCount_ = Ops::Base::FloorDiv(static_cast<int64_t>(coreNum_), td_.innerTd.lenM);
            borrowNCount_ = std::min(borrowNCount_, borrowNMax);
            DoBlockSplit(td_.innerTd.lenN, borrowNCount_, td_.innerTd.nBlockPara);
            borrowNCount_ = td_.innerTd.nBlockPara.blockCount;
            DoUbSplit(
                td_.innerTd.nBlockPara.blockFactor, static_cast<int64_t>(clSize_ / dtSizeCast_),
                td_.innerTd.nUbPara.mainCoreUbPara);
            DoUbSplit(
                td_.innerTd.nBlockPara.blockTailFactor, static_cast<int64_t>(clSize_ / dtSizeCast_),
                td_.innerTd.nUbPara.tailCoreUbPara);

            DoBlockSplit(td_.innerTd.lenR, 1, td_.innerTd.rBlockPara);
            int32_t rMaxForFullUb = LastPow2(
                MaxForFullUb(ubSize_ - UB_SS_BUFFER_SIZE - UB_SS_BR_IDX_BUFF_SIZE - UB_SS_BR_BUFF_SIZE, clSize_));
            DoUbSplit(td_.innerTd.rBlockPara.blockTailFactor, rMaxForFullUb, td_.innerTd.rUbPara.tailCoreUbPara);

            td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount * borrowNCount_;
            tilingKey_ = TILING_KEY_UB_SS_ONEWAY;
        } else { // 借轴N后不够分核
            // 借轴R分核
            borrowNCount_ = borrowNMax;
            borrowRCount_ = Ops::Base::FloorDiv(static_cast<int64_t>(coreNum_), td_.innerTd.lenM * borrowNMax);
            NGreaterClRNotFullLoadBorrowR();
        }
    }
}

void CumsumAscendcTilingImpl::NGreaterClRNotFullLoadBorrowR()
{
    td_.innerTd.nBlockPara.blockFactor =
        td_.innerTd.lenN < static_cast<int64_t>(clSize_ / dtSizeCast_) ? td_.innerTd.lenN : clSize_ / dtSizeCast_;
    td_.innerTd.nBlockPara.blockCount =
        Ops::Base::CeilDiv(td_.innerTd.lenN, static_cast<int64_t>(td_.innerTd.nBlockPara.blockFactor));
    td_.innerTd.nBlockPara.blockTailFactor = (td_.innerTd.lenN % td_.innerTd.nBlockPara.blockFactor == 0) ?
                                                 td_.innerTd.nBlockPara.blockFactor :
                                                 td_.innerTd.lenN % td_.innerTd.nBlockPara.blockFactor;
    td_.innerTd.nBlockPara.blockFactor =
        (td_.innerTd.nBlockPara.blockCount == 1) ? 0 : td_.innerTd.nBlockPara.blockFactor;
    DoUbSplit(
        td_.innerTd.nBlockPara.blockFactor, static_cast<int64_t>(clSize_ / dtSizeCast_),
        td_.innerTd.nUbPara.mainCoreUbPara);
    DoUbSplit(
        td_.innerTd.nBlockPara.blockTailFactor, static_cast<int64_t>(clSize_ / dtSizeCast_),
        td_.innerTd.nUbPara.tailCoreUbPara);

    DoBlockSplit(td_.innerTd.lenR, borrowRCount_, td_.innerTd.rBlockPara);
    borrowRCount_ = td_.innerTd.rBlockPara.blockCount;
    // 使用主核来判定R能否全载
    int32_t clNSize = dtCast_ ? clSize_ * CAST_MULT : clSize_;
    if (td_.innerTd.rBlockPara.blockFactor * clNSize <= ubSize_) { // R能全载
        UbFullLoad(td_.innerTd.rBlockPara.blockFactor, td_.innerTd.rUbPara.mainCoreUbPara);
        UbFullLoad(td_.innerTd.rBlockPara.blockTailFactor, td_.innerTd.rUbPara.tailCoreUbPara);

        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount * borrowNCount_ * borrowRCount_;
        tilingKey_ = TILING_KEY_CORE_SS_ONEWAY;
    } else { // R不能全载
        int32_t rMaxForFullUb =
            LastPow2(MaxForFullUb(ubSize_ - UB_SS_BUFFER_SIZE - UB_SS_BR_IDX_BUFF_SIZE - UB_SS_BR_BUFF_SIZE, clSize_));
        DoUbSplit(td_.innerTd.rBlockPara.blockFactor, rMaxForFullUb, td_.innerTd.rUbPara.mainCoreUbPara);
        DoUbSplit(td_.innerTd.rBlockPara.blockTailFactor, rMaxForFullUb, td_.innerTd.rUbPara.tailCoreUbPara);

        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount * borrowNCount_ * borrowRCount_;
        tilingKey_ = TILING_KEY_CORE_SS_UB_SS_ONEWAY;
    }
}

void CumsumAscendcTilingImpl::RNGreaterCl()
{
    int64_t foldN = Ops::Base::CeilAlign(td_.innerTd.lenN * dtSizeCast_, static_cast<int64_t>(blockSize_));
    JudgeSklanskyPatten(td_.innerTd.lenR, foldN);
    if (ssPatten_ == SklanskyPattern::SS_ONEWAY) { // 单向sklansky
        int32_t rMaxForFullUb = MaxForFullUb(ubSize_, alignN_);
        if (td_.innerTd.lenR <= rMaxForFullUb) { // R满足ub全载
            RNGreaterClRFullLoad();
        } else { // R不满足ub全载
            RNGreaterClRNotFullLoad();
        }
    } else {                                                      // 双向sklansky
        if (td_.innerTd.lenM >= static_cast<int64_t>(coreNum_)) { // 双向性能差，如果M够大，就用M折叠，走单向
            int64_t mAfterBorrow = CalcBorrowM();
            if (borrowMCount_ > 1) {
                return RNGreaterClBorrowM(mAfterBorrow);
            }
        }
        /* 双向skalnsky 6块buffer：
        1、折叠搬入的buffer(fp16) 2、折叠搬入的cast buffer(fp32) 3、还原折叠的buffer(fp32) 4、固定256B
        5、纵向sklansky需要的固定buffer 2K 6、 纵向sklansky需要的固定buffer的索引 8K */
        int64_t foldBufferSizeCast = foldLen_ * vRegSize_;
        int64_t foldBufferSizeOrg = dtCast_ ? foldBufferSizeCast / CONST_2 : 0;
        int64_t unfoldBufferSize = Ops::Base::CeilAlign(
            td_.innerTd.lenN * foldCount_ * foldLen_ * dtSizeCast_, static_cast<int64_t>(blockSize_));
        int64_t totalBuffSize = foldBufferSizeOrg + foldBufferSizeCast + unfoldBufferSize + vRegSize_ +
                                TWOWAY_BUFF_SIZE + TWOWAY_IDX_BUFF_SIZE;
        if (totalBuffSize <= ubSize_) { // R满足UB全载
            RNGreaterClRFullLoad();
        } else { // R不满足UB全载
            RNGreaterClRNotFullLoad();
        }
    }
}

void CumsumAscendcTilingImpl::RNLesserCl()
{
    if (td_.innerTd.lenM * td_.innerTd.lenR * td_.innerTd.lenN * dtSizeCast_ >= clSize_) { // M*R*N大于cacheline
        MRNGreaterCl();
    } else { // M*R*N小于cacheline
        // M切分策略都一样，不切核，不切UB
        CoreFullLoad(td_.innerTd.lenM, td_.innerTd.mBlockPara, td_.innerTd.mUbPara.tailCoreUbPara);
        MRNLesserCl();
    }
}

int64_t CumsumAscendcTilingImpl::CalcBorrowM()
{
    int64_t alignN = dtCast_ ? alignN_ * CONST_2 : alignN_;
    int64_t foldCountCurr = CONST_2;
    int64_t foldLenPrev = td_.innerTd.lenM;
    int64_t foldLenCurr = td_.innerTd.lenM;
    int32_t coreUseRatioMax = -1;
    while (alignN * foldCountCurr <= vRegSize_) { // 遍历所有满足的折叠次数
        foldLenCurr = Ops::Base::CeilDiv(td_.innerTd.lenM, static_cast<int64_t>(foldCountCurr));
        int32_t coreUseRatio = static_cast<int32_t>(foldLenCurr % coreNum_);
        if (coreUseRatio == 0) { // 当前折叠次数整除核数，利用率最高，无脑更新为最优策略
            borrowMCount_ = foldCountCurr;
            foldLenPrev = foldLenCurr;
            coreUseRatioMax = coreUseRatio;
        } else {                        // 当前折叠次数不能整除核数
            if (coreUseRatioMax != 0) { // 历史记录中，没有整除的，比较利用率，更新最优策略；有整除的，不更新
                borrowMCount_ = coreUseRatio > coreUseRatioMax ? foldCountCurr : borrowMCount_;
                foldLenPrev = coreUseRatio > coreUseRatioMax ? foldLenCurr : foldLenPrev;
                coreUseRatioMax = coreUseRatio;
            }
        }
        foldCountCurr++;
    }

    return foldLenPrev;
}

void CumsumAscendcTilingImpl::RNGreaterClBorrowM(int64_t mAfterBorrow)
{
    // 借M场景下，R不分核
    DoBlockSplit(td_.innerTd.lenR, 1, td_.innerTd.rBlockPara);

    // 借轴M后，剩下的M用来分核
    DoBlockSplit(mAfterBorrow, coreNum_, td_.innerTd.mBlockPara);
    td_.innerTd.mBlockPara.blockFactor *= borrowMCount_;
    int32_t borrowM = td_.innerTd.lenM % borrowMCount_ == 0 ? borrowMCount_ : td_.innerTd.lenM % borrowMCount_;
    td_.innerTd.mBlockPara.blockTailFactor = (td_.innerTd.mBlockPara.blockTailFactor - 1) * borrowMCount_ + borrowM;
    int64_t rMaxForFullUb = MaxForFullUb(ubSize_, static_cast<int32_t>(borrowMCount_ * alignN_));
    if (td_.innerTd.lenR > rMaxForFullUb) { // R无法UB全载，R切UB，M每次UB载入1个borrowMCount_
        rMaxForFullUb = MaxForFullUb(
            ubSize_ - UB_SS_BUFFER_SIZE - UB_SS_BR_BUFF_SIZE - UB_SS_BR_IDX_BUFF_SIZE,
            static_cast<int32_t>(borrowMCount_ * alignN_));
        DoUbSplit(td_.innerTd.rBlockPara.blockTailFactor, rMaxForFullUb, td_.innerTd.rUbPara.tailCoreUbPara);
        DoUbSplit(td_.innerTd.mBlockPara.blockFactor, borrowMCount_, td_.innerTd.mUbPara.mainCoreUbPara);
        DoUbSplit(td_.innerTd.mBlockPara.blockTailFactor, borrowMCount_, td_.innerTd.mUbPara.tailCoreUbPara);
        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount;
        tilingKey_ = TILING_KEY_UB_SS_ONEWAY;
    } else { // R UB全载，M按照borrowMCount_为单元，尽量多的载入UB
        UbFullLoad(td_.innerTd.rBlockPara.blockTailFactor, td_.innerTd.rUbPara.tailCoreUbPara);
        int64_t mMaxForFullUb = MaxForFullUb(
            ubSize_, static_cast<int32_t>(td_.innerTd.lenR * borrowMCount_ * alignN_));
        DoUbSplit(
            td_.innerTd.mBlockPara.blockFactor, borrowMCount_ * mMaxForFullUb, td_.innerTd.mUbPara.mainCoreUbPara);
        DoUbSplit(
            td_.innerTd.mBlockPara.blockTailFactor, borrowMCount_ * mMaxForFullUb, td_.innerTd.mUbPara.tailCoreUbPara);
        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount;
        tilingKey_ = TILING_KEY_ONEWAY;
    }
}

void CumsumAscendcTilingImpl::RNGreaterClRFullLoad()
{
    // R不分核，不切UB
    CoreFullLoad(td_.innerTd.lenR, td_.innerTd.rBlockPara, td_.innerTd.rUbPara.tailCoreUbPara);

    // M先分核
    DoBlockSplit(td_.innerTd.lenM, coreNum_, td_.innerTd.mBlockPara);
    if (ssPatten_ == SklanskyPattern::SS_ONEWAY) { // 单向sklansky
        // 分核后的M，再尽量载入UB
        int32_t mMaxForFullUb = MaxForFullUb(ubSize_, static_cast<int32_t>(td_.innerTd.lenR * alignN_));
        DoUbSplit(td_.innerTd.mBlockPara.blockFactor, mMaxForFullUb, td_.innerTd.mUbPara.mainCoreUbPara);
        DoUbSplit(td_.innerTd.mBlockPara.blockTailFactor, mMaxForFullUb, td_.innerTd.mUbPara.tailCoreUbPara);

        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount;
        tilingKey_ = TILING_KEY_ONEWAY;
    } else { // 双向sklansky
        // 为了简化复杂度，M不切UB，一次UB只载入长度为1的M
        DoUbSplit(td_.innerTd.mBlockPara.blockFactor, 1, td_.innerTd.mUbPara.mainCoreUbPara);
        DoUbSplit(td_.innerTd.mBlockPara.blockTailFactor, 1, td_.innerTd.mUbPara.tailCoreUbPara);

        td_.innerTd.tailCoreTailUbFoldPara.foldCount = foldCount_;
        td_.innerTd.tailCoreTailUbFoldPara.foldLen = foldLen_;
        td_.innerTd.tailCoreTailUbFoldPara.sklanskyIter = std::log2(td_.innerTd.tailCoreTailUbFoldPara.foldLen);

        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount;
        tilingKey_ = TILING_KEY_TWOWAY;
    }
}

void CumsumAscendcTilingImpl::RNGreaterClRNotFullLoad()
{
    if (td_.innerTd.lenM > static_cast<int64_t>(coreNum_) / CONST_2) { // M可以保证核利用率，不用借轴
        RNGreaterClRNotFullLoadNotBorrowR();
    } else { // M不能保证核利用率，需要借轴
        RNGreaterClRNotFullLoadBorrowR();
    }
}

void CumsumAscendcTilingImpl::RNGreaterClRNotFullLoadNotBorrowR()
{
    // M切多核，一次UB只载入长度为1的M
    DoBlockSplit(td_.innerTd.lenM, coreNum_, td_.innerTd.mBlockPara);
    DoUbSplit(td_.innerTd.mBlockPara.blockFactor, 1, td_.innerTd.mUbPara.mainCoreUbPara);
    DoUbSplit(td_.innerTd.mBlockPara.blockTailFactor, 1, td_.innerTd.mUbPara.tailCoreUbPara);

    // R不分核
    DoBlockSplit(td_.innerTd.lenR, 1, td_.innerTd.rBlockPara);

    if (ssPatten_ == SklanskyPattern::SS_ONEWAY) { // 单向sklansky
        // R切分UB，factor满足2^k
        int32_t rMaxForFullUb =
            MaxForFullUb(ubSize_ - UB_SS_BUFFER_SIZE - UB_SS_BR_IDX_BUFF_SIZE - UB_SS_BR_BUFF_SIZE, alignN_);
        int32_t rFactor = LastPow2(rMaxForFullUb);
        DoUbSplit(td_.innerTd.rBlockPara.blockTailFactor, rFactor, td_.innerTd.rUbPara.tailCoreUbPara);

        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount;
        tilingKey_ = TILING_KEY_UB_SS_ONEWAY;
    } else { // 双向sklansky
        // R切分UB，factor满足2^k
        int64_t alignN = dtCast_ ? vRegSize_ + vRegSize_ / CONST_2 : vRegSize_;
        alignN += td_.innerTd.lenN * foldCount_ * dtSizeCast_;
        int64_t totalBuffSize = static_cast<int64_t>(
            ubSize_ - UB_SS_BUFFER_SIZE - UB_SS_BR_IDX_BUFF_SIZE - UB_SS_BR_BUFF_SIZE - vRegSize_ - blockSize_ -
            TWOWAY_BUFF_SIZE - TWOWAY_IDX_BUFF_SIZE);
        int32_t rMaxForFullUb = Ops::Base::FloorDiv(totalBuffSize, alignN);
        rMaxForFullUb *= foldCount_;
        int32_t rFactor = LastPow2(rMaxForFullUb);
        DoUbSplit(td_.innerTd.rBlockPara.blockTailFactor, rFactor, td_.innerTd.rUbPara.tailCoreUbPara);

        int64_t foldN = Ops::Base::CeilAlign(td_.innerTd.lenN * dtSizeCast_, static_cast<int64_t>(blockSize_));
        DoFold(
            td_.innerTd.rUbPara.tailCoreUbPara.ubFactor, foldN, td_.innerTd.tailCoreMainUbFoldPara.foldCount,
            td_.innerTd.tailCoreMainUbFoldPara.foldLen);
        td_.innerTd.tailCoreMainUbFoldPara.sklanskyIter = std::log2(td_.innerTd.tailCoreMainUbFoldPara.foldLen);
        DoFold(
            td_.innerTd.rUbPara.tailCoreUbPara.ubTailFactor, foldN, td_.innerTd.tailCoreTailUbFoldPara.foldCount,
            td_.innerTd.tailCoreTailUbFoldPara.foldLen);
        td_.innerTd.tailCoreTailUbFoldPara.sklanskyIter = std::log2(td_.innerTd.tailCoreTailUbFoldPara.foldLen);

        td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount;
        tilingKey_ = TILING_KEY_UB_SS_TWOWAY;
    }
}

void CumsumAscendcTilingImpl::RNGreaterClRNotFullLoadBorrowR()
{
    // M切多核，一次UB只载入长度为1的M
    DoBlockSplit(td_.innerTd.lenM, coreNum_, td_.innerTd.mBlockPara);
    DoUbSplit(td_.innerTd.mBlockPara.blockFactor, 1, td_.innerTd.mUbPara.mainCoreUbPara);
    DoUbSplit(td_.innerTd.mBlockPara.blockTailFactor, 1, td_.innerTd.mUbPara.tailCoreUbPara);

    // 借轴R
    borrowRCount_ = Ops::Base::FloorDiv(static_cast<int64_t>(coreNum_), td_.innerTd.lenM);
    DoBlockSplit(td_.innerTd.lenR, borrowRCount_, td_.innerTd.rBlockPara);
    borrowRCount_ = td_.innerTd.rBlockPara.blockCount;

    td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount * borrowRCount_;

    // 用主核的长度来判定是双向还是单向
    int64_t foldN = Ops::Base::CeilAlign(td_.innerTd.lenN * dtSizeCast_, static_cast<int64_t>(blockSize_));
    JudgeSklanskyPatten(td_.innerTd.rBlockPara.blockFactor, foldN);
    if (ssPatten_ == SklanskyPattern::SS_ONEWAY) { // 单向sklansky
        int32_t rMaxForFullUb = MaxForFullUb(ubSize_, alignN_);
        // 用主核的长度来判定R是不是全载
        if (td_.innerTd.rBlockPara.blockFactor <= rMaxForFullUb) { // R满足UB全载
            UbFullLoad(td_.innerTd.rBlockPara.blockFactor, td_.innerTd.rUbPara.mainCoreUbPara);
            UbFullLoad(td_.innerTd.rBlockPara.blockTailFactor, td_.innerTd.rUbPara.tailCoreUbPara);

            tilingKey_ = TILING_KEY_CORE_SS_ONEWAY;
        } else { // R不满足UB全载
            // R切分UB，factor满足2^k
            rMaxForFullUb =
                MaxForFullUb(ubSize_ - UB_SS_BUFFER_SIZE - UB_SS_BR_IDX_BUFF_SIZE - UB_SS_BR_BUFF_SIZE, alignN_);
            int32_t rFactor = LastPow2(rMaxForFullUb);
            DoUbSplit(td_.innerTd.rBlockPara.blockFactor, rFactor, td_.innerTd.rUbPara.mainCoreUbPara);
            DoUbSplit(td_.innerTd.rBlockPara.blockTailFactor, rFactor, td_.innerTd.rUbPara.tailCoreUbPara);

            tilingKey_ = TILING_KEY_CORE_SS_UB_SS_ONEWAY;
        }
    } else { // 双向sklansky
        RNGreaterClRNotFullLoadBorrowRTwoway();
    }
}

void CumsumAscendcTilingImpl::RNGreaterClRNotFullLoadBorrowRTwoway()
{
    int64_t foldN = Ops::Base::CeilAlign(td_.innerTd.lenN * dtSizeCast_, static_cast<int64_t>(blockSize_));
    /* 双向skalnsky 6块buffer：
    1、折叠搬入的buffer(fp16) 2、折叠搬入的cast buffer(fp32) 3、还原折叠的buffer(fp32) 4、固定256B
    5、纵向sklansky需要的固定buffer 2K 6、 纵向sklansky需要的固定buffer的索引 8K */
    int64_t foldBufferSizeCast = foldLen_ * vRegSize_;
    int64_t foldBufferSizeOrg = dtCast_ ? foldBufferSizeCast / CONST_2 : 0;
    int64_t unfoldBufferSize =
        Ops::Base::CeilAlign(td_.innerTd.lenN * foldCount_ * foldLen_ * dtSizeCast_, static_cast<int64_t>(blockSize_));
    int64_t totalBuffSize =
        foldBufferSizeOrg + foldBufferSizeCast + unfoldBufferSize + vRegSize_ + TWOWAY_BUFF_SIZE + TWOWAY_IDX_BUFF_SIZE;
    // 用主核的长度来判定R是否全载
    if (totalBuffSize <= ubSize_) { // R满足UB全载
        UbFullLoad(td_.innerTd.rBlockPara.blockFactor, td_.innerTd.rUbPara.mainCoreUbPara);
        UbFullLoad(td_.innerTd.rBlockPara.blockTailFactor, td_.innerTd.rUbPara.tailCoreUbPara);

        DoFold(
            td_.innerTd.rUbPara.mainCoreUbPara.ubTailFactor, foldN, td_.innerTd.mainCoreTailUbFoldPara.foldCount,
            td_.innerTd.mainCoreTailUbFoldPara.foldLen);
        td_.innerTd.mainCoreTailUbFoldPara.sklanskyIter = std::log2(td_.innerTd.mainCoreTailUbFoldPara.foldLen);
        DoFold(
            td_.innerTd.rUbPara.tailCoreUbPara.ubTailFactor, foldN, td_.innerTd.tailCoreTailUbFoldPara.foldCount,
            td_.innerTd.tailCoreTailUbFoldPara.foldLen);
        td_.innerTd.tailCoreTailUbFoldPara.sklanskyIter = std::log2(td_.innerTd.tailCoreTailUbFoldPara.foldLen);

        tilingKey_ = TILING_KEY_CORE_SS_TWOWAY;
    } else { // R不满足UB全载
        // R切分UB，factor满足2^k
        int64_t alignN = dtCast_ ? vRegSize_ + vRegSize_ / CONST_2 : vRegSize_;
        alignN += td_.innerTd.lenN * foldCount_ * dtSizeCast_;
        totalBuffSize = static_cast<int64_t>(
            ubSize_ - UB_SS_BUFFER_SIZE - UB_SS_BR_IDX_BUFF_SIZE - UB_SS_BR_BUFF_SIZE - vRegSize_ - blockSize_ -
            TWOWAY_BUFF_SIZE - TWOWAY_IDX_BUFF_SIZE);
        int32_t rMaxForFullUb = Ops::Base::FloorDiv(totalBuffSize, alignN);
        rMaxForFullUb *= foldCount_;
        int32_t rFactor = LastPow2(rMaxForFullUb);
        DoUbSplit(td_.innerTd.rBlockPara.blockFactor, rFactor, td_.innerTd.rUbPara.mainCoreUbPara);
        DoUbSplit(td_.innerTd.rBlockPara.blockTailFactor, rFactor, td_.innerTd.rUbPara.tailCoreUbPara);

        DoFold(
            td_.innerTd.rUbPara.mainCoreUbPara.ubFactor, foldN, td_.innerTd.mainCoreMainUbFoldPara.foldCount,
            td_.innerTd.mainCoreMainUbFoldPara.foldLen);
        td_.innerTd.mainCoreMainUbFoldPara.sklanskyIter = std::log2(td_.innerTd.mainCoreMainUbFoldPara.foldLen);
        DoFold(
            td_.innerTd.rUbPara.mainCoreUbPara.ubTailFactor, foldN, td_.innerTd.mainCoreTailUbFoldPara.foldCount,
            td_.innerTd.mainCoreTailUbFoldPara.foldLen);
        td_.innerTd.mainCoreTailUbFoldPara.sklanskyIter = std::log2(td_.innerTd.mainCoreTailUbFoldPara.foldLen);
        DoFold(
            td_.innerTd.rUbPara.tailCoreUbPara.ubFactor, foldN, td_.innerTd.tailCoreMainUbFoldPara.foldCount,
            td_.innerTd.tailCoreMainUbFoldPara.foldLen);
        td_.innerTd.tailCoreMainUbFoldPara.sklanskyIter = std::log2(td_.innerTd.tailCoreMainUbFoldPara.foldLen);
        DoFold(
            td_.innerTd.rUbPara.tailCoreUbPara.ubTailFactor, foldN, td_.innerTd.tailCoreTailUbFoldPara.foldCount,
            td_.innerTd.tailCoreTailUbFoldPara.foldLen);
        td_.innerTd.tailCoreTailUbFoldPara.sklanskyIter = std::log2(td_.innerTd.tailCoreTailUbFoldPara.foldLen);

        tilingKey_ = TILING_KEY_CORE_SS_UB_SS_TWOWAY;
    }
}

void CumsumAscendcTilingImpl::MRNGreaterCl()
{
    int32_t minMForCl =
        Ops::Base::CeilDiv(static_cast<int64_t>(clSize_), td_.innerTd.lenR * td_.innerTd.lenN * dtSize_);
    int64_t blockCountMinMForCl = Ops::Base::CeilDiv(td_.innerTd.lenM, static_cast<int64_t>(minMForCl));
    if (blockCountMinMForCl <= coreNum_) {
        DoBlockSplit(td_.innerTd.lenM, blockCountMinMForCl, td_.innerTd.mBlockPara);
        UbFullLoad(td_.innerTd.mBlockPara.blockFactor, td_.innerTd.mUbPara.mainCoreUbPara);
        UbFullLoad(td_.innerTd.mBlockPara.blockTailFactor, td_.innerTd.mUbPara.tailCoreUbPara);
    } else {
        DoBlockSplit(td_.innerTd.lenM, coreNum_, td_.innerTd.mBlockPara);
        int32_t mMaxForFullUb = MaxForFullUb(ubSize_, static_cast<int32_t>(td_.innerTd.lenR * alignN_));
        DoUbSplit(td_.innerTd.mBlockPara.blockFactor, mMaxForFullUb, td_.innerTd.mUbPara.mainCoreUbPara);
        DoUbSplit(td_.innerTd.mBlockPara.blockTailFactor, mMaxForFullUb, td_.innerTd.mUbPara.tailCoreUbPara);
    }

    td_.innerTd.realCoreNum = td_.innerTd.mBlockPara.blockCount;
    tilingKey_ = TILING_KEY_ONEWAY;
}

void CumsumAscendcTilingImpl::MRNLesserCl()
{
    td_.innerTd.realCoreNum = 1;
    tilingKey_ = TILING_KEY_ONEWAY;
}

void CumsumAscendcTilingImpl::CalcBufferSize()
{
    switch (tilingKey_) {
        case TILING_KEY_ONEWAY:
        case TILING_KEY_CORE_SS_ONEWAY: {
            td_.innerTd.xBufSize = CalcXBufSize(ubSize_);
            break;
        }
        case TILING_KEY_TWOWAY:
        case TILING_KEY_CORE_SS_TWOWAY: {
            td_.innerTd.xUnfoldBufSize = CalcXUnfoldBufSize();
            td_.innerTd.xBufSize = CalcXBufSize(
                ubSize_ - td_.innerTd.xUnfoldBufSize - vRegSize_ - TWOWAY_BUFF_SIZE - TWOWAY_IDX_BUFF_SIZE);
            break;
        }
        case TILING_KEY_UB_SS_ONEWAY:
        case TILING_KEY_CORE_SS_UB_SS_ONEWAY: {
            td_.innerTd.ubSklanskyBufSize = UB_SS_BUFFER_SIZE;
            td_.innerTd.xBufSize =
                CalcXBufSize(ubSize_ - td_.innerTd.ubSklanskyBufSize - UB_SS_BR_IDX_BUFF_SIZE - UB_SS_BR_BUFF_SIZE);
            break;
        }
        case TILING_KEY_UB_SS_TWOWAY:
        case TILING_KEY_CORE_SS_UB_SS_TWOWAY: {
            td_.innerTd.ubSklanskyBufSize = UB_SS_BUFFER_SIZE;
            td_.innerTd.xUnfoldBufSize = CalcXUnfoldBufSize();
            td_.innerTd.xBufSize = CalcXBufSize(
                ubSize_ - td_.innerTd.ubSklanskyBufSize - UB_SS_BR_IDX_BUFF_SIZE - UB_SS_BR_BUFF_SIZE -
                td_.innerTd.xUnfoldBufSize - vRegSize_ - TWOWAY_BUFF_SIZE - TWOWAY_IDX_BUFF_SIZE);
            break;
        }
        default: {
            break;
        }
    }
}

int32_t CumsumAscendcTilingImpl::CalcXUnfoldBufSize()
{
    int64_t unfoldR = CalcUnfoldR(td_.innerTd);
    int32_t unfoldN = CalcUnfoldN(td_.innerTd.nUbPara);
    return Ops::Base::CeilAlign(static_cast<int32_t>(unfoldR * unfoldN * dtSizeCast_), blockSize_);
}

int32_t CumsumAscendcTilingImpl::CalcXBufSize(int32_t xBufferMax)
{
    return dtCast_ ? xBufferMax / CAST_MULT / blockSize_ * blockSize_ : xBufferMax;
}

int64_t CumsumAscendcTilingImpl::CalcUnfoldR(CumsumCoreInnerTilingData& innerTd)
{
    if (innerTd.mainCoreMainUbFoldPara.foldLen != 0) {
        return innerTd.mainCoreMainUbFoldPara.foldLen * innerTd.mainCoreMainUbFoldPara.foldCount;
    } else {
        if (innerTd.mainCoreTailUbFoldPara.foldLen != 0) {
            return innerTd.mainCoreTailUbFoldPara.foldLen * innerTd.mainCoreTailUbFoldPara.foldCount;
        } else {
            if (innerTd.tailCoreMainUbFoldPara.foldLen != 0) {
                return innerTd.tailCoreMainUbFoldPara.foldLen * innerTd.tailCoreMainUbFoldPara.foldCount;
            } else {
                return innerTd.tailCoreTailUbFoldPara.foldLen * innerTd.tailCoreTailUbFoldPara.foldCount;
            }
        }
    }
}

int32_t CumsumAscendcTilingImpl::CalcUnfoldN(UbPara& ubPara)
{
    if (ubPara.mainCoreUbPara.ubFactor != 0) {
        return ubPara.mainCoreUbPara.ubFactor;
    } else {
        if (ubPara.mainCoreUbPara.ubTailFactor != 0) {
            return ubPara.mainCoreUbPara.ubTailFactor;
        } else {
            if (ubPara.tailCoreUbPara.ubFactor != 0) {
                return ubPara.tailCoreUbPara.ubFactor;
            } else {
                return ubPara.tailCoreUbPara.ubTailFactor;
            }
        }
    }
}

void CumsumAscendcTilingImpl::TilingStrategyOuterTd()
{
    if (tilingKey_ != TILING_KEY_CORE_SS_ONEWAY && tilingKey_ != TILING_KEY_CORE_SS_TWOWAY &&
        tilingKey_ != TILING_KEY_CORE_SS_UB_SS_ONEWAY && tilingKey_ != TILING_KEY_CORE_SS_UB_SS_TWOWAY) {
        return;
    }

    td_.outerTd.sklanskyItersCount = ceil(log2(borrowRCount_));
    td_.outerTd.addFactorRepeatTimesMain =
        td_.innerTd.nUbPara.mainCoreUbPara.ubTailFactor == 0 ?
            0 :
            Ops::Base::FloorDiv(vRegSize_ * CONST_3 / dtSizeCast_, td_.innerTd.nUbPara.mainCoreUbPara.ubTailFactor);
    td_.outerTd.addFactorRepeatTimesTail =
        td_.innerTd.nUbPara.tailCoreUbPara.ubTailFactor == 0 ?
            0 :
            Ops::Base::FloorDiv(vRegSize_ * CONST_3 / dtSizeCast_, td_.innerTd.nUbPara.tailCoreUbPara.ubTailFactor);
    td_.outerTd.coreGroupCount = static_cast<int32_t>(td_.innerTd.lenM) * borrowNCount_;
    td_.outerTd.coreGroupCoreNum = Ops::Base::FloorDiv(coreNum_, td_.outerTd.coreGroupCount);
    td_.outerTd.addFactorBufferSize = vRegSize_ * CONST_3;
    td_.outerTd.addFactorOffsetBufferSize = ADD_FACTOR_OFFSET_MAX;
    if (dtCast_) {
        td_.outerTd.addDataBufferSize =
            (ubSize_ - td_.outerTd.addFactorBufferSize * CONST_2 - td_.outerTd.addFactorOffsetBufferSize) / CAST_MULT /
            blockSize_ * blockSize_;
    } else {
        td_.outerTd.addDataBufferSize =
            ubSize_ - td_.outerTd.addFactorBufferSize - td_.outerTd.addFactorOffsetBufferSize;
    }

    TilingStrategyOuterTdSklanskyItersTiling();
}

void CumsumAscendcTilingImpl::TilingStrategyOuterTdSklanskyItersTiling()
{
    for (int32_t i = 0; i < td_.outerTd.sklanskyItersCount; i++) {
        SklanskyItersTiling& ssiTiling = td_.outerTd.sklanskyItersTiling[i];
        ssiTiling.iterGroupAddCount = 1;
        for (int32_t j = 0; j < i; j++) {
            ssiTiling.iterGroupAddCount *= CONST_2;
        }

        ssiTiling.iterGroupStartOffset = ssiTiling.iterGroupAddCount - 1;
        ssiTiling.iterGroupAddOffset = CONST_2 * ssiTiling.iterGroupAddCount;
        int32_t mainIterGroupCount = borrowRCount_ % ssiTiling.iterGroupAddOffset == 0 ?
                                         Ops::Base::FloorDiv(borrowRCount_, ssiTiling.iterGroupAddOffset) - 1 :
                                         Ops::Base::FloorDiv(borrowRCount_, ssiTiling.iterGroupAddOffset);
        int32_t tailIterGroupCount =
            borrowRCount_ % ssiTiling.iterGroupAddOffset <= ssiTiling.iterGroupAddCount ? 0 : 1;
        tailIterGroupCount = borrowRCount_ % ssiTiling.iterGroupAddOffset == 0 ? 1 : tailIterGroupCount;
        ssiTiling.iterGroupCount = mainIterGroupCount + tailIterGroupCount;
        ssiTiling.iterGroupCoreNum = Ops::Base::FloorDiv(td_.outerTd.coreGroupCoreNum, ssiTiling.iterGroupCount);

        // 主块分组
        if (mainIterGroupCount > 0) {
            InitIterGroupTiling(
                ssiTiling.iterGroupAddCount * td_.innerTd.rBlockPara.blockFactor, ssiTiling.iterGroupCoreNum,
                ssiTiling.mainIterGroupMainNTiling, ssiTiling.mainIterGroupTailNTiling);
        }

        // 尾块分组
        if (tailIterGroupCount > 0) { // 如果有尾块，直接用尾块的R计算切分策略
            int32_t mainRCnt = borrowRCount_ % ssiTiling.iterGroupAddOffset == 0 ?
                                   ssiTiling.iterGroupAddOffset - ssiTiling.iterGroupAddCount - 1 :
                                   borrowRCount_ % ssiTiling.iterGroupAddOffset - ssiTiling.iterGroupAddCount - 1;
            InitIterGroupTiling(
                mainRCnt * td_.innerTd.rBlockPara.blockFactor + td_.innerTd.rBlockPara.blockTailFactor,
                ssiTiling.iterGroupCoreNum, ssiTiling.tailIterGroupMainNTiling, ssiTiling.tailIterGroupTailNTiling);
        } else { // 如果没有尾块，尾块的切分策略就等于主块的切分策略
            ssiTiling.tailIterGroupMainNTiling = ssiTiling.mainIterGroupMainNTiling;
            ssiTiling.tailIterGroupTailNTiling = ssiTiling.mainIterGroupTailNTiling;
        }
    }
}

void CumsumAscendcTilingImpl::InitIterGroupTiling(
    int64_t rLen, int32_t coreNum, IterGroupTiling& mainNTiling, IterGroupTiling& tailNTiling)
{
    DoBlockSplitBalance(
        rLen, coreNum, mainNTiling.iterGroupMainBlockCount, mainNTiling.iterGroupTailBlockCount,
        mainNTiling.iterGroupBlockFactor);

    // 主块N和尾块N切分策略是一样的
    tailNTiling.iterGroupMainBlockCount = mainNTiling.iterGroupMainBlockCount;
    tailNTiling.iterGroupTailBlockCount = mainNTiling.iterGroupTailBlockCount;
    tailNTiling.iterGroupBlockFactor = mainNTiling.iterGroupBlockFactor;

    // 主块N的UB切分策略
    if (td_.innerTd.nUbPara.mainCoreUbPara.ubTailFactor != 0) {
        InitIterGroupUbTiling(td_.innerTd.nUbPara.mainCoreUbPara.ubTailFactor, mainNTiling);
    }

    // 尾块N的UB切分策略
    if (td_.innerTd.nUbPara.tailCoreUbPara.ubTailFactor != 0) {
        InitIterGroupUbTiling(td_.innerTd.nUbPara.tailCoreUbPara.ubTailFactor, tailNTiling);
    }
}

void CumsumAscendcTilingImpl::InitIterGroupUbTiling(int32_t nFactor, IterGroupTiling& iterGroupTiling)
{
    // 主核UB切分策略
    if (iterGroupTiling.iterGroupMainBlockCount > 0) {
        __UbPara ubPara;
        DoUbSplit(
            iterGroupTiling.iterGroupBlockFactor,
            Ops::Base::FloorDiv(td_.outerTd.addDataBufferSize - blockSize_, nFactor * dtSize_), ubPara);
        iterGroupTiling.iterGroupMainCoreUbFactor = ubPara.ubFactor;
        iterGroupTiling.iterGroupMainCoreUbTailFactor = ubPara.ubTailFactor;
        iterGroupTiling.iterGroupMainCoreUbFactorCount = ubPara.ubCount;
    }

    // 尾核UB切分策略
    if (iterGroupTiling.iterGroupTailBlockCount > 0) {
        __UbPara ubPara;
        DoUbSplit(
            iterGroupTiling.iterGroupBlockFactor - 1,
            Ops::Base::FloorDiv(td_.outerTd.addDataBufferSize - blockSize_, nFactor * dtSize_), ubPara);
        iterGroupTiling.iterGroupTailCoreUbFactor = ubPara.ubFactor;
        iterGroupTiling.iterGroupTailCoreUbTailFactor = ubPara.ubTailFactor;
        iterGroupTiling.iterGroupTailCoreUbFactorCount = ubPara.ubCount;
    }
}

void CumsumAscendcTilingImpl::DoBlockSplitBalance(
    int64_t splitLen, int32_t coreNum, int32_t& mainBlockNum, int32_t& tailBlockNum, int64_t& blockFactor)
{
    if (splitLen <= 0) {
        return;
    }
    int32_t blockNum = splitLen < coreNum ? splitLen : coreNum;
    blockFactor = Ops::Base::CeilDiv(splitLen, static_cast<int64_t>(blockNum));
    mainBlockNum = (splitLen % blockNum == 0) ? blockNum : splitLen % blockNum;
    tailBlockNum = blockNum - mainBlockNum;
}

void CumsumAscendcTilingImpl::JudgeSklanskyPatten(int64_t rLen, int64_t alignN)
{
    if (alignN > static_cast<int64_t>(vRegSize_) / CONST_4) {
        ssPatten_ = SklanskyPattern::SS_ONEWAY;
        return;
    }

    DoFold(rLen, alignN, foldCount_, foldLen_);

    if (foldCount_ == 1) {
        ssPatten_ = SklanskyPattern::SS_ONEWAY;
        return;
    }

    ssPatten_ = SklanskyPattern::SS_TWOWAY;
}

void CumsumAscendcTilingImpl::DoFold(int64_t rLen, int64_t alignN, int32_t& foldCount, int64_t& foldLen)
{
    int32_t foldCountPrev = 1;
    int32_t foldCountCurr = 1;
    int64_t foldLenTemp = LastPow2(rLen);
    int64_t foldLenPrev = foldLenTemp == rLen ? rLen : foldLenTemp << 1;
    int64_t foldLenCurr = foldLenPrev;
    while (alignN * foldCountCurr <= vRegSize_ && foldLenCurr >= FOLD_LEN_MIN) {
        if (dtCast_ && foldCountCurr > FP16_FOLD_MAX) { // fp16场景，折叠4次以上，原始buffer的大小不够32B对齐
            break;
        }

        foldCountPrev = foldCountCurr;
        foldLenPrev = foldLenCurr;

        foldCountCurr = foldCountPrev * CONST_2;
        foldLenCurr = Ops::Base::CeilDiv(rLen, static_cast<int64_t>(foldCountCurr));
        foldLenTemp = LastPow2(foldLenCurr);
        foldLenCurr = foldLenTemp == foldLenCurr ? foldLenCurr : foldLenTemp << 1;
    }

    foldCount = foldCountPrev;
    foldLen = foldLenPrev;
}

void CumsumAscendcTilingImpl::CoreFullLoad(int64_t len, BlockPara& blockPara, __UbPara& ubPara)
{
    DoBlockSplit(len, 1, blockPara);
    DoUbSplit(blockPara.blockTailFactor, blockPara.blockTailFactor, ubPara);
}

void CumsumAscendcTilingImpl::UbFullLoad(int64_t len, __UbPara& ubPara)
{
    DoUbSplit(len, len, ubPara);
}

int32_t CumsumAscendcTilingImpl::MaxForFullUb(int32_t bufferMax, int32_t alignLen)
{
    return Ops::Base::FloorDiv(bufferMax, dtCast_ ? alignLen * CAST_MULT : alignLen);
}

void CumsumAscendcTilingImpl::DoBlockSplit(int64_t splitLen, int32_t coreNum, BlockPara& blockPara)
{
    if (splitLen <= 0) {
        return;
    }
    blockPara.blockCount = splitLen < coreNum ? splitLen : coreNum;
    blockPara.blockFactor = Ops::Base::CeilDiv(splitLen, static_cast<int64_t>(blockPara.blockCount));
    blockPara.blockCount = Ops::Base::CeilDiv(splitLen, blockPara.blockFactor);
    blockPara.blockTailFactor =
        (splitLen % blockPara.blockFactor == 0) ? blockPara.blockFactor : splitLen % blockPara.blockFactor;
    blockPara.blockFactor = (blockPara.blockCount == 1) ? 0 : blockPara.blockFactor;
}

void CumsumAscendcTilingImpl::DoUbSplit(int64_t splitLen, int64_t splitFactor, __UbPara& ubPara)
{
    if (splitLen <= 0) {
        return;
    }
    ubPara.ubFactor = splitLen < splitFactor ? splitLen : splitFactor;
    ubPara.ubCount = Ops::Base::CeilDiv(splitLen, static_cast<int64_t>(ubPara.ubFactor));
    ubPara.ubTailFactor = (splitLen % ubPara.ubFactor == 0) ? ubPara.ubFactor : splitLen % ubPara.ubFactor;
    ubPara.ubFactor = (ubPara.ubCount == 1) ? 0 : ubPara.ubFactor;
}

int64_t CumsumAscendcTilingImpl::LastPow2(int64_t n)
{
    n |= (n >> 1);
    n |= (n >> CONST_2);
    n |= (n >> CONST_4);
    n |= (n >> CONST_8);
    n |= (n >> CONST_16);
    n |= (n >> CONST_32);

    return std::max(1L, n - (n >> 1));
}

void CumsumAscendcTilingImpl::FillTilingData()
{
    CumsumSklanskyTilingData td;
    td.set_exclusive(td_.innerTd.exclusive ? 1 : 0);
    td.set_reverse(td_.innerTd.reverse ? 1 : 0);
    td.set_realCoreNum(td_.innerTd.realCoreNum);
    td.set_lenM(td_.innerTd.lenM);
    td.set_lenR(td_.innerTd.lenR);
    td.set_lenN(td_.innerTd.lenN);
    td.set_xBufSize(td_.innerTd.xBufSize);
    td.set_xUnfoldBufSize(td_.innerTd.xUnfoldBufSize);
    td.set_ubSklanskyBufSize(td_.innerTd.ubSklanskyBufSize);
    td.set_addFactorRepeatTimesMain(td_.outerTd.addFactorRepeatTimesMain);
    td.set_addFactorRepeatTimesTail(td_.outerTd.addFactorRepeatTimesTail);
    td.set_coreGroupCount(td_.outerTd.coreGroupCount);
    td.set_coreGroupCoreNum(td_.outerTd.coreGroupCoreNum);
    td.set_sklanskyItersCount(td_.outerTd.sklanskyItersCount);
    td.set_addFactorBufferSize(td_.outerTd.addFactorBufferSize);
    td.set_addDataBufferSize(td_.outerTd.addDataBufferSize);
    td.set_addFactorOffsetBufferSize(td_.outerTd.addFactorOffsetBufferSize);
    td.set_nBorrow(borrowNCount_);
    td.set_rBorrow(borrowRCount_);
    td.set_mBorrow(borrowMCount_);

    FillFoldPara(td.mainCoreMainUbFoldPara, td_.innerTd.mainCoreMainUbFoldPara);
    FillFoldPara(td.mainCoreTailUbFoldPara, td_.innerTd.mainCoreTailUbFoldPara);
    FillFoldPara(td.tailCoreMainUbFoldPara, td_.innerTd.tailCoreMainUbFoldPara);
    FillFoldPara(td.tailCoreTailUbFoldPara, td_.innerTd.tailCoreTailUbFoldPara);
    FillBlockPara(td.mBlockPara, td_.innerTd.mBlockPara);
    FillBlockPara(td.rBlockPara, td_.innerTd.rBlockPara);
    FillBlockPara(td.nBlockPara, td_.innerTd.nBlockPara);
    FillUbPara(td.mUbPara, td_.innerTd.mUbPara);
    FillUbPara(td.rUbPara, td_.innerTd.rUbPara);
    FillUbPara(td.nUbPara, td_.innerTd.nUbPara);
    FillSklanskyItersTiling(td.sklanskyItersTiling0, td_.outerTd.sklanskyItersTiling[SS_ITER_0]);
    FillSklanskyItersTiling(td.sklanskyItersTiling1, td_.outerTd.sklanskyItersTiling[SS_ITER_1]);
    FillSklanskyItersTiling(td.sklanskyItersTiling2, td_.outerTd.sklanskyItersTiling[SS_ITER_2]);
    FillSklanskyItersTiling(td.sklanskyItersTiling3, td_.outerTd.sklanskyItersTiling[SS_ITER_3]);
    FillSklanskyItersTiling(td.sklanskyItersTiling4, td_.outerTd.sklanskyItersTiling[SS_ITER_4]);
    FillSklanskyItersTiling(td.sklanskyItersTiling5, td_.outerTd.sklanskyItersTiling[SS_ITER_5]);

    td.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td.GetDataSize());
}

void CumsumAscendcTilingImpl::FillFoldPara(CumsumFoldPara& dstFoldPara, FoldPara& srcFoldPara)
{
    dstFoldPara.set_foldCount(srcFoldPara.foldCount);
    dstFoldPara.set_foldLen(srcFoldPara.foldLen);
    dstFoldPara.set_sklanskyIter(srcFoldPara.sklanskyIter);
}

void CumsumAscendcTilingImpl::FillBlockPara(CumsumBlockPara& dstBlockPara, BlockPara& srcBlockPara)
{
    dstBlockPara.set_blockCount(srcBlockPara.blockCount);
    dstBlockPara.set_blockFactor(srcBlockPara.blockFactor);
    dstBlockPara.set_blockTailFactor(srcBlockPara.blockTailFactor);
}

void CumsumAscendcTilingImpl::FillUbPara(CumsumUbPara& dstUbPara, UbPara& srcUbPara)
{
    dstUbPara.mainCoreUbPara.set_ubCount(srcUbPara.mainCoreUbPara.ubCount);
    dstUbPara.mainCoreUbPara.set_ubFactor(srcUbPara.mainCoreUbPara.ubFactor);
    dstUbPara.mainCoreUbPara.set_ubTailFactor(srcUbPara.mainCoreUbPara.ubTailFactor);
    dstUbPara.tailCoreUbPara.set_ubCount(srcUbPara.tailCoreUbPara.ubCount);
    dstUbPara.tailCoreUbPara.set_ubFactor(srcUbPara.tailCoreUbPara.ubFactor);
    dstUbPara.tailCoreUbPara.set_ubTailFactor(srcUbPara.tailCoreUbPara.ubTailFactor);
}

void CumsumAscendcTilingImpl::FillSklanskyItersTiling(
    CumsumSklanskyItersTiling& cumsumSklanskyItersTiling, SklanskyItersTiling& sklanskyItersTiling)
{
    cumsumSklanskyItersTiling.set_iterGroupCount(sklanskyItersTiling.iterGroupCount);
    cumsumSklanskyItersTiling.set_iterGroupAddCount(sklanskyItersTiling.iterGroupAddCount);
    cumsumSklanskyItersTiling.set_iterGroupAddOffset(sklanskyItersTiling.iterGroupAddOffset);
    cumsumSklanskyItersTiling.set_iterGroupStartOffset(sklanskyItersTiling.iterGroupStartOffset);
    cumsumSklanskyItersTiling.set_iterGroupCoreNum(sklanskyItersTiling.iterGroupCoreNum);
    FillIterGroupTiling(
        cumsumSklanskyItersTiling.mainIterGroupMainNTiling, sklanskyItersTiling.mainIterGroupMainNTiling);
    FillIterGroupTiling(
        cumsumSklanskyItersTiling.tailIterGroupMainNTiling, sklanskyItersTiling.tailIterGroupMainNTiling);
    FillIterGroupTiling(
        cumsumSklanskyItersTiling.mainIterGroupTailNTiling, sklanskyItersTiling.mainIterGroupTailNTiling);
    FillIterGroupTiling(
        cumsumSklanskyItersTiling.tailIterGroupTailNTiling, sklanskyItersTiling.tailIterGroupTailNTiling);
}

void CumsumAscendcTilingImpl::FillIterGroupTiling(
    CumsumIterGroupTiling& dstIterGroupTiling, IterGroupTiling& srcIterGroupTiling)
{
    dstIterGroupTiling.set_iterGroupBlockFactor(srcIterGroupTiling.iterGroupBlockFactor);
    dstIterGroupTiling.set_iterGroupMainBlockCount(srcIterGroupTiling.iterGroupMainBlockCount);
    dstIterGroupTiling.set_iterGroupTailBlockCount(srcIterGroupTiling.iterGroupTailBlockCount);
    dstIterGroupTiling.set_iterGroupMainCoreUbFactorCount(srcIterGroupTiling.iterGroupMainCoreUbFactorCount);
    dstIterGroupTiling.set_iterGroupMainCoreUbFactor(srcIterGroupTiling.iterGroupMainCoreUbFactor);
    dstIterGroupTiling.set_iterGroupMainCoreUbTailFactor(srcIterGroupTiling.iterGroupMainCoreUbTailFactor);
    dstIterGroupTiling.set_iterGroupTailCoreUbFactorCount(srcIterGroupTiling.iterGroupTailCoreUbFactorCount);
    dstIterGroupTiling.set_iterGroupTailCoreUbFactor(srcIterGroupTiling.iterGroupTailCoreUbFactor);
    dstIterGroupTiling.set_iterGroupTailCoreUbTailFactor(srcIterGroupTiling.iterGroupTailCoreUbTailFactor);
}

void CumsumAscendcTilingImpl::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is tilingKey:%lu, exclusive:%d, reverse:%d, realCoreNum:%d, lenM:%ld, lenR:%ld, lenN:%ld, \
        xBufSize:%d, xUnfoldBufSize:%d, ubSklanskyBufSize:%d, borrowN:%d, borrowR:%d, borrowM:%d",
        tilingKey_, td_.innerTd.exclusive, td_.innerTd.reverse, td_.innerTd.realCoreNum, td_.innerTd.lenM,
        td_.innerTd.lenR, td_.innerTd.lenN, td_.innerTd.xBufSize, td_.innerTd.xUnfoldBufSize,
        td_.innerTd.ubSklanskyBufSize, borrowNCount_, borrowRCount_, borrowMCount_);

    PrintFoldPara();
    PrintBlockPara();
    PrintUbPara();

    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is addFactorRepeatTimesMain:%d, addFactorRepeatTimesTail:%d, coreGroupCount:%d, \
        coreGroupCoreNum:%d, sklanskyItersCount:%d, addFactorBufferSize:%d, addDataBufferSize:%d, \
        addFactorOffsetBufferSize:%d",
        td_.outerTd.addFactorRepeatTimesMain, td_.outerTd.addFactorRepeatTimesTail, td_.outerTd.coreGroupCount,
        td_.outerTd.coreGroupCoreNum, td_.outerTd.sklanskyItersCount, td_.outerTd.addFactorBufferSize,
        td_.outerTd.addDataBufferSize, td_.outerTd.addFactorOffsetBufferSize);

    PrintSklanskyItersTiling();
}

void CumsumAscendcTilingImpl::PrintFoldPara()
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is mainCoreMainUbFoldPara:%d %ld %d, mainCoreTailUbFoldPara:%d %ld %d, \
        tailCoreMainUbFoldPara:%d %ld %d, tailCoreTailUbFoldPara:%d %ld %d",
        td_.innerTd.mainCoreMainUbFoldPara.foldCount, td_.innerTd.mainCoreMainUbFoldPara.foldLen,
        td_.innerTd.mainCoreMainUbFoldPara.sklanskyIter, td_.innerTd.mainCoreTailUbFoldPara.foldCount,
        td_.innerTd.mainCoreTailUbFoldPara.foldLen, td_.innerTd.mainCoreTailUbFoldPara.sklanskyIter,
        td_.innerTd.tailCoreMainUbFoldPara.foldCount, td_.innerTd.tailCoreMainUbFoldPara.foldLen,
        td_.innerTd.tailCoreMainUbFoldPara.sklanskyIter, td_.innerTd.tailCoreTailUbFoldPara.foldCount,
        td_.innerTd.tailCoreTailUbFoldPara.foldLen, td_.innerTd.tailCoreTailUbFoldPara.sklanskyIter);
}

void CumsumAscendcTilingImpl::PrintBlockPara()
{
    OP_LOGI(
        context_->GetNodeName(), "tilingData is mBlockPara:%d %ld %ld, rBlockPara:%d %ld %ld, nBlockPara:%d %ld %ld",
        td_.innerTd.mBlockPara.blockCount, td_.innerTd.mBlockPara.blockFactor, td_.innerTd.mBlockPara.blockTailFactor,
        td_.innerTd.rBlockPara.blockCount, td_.innerTd.rBlockPara.blockFactor, td_.innerTd.rBlockPara.blockTailFactor,
        td_.innerTd.nBlockPara.blockCount, td_.innerTd.nBlockPara.blockFactor, td_.innerTd.nBlockPara.blockTailFactor);
}

void CumsumAscendcTilingImpl::PrintUbPara()
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is mUbPara:%d %d %d %d %d %d, rUbPara:%d %d %d %d %d %d, nUbPara:%d %d %d %d %d %d",
        td_.innerTd.mUbPara.mainCoreUbPara.ubCount, td_.innerTd.mUbPara.mainCoreUbPara.ubFactor,
        td_.innerTd.mUbPara.mainCoreUbPara.ubTailFactor, td_.innerTd.mUbPara.tailCoreUbPara.ubCount,
        td_.innerTd.mUbPara.tailCoreUbPara.ubFactor, td_.innerTd.mUbPara.tailCoreUbPara.ubTailFactor,
        td_.innerTd.rUbPara.mainCoreUbPara.ubCount, td_.innerTd.rUbPara.mainCoreUbPara.ubFactor,
        td_.innerTd.rUbPara.mainCoreUbPara.ubTailFactor, td_.innerTd.rUbPara.tailCoreUbPara.ubCount,
        td_.innerTd.rUbPara.tailCoreUbPara.ubFactor, td_.innerTd.rUbPara.tailCoreUbPara.ubTailFactor,
        td_.innerTd.nUbPara.mainCoreUbPara.ubCount, td_.innerTd.nUbPara.mainCoreUbPara.ubFactor,
        td_.innerTd.nUbPara.mainCoreUbPara.ubTailFactor, td_.innerTd.nUbPara.tailCoreUbPara.ubCount,
        td_.innerTd.nUbPara.tailCoreUbPara.ubFactor, td_.innerTd.nUbPara.tailCoreUbPara.ubTailFactor);
}

void CumsumAscendcTilingImpl::PrintSklanskyItersTiling()
{
    for (int32_t i = 0; i < SS_ITER_MAX; i++) {
        OP_LOGI(
            context_->GetNodeName(),
            "tilingData is iterGroupCount:%d, iterGroupAddCount:%d, iterGroupAddOffset:%d, iterGroupStartOffset:%d, \
            iterGroupCoreNum:%d",
            td_.outerTd.sklanskyItersTiling[i].iterGroupCount, td_.outerTd.sklanskyItersTiling[i].iterGroupAddCount,
            td_.outerTd.sklanskyItersTiling[i].iterGroupAddOffset,
            td_.outerTd.sklanskyItersTiling[i].iterGroupStartOffset,
            td_.outerTd.sklanskyItersTiling[i].iterGroupCoreNum);

        PrintIterGroupTiling(td_.outerTd.sklanskyItersTiling[i].mainIterGroupMainNTiling);
        PrintIterGroupTiling(td_.outerTd.sklanskyItersTiling[i].tailIterGroupMainNTiling);
        PrintIterGroupTiling(td_.outerTd.sklanskyItersTiling[i].mainIterGroupTailNTiling);
        PrintIterGroupTiling(td_.outerTd.sklanskyItersTiling[i].tailIterGroupTailNTiling);
    }
}

void CumsumAscendcTilingImpl::PrintIterGroupTiling(IterGroupTiling& iterGroupTiling)
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is iterGroupBlockFactor:%ld, iterGroupMainBlockCount:%d, iterGroupTailBlockCount:%d, \
        iterGroupMainCoreUbFactorCount:%d, iterGroupMainCoreUbFactor:%d, iterGroupMainCoreUbTailFactor:%d, \
        iterGroupTailCoreUbFactorCount:%d, iterGroupTailCoreUbFactor:%d, iterGroupTailCoreUbTailFactor:%d",
        iterGroupTiling.iterGroupBlockFactor, iterGroupTiling.iterGroupMainBlockCount,
        iterGroupTiling.iterGroupTailBlockCount, iterGroupTiling.iterGroupMainCoreUbFactorCount,
        iterGroupTiling.iterGroupMainCoreUbFactor, iterGroupTiling.iterGroupMainCoreUbTailFactor,
        iterGroupTiling.iterGroupTailCoreUbFactorCount, iterGroupTiling.iterGroupTailCoreUbFactor,
        iterGroupTiling.iterGroupTailCoreUbTailFactor);
}

ge::graphStatus TilingCumsumAscendc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingCumsumAscendc.");

    CumsumAscendcTilingImpl tilingImpl = CumsumAscendcTilingImpl(context);
    auto compileInfo = reinterpret_cast<const CumsumCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    if (tilingImpl.Init(compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "TilingCumsumAscendc init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "TilingCumsumAscendc DoTiling failed.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Exit TilingCumsumAscendc.");
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling