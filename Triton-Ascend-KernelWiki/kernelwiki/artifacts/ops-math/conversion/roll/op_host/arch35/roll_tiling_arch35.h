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
 * \file roll_tiling_arch35.h
 * \brief
 */
#ifndef __ROLL_TILING_H__
#define __ROLL_TILING_H__

#include "op_host/tiling_base_class.h"
#include "conversion/roll/op_kernel/arch35/roll_struct.h"
#include "platform/platform_ascendc.h"

namespace optiling {

struct RollCompileInfoArch35 {
    int32_t core_num;
    int32_t ub_size;
    int32_t ub_num;
    bool is_ascendc{false};
    platform_ascendc::SocVersion socVersion;
};

class RollTilingClass : public Ops::Base::TilingBaseClass {
public:
    explicit RollTilingClass(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        Ops::Base::TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    // Private method for check and get shape attrs info
    ge::graphStatus CheckAndGetInputParam();
    ge::graphStatus CheckAttr();

    void MergeAxes(int64_t shape[], int64_t shifts[], int64_t& dimNum);
    void RemoveShapeOne(int64_t shape[], int64_t shifts[], int64_t& dimNum);
    void SplitCore();
    void SplitCoreforSimd();
    void SplitUb(UbParam& ubparam, bool isTail);
    void CalMoveParam();
    void upDateParam(
        int64_t index_, int64_t srcOffset_, int64_t blockCount_, int64_t blockLen_, int64_t srcStride_,
        int64_t dstOffeset_);
    void PrintTiling() const;

    const gert::StorageShape* xShapePtr_ = nullptr;
    const gert::StorageShape* yShapePtr_ = nullptr;
    const gert::ContinuousVector* shiftsListPtr_ = nullptr;
    const gert::ContinuousVector* dimsListPtr_ = nullptr;

    ge::DataType dataType_ = ge::DT_UNDEFINED;
    int64_t dtypeSize_{0};
    int64_t totalEmelents_{0};
    int64_t needCoreNum_{0};
    int64_t perCoreElements_{0};
    int64_t lastCoreElements_{0};
    int64_t basicElements_{0};
    int64_t maxElements_{0};
    int64_t dimNum_{0};
    int64_t UbSplitAxis_{0};
    int64_t blockSplitAxis_{-1};
    int64_t blockFactor_{0};
    int64_t blockCount_{0};
    int64_t blockTailFactor_{0};
    int64_t tilingKey{0};

    int64_t cacheLineSize_{0};
    int64_t vectorSize_{0};

    UbParam mainCoreUbParam_;
    UbParam tailCoreUbParam_;

    int64_t perCorePerElements_{0};
    int64_t perCoreLastElements_{0};

    int64_t shapes_[MAX_DIM_NUM] = {0};
    int64_t strides_[MAX_DIM_NUM] = {0};
    int64_t shifts_[MAX_DIM_NUM] = {0};
    RollTilingData* tilingData_{nullptr};
    MoveParam moveparam;
};

} // namespace optiling

#endif // __ROLL_TILING_H__