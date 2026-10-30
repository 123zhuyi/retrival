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
 * \file dynamic_stitch_tiling_arch35.h
 * \brief
 */
#ifndef __DYNAMIC_STITCH_TILING_ARCH35_H__
#define __DYNAMIC_STITCH_TILING_ARCH35_H__

#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"
#include "../../op_kernel/arch35/dynamic_stitch_tiling_def.h"
#include "platform/platform_info.h"
#include "op_common/op_host/util/math_util.h"
#include "op_host/tiling_base_util.h"


namespace optiling {
struct DynamicStitchCompileInfo {
    uint64_t blockDim;
    uint64_t ubSize;
};

enum class SliceDivisorType : int64_t
{
    SLICE_ONE = 1,
    SLICE_TWO = 2,
    SLICE_FOUR = 4,
    SLICE_EIGHT = 8
};

class DynamicStitchTilingClass : public Ops::Base::TilingBaseClass
{
public:
    explicit DynamicStitchTilingClass(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
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
        return Ops::Base::IsRegbaseSocVersion(context_);
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
    ge::graphStatus CheckAndGetParam();
    ge::graphStatus CheckAndGetIndiceInputList();
    ge::graphStatus CheckAndGetXInputList();
    ge::graphStatus CheckAndGetOutput();
    ge::graphStatus CheckAttr() const;
    ge::graphStatus CheckShapeAllNonNeg(const gert::Shape& shape) const;
    ge::graphStatus CheckAndGetSliceSize();

    // Private method for handle slice shape
    std::vector<int64_t> GetSliceShapeFromIndiceAndXShape(
        const gert::Shape& indiceShape, const gert::Shape& xShape) const;
    bool IsTwoSliceShapeEqual(const std::vector<int64_t>& sliceShape1, const std::vector<int64_t>& sliceShape2) const;

    // Other private method for tiling
    void AssignDataToEachCore();
    void ClassifySliceType();
    void PrintTiling() const;
    bool IsBigSliceSize() const;

    ge::DataType dataType_ = ge::DT_UNDEFINED;
    int64_t clrBlockNum_{0};
    int64_t clrBlockWsSize_{0};
    int64_t clrTailBlockWsSize_{0};
    int64_t writeBackBlockNum_{0};
    int64_t writeBackBlockSize_{0};
    int64_t writeBackTailBlockSize_{0};
    int64_t usedCoreNum_{0};
    int64_t blockFactor_{0};
    int64_t tailBlockFactor_{0};
    int64_t maxIndex_{0};
    int64_t sliceSize_{0};
    std::vector<int64_t> sliceShape_;
    SliceDivisorType sliceType_ = SliceDivisorType::SLICE_ONE;
    int64_t indicesBufferSize_{0};
    int32_t ubFactor_{0};
    int64_t ubLoopTimes_{0};
    int32_t ubTailFactor_{0};
    int64_t totalTensorSum_{0};
    int64_t totalTensorCnt_{0};
    int64_t indicesStartOffset_[MAX_CORE_CONT] = {0};
    uint16_t tensorStartList_[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList_[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList_[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList_[MAX_CORE_CONT] = {0};
    int64_t tensorCntList_[MAX_LIST_TENSOR_CNT] = {0};
    int64_t tensorCumsumList_[MAX_LIST_TENSOR_CNT + 1] = {0};
    DynamicStitchTilingData* tilingData_{nullptr};
};

} // namespace optiling

#endif // __DYNAMIC_STITCH_TILING_ARCH35_H__
