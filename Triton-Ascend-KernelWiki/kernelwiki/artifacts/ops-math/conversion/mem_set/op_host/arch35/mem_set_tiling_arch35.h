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
 * \file mem_set_tiling_arch35.h
 * \brief tiling for mem set
 */

#ifndef AIR_CXX_RUNTIME_OP_IMPL_MEM_SET_H
#define AIR_CXX_RUNTIME_OP_IMPL_MEM_SET_H

#include <vector>
#include "op_host/tiling_base_class.h"
#include "conversion/mem_set/op_kernel/arch35/mem_set_struct.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"


namespace optiling {

struct MemSetCompileInfoArch35 {
    uint32_t coreNum = 0;
    uint32_t ubSize = 0;
};

class MemSetTilingClass : public Ops::Base::TilingBaseClass {
public:
    explicit MemSetTilingClass(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
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
    void AllocTilingStruct();
    ge::graphStatus SetShapeAttrsInfo(bool isGE);
    template <uint16_t Count>
    void PostDo();
    template <uint16_t Count>
    void CheckTilingData(MemSetTilingData<Count>* tilingDataPostAtr);

private:
    void* tilingDataPostAtr_ = nullptr;
    std::vector<int64_t> perCoreSizes_;
    std::vector<int64_t> lastCoreSizes_;
    std::vector<int64_t> intValue_;
    std::vector<float> floatValue_;
    std::vector<int16_t> listType_;
    std::vector<int16_t> useCore_;
    std::vector<int64_t> sizes_;
    uint16_t cacheLineSize_ = 0;
    uint16_t needCore_ = 0;
    int halfUbSize_ = 0;
    uint16_t inputCount_ = 0;
    uint16_t TilingKey_ = 0;
    bool isDynamic_ = false;
};

} // namespace optiling

namespace ops {
class AtomicCleanTilingContext : public gert::TilingContext {
public:
    /**
     * Get workspaceSize list
     * @return workspaceSize list
     */
    const gert::ContinuousVector* GetCleanWorkspaceSizes() const
    {
        return GetInputPointer<gert::ContinuousVector>(0);
    }

    /**
     * Get the size of the output memory to be cleared by the output index
     * @param index output index
     * @return Size of output memory to be cleaned
     */
    uint64_t GetCleanOutputSize(size_t index) const
    {
        return GetInputValue<uint64_t>(index + 1U);
    }
};
static_assert(
    std::is_standard_layout<AtomicCleanTilingContext>::value, "The class AtomicCleanTilingContext must be a POD");
} // namespace ops
#endif