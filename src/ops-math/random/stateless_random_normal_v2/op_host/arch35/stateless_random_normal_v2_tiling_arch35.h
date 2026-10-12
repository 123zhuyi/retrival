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
 * \file stateless_random_normal_v2_tiling_arch35.h
 * \brief
 */

#pragma once

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"

namespace optiling {
constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;

BEGIN_TILING_DATA_DEF(StatelessRandomNormalV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, blockTilingSize);
TILING_DATA_FIELD_DEF(uint32_t, tailBlockTilingSize);
TILING_DATA_FIELD_DEF(uint32_t, ubTilingSize);
TILING_DATA_FIELD_DEF(uint32_t, alg);
TILING_DATA_FIELD_DEF_ARR(uint32_t, ALG_KEY_SIZE, key);
TILING_DATA_FIELD_DEF_ARR(uint32_t, ALG_COUNTER_SIZE, counter);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StatelessRandomNormalV2, StatelessRandomNormalV2TilingData)

struct StatelessRandomNormalV2CompileInfo {
    uint64_t aivNum;
    uint64_t ubSize;
};

class StatelessRandomNormalV2Tiling : public Ops::Base::TilingBaseClass {
public:
    explicit StatelessRandomNormalV2Tiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

    enum class Algorithm : int32_t
    {
        RNG_ALG_PHILOX = 1,
        RNG_ALG_THREEFRY = 2,
        RNG_ALG_AUTO_SELECT = 3
    };

protected:
    bool IsCapable() override
    {
        return true;
    }

    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    static constexpr uint32_t CORE_ALIGN_SIZE = 512;
    static constexpr uint32_t BLOCK_SIZE_BYTES = 32;
    static constexpr uint32_t MIN_TILING_SIZE = 256;
    static constexpr uint32_t REGBASE_CCEC_CACHE_SIZE = 8 * 1024;
    static constexpr uint32_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;

    template <typename T1, typename T2>
    inline T1 CeilDiv(const T1 a, const T2 b) const;
    ge::graphStatus GetInputInfo();
    ge::graphStatus GetOutputInfo();
    void BlockTiling();
    ge::graphStatus UbTiling();
    void SetTilingData();

private:
    uint32_t coreNum_ = 0;
    uint32_t ubSize_ = 0;
    ge::DataType outputDtype_ = ge::DT_UNDEFINED;
    uint32_t outputDtypeVal_ = 0;
    uint64_t outputSize_ = 1;
    uint32_t outputDtypeSize_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t blockTilingSize_ = 0;
    uint32_t tailBlockTilingSize_ = 0;
    uint32_t ubTilingSize_ = 0;
    Algorithm alg_ = Algorithm::RNG_ALG_PHILOX;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};
    const char* opName = "StatelessRandomNormalV2";
    StatelessRandomNormalV2TilingData tilingData;
};

} // namespace optiling