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
 * \file add_lora_tiling.h
 * \brief
 */
#ifndef ADD_LORA_TILING_H
#define ADD_LORA_TILING_H
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"

constexpr uint64_t WORKSIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t SOC_TILINGKEY_OFFSET = 100000UL;
constexpr uint32_t VECTORDOUBLE = 2;
constexpr uint32_t CUBESIZE = 16;

constexpr uint32_t Y_INPUT_INDEX = 0;
constexpr uint32_t X_INPUT_INDEX = 1;
constexpr uint32_t WEIGHTB_INPUT_INDEX = 2;
constexpr uint32_t INDICES_INPUT_INDEX = 3;
constexpr uint32_t WEIGHTA_INPUT_INDEX = 4;
constexpr uint32_t LAYER_IDX_ATTR_INDEX = 0;
constexpr uint32_t SCALE_ATTR_INDEX = 1;
constexpr uint32_t Y_OFFSET_ATTR_INDEX = 2;
constexpr uint32_t Y_SLICE_SIZE_ATTR_INDEX = 3;
constexpr uint32_t OUTPUT_INDEX = 0;

constexpr uint32_t DIM_INDEX_ZERO = 0;
constexpr uint32_t DIM_INDEX_ONE = 1;
constexpr uint32_t DIM_INDEX_TWO = 2;
constexpr uint32_t DIM_INDEX_THREE = 3;

constexpr uint32_t MAX_BATCH_SIZE = 65536;
constexpr uint32_t MAX_RANK_SIZE = 128;
constexpr uint32_t MAX_WEIGHT_NUM = 32;
constexpr uint32_t MAX_LAYER_NUM = 32;
constexpr uint32_t MAX_HIDDEN_SIZE = 131072;
constexpr uint32_t BATCH_MODE = 1;
constexpr int64_t Y_SLICE_SIZE_DEFAULT = -1;

namespace optiling {
enum class TilingKeyInfo : int {
    KEY_DEFAULT_SCENE = 0,
    KEY_SPARSE_SCENE = 1,
    KEY_BGMV_SCENE = 10
};

enum class SocVersionKey : int {
    KEY_SOC_VERSION_910 = 1,
    KEY_SOC_VERSION_310 = 2
};

BEGIN_TILING_DATA_DEF(AddLoraTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, layer);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, H1);
TILING_DATA_FIELD_DEF(uint32_t, H2);
TILING_DATA_FIELD_DEF(uint32_t, R);
TILING_DATA_FIELD_DEF(uint32_t, wBatch);
TILING_DATA_FIELD_DEF(uint32_t, layer_idx);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(uint32_t, y_offset);
TILING_DATA_FIELD_DEF(uint32_t, y_slice_size);
TILING_DATA_FIELD_DEF(uint32_t, taskNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, H2PerCore);
TILING_DATA_FIELD_DEF(uint32_t, addLoraFlag);
TILING_DATA_FIELD_DEF(uint32_t, y_column);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddLora, AddLoraTilingData)

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct Tiling4AddLoraCompileInfo {
};

class AddLoraTiling : public Ops::Base::TilingBaseClass
{
public:
    explicit AddLoraTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {
        Reset();
    }
    ~AddLoraTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus CheckInputDataType();
    ge::graphStatus CheckInputTensorShape();
    ge::graphStatus CheckOptionalInputTensorShape();
    ge::graphStatus CheckInputAttrLimit();
    ge::graphStatus CheckYSliceAttrs();
    ge::graphStatus CheckInputLimits();
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

    void Reset();

    void PrintTilingData();
    RequiredParaInfo yParaInfo_ = {nullptr, nullptr};
    RequiredParaInfo xParaInfo_ = {nullptr, nullptr};
    RequiredParaInfo weightBParaInfo_ = {nullptr, nullptr};
    RequiredParaInfo indicesParaInfo_ = {nullptr, nullptr};
    OptionalParaInfo weightAParaInfo_ = {nullptr, nullptr};
    RequiredParaInfo yOutParaInfo_ = {nullptr, nullptr};
    const gert::RuntimeAttrs *attrs_ = nullptr;
    AddLoraTilingData tilingData_;
    bool isAscend310P_ = 0;
    std::uint32_t coreNum_ = 64;
    uint32_t batchSize_ = 2;
    uint32_t H3_ = 16;
    uint32_t H1_ = 16;
    uint32_t wBatch_ = 2;
    uint32_t H2_ = 16;
    uint32_t R_ = 16;
    uint32_t L_ = 16;
    bool addLoraFlag_ = 0;
    uint32_t layerIdx_ = 0;
    float scale_ = 2.0;
    uint32_t yOffset_ = 0;
    uint32_t ySliceSize_ = 16;
};

} // namespace optiling
#endif // ADD_LORA_TILING_H