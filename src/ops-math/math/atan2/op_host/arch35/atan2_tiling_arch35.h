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
 * \file atan2_tiling_arch35.h
 * \brief atan2_tiling head file
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ATAN2_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ATAN2_TILING_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"

namespace optiling {

class Atan2Tiling : public Ops::Base::TilingBaseClass {
public:
    explicit Atan2Tiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t tilingKey = 0;
    bool CheckDtype(const ge::DataType& input0Dtype, const ge::DataType& input1Dtype,
                    const ge::DataType& outputDtype) const;
};

}  // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADD_TILING_H
