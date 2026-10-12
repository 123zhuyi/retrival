/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_arch_helper.cpp
 * \brief add aicore config by npu arch
 */

#include "op_common/op_host/util/op_arch_helper.h"

#include <map>
#include <string>
#include <vector>

namespace opbase {
namespace {
const std::map<NpuArch, std::vector<std::string>> g_archToSocNames = {
    {Ops::Base::DAV_1001, {"ascend910"}},
    {Ops::Base::DAV_2002, {"ascend310p"}},
    {Ops::Base::DAV_2201, {"ascend910b", "ascend910_93"}},
    {Ops::Base::DAV_3002, {"ascend310b"}},
    {Ops::Base::DAV_3102, {"ascend610lite"}},
    {Ops::Base::DAV_3510, {"ascend950"}},
};
} // namespace

void ArchConfigHelper::AddConfigByArch(ops::OpAICoreDef& aicore, NpuArch arch, ops::OpAICoreConfig& aicoreConfig)
{
    auto iter = g_archToSocNames.find(arch);
    if (iter == g_archToSocNames.end()) {
        return;
    }
    for (const auto& socName : iter->second) {
        aicore.AddConfig(socName.c_str(), aicoreConfig);
    }
}

void ArchConfigHelper::AddConfigByArch(ops::OpAICoreDef& aicore, NpuArch arch)
{
    auto iter = g_archToSocNames.find(arch);
    if (iter == g_archToSocNames.end()) {
        return;
    }
    for (const auto& socName : iter->second) {
        aicore.AddConfig(socName.c_str());
    }
}
} // namespace opbase
