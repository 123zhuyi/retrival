/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform.h"
#include <iostream>

namespace op {

SocVersion g_socVersion = SocVersion::ASCEND910B;
// 显式设置的 NpuArch 覆盖值，DAV_RESV 表示未设置，此时回退到由 g_socVersion 推导。
NpuArch g_npuArchOverride = NpuArch::DAV_RESV;
PlatformInfo* g_platformInfo = new PlatformInfo();

bool PlatformInfo::CheckSupport(SocSpec socSpec, SocSpecAbility ability) const { return true; }

PlatformInfo::~PlatformInfo()
{
    if (impl_ != nullptr) {
        delete impl_;
    }
}

bool PlatformInfo::Valid() const { return valid_; }

void PlatformInfo::SetPlatformImpl(PlatformInfoImpl* impl)
{
    impl_ = impl;
    valid_ = true;
}

SocVersion PlatformInfo::GetSocVersion() const { return g_socVersion; }

const std::string& PlatformInfo::GetSocLongVersion() const { return ""; }

int32_t PlatformInfo::GetDeviceId() const { return 0; }

int64_t PlatformInfo::GetBlockSize() const { return 0; }

uint32_t PlatformInfo::GetCubeCoreNum() const { return 0; }

uint32_t PlatformInfo::GetVectorCoreNum() const { return 0; }

bool PlatformInfo::GetFftsPlusMode() const { return true; }

fe::PlatFormInfos* PlatformInfo::GetPlatformInfos() const { return nullptr; }

const PlatformInfo& GetCurrentPlatformInfo() { return *g_platformInfo; }

NpuArch PlatformInfo::GetCurNpuArch() const
{
    // 优先使用显式设置的 NpuArch，未设置时再由 SocVersion 推导，保证向后兼容。
    if (g_npuArchOverride != NpuArch::DAV_RESV) {
        return g_npuArchOverride;
    }
    static const std::map<SocVersion, NpuArch> soc2ArchMap = {
        {SocVersion::ASCEND910, NpuArch::DAV_1001},    {SocVersion::ASCEND910B, NpuArch::DAV_2201},
        {SocVersion::ASCEND910_93, NpuArch::DAV_2201}, {SocVersion::ASCEND950, NpuArch::DAV_3510},
        {SocVersion::ASCEND310P, NpuArch::DAV_2002},   {SocVersion::ASCEND310B, NpuArch::DAV_3002},
        {SocVersion::ASCEND610LITE, NpuArch::DAV_3102}};
    const auto it = soc2ArchMap.find(g_socVersion);
    if (it != soc2ArchMap.end()) {
        return it->second;
    }
    std::cout << "Error, Unsupported SocVersion, plz modyfy this function" << std::endl;
    return NpuArch::DAV_RESV;
}

ge::AscendString ToString(SocVersion socVersion)
{
    static const std::map<SocVersion, std::string> kSocVersionMap = {
        {SocVersion::ASCEND910, "Ascend910"},
        {SocVersion::ASCEND910B, "Ascend910B"},
        {SocVersion::ASCEND910_93, "Ascend910_93"},
        {SocVersion::ASCEND950, "Ascend950"},
        {SocVersion::ASCEND910E, "Ascend910E"},
        {SocVersion::ASCEND310, "Ascend310"},
        {SocVersion::ASCEND310P, "Ascend310P"},
        {SocVersion::ASCEND310B, "Ascend310B"},
        {SocVersion::ASCEND310C, "Ascend310C"},
        {SocVersion::ASCEND610LITE, "Ascend610LITE"},
        {SocVersion::KIRINX90, "KirinX90"},
        {SocVersion::KIRIN9030, "Kirin9030"},
        {SocVersion::RESERVED_VERSION, "UnknowSocVersion"},
    };
    static const std::string reserved("UnknowSocVersion");
    const auto it = kSocVersionMap.find(socVersion);
    if (it != kSocVersionMap.end()) {
        return ge::AscendString((it->second).c_str());
    } else {
        return ge::AscendString(reserved.c_str());
    }
}

void SetPlatformSocVersion(SocVersion socVersion)
{
    g_socVersion = socVersion;
    // 设置 SocVersion 时复位 NpuArch 覆盖值，避免两者相互污染。
    g_npuArchOverride = NpuArch::DAV_RESV;
}

void SetPlatformNpuArch(NpuArch npuArch)
{
    g_npuArchOverride = npuArch;
    // 顺带把 SocVersion 设为该 arch 对应的代表 Soc，保证 GetSocVersion 与 GetCurNpuArch 一致。
    // 注意：DAV_2201 同时对应 ASCEND910B / ASCEND910_93，此处取 ASCEND910B 作为代表值。
    static const std::map<NpuArch, SocVersion> arch2SocMap = {
        {NpuArch::DAV_1001, SocVersion::ASCEND910},     {NpuArch::DAV_2201, SocVersion::ASCEND910B},
        {NpuArch::DAV_3510, SocVersion::ASCEND950},     {NpuArch::DAV_2002, SocVersion::ASCEND310P},
        {NpuArch::DAV_3002, SocVersion::ASCEND310B},    {NpuArch::DAV_3102, SocVersion::ASCEND610LITE}};
    const auto it = arch2SocMap.find(npuArch);
    if (it != arch2SocMap.end()) {
        g_socVersion = it->second;
    }
}

} // namespace op