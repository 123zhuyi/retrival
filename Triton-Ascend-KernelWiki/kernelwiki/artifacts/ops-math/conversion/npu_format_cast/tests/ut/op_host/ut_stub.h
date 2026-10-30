/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gmock/gmock.h>
#include <limits.h>
#include <map>
#include <stdlib.h>
#include "securec.h"
#include "runtime/base.h"
#include "runtime/kernel.h"
#include "runtime/rts/rts_kernel.h"
#include "runtime/mem.h"
#include "runtime/context.h"
#include "runtime/stream.h"

using namespace std;
class UtStub {
public:
    // runtime
    virtual rtError_t rtGetSocVersion(char* version, const uint32_t maxLen);
    virtual rtError_t rtGetDeviceInfo(uint32_t device_id, int32_t module_type, int32_t info_type, int64_t* val);
    virtual rtError_t rtCtxGetDevice(int32_t* device);
    virtual rtError_t rtGetDevice(int32_t* device);
    virtual rtError_t rtGetDevicePhyIdByIndex(uint32_t device_id, uint32_t* phy_id);
    virtual rtError_t rtAicpuKernelLaunchExWithArgs(
        const uint32_t kernelType, const char* const opName, const uint32_t numBlocks, const rtAicpuArgsEx_t* argsInfo,
        rtSmDesc_t* const smDesc, const rtStream_t stm, const uint32_t flags);
    virtual rtError_t rtsBinaryLoadFromFile(
        const char* const binPath, const rtLoadBinaryConfig_t* const optionalCfg, rtBinHandle* binHandle);
    virtual rtError_t rtsFuncGetByName(const rtBinHandle binHandle, const char* kernelName, rtFuncHandle* funcHandle);
    virtual rtError_t rtsLaunchCpuKernel(
        const rtFuncHandle funcHandle, const uint32_t numBlocks, rtStream_t st, const rtKernelLaunchCfg_t* cfg,
        rtCpuKernelArgs_t* argsInfo);
    // PlatformInfoManager
    virtual string GetSoFilePath();
};

class UtMock : public UtStub {
public:
    UtMock()
    {}
    static UtMock& GetInstance()
    {
        static UtMock instance;
        return instance;
    }
    void SetAscendPath(const char* p)
    {
        ascend_latest_.assign(RealPath(string(p)));
    }
    void SetSocVersion(const string& full_soc_version)
    {
        full_soc_version_.assign(full_soc_version);
    }
    string GetSocVersion() const
    {
        return full_soc_version_;
    }
    bool Delegated()
    {
        return delegated_;
    }

    void DelegateToFake()
    {
        EXPECT_CALL(*this, rtGetSocVersion).WillRepeatedly([this](char* version, const uint32_t maxLen) {
            memcpy_s(version, maxLen, full_soc_version_.c_str(), full_soc_version_.size() + 1);
            return RT_ERROR_NONE;
        });
        EXPECT_CALL(*this, rtGetDeviceInfo)
            .WillRepeatedly(
                [this]([[maybe_unused]] uint32_t device_id, int32_t module_type, int32_t info_type, int64_t* val) {
                    static map<string, map<string, int64_t>> device_info_map{
                        {"Ascend910A", {{"ai_core_cnt", 32}, {"vec_core_cnt", 0}}},
                        {"Ascend910B2", {{"ai_core_cnt", 24}, {"vec_core_cnt", 48}}},
                        {"Ascend310P1", {{"ai_core_cnt", 10}, {"vec_core_cnt", 8}}},
                        {"Ascend910_9382", {{"ai_core_cnt", 24}, {"vec_core_cnt", 48}}},
                        {"Ascend310B1", {{"ai_core_cnt", 1}, {"vec_core_cnt", 1}}},
                        {"Ascend610Lite", {{"ai_core_cnt", 4}, {"vec_core_cnt", 4}}},
                        {"Ascend950PR_9589", {{"ai_core_cnt", 32}, {"vec_core_cnt", 64}}},
                    };

                    if (info_type == 3) {       // core num
                        if (module_type == 4) { // ai core
                            *val = device_info_map[full_soc_version_]["ai_core_cnt"];
                        } else if (module_type == 7) { // vector core
                            *val = device_info_map[full_soc_version_]["vec_core_cnt"];
                        }
                    }
                    return RT_ERROR_NONE;
                });
        EXPECT_CALL(*this, rtCtxGetDevice).WillRepeatedly([this](int32_t* device) {
            *device = device_id_;
            return RT_ERROR_NONE;
        });
        EXPECT_CALL(*this, rtGetDevice).WillRepeatedly([this](int32_t* device) {
            *device = device_id_;
            return RT_ERROR_NONE;
        });
        EXPECT_CALL(*this, rtGetDevicePhyIdByIndex).WillRepeatedly([this](uint32_t device_id, uint32_t* phy_id) {
            *phy_id = device_id_;
            return RT_ERROR_NONE;
        });
        EXPECT_CALL(*this, rtAicpuKernelLaunchExWithArgs)
            .WillRepeatedly([this](
                                [[maybe_unused]] const uint32_t kernelType, [[maybe_unused]] const char* const opName,
                                [[maybe_unused]] const uint32_t numBlocks,
                                [[maybe_unused]] const rtAicpuArgsEx_t* argsInfo,
                                [[maybe_unused]] rtSmDesc_t* const smDesc, [[maybe_unused]] const rtStream_t stm,
                                [[maybe_unused]] const uint32_t flags) {
                cout << "[WARNING] AiCpu kernel launch is not supported in simulator yet !!!" << endl;
                setenv("NEED_AICPU_SIMULATOR", "1", 1);
                return 1;
            });
        EXPECT_CALL(*this, GetSoFilePath).WillRepeatedly([this]() { return ascend_latest_ + "/compiler/lib64/"; });
        EXPECT_CALL(*this, rtsBinaryLoadFromFile)
            .WillRepeatedly(
                [this](
                    const char* const binPath, const rtLoadBinaryConfig_t* const optionalCfg, rtBinHandle* binHandle) {
                    rtBinHandle tmp_binHandle = nullptr;
                    *binHandle = &tmp_binHandle;
                    return RT_ERROR_NONE;
                });
        EXPECT_CALL(*this, rtsFuncGetByName)
            .WillRepeatedly([this](const rtBinHandle binHandle, const char* kernelName, rtFuncHandle* funcHandle) {
                rtFuncHandle tmp_funcHandle = nullptr;
                *funcHandle = &tmp_funcHandle;
                return RT_ERROR_NONE;
            });
        EXPECT_CALL(*this, rtsLaunchCpuKernel)
            .WillRepeatedly([this](
                                const rtFuncHandle funcHandle, const uint32_t numBlocks, rtStream_t st,
                                const rtKernelLaunchCfg_t* cfg, rtCpuKernelArgs_t* argsInfo) { return RT_ERROR_NONE; });
        delegated_ = true;
    }

    // runtime
    MOCK_METHOD(rtError_t, rtGetSocVersion, (char*, const uint32_t));
    MOCK_METHOD(rtError_t, rtGetDeviceInfo, (uint32_t, int32_t, int32_t, int64_t*));
    MOCK_METHOD(rtError_t, rtCtxGetDevice, (int32_t*));
    MOCK_METHOD(rtError_t, rtGetDevice, (int32_t*));
    MOCK_METHOD(rtError_t, rtGetDevicePhyIdByIndex, (uint32_t, uint32_t*));
    MOCK_METHOD(
        rtError_t, rtAicpuKernelLaunchExWithArgs,
        (const uint32_t, const char* const, const uint32_t, const rtAicpuArgsEx_t*, rtSmDesc_t* const, const rtStream_t,
         const uint32_t));
    MOCK_METHOD(
        rtError_t, rtsBinaryLoadFromFile,
        (const char* const binPath, const rtLoadBinaryConfig_t* const optionalCfg, rtBinHandle* binHandle));
    MOCK_METHOD(
        rtError_t, rtsFuncGetByName, (const rtBinHandle binHandle, const char* kernelName, rtFuncHandle* funcHandle));
    MOCK_METHOD(
        rtError_t, rtsLaunchCpuKernel,
        (const rtFuncHandle funcHandle, const uint32_t numBlocks, rtStream_t st, const rtKernelLaunchCfg_t* cfg,
         rtCpuKernelArgs_t* argsInfo));
    // PlatformInfoManager
    MOCK_METHOD(string, GetSoFilePath, ());

private:
    string RealPath(const string& path)
    {
        // Nullptr is returned when the path does not exist or there is no permission
        // Return absolute path when path is accessible
        string res;
        char resolved_path[PATH_MAX] = {};
        const char* p = realpath(path.c_str(), resolved_path);
        if (p == nullptr) {
            res.assign("");
        } else {
            res.assign(resolved_path);
        }
        return res;
    }

private:
    string ascend_latest_{"/usr/local/Ascend/latest"};
    string full_soc_version_{"Ascend910A"};
    int32_t device_id_{0};
    bool delegated_{false};
};
