/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_bernoulli.h"

namespace {
constexpr uint8_t GUARD = 0xA5U;
constexpr int64_t DEFAULT_SEED = 20260725;
constexpr int64_t DEFAULT_OFFSET = 4;
constexpr uint64_t MAX_ALIAS_WORKSPACE_BYTES = 4096;

enum class ValueKind {
    FP16,
    FP32,
    FP64,
    BF16,
    U8,
    I8,
    I16,
    I32,
    I64,
    BOOL,
};

struct DType {
    const char* name;
    aclDataType aclType;
    size_t bytes;
    ValueKind kind;
};

constexpr std::array<DType, 10> DTYPES{{
    {"fp16", ACL_FLOAT16, 2, ValueKind::FP16},
    {"fp32", ACL_FLOAT, 4, ValueKind::FP32},
    {"fp64", ACL_DOUBLE, 8, ValueKind::FP64},
    {"bf16", ACL_BF16, 2, ValueKind::BF16},
    {"uint8", ACL_UINT8, 1, ValueKind::U8},
    {"int8", ACL_INT8, 1, ValueKind::I8},
    {"int16", ACL_INT16, 2, ValueKind::I16},
    {"int32", ACL_INT32, 4, ValueKind::I32},
    {"int64", ACL_INT64, 8, ValueKind::I64},
    {"bool", ACL_BOOL, 1, ValueKind::BOOL},
}};

constexpr std::array<DType, 4> PROB_DTYPES{{
    {"fp16", ACL_FLOAT16, 2, ValueKind::FP16},
    {"fp32", ACL_FLOAT, 4, ValueKind::FP32},
    {"fp64", ACL_DOUBLE, 8, ValueKind::FP64},
    {"bf16", ACL_BF16, 2, ValueKind::BF16},
}};

std::string RecentError()
{
    const char* message = aclGetRecentErrMsg();
    return message == nullptr ? std::string{} : std::string(message);
}

void Check(aclError status, const char* operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status) + " " + RecentError());
    }
}

size_t Elements(const std::vector<int64_t>& shape)
{
    size_t elements = 1;
    for (int64_t dim : shape) {
        if (dim == 0) {
            return 0;
        }
        if (dim < 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / elements) {
            throw std::runtime_error("invalid or overflowing shape");
        }
        elements *= static_cast<size_t>(dim);
    }
    return elements;
}

std::vector<int64_t> DenseStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    int64_t running = 1;
    for (size_t reverse = shape.size(); reverse > 0; --reverse) {
        const size_t index = reverse - 1;
        strides[index] = running;
        running *= shape[index];
    }
    return strides;
}

size_t RequiredStorage(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, int64_t offset)
{
    if (Elements(shape) == 0) {
        return static_cast<size_t>(offset);
    }
    size_t maximum = static_cast<size_t>(offset);
    for (size_t index = 0; index < shape.size(); ++index) {
        maximum += static_cast<size_t>(shape[index] - 1) * static_cast<size_t>(strides[index]);
    }
    return maximum + 1;
}

std::vector<size_t> LogicalIndices(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                                   int64_t offset)
{
    std::vector<size_t> indices(Elements(shape), static_cast<size_t>(offset));
    for (size_t logical = 0; logical < indices.size(); ++logical) {
        size_t remaining = logical;
        size_t storage = static_cast<size_t>(offset);
        for (size_t reverse = shape.size(); reverse > 0; --reverse) {
            const size_t dimension = reverse - 1;
            const size_t extent = static_cast<size_t>(shape[dimension]);
            const size_t coordinate = extent == 0 ? 0 : remaining % extent;
            remaining = extent == 0 ? 0 : remaining / extent;
            storage += coordinate * static_cast<size_t>(strides[dimension]);
        }
        indices[logical] = storage;
    }
    return indices;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x007FFFFFU;
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x00800000U;
        return static_cast<uint16_t>(sign | (mantissa >> static_cast<uint32_t>(14 - exponent)));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    mantissa = (mantissa + 0x00001000U) >> 13U;
    if (mantissa == 0x400U) {
        mantissa = 0;
        ++exponent;
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) | (mantissa & 0x03FFU));
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t bias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + bias) >> 16U);
}

class Runtime {
public:
    explicit Runtime(int32_t device) : device_(device)
    {
        Check(aclInit(nullptr), "aclInit");
        initialized_ = true;
        Check(aclrtSetDevice(device_), "aclrtSetDevice");
        deviceSet_ = true;
        Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    ~Runtime()
    {
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            (void)aclrtResetDevice(device_);
        }
        if (initialized_) {
            (void)aclFinalize();
        }
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t device_ = 0;
    bool initialized_ = false;
    bool deviceSet_ = false;
    aclrtStream stream_ = nullptr;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes)
    {
        capacity_ = std::max<size_t>(bytes, 1);
        Check(aclrtMalloc(&data_, capacity_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            (void)aclrtFree(data_);
        }
    }

    void Fill(uint8_t value)
    {
        std::vector<uint8_t> host(capacity_, value);
        Check(aclrtMemcpy(data_, capacity_, host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)");
    }

    std::vector<uint8_t> CopyToHost(size_t bytes) const
    {
        std::vector<uint8_t> host(bytes);
        if (bytes > 0) {
            Check(aclrtMemcpy(host.data(), bytes, data_, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H)");
        }
        return host;
    }

    void* Data() const { return data_; }

private:
    void* data_ = nullptr;
    size_t capacity_ = 0;
};

class Tensor {
public:
    Tensor(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, int64_t offset,
           const std::vector<int64_t>& storageShape, aclDataType dtype, void* data)
    {
        tensor_ = aclCreateTensor(shape.empty() ? nullptr : shape.data(), shape.size(), dtype,
                                  strides.empty() ? nullptr : strides.data(), offset, ACL_FORMAT_ND,
                                  storageShape.empty() ? nullptr : storageShape.data(), storageShape.size(), data);
        if (tensor_ == nullptr) {
            throw std::runtime_error("aclCreateTensor failed: " + RecentError());
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    ~Tensor()
    {
        if (tensor_ != nullptr) {
            (void)aclDestroyTensor(tensor_);
        }
    }

    aclTensor* Get() const { return tensor_; }

private:
    aclTensor* tensor_ = nullptr;
};

class Scalar {
public:
    Scalar(double probability, const DType& dtype)
    {
        if (dtype.kind == ValueKind::FP16) {
            const uint16_t value = FloatToHalf(static_cast<float>(probability));
            std::memcpy(storage_.data(), &value, sizeof(value));
        } else if (dtype.kind == ValueKind::FP32) {
            const float value = static_cast<float>(probability);
            std::memcpy(storage_.data(), &value, sizeof(value));
        } else if (dtype.kind == ValueKind::FP64) {
            std::memcpy(storage_.data(), &probability, sizeof(probability));
        } else if (dtype.kind == ValueKind::BF16) {
            const uint16_t value = FloatToBFloat16(static_cast<float>(probability));
            std::memcpy(storage_.data(), &value, sizeof(value));
        } else {
            throw std::runtime_error("unsupported probability dtype");
        }
        scalar_ = aclCreateScalar(storage_.data(), dtype.aclType);
        if (scalar_ == nullptr) {
            throw std::runtime_error("aclCreateScalar failed: " + RecentError());
        }
    }

    Scalar(const Scalar&) = delete;
    Scalar& operator=(const Scalar&) = delete;

    ~Scalar()
    {
        if (scalar_ != nullptr) {
            (void)aclDestroyScalar(scalar_);
        }
    }

    aclScalar* Get() const { return scalar_; }

private:
    alignas(8) std::array<uint8_t, 8> storage_{};
    aclScalar* scalar_ = nullptr;
};

struct Case {
    const DType* dtype = nullptr;
    const DType* probDtype = &PROB_DTYPES[1];
    std::vector<int64_t> shape{129};
    std::vector<int64_t> strides;
    int64_t storageOffset = 0;
    size_t storageElements = 0;
    double probability = 0.37;
    int64_t seed = DEFAULT_SEED;
    int64_t offset = DEFAULT_OFFSET;
    bool inplace = false;
};

struct Result {
    std::vector<uint8_t> logical;
    size_t zeros = 0;
    size_t ones = 0;
    size_t other = 0;
    size_t changedGuardBytes = 0;
    uint64_t workspaceBytes = 0;
};

struct AliasThreshold {
    const char* widthName;
    const DType* dtype;
    int64_t fallbackElements;
    int64_t aliasElements;
};

const std::array<AliasThreshold, 4> ALIAS_THRESHOLDS{{
    {"byte1", &DTYPES[4], 15, 16},
    {"byte2", &DTYPES[0], 7, 8},
    {"byte4", &DTYPES[1], 3, 4},
    {"byte8", &DTYPES[2], 1, 2},
}};

template <typename T>
T Read(const uint8_t* data)
{
    T value{};
    std::memcpy(&value, data, sizeof(value));
    return value;
}

int Classify(const uint8_t* data, ValueKind kind)
{
    switch (kind) {
        case ValueKind::FP16: {
            const uint16_t value = Read<uint16_t>(data);
            return value == 0U ? 0 : (value == 0x3C00U ? 1 : 2);
        }
        case ValueKind::FP32: {
            const float value = Read<float>(data);
            return value == 0.0F ? 0 : (value == 1.0F ? 1 : 2);
        }
        case ValueKind::FP64: {
            const double value = Read<double>(data);
            return value == 0.0 ? 0 : (value == 1.0 ? 1 : 2);
        }
        case ValueKind::BF16: {
            const uint16_t value = Read<uint16_t>(data);
            return value == 0U ? 0 : (value == 0x3F80U ? 1 : 2);
        }
        case ValueKind::U8:
        case ValueKind::BOOL: {
            const uint8_t value = Read<uint8_t>(data);
            return value == 0U ? 0 : (value == 1U ? 1 : 2);
        }
        case ValueKind::I8: {
            const int8_t value = Read<int8_t>(data);
            return value == 0 ? 0 : (value == 1 ? 1 : 2);
        }
        case ValueKind::I16: {
            const int16_t value = Read<int16_t>(data);
            return value == 0 ? 0 : (value == 1 ? 1 : 2);
        }
        case ValueKind::I32: {
            const int32_t value = Read<int32_t>(data);
            return value == 0 ? 0 : (value == 1 ? 1 : 2);
        }
        case ValueKind::I64: {
            const int64_t value = Read<int64_t>(data);
            return value == 0 ? 0 : (value == 1 ? 1 : 2);
        }
    }
    return 2;
}

Result Run(const Case& test, Runtime& runtime)
{
    if (test.dtype == nullptr || test.probDtype == nullptr) {
        throw std::runtime_error("test dtype is null");
    }
    const std::vector<int64_t> strides = test.strides.empty() ? DenseStrides(test.shape) : test.strides;
    const size_t required = RequiredStorage(test.shape, strides, test.storageOffset);
    const size_t storageElements = test.storageElements == 0 ? required : test.storageElements;
    if (storageElements < required) {
        throw std::runtime_error("storage is smaller than the view span");
    }
    const size_t storageBytes = storageElements * test.dtype->bytes;
    const bool dense = test.strides.empty() && test.storageOffset == 0 && test.storageElements == 0;
    const std::vector<int64_t> storageShape = dense ? test.shape :
                                                      std::vector<int64_t>{static_cast<int64_t>(storageElements)};
    const std::vector<size_t> logicalIndices = LogicalIndices(test.shape, strides, test.storageOffset);

    DeviceBuffer selfBuffer(storageBytes);
    DeviceBuffer outBuffer(storageBytes);
    selfBuffer.Fill(GUARD);
    outBuffer.Fill(GUARD);
    Tensor self(test.shape, strides, test.storageOffset, storageShape, test.dtype->aclType, selfBuffer.Data());
    Tensor out(test.shape, strides, test.storageOffset, storageShape, test.dtype->aclType, outBuffer.Data());
    Scalar probability(test.probability, *test.probDtype);

    uint64_t workspaceBytes = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus status = ACL_SUCCESS;
    if (test.inplace) {
        status = aclnnInplaceBernoulliGetWorkspaceSize(self.Get(), probability.Get(), test.seed, test.offset,
                                                       &workspaceBytes, &executor);
    } else {
        status = aclnnBernoulliGetWorkspaceSize(self.Get(), probability.Get(), test.seed, test.offset, out.Get(),
                                                &workspaceBytes, &executor);
    }
    if (status != ACL_SUCCESS || executor == nullptr) {
        if (executor != nullptr) {
            (void)aclDestroyAclOpExecutor(executor);
        }
        throw std::runtime_error("GetWorkspaceSize failed: " + std::to_string(status) + " " + RecentError());
    }

    DeviceBuffer workspace(static_cast<size_t>(workspaceBytes));
    if (test.inplace) {
        status = aclnnInplaceBernoulli(workspaceBytes == 0 ? nullptr : workspace.Data(), workspaceBytes, executor,
                                       runtime.Stream());
    } else {
        status = aclnnBernoulli(workspaceBytes == 0 ? nullptr : workspace.Data(), workspaceBytes, executor,
                                runtime.Stream());
    }
    if (status != ACL_SUCCESS) {
        throw std::runtime_error("aclnnBernoulli phase 2 failed: " + std::to_string(status) + " " + RecentError());
    }
    Check(aclrtSynchronizeStream(runtime.Stream()), "aclrtSynchronizeStream");

    const std::vector<uint8_t> storage = (test.inplace ? selfBuffer : outBuffer).CopyToHost(storageBytes);
    Result result;
    result.workspaceBytes = workspaceBytes;
    result.logical.resize(logicalIndices.size() * test.dtype->bytes);
    std::vector<uint8_t> logicalMask(storageBytes, 0U);
    for (size_t logical = 0; logical < logicalIndices.size(); ++logical) {
        const size_t storageByte = logicalIndices[logical] * test.dtype->bytes;
        const size_t logicalByte = logical * test.dtype->bytes;
        std::memcpy(result.logical.data() + logicalByte, storage.data() + storageByte, test.dtype->bytes);
        std::fill_n(logicalMask.begin() + static_cast<std::ptrdiff_t>(storageByte), test.dtype->bytes, 1U);
        const int category = Classify(result.logical.data() + logicalByte, test.dtype->kind);
        result.zeros += category == 0 ? 1U : 0U;
        result.ones += category == 1 ? 1U : 0U;
        result.other += category == 2 ? 1U : 0U;
    }
    for (size_t byte = 0; byte < storage.size(); ++byte) {
        if (logicalMask[byte] == 0U && storage[byte] != GUARD) {
            ++result.changedGuardBytes;
        }
    }
    return result;
}

uint64_t Fnv1a(const std::vector<uint8_t>& bytes)
{
    uint64_t hash = 14695981039346656037ULL;
    for (uint8_t byte : bytes) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool InvalidOffsetIsRejected(Runtime& runtime)
{
    const DType& dtype = DTYPES[1];
    const std::vector<int64_t> shape{16};
    const std::vector<int64_t> strides{1};
    DeviceBuffer selfBuffer(16 * dtype.bytes);
    DeviceBuffer outBuffer(16 * dtype.bytes);
    Tensor self(shape, strides, 0, shape, dtype.aclType, selfBuffer.Data());
    Tensor out(shape, strides, 0, shape, dtype.aclType, outBuffer.Data());
    Scalar probability(0.5, PROB_DTYPES[1]);
    uint64_t workspaceBytes = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status = aclnnBernoulliGetWorkspaceSize(self.Get(), probability.Get(), DEFAULT_SEED, 2, out.Get(),
                                                              &workspaceBytes, &executor);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
    (void)runtime;
    return status != ACL_SUCCESS;
}

bool InvalidProbabilityIsRejected(double value, Runtime& runtime)
{
    const DType& dtype = DTYPES[1];
    const std::vector<int64_t> shape{16};
    const std::vector<int64_t> strides{1};
    DeviceBuffer selfBuffer(16 * dtype.bytes);
    DeviceBuffer outBuffer(16 * dtype.bytes);
    Tensor self(shape, strides, 0, shape, dtype.aclType, selfBuffer.Data());
    Tensor out(shape, strides, 0, shape, dtype.aclType, outBuffer.Data());
    Scalar probability(value, PROB_DTYPES[1]);
    uint64_t workspaceBytes = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus status = aclnnBernoulliGetWorkspaceSize(self.Get(), probability.Get(), DEFAULT_SEED,
                                                              DEFAULT_OFFSET, out.Get(), &workspaceBytes, &executor);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
    (void)runtime;
    return status != ACL_SUCCESS;
}

bool DistributionIsPlausible(const Result& result, size_t elements, double probability)
{
    const double expected = static_cast<double>(elements) * probability;
    const double sigma = std::sqrt(static_cast<double>(elements) * probability * (1.0 - probability));
    return result.other == 0 && std::abs(static_cast<double>(result.ones) - expected) <= 6.0 * sigma;
}

void PrintWorkspaceMetric(const std::string& name, const Result& result)
{
    std::printf("[METRIC] case=%s workspace_bytes=%llu\n", name.c_str(),
                static_cast<unsigned long long>(result.workspaceBytes));
}

class Reporter {
public:
    void CheckCase(bool passed, const std::string& name)
    {
        ++total_;
        if (!passed) {
            ++failed_;
        }
        std::printf("[%s] %s\n", passed ? "PASS" : "FAIL", name.c_str());
    }

    int ExitCode() const
    {
        std::printf("SUMMARY total=%zu passed=%zu failed=%zu\n", total_, total_ - failed_, failed_);
        return failed_ == 0 ? 0 : 1;
    }

private:
    size_t total_ = 0;
    size_t failed_ = 0;
};

template <typename Function>
void Guarded(Reporter& reporter, const std::string& name, Function&& function)
{
    try {
        reporter.CheckCase(function(), name);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[FAIL] %s: %s\n", name.c_str(), error.what());
        reporter.CheckCase(false, name);
    }
}
} // namespace

int main()
{
    int32_t device = 0;
    if (const char* value = std::getenv("BERNOULLI_ST_DEVICE_ID"); value != nullptr) {
        device = std::stoi(value);
    }

    try {
        Runtime runtime(device);
        Reporter reporter;

        for (const DType& dtype : DTYPES) {
            const std::string name = std::string("contiguous_general_") + dtype.name;
            Guarded(reporter, name, [&] {
                Case test;
                test.dtype = &dtype;
                const Result result = Run(test, runtime);
                PrintWorkspaceMetric(name, result);
                return result.other == 0 && result.zeros + result.ones == Elements(test.shape) &&
                       result.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES;
            });
        }

        for (const DType& dtype : DTYPES) {
            const std::string name = std::string("alias_syncall_boundary_n257_") + dtype.name;
            Guarded(reporter, name, [&] {
                Case test;
                test.dtype = &dtype;
                test.shape = {257};
                const Result result = Run(test, runtime);
                PrintWorkspaceMetric(name, result);
                return result.other == 0 && result.zeros + result.ones == Elements(test.shape) &&
                       result.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES;
            });
        }

        Guarded(reporter, "alias_large_multicore_n1000000_fp32", [&] {
            Case test;
            test.dtype = &DTYPES[1];
            test.shape = {1000000};
            const Result result = Run(test, runtime);
            PrintWorkspaceMetric("alias_large_multicore_n1000000_fp32", result);
            return DistributionIsPlausible(result, Elements(test.shape), test.probability) &&
                   result.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES;
        });

        for (const AliasThreshold& threshold : ALIAS_THRESHOLDS) {
            uint64_t fallbackWorkspace = 0;
            const std::string fallbackName = std::string("workspace_") + threshold.widthName + "_fallback_" +
                                             std::to_string(threshold.fallbackElements);
            Guarded(reporter, fallbackName, [&] {
                Case test;
                test.dtype = threshold.dtype;
                test.shape = {threshold.fallbackElements};
                const Result result = Run(test, runtime);
                PrintWorkspaceMetric(fallbackName, result);
                fallbackWorkspace = result.workspaceBytes;
                return result.other == 0;
            });

            const std::string aliasName = std::string("workspace_") + threshold.widthName + "_alias_" +
                                          std::to_string(threshold.aliasElements);
            Guarded(reporter, aliasName, [&] {
                Case test;
                test.dtype = threshold.dtype;
                test.shape = {threshold.aliasElements};
                const Result result = Run(test, runtime);
                PrintWorkspaceMetric(aliasName, result);
                return result.other == 0 && result.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES &&
                       result.workspaceBytes < fallbackWorkspace;
            });
        }

        for (int64_t elements : {127, 128, 129, 255, 256, 257}) {
            const std::string name = std::string("alias_mask_boundary_n") + std::to_string(elements);
            Guarded(reporter, name, [&] {
                Case test;
                test.dtype = &DTYPES[1];
                test.shape = {elements};
                const Result result = Run(test, runtime);
                PrintWorkspaceMetric(name, result);
                return result.other == 0 && result.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES;
            });
        }

        for (const DType& dtype : DTYPES) {
            for (bool inplace : {false, true}) {
                const std::string name = std::string("noncontiguous_") + (inplace ? "inplace_" : "outplace_") +
                                         dtype.name;
                Guarded(reporter, name, [&] {
                    Case test;
                    test.dtype = &dtype;
                    test.shape = {4, 5};
                    test.strides = {10, 2};
                    test.storageOffset = 3;
                    test.storageElements = 42;
                    test.inplace = inplace;
                    const Result result = Run(test, runtime);
                    return result.other == 0 && result.changedGuardBytes == 0 && result.workspaceBytes > 0;
                });
            }
        }

        for (bool inplace : {false, true}) {
            const std::string name = std::string("transposed_") + (inplace ? "inplace" : "outplace");
            Guarded(reporter, name, [&] {
                Case test;
                test.dtype = &DTYPES[1];
                test.shape = {3, 4};
                test.strides = {1, 3};
                test.storageElements = 12;
                test.inplace = inplace;
                const Result result = Run(test, runtime);
                return result.other == 0 && result.changedGuardBytes == 0;
            });
        }

        for (const DType* dtype : {&DTYPES[4], &DTYPES[0], &DTYPES[1], &DTYPES[2]}) {
            for (bool inplace : {false, true}) {
                const std::string name = std::string("dense_oversized_storage_") +
                                         (inplace ? "inplace_" : "outplace_") + dtype->name;
                Guarded(reporter, name, [&] {
                    Case test;
                    test.dtype = dtype;
                    test.shape = {129};
                    test.storageElements = 200;
                    test.inplace = inplace;
                    const Result result = Run(test, runtime);
                    return result.other == 0 && result.zeros + result.ones == Elements(test.shape) &&
                           result.changedGuardBytes == 0 && result.workspaceBytes > 0;
                });
            }
        }

        for (bool inplace : {false, true}) {
            const std::string name = std::string("dense_flattened_equal_storage_") + (inplace ? "inplace" : "outplace");
            Guarded(reporter, name, [&] {
                Case test;
                test.dtype = &DTYPES[1];
                test.shape = {3, 43};
                test.storageElements = 129;
                test.inplace = inplace;
                const Result result = Run(test, runtime);
                return result.other == 0 && result.zeros + result.ones == Elements(test.shape) &&
                       result.changedGuardBytes == 0 && result.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES;
            });
        }

        for (bool inplace : {false, true}) {
            const std::string name = std::string("dense_nonzero_offset_") + (inplace ? "inplace" : "outplace");
            Guarded(reporter, name, [&] {
                Case test;
                test.dtype = &DTYPES[1];
                test.shape = {129};
                test.strides = {1};
                test.storageOffset = 5;
                test.storageElements = 134;
                test.inplace = inplace;
                const Result result = Run(test, runtime);
                return result.other == 0 && result.zeros + result.ones == Elements(test.shape) &&
                       result.changedGuardBytes == 0 && result.workspaceBytes > 0;
            });
        }

        for (bool inplace : {false, true}) {
            const std::string name = std::string("singleton_zero_stride_") + (inplace ? "inplace" : "outplace");
            Guarded(reporter, name, [&] {
                Case test;
                test.dtype = &DTYPES[1];
                test.shape = {1, 129};
                test.strides = {0, 1};
                test.storageElements = 129;
                test.inplace = inplace;
                const Result result = Run(test, runtime);
                return result.other == 0 && result.zeros + result.ones == Elements(test.shape) &&
                       result.changedGuardBytes == 0 && result.workspaceBytes > 0;
            });
        }

        for (const DType& probDtype : PROB_DTYPES) {
            Guarded(reporter, std::string("prob_dtype_") + probDtype.name, [&] {
                Case test;
                test.dtype = &DTYPES[1];
                test.probDtype = &probDtype;
                test.probability = 0.5;
                const Result result = Run(test, runtime);
                return result.other == 0;
            });
        }

        for (size_t rank = 0; rank <= 8; ++rank) {
            Guarded(reporter, std::string("rank") + std::to_string(rank), [&] {
                Case test;
                test.dtype = &DTYPES[1];
                test.shape.assign(rank, 1);
                if (rank > 0) {
                    test.shape.back() = 2;
                }
                return Run(test, runtime).other == 0;
            });
        }
        Guarded(reporter, "empty_tensor", [&] {
            Case test;
            test.dtype = &DTYPES[1];
            test.shape = {2, 0, 3};
            const Result result = Run(test, runtime);
            return result.logical.empty() && result.other == 0;
        });
        Guarded(reporter, "prob_zero", [&] {
            Case test;
            test.dtype = &DTYPES[1];
            test.probability = 0.0;
            const Result result = Run(test, runtime);
            return result.zeros == Elements(test.shape) && result.other == 0;
        });
        Guarded(reporter, "prob_one", [&] {
            Case test;
            test.dtype = &DTYPES[1];
            test.probability = 1.0;
            const Result result = Run(test, runtime);
            return result.ones == Elements(test.shape) && result.other == 0;
        });
        for (double probability : {0.001, 0.999}) {
            const std::string name = probability < 0.5 ? "prob_near_zero" : "prob_near_one";
            Guarded(reporter, name, [&] {
                Case test;
                test.dtype = &DTYPES[1];
                test.shape = {262144};
                test.probability = probability;
                const Result result = Run(test, runtime);
                PrintWorkspaceMetric(name, result);
                return DistributionIsPlausible(result, Elements(test.shape), test.probability) &&
                       result.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES;
            });
        }
        for (int64_t offset : {0, 4, 8}) {
            Guarded(reporter, std::string("legal_offset_") + std::to_string(offset), [&] {
                Case test;
                test.dtype = &DTYPES[1];
                test.offset = offset;
                return Run(test, runtime).other == 0;
            });
        }
        Guarded(reporter, "seed_offset_reproducibility_and_sensitivity", [&] {
            Case test;
            test.dtype = &DTYPES[1];
            test.shape = {4096};
            const Result first = Run(test, runtime);
            const Result second = Run(test, runtime);
            Case changedOffset = test;
            changedOffset.offset += 4;
            const Result offsetResult = Run(changedOffset, runtime);
            Case changedSeed = test;
            changedSeed.seed += 1;
            const Result seedResult = Run(changedSeed, runtime);
            const double mean = static_cast<double>(first.ones) / static_cast<double>(Elements(test.shape));
            const double sigma = std::sqrt(test.probability * (1.0 - test.probability) / Elements(test.shape));
            PrintWorkspaceMetric("seed_offset_reproducibility_and_sensitivity", first);
            return first.other == 0 && first.logical == second.logical &&
                   Fnv1a(first.logical) != Fnv1a(offsetResult.logical) &&
                   Fnv1a(first.logical) != Fnv1a(seedResult.logical) &&
                   std::abs(mean - test.probability) <= 6.0 * sigma &&
                   first.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES &&
                   second.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES &&
                   offsetResult.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES &&
                   seedResult.workspaceBytes <= MAX_ALIAS_WORKSPACE_BYTES;
        });
        Guarded(reporter, "invalid_offset_rejected", [&] { return InvalidOffsetIsRejected(runtime); });
        Guarded(reporter, "prob_below_zero_rejected", [&] { return InvalidProbabilityIsRejected(-0.001, runtime); });
        Guarded(reporter, "prob_above_one_rejected", [&] { return InvalidProbabilityIsRejected(1.001, runtime); });
        Guarded(reporter, "prob_nan_rejected",
                [&] { return InvalidProbabilityIsRejected(std::numeric_limits<double>::quiet_NaN(), runtime); });

        return reporter.ExitCode();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FATAL: %s\n", error.what());
        return 1;
    }
}
