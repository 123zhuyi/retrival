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
 * \file acl_stft.cpp
 * \brief
 */

#include <cmath>
#include <mutex>
#include <list>
#include <unordered_map>
#include <string>
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/framework_op.h"
#include "platform/platform_info.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "conversion/pad_v3/op_api/padv3.h"
#include "math/mul/op_api/mul.h"
#include "math/ones_like/op_api/ones_like.h"
#include "stft.h"
#include "acl_stft.h"

using namespace op;

static const uint64_t STFT_MIN_INPUT_DIM = 1;
static const uint64_t STFT_MAX_INPUT_DIM = 2;
static const uint64_t STFT_WINDOW_DIM = 1;
static const uint64_t STFT_MIN_OUTPUT_DIM = 2;
static const uint64_t STFT_MAX_OUTPUT_DIM = 4;
static const int64_t PAD_VALUE = 0;
static const std::string PAD_MODE = "constant";
static const float K2PI = 6.2831853071795864769252867665590057683943388f;
static const int QUADRANT_ONE = 1;
static const int QUADRANT_TWO = 2;
static const int QUADRANT_FOUR = 4;
static const int REAL_IMAG_NUM = 2;
static const int64_t DEFAULT_DFT_CACHE_MAX_MEMORY = 8LL * 1024 * 1024 * 1024; // 8GB，可覆盖nFft到32768的所有常见尺度
static const int FP32_DIVIDE_FP16 = 2;
static const int FP16_NUM_PER_BLOCK = 16;
static const int X1_NFFT = 400;
static const int X1_HOP = 160;
static const int X1_ROW_SIZE = 201;
static const int X1_BATCH = 16;
static const int ROW_SIZE_DIVIDE = 3;
static const int ROW_SIZE_DIVIDE_B3 = 5;
static const int SECOND_ROW_SIZE_DIVIDE = 2;
static const int BLOCK_SIZE = 32;
static const int PACKAGE_SIZE = 128;
static const int FP32_BYTES = 4;

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128};

// DFT矩阵缓存键：DFT矩阵元素值W[k,n]=exp(-j*2π*k*n/nFft)仅由K和nFft决定，
// 矩阵物理布局(列数colSizeAlign)由nfftAlignBytes决定。
// hopLength/winLength/normalized/onesided/returnComplex仅通过nfftAlignBytes间接影响矩阵布局，
// nfftAlignBytes只有2种取值(32或128)，因此4字段键可完整覆盖8字段的所有场景。
struct DftCacheKey {
    int64_t K;              // onesided ? (nFft/2+1) : nFft，决定矩阵行数和元素值
    int64_t nFft;           // 决定矩阵元素值
    int64_t nfftAlignBytes; // 决定colSizeAlign(矩阵物理列数/padding)，仅32或128两种取值
    int32_t deviceId;

    bool operator==(const DftCacheKey& other) const
    {
        return K == other.K && nFft == other.nFft && nfftAlignBytes == other.nfftAlignBytes &&
               deviceId == other.deviceId;
    }
};

struct DftCacheKeyHash {
    std::size_t operator()(const DftCacheKey& key) const
    {
        return std::hash<int64_t>()(key.K) ^ (std::hash<int64_t>()(key.nFft) << 1) ^
               (std::hash<int64_t>()(key.nfftAlignBytes) << 2) ^ std::hash<int32_t>()(key.deviceId);
    }
};

// DFT矩阵缓存条目
struct DftCacheEntry {
    void* devicePtr;    // NPU上的DFT矩阵指针
    int64_t matrixSize; // 矩阵占用的显存大小（字节）
};

// DFT矩阵缓存：基于LRU淘汰策略，按总显存预算控制容量
// 缓存仅服务于AiCore路径（AiCpu路径不构造DFT矩阵）
class DftMatrixCache {
public:
    static DftMatrixCache& GetInstance()
    {
        static DftMatrixCache instance;
        return instance;
    }

    // 查找缓存，命中时更新LRU顺序
    void* Find(int64_t K, int64_t nFft, int64_t nfftAlignBytes, int32_t deviceId)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        DftCacheKey key = {K, nFft, nfftAlignBytes, deviceId};
        auto it = cacheMap_.find(key);
        if (it == cacheMap_.end()) {
            return nullptr;
        }
        // 命中：移到LRU链表头部（最近使用）
        lruList_.splice(lruList_.begin(), lruList_, it->second.lruIt);
        return it->second.entry.devicePtr;
    }

    // 插入缓存，返回true表示已缓存，false表示不缓存（矩阵太大）
    bool Insert(int64_t K, int64_t nFft, int64_t nfftAlignBytes, int32_t deviceId, void* devicePtr, int64_t matrixSize)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        DftCacheKey key = {K, nFft, nfftAlignBytes, deviceId};

        // 如果已存在，更新LRU顺序
        auto it = cacheMap_.find(key);
        if (it != cacheMap_.end()) {
            lruList_.splice(lruList_.begin(), lruList_, it->second.lruIt);
            return true;
        }

        // 大矩阵直通：单矩阵超过预算50%时不缓存
        // 避免一个大矩阵占满预算，导致其他小矩阵全部被淘汰
        if (matrixSize > maxMemory_ / 2) {
            return false;
        }

        // 淘汰直到有足够空间
        while (usedMemory_ + matrixSize > maxMemory_ && !lruList_.empty()) {
            EvictOne();
        }

        // 插入新条目
        lruList_.push_front(key);
        CacheMapValue value;
        value.entry = {devicePtr, matrixSize};
        value.lruIt = lruList_.begin();
        cacheMap_[key] = value;
        usedMemory_ += matrixSize;
        return true;
    }

private:
    DftMatrixCache() : maxMemory_(DEFAULT_DFT_CACHE_MAX_MEMORY), usedMemory_(0) {}

    void EvictOne()
    {
        // 淘汰LRU链表尾部（最久未使用）
        auto& evictKey = lruList_.back();
        auto it = cacheMap_.find(evictKey);
        if (it != cacheMap_.end()) {
            usedMemory_ -= it->second.entry.matrixSize;
            // 注意：devicePtr的释放由executor管理，这里仅从缓存索引中移除
            cacheMap_.erase(it);
        }
        lruList_.pop_back();
    }

    struct CacheMapValue {
        DftCacheEntry entry;
        std::list<DftCacheKey>::iterator lruIt;
    };

    std::mutex mutex_;
    int64_t maxMemory_;              // 显存预算上限
    int64_t usedMemory_;             // 当前已用显存
    std::list<DftCacheKey> lruList_; // LRU链表，头部=最近使用
    std::unordered_map<DftCacheKey, CacheMapValue, DftCacheKeyHash> cacheMap_;
};

static int64_t nFftToAlign(const aclTensor* self, int64_t nfft, int alignBytes)
{
    int64_t nFftAlign = 0;
    switch (self->GetDataType()) {
        case DataType::DT_FLOAT: {
            int alignNum = alignBytes / FP32_BYTES;
            nFftAlign = (nfft + alignNum - 1) / alignNum * alignNum;
            break;
        }
        default:
            break;
    }

    return nFftAlign;
}

static int NfftAlignBytes(int64_t nfft, int64_t hopLength, bool normalized, bool onesided, bool returnComplex)
{
    if (nfft == X1_NFFT && hopLength == X1_HOP && normalized == false && onesided == true && returnComplex == false) {
        return BLOCK_SIZE;
    }
    return PACKAGE_SIZE;
}

static float Mul2Pi(int m, int n)
{
    if (n == 0) {
        return -1.0f;
    }
    return ((K2PI * (m)) / (n));
}

static void CalcRealAndImag(int m, int n, float* out)
{
    int m0 = m;
    int n0 = n;
    float* out0 = out;
    float theta, c, s, t;
    unsigned int octant = 0;
    int size = n0;

    m0 = m0 % n0;
    n0 += n0;
    n0 += n0;
    m0 += m0;
    m0 += m0;

    if (m0 < 0) {
        m0 += n0;
    }
    if (m0 > n0 - m0) {
        m0 = n0 - m0;
        octant |= static_cast<unsigned int>(QUADRANT_FOUR);
    }
    if (m0 > size) {
        m0 = m0 - size;
        octant |= static_cast<unsigned int>(QUADRANT_TWO);
    }
    if (m0 > size - m0) {
        m0 = size - m0;
        octant |= static_cast<unsigned int>(QUADRANT_ONE);
    }

    theta = Mul2Pi(m0, n0);
    c = cos(theta);
    s = sin(theta);

    if ((octant & static_cast<unsigned int>(QUADRANT_ONE)) != 0U) {
        t = c;
        c = s;
        s = t;
    }
    if ((octant & static_cast<unsigned int>(QUADRANT_TWO)) != 0U) {
        t = c;
        c = -s;
        s = t;
    }
    if ((octant & static_cast<unsigned int>(QUADRANT_FOUR)) != 0U) {
        s = -s;
    }
    out0[0] = c;
    out0[1] = s;
}

static bool HasEmptyTensor(const aclTensor* self)
{
    // 检查张量是否存在空维
    if (self->IsEmpty()) {
        return true;
    }

    return false;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* window, const aclTensor* out)
{
    // 检查self, window, out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, return false);
    if (window != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(window, ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SAME(self, window, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckFormat(const aclTensor* self)
{
    // self格式是ND
    if (self->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input format only support ND");
        return false;
    }
    return true;
}

static op::Shape GetOutputShape(const aclTensor* self, bool onesided, bool returnComplex, int64_t hopLength,
                                int64_t nFft)
{
    op::Shape selfShape = self->GetViewShape();
    auto dimNum = selfShape.GetDimNum();
    int64_t batch = dimNum == STFT_MAX_INPUT_DIM ? selfShape.GetDim(0) : 0;
    int64_t len = dimNum == STFT_MAX_INPUT_DIM ? selfShape.GetDim(1) : selfShape.GetDim(0);
    if (hopLength <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect hopLength > 0,  change hopLength = 1");
        hopLength = 1;
    }
    int64_t frames = (len - nFft) / hopLength + 1;
    int64_t n = onesided == true ? nFft / REAL_IMAG_NUM + 1 : nFft;

    op::Shape outShape;
    op::Shape outShapeComplexWithBatch = {batch, n, frames};
    op::Shape outShapeComplex = {n, frames};
    op::Shape outShapeRealWithBatch = {batch, n, frames, REAL_IMAG_NUM};
    op::Shape outShapeReal = {n, frames, REAL_IMAG_NUM};

    if (returnComplex) {
        outShape = batch > 0 ? outShapeComplexWithBatch : outShapeComplex;
    } else {
        outShape = batch > 0 ? outShapeRealWithBatch : outShapeReal;
    }
    return outShape;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out, const aclTensor* window, int64_t hopLength,
                       int64_t winLength, int64_t nFft, bool onesided, bool returnComplex)
{
    // input dim: 1~2
    OP_CHECK_MIN_DIM(self, STFT_MIN_INPUT_DIM, return false);
    OP_CHECK_MAX_DIM(self, STFT_MAX_INPUT_DIM, return false);

    // output dim: 2~4
    OP_CHECK_MIN_DIM(out, STFT_MIN_OUTPUT_DIM, return false);
    OP_CHECK_MAX_DIM(out, STFT_MAX_OUTPUT_DIM, return false);

    op::Shape selfShape = self->GetViewShape();
    auto dimNum = selfShape.GetDimNum();
    int64_t len = dimNum == STFT_MAX_INPUT_DIM ? selfShape.GetDim(1) : selfShape.GetDim(0);
    if (nFft <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect nFft > 0");
        return false;
    }
    if (len < nFft) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect input length >= nFft");
        return false;
    }
    if (hopLength <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect hopLength > 0");
        return false;
    }
    if (winLength <= 0 || winLength > nFft) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect 0 < winLength <= nFft");
        return false;
    }
    bool isInputComplex = false;
    if (self->GetDataType() == DataType::DT_COMPLEX64 || self->GetDataType() == DataType::DT_COMPLEX128) {
        isInputComplex = true;
    }
    if (window) {
        OP_CHECK_MIN_DIM(window, STFT_WINDOW_DIM, return false);
        OP_CHECK_MAX_DIM(window, STFT_WINDOW_DIM, return false);
        // winLength不等于nfft时需要和window的shape相同
        if (winLength != nFft && window->GetViewShape().GetDim(0) != winLength) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect winLength and window size should be equal");
            return false;
        }
        if (window->GetDataType() == DataType::DT_COMPLEX64 || window->GetDataType() == DataType::DT_COMPLEX128) {
            isInputComplex = true;
        }
    }
    // if input is complex, onesided can't be true
    if (isInputComplex && onesided) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when input is complex, onesided can't be true");
        return false;
    }
    op::Shape outShape = GetOutputShape(self, onesided, returnComplex, hopLength, nFft);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);

    return true;
}

static bool CheckPlatform()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
        return true;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "STFT is not supported on this platform");
        return false;
    }
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out, const aclTensor* window, int64_t hopLength,
                               int64_t winLength, int64_t nFft, bool onesided, bool returnComplex)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, window, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查格式是否支持
    CHECK_RET(CheckFormat(self), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out, window, hopLength, winLength, nFft, onesided, returnComplex),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GeneratePadWindow(const aclTensor* self, const aclTensor* window, int64_t winLength,
                                          int64_t nFft, int nfftAlignBytes, aclOpExecutor* executor)
{
    int64_t left = (nFft - winLength) / 2;

    // nFft按照block对齐，即nFft -> nFft_align
    int64_t nFftAlign = nFftToAlign(self, nFft, nfftAlignBytes);
    int64_t right = nFftAlign - winLength - left;
    if (window == nullptr) {
        auto assist = executor->AllocHostTensor({winLength}, DataType::DT_FLOAT);
        window = l0op::OnesLike(assist, executor);
    }
    // 生成填充tensor
    size_t dims = 2;
    std::vector<int64_t> padVec = {left, right};
    auto padArray = executor->AllocIntArray(padVec.data(), dims);
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }
    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);

    const aclTensor* valueTensor = executor->ConvertToTensor(executor->AllocScalar(PAD_VALUE), window->GetDataType());
    if (valueTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try convert PAD_VALUE pad tensor failed.");
        return nullptr;
    }
    return l0op::PadV3(window, padTensor, valueTensor, PAD_MODE, true, executor);
}

static const aclTensor* GenerateDftMatrix(const aclTensor* self, int64_t rowSize, int64_t colSize, int nfftAlignBytes,
                                          aclOpExecutor* executor)
{
    // colSize按照block对齐，即(K, nFft) -> (K, nFft_align)
    int64_t colSizeAlign = nFftToAlign(self, colSize, nfftAlignBytes);
    auto deviceId = GetCurrentPlatformInfo().GetDeviceId();
    // 缓存键：DFT矩阵由K(rowSize)、nFft(colSize)、nfftAlignBytes(决定物理布局)唯一确定
    void* planDevice = DftMatrixCache::GetInstance().Find(rowSize, colSize, nfftAlignBytes, deviceId);

    // 命中plan cache
    if (planDevice != nullptr) {
        OP_LOGI("DftMatrix cache HIT: K=%lld, nFft=%lld, alignBytes=%d, deviceId=%d", (long long)rowSize,
                (long long)colSize, nfftAlignBytes, deviceId);
        auto dft = executor->AllocTensor({REAL_IMAG_NUM, rowSize, colSizeAlign}, op::DataType::DT_FLOAT);
        dft->SetFromWorkspace(false);
        dft->SetStorageAddr(planDevice);
        executor->AbandonCache();
        return dft;
    }

    // 未命中plan cache
    OP_LOGI("DftMatrix cache MISS: K=%lld, nFft=%lld, alignBytes=%d, deviceId=%d, constructing matrix...",
            (long long)rowSize, (long long)colSize, nfftAlignBytes, deviceId);
    auto dftMatrix = executor->AllocHostTensor({2, rowSize, colSizeAlign}, op::DataType::DT_FLOAT);
    float* addrReal = static_cast<float*>(dftMatrix->GetStorageAddr());
    float* addrImag = static_cast<float*>(dftMatrix->GetStorageAddr()) + colSizeAlign;
    float out[2];

    // 实部及虚部按频率行交错排布: [R0, I0, R1, I1, ...]。
    // generic kernel(tiling key 1)以 planOffset = IMAG_AND_REAL * mOffset * nfftAlign 读取
    // 第m个频率的real/imag行(位于第2m/2m+1行), 且Gather mask按相邻real/imag行配对抽取,
    // 两平面布局会导致输出频率行错位(实部读到real平面全部行, 虚部被挤到额外行)。
    for (int i = 0; i < rowSize; i++) {
        if (i > 0) {
            addrReal += colSizeAlign;
            addrImag += colSizeAlign;
        }
        for (int j = 0; j < colSizeAlign; j++) {
            if (j < colSize) {
                CalcRealAndImag(-1 * i * j, colSize, out);
                *addrReal = out[0];
                *addrImag = out[1];
            } else {
                *addrReal = 0;
                *addrImag = 0;
            }
            addrReal++;
            addrImag++;
        }
    }

    // 同步拷贝到NPU
    auto deviceTensor = op::CopyToNpuSync(dftMatrix, executor);
    CHECK_RET(deviceTensor != nullptr, nullptr);

    // 计算矩阵大小并尝试缓存（大矩阵可能不被缓存）
    int64_t matrixSize = 2 * rowSize * colSizeAlign * sizeof(float);
    bool cached = DftMatrixCache::GetInstance().Insert(rowSize, colSize, nfftAlignBytes, deviceId,
                                                       deviceTensor->GetData(), matrixSize);
    OP_LOGI("DftMatrix constructed: K=%lld, nFft=%lld, matrixSize=%lldMB, cached=%s", (long long)rowSize,
            (long long)colSize, (long long)(matrixSize / 1024 / 1024), cached ? "yes" : "no(too large)");

    return deviceTensor;
}

aclnnStatus aclStftGetWorkspaceSize(const aclTensor* self, const aclTensor* windowOptional, aclTensor* out,
                                    int64_t nFft, int64_t hopLength, int64_t winLength, bool normalized, bool onesided,
                                    bool returnComplex, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGI("aclStftGetWorkspaceSize ENTER: nFft=%lld, hopLength=%lld, winLength=%lld, onesided=%d, returnComplex=%d",
            (long long)nFft, (long long)hopLength, (long long)winLength, onesided, returnComplex);
    L2_DFX_PHASE_1(aclStft,
                   DFX_IN(self, windowOptional, nFft, hopLength, winLength, normalized, onesided, returnComplex),
                   DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    bool result = CheckPlatform();
    CHECK_RET(result == true, ACLNN_ERR_PARAM_INVALID);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out, windowOptional, hopLength, winLength, nFft, onesided, returnComplex);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (HasEmptyTensor(self)) {
        *workspaceSize = 0U;
        uniqueExecutor.ReleaseTo(executor);
        OP_LOGD("self: nullptr, return");
        return ACLNN_SUCCESS;
    }

    int nfftAlignBytes = NfftAlignBytes(nFft, hopLength, normalized, onesided, returnComplex);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (!l0op::IsStftAiCoreSupported(selfContiguous, windowOptional, nFft, hopLength, winLength, normalized, onesided,
                                     returnComplex)) {
        // aicpu
        OP_LOGI("aclStft path=AiCpu: nFft=%lld, hopLength=%lld, winLength=%lld, onesided=%d, returnComplex=%d",
                (long long)nFft, (long long)hopLength, (long long)winLength, onesided, returnComplex);
        auto stftResult = l0op::Stft(selfContiguous, nullptr, windowOptional, nFft, hopLength, winLength, normalized,
                                     onesided, returnComplex, uniqueExecutor.get());
        CHECK_RET(stftResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(stftResult, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // window length < nFft, need to pad window
        OP_LOGI("aclStft path=AiCore: nFft=%lld, hopLength=%lld, winLength=%lld, onesided=%d, returnComplex=%d",
                (long long)nFft, (long long)hopLength, (long long)winLength, onesided, returnComplex);
        const aclTensor* windowPad;
        int64_t nFftAlign = nFftToAlign(self, nFft, nfftAlignBytes);
        if (winLength < nFftAlign) {
            windowPad = GeneratePadWindow(self, windowOptional, winLength, nFft, nfftAlignBytes, uniqueExecutor.get());
        } else {
            windowPad = windowOptional;
        }

        // 生成辅助矩阵W：w_real（K，N）+ w_imag（K，N）
        const int64_t K = onesided ? (nFft / 2) + 1 : nFft;
        const int64_t N = nFft;
        const aclTensor* dftMatrix = GenerateDftMatrix(self, K, N, nfftAlignBytes, uniqueExecutor.get());

        const aclTensor* stftResult;
        if (nFft == X1_NFFT && hopLength == X1_HOP && normalized == false && onesided == true &&
            returnComplex == false) {
            // mul(dftMatrix, windowPad)
            const aclTensor* w = windowPad == nullptr ? dftMatrix :
                                                        l0op::Mul(dftMatrix, windowPad, uniqueExecutor.get());
            // stft
            stftResult = l0op::Stft(selfContiguous, w, nullptr, nFft, hopLength, winLength, normalized, onesided,
                                    returnComplex, uniqueExecutor.get());
        } else {
            // stft
            stftResult = l0op::Stft(selfContiguous, dftMatrix, windowPad, nFft, hopLength, winLength, normalized,
                                    onesided, returnComplex, uniqueExecutor.get());
        }
        CHECK_RET(stftResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(stftResult, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclStft(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclStft);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
