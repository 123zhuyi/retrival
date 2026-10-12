/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/transdata.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

namespace l0op {
static constexpr int32_t kDataTypeSizeBitOffset = 1000;
static constexpr uint32_t kBitNumOfOneByte = 8U;
static constexpr uint32_t kBitThreeBytes = 24U;
static constexpr int64_t NUM_8 = 8;
static constexpr int64_t NUM_4 = 4;
static constexpr int64_t NUM_2 = 2;
static constexpr int64_t MAX_GROUPS = 0xffff;

static int8_t kSupportMap[14][14] = {
    /*  0   , 1   , 2   , 3  , 4 , 5 , 6 , 7    , 8    , 9    , 10 , 11  ,  12,      13 */
    /*  NCHW, NHWC, HWCN, 5HD, FZ, ND, NZ, NCDHW, NDHWC, DHWCN, 6HD, FZ_3D, 5HD_C04, FZ_C04 */
    {0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1}, /* NCHW */
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, /* NHWC */
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1}, /* HWCN */
    {1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, /* 5HD */
    {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, /* FZ  */
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, /* ND */
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, /* NZ */
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0}, /* NCDHW */
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, /* NDHWC */
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, /* DHWCN */
    {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0}, /* 6HD */
    {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0}, /* FZ_3D */
    {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, /* 5HD_C04 */
    {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  /* FZ_C04 */
};
/*  0   , 1   , 2   , 3  , 4 , 5 , 6 , 7    , 8    , 9    , 10 , 11  ,  12,      13    , 14  ,15       ,16,       17,      18*/
/*  NCHW, NHWC, HWCN, 5HD, FZ, ND, NZ, NCDHW, NDHWC, DHWCN, 6HD, FZ_3D, 5HD_C04, FZ_C04, NCL, NZ_C0_16, NZ_C0_32, NZ_C0_2, NZ_C0_4 */
static int64_t kFormatRank[] = {4, 4, 4, 5, 4, -1, -1, 5, 5, 5, 6, 4, 5, 4, 3, -1, -1, -1, -1};

static int8_t kFormatIndex[] = {
    0,  //         FORMAT_NCHW = 0,   // NCHW
    1,  //         FORMAT_NHWC,       // NHWC
    5,  //         FORMAT_ND,         // Nd Tensor
    3,  //         FORMAT_NC1HWC0,    // NC1HWC0
    4,  //         FORMAT_FRACTAL_Z,  // FRACTAL_Z
    -1, //        FORMAT_NC1C0HWPAD = 5,
    -1, //        FORMAT_NHWC1C0,
    -1, //        FORMAT_FSR_NCHW,
    -1, //        FORMAT_FRACTAL_DECONV,
    -1, //        FORMAT_C1HWNC0,
    -1, //        FORMAT_FRACTAL_DECONV_TRANSPOSE = 10,
    -1, //        FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
    12, //        FORMAT_NC1HWC0_C04,    // NC1HWC0, C0 is 4
    13, //        FORMAT_FRACTAL_Z_C04,  // FRACZ, C0 is 4
    -1, //        FORMAT_CHWN,
    -1, //        FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15,
    2,  //         FORMAT_HWCN,
    -1, //        FORMAT_NC1KHKWHWC0,  // KH,KW kernel h& kernel w maxpooling max output format
    -1, //        FORMAT_BN_WEIGHT,
    -1, //        FORMAT_FILTER_HWCK,  // filter input tensor format
    -1, //        FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20,
    -1, //        FORMAT_HASHTABLE_LOOKUP_KEYS,
    -1, //        FORMAT_HASHTABLE_LOOKUP_VALUE,
    -1, //        FORMAT_HASHTABLE_LOOKUP_OUTPUT,
    -1, //        FORMAT_HASHTABLE_LOOKUP_HITS,
    -1, //        FORMAT_C1HWNCoC0 = 25,
    -1, //        FORMAT_MD,
    8,  //         FORMAT_NDHWC,
    -1, //        FORMAT_FRACTAL_ZZ,
    6,  //         FORMAT_FRACTAL_NZ,
    7,  //         FORMAT_NCDHW = 30,
    9,  //         FORMAT_DHWCN,  // 3D filter input tensor format
    10, //        FORMAT_NDC1HWC0,
    11, //        FORMAT_FRACTAL_Z_3D,
    -1, //        FORMAT_CN,
    -1, //        FORMAT_NC = 35,
    -1, //        FORMAT_DHWNC,
    -1, //        FORMAT_FRACTAL_Z_3D_TRANSPOSE, // 3D filter(transpose) input tensor format
    -1, //        FORMAT_FRACTAL_ZN_LSTM,
    -1, //        FORMAT_FRACTAL_Z_G,
    -1, //        FORMAT_RESERVED = 40,
    -1, //        FORMAT_ALL,
    -1, //        FORMAT_NULL,
    -1, //        FORMAT_ND_RNN_BIAS,
    -1, //        FORMAT_FRACTAL_ZN_RNN,
    -1, //        FORMAT_NYUV 45,
    -1, //        FORMAT_NYUV_A,
    14, //        FORMAT_NCL,
    -1, //        FRACTAL_Z_WINO,
    -1, //        C1HWC0,
    15, //        FRACTAL_NZ_C0_16,
    16, //        FRACTAL_NZ_C0_32,
    17, //        FRACTAL_NZ_C0_2,
    18, //        FRACTAL_NZ_C0_4,
    // Add new formats definition here
    -1, //        FORMAT_END,
    // FORMAT_MAX defines the max value of Format.
    // Any Format should not exceed the value of FORMAT_MAX.
    // ** Attention ** : FORMAT_MAX stands for the SPEC of enum Format and almost SHOULD NOT be used in code.
    //                   If you want to judge the range of Format, you can use FORMAT_END.
    -1 //        FORMAT_MAX = 0xff
};

/*                                NCHW, NHWC, HWCN, 5HD, FZ, ND, NZ, NCDHW, NDHWC, DHWCN, 6HD, FZ_3D, 5HD_C04, FZ_C04 */
static int64_t kFormatDimCIndex[] = {1, 3, 2, -1, -1, -1, -1, 1, 4, 3, -1, -1, -1, -1};

static inline int8_t GetFormatIdx(op::Format format)
{
    CHECK_RET(format >= 0 && format < sizeof(kFormatIndex) / sizeof(kFormatIndex[0]), -1);
    return kFormatIndex[format];
}

static inline int8_t GetDimCIdx(int8_t formatIdx)
{
    CHECK_RET(
        formatIdx >= 0 && static_cast<uint8_t>(formatIdx) < (sizeof(kFormatDimCIndex) / sizeof(kFormatDimCIndex[0])),
        -1);
    return kFormatDimCIndex[formatIdx];
}

static inline int64_t GetDimCFromX(const aclTensor* x)
{
    auto oriPrimaryFormat = op::GetPrimaryFormat(x->GetOriginalFormat());
    auto formatIdx = GetFormatIdx(oriPrimaryFormat);
    auto dimCIdx = GetDimCIdx(formatIdx);
    auto& oriShape = x->GetOriginalShape();
    CHECK_RET(dimCIdx >= 0 && static_cast<uint8_t>(dimCIdx) < oriShape.GetDimNum(), 0);
    return oriShape[dimCIdx];
}

const aclTensor* ReFormat(const aclTensor* x, const op::Format& format, aclOpExecutor* executor)
{
    auto formatIdx = kFormatIndex[format];
    if (formatIdx != -1) {
        auto formatRank = kFormatRank[formatIdx];
        const auto& viewShape = x->GetViewShape();
        auto shapeRank = static_cast<int64_t>(viewShape.GetDimNum());
        if (formatRank != -1 && formatRank != shapeRank) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "input tensor shape's rank does not match format, format is: %s, shape is: %s.",
                op::ToString(format).GetString(), op::ToString(viewShape).GetString());
            return nullptr;
        }
    }
    auto formatTensor = executor == nullptr ? const_cast<aclTensor*>(x) :
                                              executor->CreateView(x, x->GetViewShape(), x->GetViewOffset());
    formatTensor->SetViewFormat(format);
    formatTensor->SetOriginalFormat(format);
    formatTensor->SetStorageFormat(format);
    return formatTensor;
}

inline int64_t CalcC0Format(op::DataType dataType)
{
    int64_t blockSize = op::GetCurrentPlatformInfo().GetBlockSize();
    size_t typeSize = op::TypeSize(dataType);
    if (typeSize > kDataTypeSizeBitOffset) {
        blockSize = blockSize * NUM_8 / (typeSize - kDataTypeSizeBitOffset);
    } else if (typeSize > 0) {
        blockSize = blockSize / typeSize;
    }
    // blockSize is 2 ** (c0Format - 1)
    int64_t res = 1;
    while (blockSize > 1) {
        blockSize /= NUM_2;
        res += 1;
    }
    return res;
}

inline int64_t CalcC0FormatSpecial(op::DataType dataType)
{
    int64_t blockSize = op::GetCurrentPlatformInfo().GetBlockSize();
    size_t typeSize = op::TypeSize(dataType);
    if (typeSize > kDataTypeSizeBitOffset) {
        blockSize = blockSize * NUM_8 / (typeSize - kDataTypeSizeBitOffset);
    } else if (typeSize > 0) {
        if (typeSize == NUM_4) {
            blockSize = blockSize / NUM_2;
        } else {
            blockSize = blockSize / typeSize;
        }
    }
    int64_t res = 1;
    while (blockSize > 1) {
        blockSize /= NUM_2;
        res += 1;
    }
    return res;
}

static inline int64_t CalcC0FormatSpecialNZ(op::Format format)
{
    int64_t c0Size = 16;
    if (format == op::Format::FORMAT_FRACTAL_NZ_C0_32) {
        c0Size = c0Size * NUM_2;
    }
    int64_t res = 1;
    while (c0Size > 1) {
        c0Size /= NUM_2;
        res += 1;
    }
    return res;
}

inline int64_t MergeFormatSubFormatC0Format(op::Format dstPrimaryFormat, int64_t group, int64_t c0)
{
    return static_cast<int64_t>(
        static_cast<uint32_t>(dstPrimaryFormat) | static_cast<uint32_t>(group) << kBitNumOfOneByte |
        static_cast<uint32_t>(c0) << kBitThreeBytes);
}

static inline bool CheckPrimaryFormatValid(op::Format srcPrimaryFormat, op::Format dstPrimaryFormat)
{
    if ((sizeof(kFormatIndex) / sizeof(kFormatIndex[0])) < srcPrimaryFormat ||
        (sizeof(kFormatIndex) / sizeof(kFormatIndex[0])) < dstPrimaryFormat || kFormatIndex[srcPrimaryFormat] == -1 ||
        kFormatIndex[dstPrimaryFormat] == -1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "TransData not support: %s -> %s", op::ToString(srcPrimaryFormat).GetString(),
            op::ToString(dstPrimaryFormat).GetString());
        return false;
    }
    return true;
}

static inline bool CheckTransDataSupport(op::Format srcPrimaryFormat, op::Format dstPrimaryFormat)
{
    int8_t isSupport = 0;
    if (op::IsRegBase()) {
        bool isNd2Nz = (srcPrimaryFormat == op::Format::FORMAT_ND || srcPrimaryFormat == op::Format::FORMAT_NCL) &&
            (dstPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ ||
             dstPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16 ||
             dstPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ_C0_32);

        bool isNz2Nd = dstPrimaryFormat == op::Format::FORMAT_ND && 
            (srcPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ ||
             srcPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ_C0_2 ||
             srcPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ_C0_4 ||
             srcPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16 ||
             srcPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ_C0_32);
        if (isNd2Nz || isNz2Nd) {
            isSupport = 1;
        }
    } else if (
        dstPrimaryFormat != op::Format::FORMAT_FRACTAL_NZ_C0_16 &&
        dstPrimaryFormat != op::Format::FORMAT_FRACTAL_NZ_C0_32) {
        // check format trans in format list
        auto srcFormatIndex = kFormatIndex[srcPrimaryFormat];
        auto dstFormatIndex = kFormatIndex[dstPrimaryFormat];
        isSupport = kSupportMap[srcFormatIndex][dstFormatIndex];
    }
    if (isSupport == 0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "TransData not support: %s -> %s", op::ToString(srcPrimaryFormat).GetString(),
            op::ToString(dstPrimaryFormat).GetString());
        return false;
    }
    return true;
}

static inline bool CheckFormatShapeMatch(const op::Shape shape, op::Format primaryFormat)
{
    auto srcFormatIndex = kFormatIndex[primaryFormat];
    auto matchFormatRank = kFormatRank[srcFormatIndex];
    if (matchFormatRank != -1 && matchFormatRank != static_cast<int64_t>(shape.GetDimNum())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input tensor format not match it's shape, format is: %s, shape is: %s.",
            op::ToString(primaryFormat).GetString(), op::ToString(shape).GetString());
        return false;
    }
    return true;
}

static inline bool CheckSrcTensorValid(const aclTensor* x, op::Format srcPrimaryFormat)
{
    // check storage format shape match
    auto& storageShape = x->GetStorageShape();
    CHECK_RET(CheckFormatShapeMatch(storageShape, srcPrimaryFormat), false);

    auto& oriShape = x->GetOriginalShape();
    auto oriPrimaryFormat = op::GetPrimaryFormat(x->GetOriginalFormat());
    CHECK_RET(CheckFormatShapeMatch(oriShape, oriPrimaryFormat), false);
    return true;
}

static inline bool CheckTransDataParams(const aclTensor* x, op::Format srcPrimaryFormat, op::Format dstPrimaryFormat)
{
    // check primary format valid
    CHECK_RET(CheckPrimaryFormatValid(srcPrimaryFormat, dstPrimaryFormat), false);
    // check input tensor valid
    CHECK_RET(CheckSrcTensorValid(x, srcPrimaryFormat), false);
    // check transdata ability
    CHECK_RET(CheckTransDataSupport(srcPrimaryFormat, dstPrimaryFormat), false);
    return true;
}

static inline op::Format BuildDstFormat(op::Format dstPrimaryFormat, int64_t groups, op::DataType dataType)
{
    int64_t mergedFormat;
    if (dstPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16 ||
        dstPrimaryFormat == op::Format::FORMAT_FRACTAL_NZ_C0_32) {
        mergedFormat = MergeFormatSubFormatC0Format(dstPrimaryFormat, groups, CalcC0FormatSpecialNZ(dstPrimaryFormat));
    } else if (op::IsPrivateFormat(dstPrimaryFormat)) {
        mergedFormat = MergeFormatSubFormatC0Format(dstPrimaryFormat, groups, CalcC0Format(dataType));
    } else {
        mergedFormat = dstPrimaryFormat;
    }
    return static_cast<op::Format>(mergedFormat);
}

static inline op::Format BuildDstFormatSpecial(op::Format dstPrimaryFormat, int64_t groups, op::DataType dataType)
{
    int64_t mergedFormat;
    if (op::IsPrivateFormat(dstPrimaryFormat)) {
        mergedFormat = MergeFormatSubFormatC0Format(dstPrimaryFormat, groups, CalcC0FormatSpecial(dataType));
    } else {
        mergedFormat = dstPrimaryFormat;
    }
    return static_cast<op::Format>(mergedFormat);
}

OP_TYPE_REGISTER(TransData);

static inline bool IsTransDataFz(const aclTensor* x, op::Format dstPrimaryFormat, int64_t groups)
{
    auto srcFormat = x->GetStorageFormat();
    auto srcPrimaryFormat = op::GetPrimaryFormat(srcFormat);
    auto srcOriginFormat = x->GetOriginalFormat();
    if (srcOriginFormat != op::Format::FORMAT_NCHW && srcOriginFormat != op::Format::FORMAT_NCDHW) {
        return false;
    }
    if ((srcPrimaryFormat == op::Format::FORMAT_FRACTAL_Z || srcPrimaryFormat == op::Format::FORMAT_FRACTAL_Z_3D) &&
        op::GetSubFormat(srcFormat) > 1) {
        return true;
    }
    if ((dstPrimaryFormat == op::Format::FORMAT_FRACTAL_Z || dstPrimaryFormat == op::Format::FORMAT_FRACTAL_Z_3D) &&
        groups > 1) {
        return true;
    }

    return false;
}

static const aclTensor* TransDataToFzWithoutGroup(
    const aclTensor* x, op::Format srcPrimaryFormat, op::Format midFormat, aclOpExecutor* executor)
{
    L0_DFX(TransDataToFzWithoutGroup, x, srcPrimaryFormat);

    auto fzFormat = BuildDstFormat(midFormat, 1, x->GetDataType());
    auto out = executor->AllocTensor(x->GetDataType(), fzFormat, x->GetOriginalFormat());
    auto ret = INFER_SHAPE(
        TransData, OP_INPUT(x), OP_OUTPUT(out),
        OP_ATTR(op::ToString(srcPrimaryFormat).GetString(), op::ToString(midFormat).GetString(), 0, 0, 1));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        TransData, OP_INPUT(x), OP_OUTPUT(out),
        OP_ATTR(op::ToString(srcPrimaryFormat).GetString(), op::ToString(midFormat).GetString(), 0, 0, 1));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "TransData ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return out;
}

static const aclTensor* TransDataFzToDst(
    const aclTensor* x, op::Format dstPrimaryFormat, op::Format midFormat, int64_t groups, aclOpExecutor* executor)
{
    L0_DFX(TransDataFzToDst, x, dstPrimaryFormat, groups);

    if (groups > MAX_GROUPS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The groups %ld is larger than the max groups 65535!", groups);
        return nullptr;
    }

    auto fzFormat = BuildDstFormat(dstPrimaryFormat, groups, x->GetDataType());
    auto out = executor->AllocTensor(x->GetDataType(), fzFormat, x->GetOriginalFormat());
    // srcSubFormat, dstSubFormat is not used, use default 0
    auto ret = INFER_SHAPE(
        TransData, OP_INPUT(x), OP_OUTPUT(out),
        OP_ATTR(op::ToString(midFormat).GetString(), op::ToString(dstPrimaryFormat).GetString(), 0, 0, groups));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        TransData, OP_INPUT(x), OP_OUTPUT(out),
        OP_ATTR(op::ToString(midFormat).GetString(), op::ToString(dstPrimaryFormat).GetString(), 0, 0, groups));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "TransData ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return out;
}

static const aclTensor* TransDataFzWithGroup(
    const aclTensor* x, op::Format srcPrimaryFormat, op::Format dstPrimaryFormat, int64_t groups,
    aclOpExecutor* executor)
{
    op::Format midFormat;
    if (x->GetOriginalFormat() == op::Format::FORMAT_NCHW) {
        midFormat = op::Format::FORMAT_FRACTAL_Z;
    } else {
        midFormat = op::Format::FORMAT_FRACTAL_Z_3D;
    }

    // to FZ(group = 1)
    auto fzTensor = TransDataToFzWithoutGroup(x, srcPrimaryFormat, midFormat, executor);
    CHECK_RET(fzTensor != nullptr, nullptr);
    // to dst
    return TransDataFzToDst(fzTensor, dstPrimaryFormat, midFormat, groups, executor);
}

const aclTensor* TransData(const aclTensor* x, op::Format dstPrimaryFormat, int64_t groups, aclOpExecutor* executor)
{
    L0_DFX(TransData, x, dstPrimaryFormat, groups);
    CHECK_RET(x != nullptr, x);
    auto srcPrimaryFormat = op::GetPrimaryFormat(x->GetStorageFormat());
    dstPrimaryFormat = op::GetPrimaryFormat(dstPrimaryFormat);
    if (srcPrimaryFormat == dstPrimaryFormat) {
        return x;
    }
    if (!CheckTransDataParams(x, srcPrimaryFormat, dstPrimaryFormat)) {
        return nullptr;
    }
    if (IsTransDataFz(x, dstPrimaryFormat, groups)) {
        return TransDataFzWithGroup(x, srcPrimaryFormat, dstPrimaryFormat, groups, executor);
    }
    auto mergedFormat = BuildDstFormat(dstPrimaryFormat, groups, x->GetDataType());
    OP_LOGI("TransData out mergedFormat: %s", op::ToString(mergedFormat).GetString());
    auto out = executor->AllocTensor(x->GetDataType(), static_cast<op::Format>(mergedFormat), x->GetOriginalFormat());
    auto ret = INFER_SHAPE(
        TransData, OP_INPUT(x), OP_OUTPUT(out),
        OP_ATTR(op::ToString(srcPrimaryFormat).GetString(), op::ToString(dstPrimaryFormat).GetString(), groups));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        TransData, OP_INPUT(x), OP_OUTPUT(out),
        OP_ATTR(op::ToString(srcPrimaryFormat).GetString(), op::ToString(dstPrimaryFormat).GetString(), 0, 0, groups));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "TransData ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

OP_TYPE_REGISTER(TransDataSpecial);
/**
 * Special Transdata. Set the c0 size strictly based on the data type and chip block size.
 * this transdata c0 size rule:
 *     fp16: block_size/2
 *     fp32/int32/uint32: block_size/2
 *     int8/uint8: block_size/1
 * bool not supported, should do:
 *     (NCHW, bool)-> cast -> (NCHW, fp16) -> TransDataSpecial -> (5HD, fp16) -> cast -> (5HD, bool)
 *     (5HD, bool)-> cast -> (5HD, fp16) -> TransDataSpecial -> (NCHW, fp16) -> cast -> (NCHW, bool)
 *
 * @param x : aclTensor need to transpose
 * @param dstPrimaryFormat: dstPrimaryFormat like NC1HWC0
 * @param groups: groups
 * @param executor: executor should not be null
 * @return trans format tensor
 */
const aclTensor* TransDataSpecial(
    const aclTensor* x, op::Format dstPrimaryFormat, int64_t groups, aclOpExecutor* executor)
{
    L0_DFX(TransDataSpecial, x, dstPrimaryFormat, groups);
    CHECK_RET(x != nullptr, x);
    auto srcPrimaryFormat = op::GetPrimaryFormat(x->GetStorageFormat());
    dstPrimaryFormat = op::GetPrimaryFormat(dstPrimaryFormat);
    if (srcPrimaryFormat == dstPrimaryFormat) {
        return x;
    }
    if (!CheckTransDataParams(x, srcPrimaryFormat, dstPrimaryFormat)) {
        return nullptr;
    }
    auto mergedFormat = BuildDstFormatSpecial(dstPrimaryFormat, groups, x->GetDataType());
    OP_LOGI("TransDataSpecial out mergedFormat: %s", op::ToString(mergedFormat).GetString());
    auto out = executor->AllocTensor(x->GetDataType(), static_cast<op::Format>(mergedFormat), x->GetOriginalFormat());

    auto ret = INFER_SHAPE(
        TransData, OP_INPUT(x), OP_OUTPUT(out),
        OP_ATTR(op::ToString(srcPrimaryFormat).GetString(), op::ToString(dstPrimaryFormat).GetString(), groups));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        TransData, OP_INPUT(x), OP_OUTPUT(out),
        OP_ATTR(op::ToString(srcPrimaryFormat).GetString(), op::ToString(dstPrimaryFormat).GetString(), 0, 0, groups));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "TransDataSpecial ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

} // namespace l0op