/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include <cmath>
#include "../op_kernel/round_tiling_data.h"
#include "../op_kernel/round_tiling_key.h"
#include "util/platform_util.h"

namespace optiling {
#define UB_DATA_NUM_INT32 4U // 对应DT_INT32类型的ub分块数量：x(x2), y(x2) --> 共4个
#define UB_DATA_NUM_F16_BF16_NO_DECIMAL \
    8U // 对应DT_FLOAT16/DT_BF16（无decimals）的ub分块数量：x(x2), y(x2), round_temp(x2), x_as_float32(x2) --> 共8个
#define UB_DATA_NUM_F16_BF16_WITH_DECIMAL \
    10U // 对应DT_FLOAT16/DT_BF16（有decimals）的ub分块数量：x(x2), y(x2), round_temp(x2), x_as_float32(x2),
        // x_scaled(x2) --> 共10个
#define UB_DATA_NUM_FLOAT_NO_DECIMAL 5U // 对应DT_FLOAT（无decimals）的ub分块数量：x(x2), y(x2), round_temp --> 共5个
#define UB_DATA_NUM_FLOAT_WITH_DECIMAL \
    6U                         // 对应DT_FLOAT（有decimals）的ub分块数量：x(x2), y(x2), round_temp, x_scaled --> 共6个
const uint64_t BUFFER_NUM = 2; // 做double buffer，计算优化tips
struct RoundCompileInfo {};

static ge::graphStatus TilingParseRound([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    RoundTilingData* tiling = context->GetTilingData<RoundTilingData>();
    uint64_t ubSize = 0; // Ub大小
    uint64_t bigCoreDataNum = 0;
    uint64_t bigTileNum = 0;
    uint64_t finalBigTileNum = 0;
    uint64_t bigTailDataNum = 0;
    int64_t decimals = 0;
    uint64_t ubDataNumber = 8;
    auto blockSize = Ops::Base::GetUbBlockSize(context);

    // 获取属性值
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* decimals_ptr = attrs->GetInt(0);
        if (decimals_ptr != nullptr)
            decimals = *decimals_ptr;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()); // 获取计算平台
    // 根据计算平台计算Ub大小，计算核一次能够处理的数据收到Ub大小的限制，因此需要根据Ub大小将每个核需要处理的数据进行批次拆分
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum(); // 获取有几个AIcore可用，未来分配AICore核数不要超过物理核数

    // 从context中获取第零个输入（也就是x），获取它的形状，获取形状总尺寸，也就是x的所有分量相乘（也就是总共需要计算这么多数值）
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    uint64_t inputNum = inputShape->GetStorageShape().GetShapeSize();

    // 获取第零个输入的描述信息（GetInputDesc），获取它的数据类型（GetDataType），计算该数据类型占的内存空间，有多少字节，赋值给typeLength
    uint32_t typeLength = 0;
    // ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::TypeUtils::GetDataTypeLength(inputDesc->GetDataType(), typeLength);
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t inputLength = inputNum * typeLength; // 输入数据占内存总大小

    // 没有核心可用，退出
    if (coreNum == 0 || blockSize == 0) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dataType = inputDesc->GetDataType();
    if (dataType == ge::DT_INT32) {
        ubDataNumber = UB_DATA_NUM_INT32;
    } else if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        if (decimals) {
            ubDataNumber = UB_DATA_NUM_F16_BF16_WITH_DECIMAL;
        } else {
            ubDataNumber = UB_DATA_NUM_F16_BF16_NO_DECIMAL;
        }
    } else if (dataType == ge::DT_FLOAT) {
        if (decimals) {
            ubDataNumber = UB_DATA_NUM_FLOAT_WITH_DECIMAL;
        } else {
            ubDataNumber = UB_DATA_NUM_FLOAT_NO_DECIMAL;
        }
    } else {
        // 不支持的类型，退出
        return ge::GRAPH_FAILED;
    }
    /* 每一小块数据块大小：
        总共能存（ubSize / blockSize）个block，
        每个block做buffer优化，能存ubSize / blockSize /BUFFER_NUM /个块数据
        总共要存储ubDataNumber个变量，每个变量存储ubSize / blockSize /BUFFER_NUM /ubDataNumber个数据块
    */
    uint64_t tileBlockNum = ubSize / blockSize / ubDataNumber;

    /*
        计算每个分块有多少个数值：
        总共tileBlockNum个数据块，每个数据块占blockSize（32字节），总共能存储tileBlockNum * blockSize个字节
        每个自然数据占typeLength个字节，总共单次可计算tileDataNum个数据
    */
    uint64_t tileDataNum = (tileBlockNum * blockSize) / typeLength;

    // 对输入数据做向上32对齐，因为最小的计算单元占32字节（一个blockSize）,固定公式
    uint64_t inputLengthAlgin32 = (((inputLength + blockSize - 1) / blockSize) * blockSize);

    // 如果一个核能计算完，就单核计算
    if (inputNum <= tileDataNum) {
        coreNum = 1;
    } else {
        // 计算所需核数，若4个核心，要计算5个block，那就用4个核心算，如果4个核心要计算3个block，那就用3个核心算
        coreNum = (coreNum < inputLengthAlgin32 / blockSize) ? coreNum : inputLengthAlgin32 / blockSize;
        // 万一算出来零，我们用1个核算
        coreNum = (coreNum >= 1) ? coreNum : 1;
    }

    // 每个核心需要计算的block数目
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / blockSize / coreNum;
    // 剩下大核需要计算的block数目，比如42个block，4个核计算，那就余两个block，前面的核心没人分一个，即有两个大核
    uint64_t tailBlockNum = (inputLengthAlgin32 / blockSize) % coreNum;

    // 小核要计算的数据
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * blockSize / typeLength;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 有大核，计算大核分配，若刚好，则不计算大核的分配
    if (0 != tailBlockNum) {
        everyCoreInputBlockNum += 1;
        bigCoreDataNum = everyCoreInputBlockNum * blockSize / typeLength;
        bigTileNum = bigCoreDataNum / tileDataNum;
        finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
        bigTailDataNum = bigCoreDataNum - (tileDataNum * bigTileNum);
        bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
    }
    tiling->smallCoreDataNum = ((uint64_t)smallCoreDataNum);
    tiling->bigCoreDataNum = ((uint64_t)bigCoreDataNum);
    tiling->tileDataNum = ((uint64_t)tileDataNum);
    tiling->smallTailDataNum = ((uint64_t)smallTailDataNum);
    tiling->bigTailDataNum = ((uint64_t)bigTailDataNum);
    tiling->finalSmallTileNum = ((uint64_t)finalSmallTileNum);
    tiling->finalBigTileNum = ((uint64_t)finalBigTileNum);
    tiling->tailBlockNum = ((uint64_t)tailBlockNum);
    float decimals_float = static_cast<float>(std::pow(10, decimals));
    context->SetTilingKey(0);
    tiling->decimals = ((float)decimals_float);
    context->SetBlockDim(coreNum);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(Round).Tiling(TilingFunc).TilingParse<RoundCompileInfo>(TilingParseRound);
} // namespace optiling
