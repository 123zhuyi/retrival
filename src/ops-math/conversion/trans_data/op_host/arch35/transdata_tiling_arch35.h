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
 * \file transdata_tiling_arch35.h
 * \brief transdata tiling for ascendC impl
 */
#ifndef OPS_MATH_CONVERSION_TRANSDATA_TILING_ARCH35_H_
#define OPS_MATH_CONVERSION_TRANSDATA_TILING_ARCH35_H_

#include <cstdint>
#include <string>

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

struct TransDataCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
    int64_t blockSize;
};

BEGIN_TILING_DATA_DEF(TransDataASCTilingData)
TILING_DATA_FIELD_DEF(int64_t, c0);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, tNum); // thread number
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TransDataNzToNdTilingData)
TILING_DATA_FIELD_DEF(int64_t, c0);
TILING_DATA_FIELD_DEF(int64_t, c1);
TILING_DATA_FIELD_DEF(int64_t, n0);
TILING_DATA_FIELD_DEF(int64_t, n1);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, c);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TransData, TransDataASCTilingData);
REGISTER_TILING_DATA_CLASS(TransData_21002, TransDataNzToNdTilingData);

ge::graphStatus Tiling4TransDataAscendC(gert::TilingContext* context);

namespace transdata_asc {
constexpr int64_t TILING_MODE_SIMT = 21000;
constexpr int64_t TILING_MODE_SIMT_LARGE_SHAPE = 21001;
constexpr int64_t TILING_MODE_SIMT_NZ_TO_ND = 21002;
constexpr int64_t MAX_INT32_SIZE = 0x7fffffff;
constexpr size_t B4_BIT_SIZE = 4;
constexpr size_t nFour = 4;
constexpr size_t nThree = 3;
constexpr size_t nTwo = 2;
constexpr size_t kSyncWorkSpaceSize = static_cast<size_t>(16) * 1024 * 1024;
constexpr int64_t tNum256 = 256;
constexpr int64_t tNum512 = 512;
constexpr int64_t C0_2 = 2;
constexpr int64_t C0_4 = 4;
constexpr int64_t C0_8 = 8;
constexpr int64_t C0_16 = 16;
constexpr int64_t C0_32 = 32;
constexpr int64_t N0_16 = 16;
constexpr int64_t SIMT_RSV_SIZE = 128 * 1024L;

class TransDataTilingAscendC {
public:
    explicit TransDataTilingAscendC(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus DoTiling();
    ge::graphStatus DoNz2NdTiling();
    ge::graphStatus GetHardwareInfo();

private:
    ge::graphStatus CalcTilingData();
    bool GetTransFormatAndDType();
    bool GetTransNz2NdFormatAndDType();
    bool GetShapeInfo();
    bool CalcC0Size();
    void CalcHSize();
    void CalcNCSize();
    bool CalcNzToNdShapeSize();
    void CalcBlockAndThreadNum();
    void ReshapeInShape();
    void CalcTilingKey();
    void WriteTilingData();
    void WriteNzToNdTilingData();
    std::string PrintTilingData();
    std::string PrintNz2NdTilingData();

private:
    gert::TilingContext* context_ = nullptr;
    TransDataASCTilingData tilingData_;
    TransDataNzToNdTilingData tilingNzToNdData_;
    gert::Shape inShape_;
    gert::Shape outShape_;
    ge::Format dstFormat_{ge::FORMAT_MD};
    size_t dtypeSize_{0};
    ge::DataType srcDtype_{ge::DT_UNDEFINED};

    uint32_t coreNum_{1};
    uint32_t bNum_{0};
    int64_t ubSize_{0};
    int64_t blockSize_{0};
    int64_t tilingKey_{TILING_MODE_SIMT};

    int64_t expectC0_{0};
    int64_t c0_{0};
    int64_t h_{0};
    int64_t n_{0};
    int64_t c_{0};
    int64_t c1_{0};
    int64_t n0_{0};
    int64_t n1_{0};
    int64_t tNum_{0};
};
} // namespace transdata_asc

} // namespace optiling
#endif // OPS_MATH_CONVERSION_TRANSDATA_TILING_ARCH35_H_
