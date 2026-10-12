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
 * \file ts_nsa_compress_kernel.cpp
 * \brief NsaCompress kernel用例.
 */

#include "ts_nsa_compress.h"

namespace {

TEST_P(Ts_NsaCompress_WithParam_Ascend910B2, Tc_Kernel_NsaCompress)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_NsaCompress_Kernel_Case = ::testing::Values(

    NsaCompressCase("NsaCompress_Case_0", true, "",                         /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 1, 32,                           /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_1", true, "",                         /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 1, 1024,                         /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_2", true, "",                         /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 4, 128,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_3", true, "",                         /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 4, 192,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_4", true, "",                         /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 4, 256,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_5", true, "",                         /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 6, 128,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_6", true, "",                         /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 6, 192,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_7", true, "",                         /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 6, 256,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_random_8", true, "",                  /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(2290, 24, 192,                         /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {6,    93,   129,  184,  189,  284,  287,  323,  353,  391,  450,  495,  532,
                                      558,  657,  702,  774,  873,  945,  981,  1076, 1126, 1132, 1140, 1185, 1218,
                                      1236, 1323, 1345, 1389, 1417, 1491, 1582, 1676, 1681, 1694, 1706, 1772, 1838,
                                      1910, 1961, 2027, 2075, 2168, 2193, 2200, 2238, 2290}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     32, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_random_9", true, "",                  /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(2576, 4, 128,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {99,   178,  271,  272,  289,  302,  360,  410,  485,  576,  581,  588,  657,
                                      746,  818,  881,  930,  967,  978,  1037, 1101, 1179, 1256, 1329, 1392, 1465,
                                      1487, 1503, 1592, 1664, 1733, 1761, 1807, 1902, 1932, 1937, 2018, 2018, 2060,
                                      2062, 2073, 2169, 2254, 2328, 2393, 2404, 2501, 2576}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_random_10", true, "",                 /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(2320, 4, 192,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {93,   144,  150,  180,  204,  282,  330,  377,  417,  476,  503,  562,  643,
                                      709,  765,  798,  887,  975,  988,  1073, 1092, 1152, 1174, 1223, 1305, 1357,
                                      1384, 1389, 1399, 1403, 1442, 1454, 1510, 1606, 1696, 1767, 1810, 1856, 1917,
                                      1991, 2027, 2067, 2073, 2087, 2172, 2223, 2225, 2320}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_random_11", true, "",                 /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(1510, 4, 192,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {93,   144,  150,  180,  204,  282,  330,  377,  417,  476,  503,  562,
                                      643,  709,  765,  798,  887,  975,  988,  1073, 1092, 1152, 1174, 1223,
                                      1305, 1357, 1384, 1389, 1399, 1403, 1442, 1454, 1510}, /* ActualSeqLenList
                                                                                              */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_BF16_12", true, "",                   /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                           ExpectInfo(true,                                 /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 9, 32,                           /* T, N, D */
                                     ge::DataType::DT_BF16,                 /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /*
                          ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16, /*CompressBlockSize*/ 16, /*CompressStride*/ 0      /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_Layout_13", true, "",                 /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                           ExpectInfo(false,                                /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 17, 32,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /* ActualSeqLenList */
                                     LayoutType::BSH,                                        /* Layout */
                                     16,                                                     /*CompressBlockSize*/
                                     16,                                                     /*CompressStride*/
                                     0                                                       /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_CompressBlockSize_14", true, "",      /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                           ExpectInfo(false,                                /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 32, 32,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /* ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     7,                                                      /*CompressBlockSize*/
                                     16,                                                     /*CompressStride*/
                                     0                                                       /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_CompressStride_15", true, "",         /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                           ExpectInfo(false,                                /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 21, 32,                          /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /* ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16,                                                     /*CompressBlockSize*/
                                     31,                                                     /*CompressStride*/
                                     0                                                       /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_CompressBlockSize_CompressStride_16", true, "", /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, false),                                  /* RunTiling, RunKernel */
                           ExpectInfo(false,                                          /* ExpectSuccess */
                                      0,                                              /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)),           /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 23, 32,                                    /* T, N, D */
                                     ge::DataType::DT_FLOAT16,                        /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /* ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     16,                                                     /*CompressBlockSize*/
                                     32,                                                     /*CompressStride*/
                                     0                                                       /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_ActSeqLenType_17", true, "",          /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                           ExpectInfo(false,                                /* ExpectSuccess */
                                      0,                                    /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    NsaCompressParam(4800, 5, 32,                           /* T, N, D */
                                     ge::DataType::DT_FLOAT16,              /* Dtype */
                                     {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200, 1300,
                                      1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                                      2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
                                      4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800}, /* ActualSeqLenList */
                                     LayoutType::TND,                                        /* Layout */
                                     32,                                                     /*CompressBlockSize*/
                                     32,                                                     /*CompressStride*/
                                     1                                                       /*ActSeqLenType*/
                                     )),
    NsaCompressCase("NsaCompress_Case_EmptyOutput_18", true, "",                      /* CaseName, Enable, DebugInfo */
                    OpInfo(ControlInfo(true, false),                                  /* RunTiling, RunKernel */
                           ExpectInfo(false,                                          /* ExpectSuccess */
                                      0,                                              /* ExpectTilingKey */
                                      ExpectInfo::kInvalidTilingBlockDim)),           /* ExpectTilingBlockDim */
                    NsaCompressParam(310, 5, 32,                                      /* T, N, D */
                                     ge::DataType::DT_FLOAT16,                        /* Dtype */
                                     {31, 62, 93, 124, 155, 186, 217, 248, 279, 310}, /* ActualSeqLenList */
                                     LayoutType::TND,                                 /* Layout */
                                     32,                                              /*CompressBlockSize*/
                                     32,                                              /*CompressStride*/
                                     1                                                /*ActSeqLenType*/
                                     ))

);

INSTANTIATE_TEST_SUITE_P(NsaCompress, Ts_NsaCompress_WithParam_Ascend910B2, Tc_NsaCompress_Kernel_Case);
} // namespace
