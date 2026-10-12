/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

#include <vector>

using namespace std;
using namespace aicpu;

class TEST_PROD_ENV_MAT_A_CALC_RIJ_UT : public testing::Test {};

namespace {
constexpr int32_t kNeighborMaxNum = 1024;
constexpr int32_t kNloc = 2; // number of core-region atoms
constexpr int32_t kNnei = 2; // sum(sel_a)
constexpr int32_t kCoordinateXyzNum = 3;
constexpr int32_t kResultNum = kNloc * kNnei;                  // nlist / distance / rij_axis slot count
constexpr int32_t kRijNum = kNloc * kNnei * kCoordinateXyzNum; // rij slot count (xyz interleaved)
constexpr int32_t kMeshHeaderNum = 1;                          // leading nlocnum slot
constexpr int32_t kMeshSectionsBeforeNeigh = 2;                // ilist + numneigh sections, each of length kNloc
constexpr int32_t kMeshLen = kMeshHeaderNum + kMeshSectionsBeforeNeigh * kNloc + kNloc * kNeighborMaxNum;

#define CREATE_NODEDEF(node_def, shapes, data_types, datas, sel_a, rcut_r)     \
    NodeDefBuilder(node_def.get(), "ProdEnvMatACalcRij", "ProdEnvMatACalcRij") \
        .Input({"coord", data_types[0], shapes[0], datas[0]})                  \
        .Input({"type", data_types[1], shapes[1], datas[1]})                   \
        .Input({"natoms", data_types[2], shapes[2], datas[2]})                 \
        .Input({"box", data_types[3], shapes[3], datas[3]})                    \
        .Input({"mesh", data_types[4], shapes[4], datas[4]})                   \
        .Output({"rij", data_types[5], shapes[5], datas[5]})                   \
        .Output({"nlist", data_types[6], shapes[6], datas[6]})                 \
        .Output({"distance", data_types[7], shapes[7], datas[7]})              \
        .Output({"rij_x", data_types[8], shapes[8], datas[8]})                 \
        .Output({"rij_y", data_types[9], shapes[9], datas[9]})                 \
        .Output({"rij_z", data_types[10], shapes[10], datas[10]})              \
        .Attr("rcut_a", 1.0f)                                                  \
        .Attr("rcut_r", static_cast<float>(rcut_r))                            \
        .Attr("rcut_r_smth", 1.0f)                                             \
        .Attr("sel_a", sel_a)                                                  \
        .Attr("sel_r", std::vector<int64_t>{})

// Build the mesh (neighbor-list) buffer for a two-atom system where atom0 and atom1 are each other's only neighbor.
// Layout: [nlocnum, ilist(nlocnum), numneigh(nlocnum), firstneigh(nlocnum * kNeighborMaxNum)].
std::vector<int32_t> BuildTwoAtomMesh()
{
    const int32_t ilistOffset = kMeshHeaderNum;
    const int32_t numneighOffset = kMeshHeaderNum + kNloc;
    const int32_t firstneighOffset = kMeshHeaderNum + kMeshSectionsBeforeNeigh * kNloc;
    std::vector<int32_t> mesh(kMeshLen, 0);
    mesh[0] = kNloc;                              // nlocnum
    mesh[ilistOffset] = 0;                        // ilist[0] = atom0
    mesh[ilistOffset + 1] = 1;                    // ilist[1] = atom1
    mesh[numneighOffset] = 1;                     // atom0 has 1 neighbor
    mesh[numneighOffset + 1] = 1;                 // atom1 has 1 neighbor
    mesh[firstneighOffset] = 1;                   // firstneigh[0][0] = atom1
    mesh[firstneighOffset + kNeighborMaxNum] = 0; // firstneigh[1][0] = atom0
    return mesh;
}
} // namespace

// atom0=(0,0,0), atom1=(1,0,0); rcut_r=2 so squared distance 1 < 4 keeps them as neighbors.
// Expected (nnei=2, so slot0 is filled, slot1 keeps its init value):
//   nlist    = {1, -1, 0, -1}          (init -1)
//   distance = {1, 5,  1,  5}          (init rcut^2 + 1 = 5)
//   rij_x    = {1, 0, -1, 0}
//   rij_y/z  = {0, 0,  0, 0}
//   rij      = {1,0,0, 0,0,0, -1,0,0, 0,0,0}
TEST_F(TEST_PROD_ENV_MAT_A_CALC_RIJ_UT, DATA_TYPE_FLOAT_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT,
                                   DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 6},
                                      {1, 2},
                                      {3},
                                      {1, 9},
                                      {static_cast<int64_t>(kMeshLen)},
                                      {1, kRijNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum}};
    std::vector<int64_t> sel_a = {2};

    float coord[6] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    int32_t type[2] = {0, 0};
    int32_t natoms[3] = {2, 2, 2};
    float box[9] = {0.0f};
    std::vector<int32_t> mesh = BuildTwoAtomMesh();

    float rij[kRijNum] = {0.0f};
    int32_t nlist[kResultNum] = {0};
    float distance[kResultNum] = {0.0f};
    float rij_x[kResultNum] = {0.0f};
    float rij_y[kResultNum] = {0.0f};
    float rij_z[kResultNum] = {0.0f};

    vector<void*> datas = {(void*)coord, (void*)type,     (void*)natoms, (void*)box,   (void*)mesh.data(), (void*)rij,
                           (void*)nlist, (void*)distance, (void*)rij_x,  (void*)rij_y, (void*)rij_z};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, sel_a, 2.0f);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t expect_nlist[kResultNum] = {1, -1, 0, -1};
    float expect_distance[kResultNum] = {1.0f, 5.0f, 1.0f, 5.0f};
    float expect_rij_x[kResultNum] = {1.0f, 0.0f, -1.0f, 0.0f};
    float expect_rij[kRijNum] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int32_t i = 0; i < kResultNum; i++) {
        EXPECT_EQ(nlist[i], expect_nlist[i]);
        EXPECT_FLOAT_EQ(distance[i], expect_distance[i]);
        EXPECT_FLOAT_EQ(rij_x[i], expect_rij_x[i]);
        EXPECT_FLOAT_EQ(rij_y[i], 0.0f);
        EXPECT_FLOAT_EQ(rij_z[i], 0.0f);
    }
    for (int32_t i = 0; i < kRijNum; i++) {
        EXPECT_FLOAT_EQ(rij[i], expect_rij[i]);
    }
}

TEST_F(TEST_PROD_ENV_MAT_A_CALC_RIJ_UT, DATA_TYPE_DOUBLE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32,  DT_INT32,  DT_DOUBLE, DT_INT32, DT_DOUBLE,
                                   DT_INT32,  DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{1, 6},
                                      {1, 2},
                                      {3},
                                      {1, 9},
                                      {static_cast<int64_t>(kMeshLen)},
                                      {1, kRijNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum}};
    std::vector<int64_t> sel_a = {2};

    double coord[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    int32_t type[2] = {0, 0};
    int32_t natoms[3] = {2, 2, 2};
    double box[9] = {0.0};
    std::vector<int32_t> mesh = BuildTwoAtomMesh();

    double rij[kRijNum] = {0.0};
    int32_t nlist[kResultNum] = {0};
    double distance[kResultNum] = {0.0};
    double rij_x[kResultNum] = {0.0};
    double rij_y[kResultNum] = {0.0};
    double rij_z[kResultNum] = {0.0};

    vector<void*> datas = {(void*)coord, (void*)type,     (void*)natoms, (void*)box,   (void*)mesh.data(), (void*)rij,
                           (void*)nlist, (void*)distance, (void*)rij_x,  (void*)rij_y, (void*)rij_z};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, sel_a, 2.0f);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t expect_nlist[kResultNum] = {1, -1, 0, -1};
    double expect_distance[kResultNum] = {1.0, 5.0, 1.0, 5.0};
    double expect_rij_x[kResultNum] = {1.0, 0.0, -1.0, 0.0};
    for (int32_t i = 0; i < kResultNum; i++) {
        EXPECT_EQ(nlist[i], expect_nlist[i]);
        EXPECT_DOUBLE_EQ(distance[i], expect_distance[i]);
        EXPECT_DOUBLE_EQ(rij_x[i], expect_rij_x[i]);
    }
}

// atom1's type (5) is >= atom_types (= sel_a.size() = 1): atom0's only neighbor must be skipped instead of indexing
// sel out of range; atom1's neighbor (atom0, type 0) is still kept.
TEST_F(TEST_PROD_ENV_MAT_A_CALC_RIJ_UT, TYPE_OUT_OF_RANGE_SKIPPED_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT,
                                   DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 6},
                                      {1, 2},
                                      {3},
                                      {1, 9},
                                      {static_cast<int64_t>(kMeshLen)},
                                      {1, kRijNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum}};
    std::vector<int64_t> sel_a = {2};

    float coord[6] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    int32_t type[2] = {0, 5};
    int32_t natoms[3] = {2, 2, 2};
    float box[9] = {0.0f};
    std::vector<int32_t> mesh = BuildTwoAtomMesh();

    float rij[kRijNum] = {0.0f};
    int32_t nlist[kResultNum] = {0};
    float distance[kResultNum] = {0.0f};
    float rij_x[kResultNum] = {0.0f};
    float rij_y[kResultNum] = {0.0f};
    float rij_z[kResultNum] = {0.0f};

    vector<void*> datas = {(void*)coord, (void*)type,     (void*)natoms, (void*)box,   (void*)mesh.data(), (void*)rij,
                           (void*)nlist, (void*)distance, (void*)rij_x,  (void*)rij_y, (void*)rij_z};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, sel_a, 2.0f);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t expect_nlist[kResultNum] = {-1, -1, 0, -1};
    float expect_distance[kResultNum] = {5.0f, 5.0f, 1.0f, 5.0f};
    for (int32_t i = 0; i < kResultNum; i++) {
        EXPECT_EQ(nlist[i], expect_nlist[i]);
        EXPECT_FLOAT_EQ(distance[i], expect_distance[i]);
    }
}

// sel_a empty -> atom_types = 0, nnei = 0: a neighbor within rcut must not index the empty sel bucket (would crash).
TEST_F(TEST_PROD_ENV_MAT_A_CALC_RIJ_UT, EMPTY_SEL_A_NO_CRASH_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT,
                                   DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 6}, {1, 2}, {3},    {1, 9}, {static_cast<int64_t>(kMeshLen)}, {1, 0}, {1, 0},
                                      {1, 0}, {1, 0}, {1, 0}, {1, 0}};
    std::vector<int64_t> sel_a = {};

    float coord[6] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    int32_t type[2] = {0, 0};
    int32_t natoms[3] = {2, 2, 2};
    float box[9] = {0.0f};
    std::vector<int32_t> mesh = BuildTwoAtomMesh();

    float rij[1] = {0.0f};
    int32_t nlist[1] = {0};
    float distance[1] = {0.0f};
    float rij_x[1] = {0.0f};
    float rij_y[1] = {0.0f};
    float rij_z[1] = {0.0f};

    vector<void*> datas = {(void*)coord, (void*)type,     (void*)natoms, (void*)box,   (void*)mesh.data(), (void*)rij,
                           (void*)nlist, (void*)distance, (void*)rij_x,  (void*)rij_y, (void*)rij_z};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, sel_a, 2.0f);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_PROD_ENV_MAT_A_CALC_RIJ_UT, COORD_NOT_2D_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT,
                                   DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{6},
                                      {1, 2},
                                      {3},
                                      {1, 9},
                                      {static_cast<int64_t>(kMeshLen)},
                                      {1, kRijNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum}};
    std::vector<int64_t> sel_a = {2};

    float coord[6] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    int32_t type[2] = {0, 0};
    int32_t natoms[3] = {2, 2, 2};
    float box[9] = {0.0f};
    std::vector<int32_t> mesh = BuildTwoAtomMesh();
    float rij[kRijNum] = {0.0f};
    int32_t nlist[kResultNum] = {0};
    float distance[kResultNum] = {0.0f};
    float rij_x[kResultNum] = {0.0f};
    float rij_y[kResultNum] = {0.0f};
    float rij_z[kResultNum] = {0.0f};

    vector<void*> datas = {(void*)coord, (void*)type,     (void*)natoms, (void*)box,   (void*)mesh.data(), (void*)rij,
                           (void*)nlist, (void*)distance, (void*)rij_x,  (void*)rij_y, (void*)rij_z};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, sel_a, 2.0f);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_PROD_ENV_MAT_A_CALC_RIJ_UT, COORD_DTYPE_INVALID_FAIL)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT,
                                   DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 6},
                                      {1, 2},
                                      {3},
                                      {1, 9},
                                      {static_cast<int64_t>(kMeshLen)},
                                      {1, kRijNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum}};
    std::vector<int64_t> sel_a = {2};

    int32_t coord[6] = {0, 0, 0, 1, 0, 0};
    int32_t type[2] = {0, 0};
    int32_t natoms[3] = {2, 2, 2};
    float box[9] = {0.0f};
    std::vector<int32_t> mesh = BuildTwoAtomMesh();
    float rij[kRijNum] = {0.0f};
    int32_t nlist[kResultNum] = {0};
    float distance[kResultNum] = {0.0f};
    float rij_x[kResultNum] = {0.0f};
    float rij_y[kResultNum] = {0.0f};
    float rij_z[kResultNum] = {0.0f};

    vector<void*> datas = {(void*)coord, (void*)type,     (void*)natoms, (void*)box,   (void*)mesh.data(), (void*)rij,
                           (void*)nlist, (void*)distance, (void*)rij_x,  (void*)rij_y, (void*)rij_z};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, sel_a, 2.0f);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_PROD_ENV_MAT_A_CALC_RIJ_UT, NATOMS_TOO_SMALL_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT,
                                   DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 6},
                                      {1, 2},
                                      {2},
                                      {1, 9},
                                      {static_cast<int64_t>(kMeshLen)},
                                      {1, kRijNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum},
                                      {1, kResultNum}};
    std::vector<int64_t> sel_a = {2};

    float coord[6] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    int32_t type[2] = {0, 0};
    int32_t natoms[2] = {2, 2};
    float box[9] = {0.0f};
    std::vector<int32_t> mesh = BuildTwoAtomMesh();
    float rij[kRijNum] = {0.0f};
    int32_t nlist[kResultNum] = {0};
    float distance[kResultNum] = {0.0f};
    float rij_x[kResultNum] = {0.0f};
    float rij_y[kResultNum] = {0.0f};
    float rij_z[kResultNum] = {0.0f};

    vector<void*> datas = {(void*)coord, (void*)type,     (void*)natoms, (void*)box,   (void*)mesh.data(), (void*)rij,
                           (void*)nlist, (void*)distance, (void*)rij_x,  (void*)rij_y, (void*)rij_z};
    auto node_def = CpuKernelUtils::CreateNodeDef();
    CREATE_NODEDEF(node_def, shapes, data_types, datas, sel_a, 2.0f);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
