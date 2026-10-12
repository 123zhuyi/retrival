/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/prod_env_mat_a_calc_rij_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
// A minimal two-atom system: atom0 at (0,0,0), atom1 at (1,0,0). With rcut_r = 2 the squared distance 1 < 4, so the
// two atoms are each other's only neighbor. DOUBLE is used to exercise the operator's double kernel path on device.
constexpr int32_t kNsample = 1;
constexpr int32_t kNall = 2;
constexpr int32_t kNloc = 2;
constexpr int32_t kNnei = 2; // sum(sel_a)
constexpr int32_t kNeighborMaxNum = 1024;
constexpr int32_t kCoordinateXyzNum = 3;
constexpr int32_t kResultNum = kNloc * kNnei;
constexpr int32_t kRijNum = kNloc * kNnei * kCoordinateXyzNum;
constexpr int32_t kMeshHeaderNum = 1;           // leading nlocnum slot
constexpr int32_t kMeshSectionsBeforeNeigh = 2; // ilist + numneigh sections, each of length kNloc
constexpr int32_t kMeshLen = kMeshHeaderNum + kMeshSectionsBeforeNeigh * kNloc + kNloc * kNeighborMaxNum;
constexpr float kRcutR = 2.0f;

const double kCoord[kNall * kCoordinateXyzNum] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
const int32_t kType[kNall] = {0, 0};
const int32_t kNatoms[3] = {kNloc, kNall, kNall};
const double kBox[9] = {0.0};
const std::vector<int64_t> kSelA = {2};

// Expected outputs (slot0 filled, slot1 keeps its init value).
const int32_t kExpectNlist[kResultNum] = {1, -1, 0, -1};
const double kExpectDistance[kResultNum] = {1.0, 5.0, 1.0, 5.0};
const double kExpectRij[kRijNum] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

string DataTypeToString(DataType dt)
{
    switch (dt) {
        case DT_FLOAT:
            return "DT_FLOAT";
        case DT_DOUBLE:
            return "DT_DOUBLE";
        case DT_INT32:
            return "DT_INT32";
        default:
            return "DTYPE(" + std::to_string(static_cast<int32_t>(dt)) + ")";
    }
}

Tensor BuildTensor(const vector<int64_t>& shape, DataType dtype, const void* data, size_t byteSize)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    Tensor tensor(desc, reinterpret_cast<const uint8_t*>(data), byteSize);
    return tensor;
}

// Build the mesh (neighbor-list) buffer: [nlocnum, ilist, numneigh, firstneigh(nlocnum * kNeighborMaxNum)].
std::vector<int32_t> BuildMesh()
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

int CreateOppInGraph(vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    vector<int64_t> coordShape = {kNsample, kNall * 3};
    vector<int64_t> typeShape = {kNsample, kNall};
    vector<int64_t> natomsShape = {3};
    vector<int64_t> boxShape = {kNsample, 9};
    vector<int64_t> meshShape = {static_cast<int64_t>(kMeshLen)};

    TensorDesc coordDesc(ge::Shape(coordShape), FORMAT_ND, DT_DOUBLE);
    coordDesc.SetPlacement(ge::kPlacementHost);
    auto coordData = op::Data("coord").set_attr_index(0);
    coordData.update_input_desc_x(coordDesc);
    coordData.update_output_desc_y(coordDesc);
    input.push_back(BuildTensor(coordShape, DT_DOUBLE, kCoord, sizeof(kCoord)));
    graph.AddOp(coordData);
    inputs.push_back(coordData);

    TensorDesc typeDesc(ge::Shape(typeShape), FORMAT_ND, DT_INT32);
    typeDesc.SetPlacement(ge::kPlacementHost);
    auto typeData = op::Data("type").set_attr_index(1);
    typeData.update_input_desc_x(typeDesc);
    typeData.update_output_desc_y(typeDesc);
    input.push_back(BuildTensor(typeShape, DT_INT32, kType, sizeof(kType)));
    graph.AddOp(typeData);
    inputs.push_back(typeData);

    TensorDesc boxDesc(ge::Shape(boxShape), FORMAT_ND, DT_DOUBLE);
    boxDesc.SetPlacement(ge::kPlacementHost);
    auto boxData = op::Data("box").set_attr_index(2);
    boxData.update_input_desc_x(boxDesc);
    boxData.update_output_desc_y(boxDesc);
    input.push_back(BuildTensor(boxShape, DT_DOUBLE, kBox, sizeof(kBox)));
    graph.AddOp(boxData);
    inputs.push_back(boxData);

    std::vector<int32_t> mesh = BuildMesh();
    TensorDesc meshDesc(ge::Shape(meshShape), FORMAT_ND, DT_INT32);
    meshDesc.SetPlacement(ge::kPlacementHost);
    auto meshData = op::Data("mesh").set_attr_index(3);
    meshData.update_input_desc_x(meshDesc);
    meshData.update_output_desc_y(meshDesc);
    input.push_back(BuildTensor(meshShape, DT_INT32, mesh.data(), mesh.size() * sizeof(int32_t)));
    graph.AddOp(meshData);
    inputs.push_back(meshData);

    // natoms drives the output shapes through data-dependent InferShape, so it must be a Const in the graph.
    TensorDesc natomsDesc(ge::Shape(natomsShape), FORMAT_ND, DT_INT32);
    natomsDesc.SetPlacement(ge::kPlacementHost);
    auto natomsConst = op::Const("natoms");
    natomsConst.set_attr_value(BuildTensor(natomsShape, DT_INT32, kNatoms, sizeof(kNatoms)));
    natomsConst.update_output_desc_y(natomsDesc);
    graph.AddOp(natomsConst);

    auto rijOp = op::ProdEnvMatACalcRij("prod_env_mat_a_calc_rij");
    rijOp.set_input_coord(coordData);
    rijOp.set_input_type(typeData);
    rijOp.set_input_natoms(natomsConst);
    rijOp.set_input_box(boxData);
    rijOp.set_input_mesh(meshData);
    rijOp.update_input_desc_coord(coordDesc);
    rijOp.update_input_desc_type(typeDesc);
    rijOp.update_input_desc_natoms(natomsDesc);
    rijOp.update_input_desc_box(boxDesc);
    rijOp.update_input_desc_mesh(meshDesc);
    rijOp.set_attr_rcut_a(1.0f);
    rijOp.set_attr_rcut_r(kRcutR);
    rijOp.set_attr_rcut_r_smth(1.0f);
    rijOp.set_attr_sel_a(kSelA);
    rijOp.set_attr_sel_r(std::vector<int64_t>{});
    graph.AddOp(rijOp);

    outputs.push_back(rijOp);
    return SUCCESS;
}

int CheckOutput(vector<ge::Tensor>& output)
{
    if (output.size() != 6U) {
        std::cout << "ERROR: expect 6 outputs, but got " << output.size() << std::endl;
        return FAILED;
    }

    const auto* rij = reinterpret_cast<const double*>(output[0].GetData());
    const auto* nlist = reinterpret_cast<const int32_t*>(output[1].GetData());
    const auto* distance = reinterpret_cast<const double*>(output[2].GetData());

    for (size_t i = 0; i < output.size(); i++) {
        std::cout << "output[" << i << "] dtype: " << DataTypeToString(output[i].GetTensorDesc().GetDataType())
                  << ", shape size: " << output[i].GetTensorDesc().GetShape().GetShapeSize() << std::endl;
    }

    std::cout << "nlist   :";
    for (int32_t i = 0; i < kResultNum; i++) {
        std::cout << " " << nlist[i];
    }
    std::cout << std::endl;
    std::cout << "distance:";
    for (int32_t i = 0; i < kResultNum; i++) {
        std::cout << " " << distance[i];
    }
    std::cout << std::endl;
    std::cout << "rij     :";
    for (int32_t i = 0; i < kRijNum; i++) {
        std::cout << " " << rij[i];
    }
    std::cout << std::endl;

    int ret = SUCCESS;
    for (int32_t i = 0; i < kResultNum; i++) {
        if (nlist[i] != kExpectNlist[i]) {
            std::cout << "ERROR: nlist[" << i << "] is " << nlist[i] << ", expect " << kExpectNlist[i] << std::endl;
            ret = FAILED;
        }
        if (distance[i] != kExpectDistance[i]) {
            std::cout << "ERROR: distance[" << i << "] is " << distance[i] << ", expect " << kExpectDistance[i]
                      << std::endl;
            ret = FAILED;
        }
    }
    for (int32_t i = 0; i < kRijNum; i++) {
        if (rij[i] != kExpectRij[i]) {
            std::cout << "ERROR: rij[" << i << "] is " << rij[i] << ", expect " << kExpectRij[i] << std::endl;
            ret = FAILED;
        }
    }
    return ret;
}

void PrintGeMessage()
{
    // On success these are empty; printing them unconditionally would emit confusing blank error lines.
    std::string errorStr(ge::GEGetErrorMsgV2().GetString());
    if (!errorStr.empty()) {
        std::cout << "Error message: " << errorStr << std::endl;
    }
    std::string warningStr(ge::GEGetWarningMsgV2().GetString());
    if (!warningStr.empty()) {
        std::cout << "Warning message: " << warningStr << std::endl;
    }
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    vector<Operator> inputs{};
    vector<Operator> outputs{};
    if (CreateOppInGraph(input, inputs, outputs, graph) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> build_options = {};
    ge::Session* session = new (std::nothrow) Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
        PrintGeMessage();
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        PrintGeMessage();
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int check_ret = CheckOutput(output);
    PrintGeMessage();

    delete session;
    if (ge::GEFinalize() != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }

    if (check_ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: ProdEnvMatACalcRij example result check failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: ProdEnvMatACalcRij example result check success\n", GetTime().c_str());
    return SUCCESS;
}
