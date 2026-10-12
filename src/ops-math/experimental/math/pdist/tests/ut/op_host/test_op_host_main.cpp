#include <filesystem>
#include <iostream>
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

class OpHostUtEnvironment : public testing::Environment {
public:
    OpHostUtEnvironment(char** argv) : argv_(argv) {}

    virtual void SetUp() {
        cout << "Global Environment SetUp." << endl;
        std::filesystem::path currDir = argv_[0];
        if (currDir.is_relative()) {
            currDir = std::filesystem::weakly_canonical(std::filesystem::current_path() / currDir);
        } else {
            currDir = std::filesystem::canonical(currDir);
        }
        string opHostSoPath = currDir.parent_path().string() + string("/libpdist_op_host_ut_lib.so");
        cout << "Loading op_host .so from: " << opHostSoPath << endl;
        gert::OppSoDesc oppSoDesc({ge::AscendString(opHostSoPath.c_str())}, "op_host_so");
        shared_ptr<gert::OpImplSpaceRegistryV2> opImplSpaceRegistryV2 = make_shared<gert::OpImplSpaceRegistryV2>();
        if (opImplSpaceRegistryV2->AddSoToRegistry(oppSoDesc) == ge::GRAPH_FAILED) {
            cerr << "Failed to add .so to registry." << endl;
            return;
        }
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(opImplSpaceRegistryV2);
        cout << "OpImplSpaceRegistryV2 initialized successfully" << endl;
    }

    virtual void TearDown() {
        cout << "Global Environment TearDown" << endl;
    }

private:
    char** argv_;
};

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new OpHostUtEnvironment(argv));
    return RUN_ALL_TESTS();
}
