//===- TestBufferUtils.cpp - Pass to test buffer utils --------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "Test/TestPasses.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/BufferUtils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

#include <set>

// Define the command-line option for fusion kind
static llvm::cl::opt<std::string> BufferUtilsTestVar(
    "test-buffer-utils-var",
    llvm::cl::desc("Specify the kind for testing modification"),
    llvm::cl::value_desc("modification"), llvm::cl::init(""));

namespace bishengir_test {
using namespace mlir;
using namespace mlir::utils;
using ArgsSet = std::set<std::string>;
namespace {
ArgsSet parseBufferUtilsTestVar() {
  ArgsSet result;
  llvm::StringRef input(BufferUtilsTestVar);
  SmallVector<llvm::StringRef> splitted;
  input.split(splitted, ',', -1, false);
  for (auto split : splitted)
    result.insert(split.str());
  return result;
}
} // namespace
struct TestBufferUtilsPass
    : public PassWrapper<TestBufferUtilsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBufferUtilsPass)

  StringRef getArgument() const final { return "test-buffer-utils"; }
  StringRef getDescription() const final { return "Test buffer utils"; }

  void runOnOperation() override {
    // Get the current operation being operated on.
    ArgsSet parsedVars = parseBufferUtilsTestVar();

    mlir::ModuleOp moduleOp = getOperation();
    // Walk over the functions in the module
    moduleOp.walk([&](func::FuncOp funcOp) {
      BufferAnalysisOptions options;
      if (parsedVars.count("pass-double-to-all-args")) {
        for (auto arg : funcOp.getArguments()) {
          auto copyOpUsers =
              llvm::make_filter_range(arg.getUsers(), [](Operation *user) {
                return (isa<hfusion::LoadOp>(user) ||
                        isa<hfusion::StoreOp>(user));
              });
          if (llvm::hasSingleElement(copyOpUsers)) {
            Operation *copyOp = *(copyOpUsers.begin());
            options.multiBufferCount[copyOp->getResult(0)] =
                2; /* 2 multi buffers */
          }
        }
      }

      if (parsedVars.count("enable-dma-opt"))
        options.enableDmaOpt = true;

      options.printLiveRange = true;
      auto maxBufferOut = mlir::utils::countMaxBuffer(funcOp, options);
      llvm::outs() << funcOp.getName() << ": " << maxBufferOut << "\n";
    });
  }
};
void registerTestBufferUtilsPass() { PassRegistration<TestBufferUtilsPass>(); }
} // namespace bishengir_test
