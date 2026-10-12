//===- InstructionMarker.cpp - Pass to number instrutions -----------------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

#include <set>

namespace bishengir_test {
using namespace mlir;
using ArgsSet = std::set<std::string>;
struct InstructionMarkerPass
    : public PassWrapper<InstructionMarkerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstructionMarkerPass)

  StringRef getArgument() const final { return "instruction-marker"; }
  StringRef getDescription() const final { return "Instruction marker"; }
  static constexpr StringLiteral kDebugAttr = "debug";
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    unsigned instructionCounter = 0;
    auto ctx = &getContext();
    OpBuilder builder(ctx);
    moduleOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<ModuleOp>(op))
        return;
      op->setAttr(kDebugAttr, builder.getIndexAttr(instructionCounter++));
    });
  }
};
void registerInstructionMarkerPass() {
  PassRegistration<InstructionMarkerPass>();
}
} // namespace bishengir_test
