//===-- Passes.td - Annotation pass definition file------------*-tablegen-*-==//
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

#ifndef BISHENGIR_DIALECT_ANNOTATION_PASSES
#define BISHENGIR_DIALECT_ANNOTATION_PASSES

include "mlir/Pass/PassBase.td"

def AnnotationLowering : Pass<"annotation-lowering"> {
  let summary = "Lowering the annotation dialect.";
  let constructor = "mlir::annotation::createAnnotationLoweringPass()";
}

#endif // BISHENGIR_DIALECT_ANNOTATION_PASSES
