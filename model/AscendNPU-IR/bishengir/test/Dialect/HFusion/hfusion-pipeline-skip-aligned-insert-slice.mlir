// RUN: bishengir-opt %s --lower-hfusion-pipeline="enable-triton-kernel-compile=true" | FileCheck %s --check-prefix=TRITON
// RUN: bishengir-opt %s --lower-hfusion-pipeline | FileCheck %s --check-prefix=DEFAULT

// Pipeline wiring: enable-triton-kernel-compile must pass skipAlignedSlice into
// normalize-tensor-ops so aligned insert_slice is not folded to concat (910B UB).

// TRITON-LABEL: func.func @pipeline_skip_aligned_insert_slice
// TRITON: tensor.insert_slice
// TRITON-NOT: tensor.concat

// DEFAULT-LABEL: func.func @pipeline_skip_aligned_insert_slice
// DEFAULT: tensor.concat
module {
  func.func @pipeline_skip_aligned_insert_slice(
      %src: tensor<1xf32>, %dst: tensor<4xf32>) -> tensor<4xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %inserted_slice = tensor.insert_slice %src into %dst[0] [1] [1]
        : tensor<1xf32> into tensor<4xf32>
    return %inserted_slice : tensor<4xf32>
  }
}
