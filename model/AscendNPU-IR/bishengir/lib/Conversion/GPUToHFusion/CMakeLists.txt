add_bishengir_conversion_library(BiShengIRGPUToHFusion
  GPUToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/GPUToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRGPUDialect
  BiShengIRHFusionDialect
  MLIRTransformUtils
  MLIRRewrite
  MLIRPass
)
