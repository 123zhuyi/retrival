# BISHENG
set(BISHENG "${ASCEND_DIR}/compiler/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")

# PYTHON
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Found Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Found Python3_LIBRARIES: ${Python3_LIBRARIES}")

# TORCH
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Torch_DIR "${TORCH_CMAKE_PATH}/Torch")
find_package(Torch REQUIRED)
message(STATUS "FoundTorch_DIR: ${Torch_DIR}")
message(STATUS "FoundTorch_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "FoundTorch_INCLUDES: ${TORCH_INCLUDE_DIRS}")

# TORCH NPU
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_NPU_INCLUDE_PATH "${TORCH_NPU_PATH}/include")
set(TORCH_NPU_LIB_PATH     "${TORCH_NPU_PATH}/lib")
message(STATUS "TORCH_NPU_PATH: ${TORCH_NPU_PATH}")
message(STATUS "TORCH_NPU_INCLUDE_DIR: ${TORCH_NPU_INCLUDE_PATH}")
message(STATUS "TORCH_NPU_LIB_DIR: ${TORCH_NPU_LIB_PATH}")

set(TORCH_EXTENSION_INCLUDE_DIRS
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/interface
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/tiling
    ${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api
    ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
)

set(TORCH_EXTENSION_LINK_DIRS
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_DIR}/lib64
)

set(TORCH_EXTENSION_LINK_LIBS
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
)

set(TORCH_EXTENSION_COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O2
    -fdiagnostics-color=always
    -DTORCH_MODE
    -w
)
