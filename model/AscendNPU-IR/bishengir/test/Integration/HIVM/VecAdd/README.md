# HIVM Vector Add

This is a simple example that demonstrates:
- how to compile `.mlir` into binary that can be executed on the Ascend NPU using `bishengir-compile`
- how to write a host wrapper that can launch the program on Ascend NPU

## Requirements

To run the example on Ascend NPU, please install the CANN software package and the set the environment variable `ASCEND_HOME_PATH` to the installed path. For example:

```bash
export ASCEND_HOME_PATH=/usr/local/ascend-toolkit/latest
```

## How to build the example

1. Prepare the requirements listed above.
2. Add `-DBISHENGIR_BUILD_EXAMPLES=ON` to the cmake build options. For example:

    ```bash
    cmake -G Ninja .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="AscendNPU-IR" \ # Project root directory
    -DBSPUB_DAVINCI_BISHENGIR=ON # Mandatory! Enables AscendNPU IR extensions for third-party repositories
    [Other CMake Options as required]
    -DBISHENGIR_BUILD_EXAMPLES=ON # Build integration examples
    ```

3. Build the project. After building, the executable `hivm-vec-add` should appear in `./bin` directory.
4. Build the device binary using `bishengir-compile`
    ```bash
    bishengir-compile add.mlir -enable-hivm-compile -o kernel.o
    ```
5. Execute `hivm-vec-add`
    ```bash
    ./hivm-vec-add
    ```
6. You should see the following results (some text are omitted):
    ```bash
    i0       Expect: 1                         Result: 1
    i1       Expect: 2                         Result: 2
    i2       Expect: 3                         Result: 3
    i3       Expect: 4                         Result: 4
    ```
