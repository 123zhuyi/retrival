# AddExamplePypto

`examples/add_example` 的 PyPTO 版本：功能同为逐元素加法，区别在于 kernel 由 PyPTO DSL
(`op_kernel/add_example_pypto.py`) 编写，而不是 AscendC C++。本算子用于演示 ops-cv 中 PyPTO 算子的接入方式。

## 编译

```bash
source <ascend-toolkit>/set_env.sh
bash build.sh --pkg --ops=add_example_pypto --soc=ascend950
```

## 接入方式

在算子的 `op_host/CMakeLists.txt` 中，于 `add_modules_sources`（或 `add_all_modules_sources`）**之前**加两行：

```cmake
require_pypto_pro(add_example_pypto)
enable_pypto_kernel(add_example_pypto)

add_modules_sources(OPTYPE add_example_pypto ACLNNTYPE aclnn)
```

- `require_pypto_pro(<op_name>)`：环境中没有 `pypto_pro` 时告警并跳过本算子，不会让整个 configure 失败。
- `enable_pypto_kernel(<op_file>)`：`<op_file>` 需与 `op_kernel/<op_file>.py`、以及算子定义里
  `ExtendCfgInfo("opFile.value", ...)` 三者保持一致。

## 构建流程

`enable_pypto_kernel` 在 **configure 阶段**调用 `scripts/util/pypto_codegen.py`，对 kernel 的 `.py` 做 codegen：

| 产物 | 用途 |
| --- | --- |
| `<Tiling>_tiling.h` | tiling 结构体，host 侧 tiling 直接使用 |
| `<TilingKey>_tilingkey.h` | `ASCENDC_TPL` tiling key 声明 |
| `<op_file>_pypto_infer.cpp` | 供 `.i` infer 使用的自包含源码 |

上面两个头文件会以 `-include` 的方式**强制包含**到本算子的 `*_tiling*.cpp`（顺序：tilingkey 在前，tiling 在后），
因此 host 侧 tiling 无需显式 include，即可直接使用 `.py` 中定义的 tiling 结构体：

```cpp
// AddExamplePyptoTiling 来自 op_kernel/add_example_pypto.py，经 force-include 可见
AddExamplePyptoTiling* tilingData = context->GetTilingData<AddExamplePyptoTiling>();
```

host 与 kernel 因此共享同一份 tiling 布局与 tiling key 位分配，无需手写副本。

**build 阶段**，本算子生成的 wrapper (`<build>/tbe/dynamic/<op_file>.py`) 会调用 `pypto_compile_op`
而不是 `compile_op`，由它完成按 tiling key 的 codegen 并复用 AscendC 后端编出 kernel 二进制。

## 芯片与 arch

PyPTO 的 codegen arch 由平台自动推导（`a2`/`a3`/`a5`）。本算子只注册了 `ascend950`(a5)，与当前 toolkit
自动推导的 arch 一致。编译 kernel 二进制时，cmake 会按所编 soc 导出 `PYPTO_JIT_ARCH`
（`ascend910b`/`ascend910_93` → `a3`，`ascend950` → `a5`），避免 `pypto_compile_op` 回退到默认的 `a5`
而产生目标芯片不支持的指令。
