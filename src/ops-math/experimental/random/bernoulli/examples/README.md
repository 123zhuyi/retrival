# aclnnBernoulli 调用样例

本目录包含两个标准两段式 ACLNN 调用样例：

| 文件 | 接口 | 说明 |
| --- | --- | --- |
| `test_aclnn_bernoulli.cpp` | `aclnnBernoulliGetWorkspaceSize` / `aclnnBernoulli` | out-of-place 标量概率调用 |
| `test_aclnn_inplace_bernoulli.cpp` | `aclnnInplaceBernoulliGetWorkspaceSize` / `aclnnInplaceBernoulli` | in-place 标量概率调用 |

先在仓库根目录构建并安装当前源码生成的 experimental 包，再加载安装环境：

```bash
bash build.sh --pkg --experimental --vendor_name=experimental \
  --soc=ascend910b --ops=bernoulli -j16

./build_out/cann-ops-math-experimental_linux-<arch>.run \
  --install-path=<install-path> --force
source <install-path>/vendors/experimental_math/bin/set_env.bash
```

使用仓库统一入口编译并运行默认 eager 样例：

```bash
bash build.sh --experimental --run_example bernoulli eager cust \
  --vendor_name=experimental --soc=ascend910b
```

仅编译、不执行时增加 `--noexec`。Atlas A3 将 `--soc` 改为
`ascend910_93`。运行前确保设备可用，且 `ASCEND_CUSTOM_OPP_PATH` 和
`LD_LIBRARY_PATH` 指向刚安装的 experimental 包。
