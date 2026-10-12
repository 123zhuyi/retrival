# aclnnBernoulli ACLNN ST

本目录使用 ops-math 现有 ATK 系统测试结构：

```text
aclnnBernoulli/
|-- atk_aclnnBernoulli.json
`-- executor_aclnnBernoulli.py
```

`atk_aclnnBernoulli.json` 包含 200 个标量概率用例；
`executor_aclnnBernoulli.py` 注册 ACLNN 执行器，并使用
`torch.bernoulli` 生成统计 golden。用例覆盖：

- FP16、FP32、FP64 和 BF16 输出与概率 dtype；
- rank 1-7 和多种 mask/tile 边界 shape；
- `prob=0`、`prob=1` 和一般概率；
- `offset=0/4/8/64` 等合法随机偏移。

执行前先构建、安装当前 checkout 的 experimental 包，并加载 vendor 环境：

```bash
bash build.sh --pkg --experimental --vendor_name=experimental \
  --soc=ascend910b --ops=bernoulli -j16
./build_out/cann-ops-math-experimental_linux-<arch>.run \
  --install-path=<install-path> --force
source <install-path>/vendors/experimental_math/bin/set_env.bash
```

随后按 ops-math CI/ATK 测试环境加载本目录的 JSON 和 executor。随机算子不做
逐元素固定值比较：必须同时检查输出值域为 0/1、样本均值满足统计阈值，并在
需要验证随机状态时使用相同的 `seed/offset`。
