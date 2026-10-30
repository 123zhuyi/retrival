# AscendOpTest 官方精度/性能验收（Polar）

任务要求"精度满足 AscendOpTest 工具默认阈值"，须用**官方工具**跑（非我们 pybind 复刻）。本目录为离线准备好的资产。

## 资产

| 文件 | 作用 |
|---|---|
| `../../../S8/Polar.json` | 算子描述文件（IR/原型，复用，`-i` 传入） |
| `polar_cases.json` | 测试用例（6 例：同 shape 小/16M + 广播低→高/标量/双向 + 高维非对齐），`-c` 传入 |
| `polar_golden.py` | `expect_func` CPU 基准（numpy 广播 polar），被 case json 引用 |

**关键点**：complex64 在 AscendOpTest `accuracy_config` **无内置默认** → 每个 `output_desc` 已显式写 `"err_threshold":[0.0001,0.0001]`（fp32 分量默认；[绝对偏差,错误率]），否则 run 时 KeyError。

## 在 NPU notebook 上运行

```bash
# 0) 前置
pip install ml_dtypes
# 1) 部署自定义 Polar 算子（构建+安装 .run，同 S8/Polar 流程）
cd ~/work/S8/Polar && bash build.sh && ./build_out/*.run --quiet
# 2) 关键：使部署算子生效（README 红字强调）
source $ASCEND_HOME_PATH/opp/vendors/customize/bin/set_env.bash
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
# 3) 取官方工具
git clone https://gitcode.com/HIT1920/AscendOpTest.git && cd AscendOpTest
# 4) 跑精度（--build 首次/改用例后必加；--op-type custom 自定义算子）
python run_test.py \
  -i /home/ma-user/work/S8/Polar.json \
  -c /home/ma-user/work/case_910b/Polar/aot/polar_cases.json \
  --op-type custom --build
# 5) 跑性能（msprof application 形式）
python run_test.py -i .../Polar.json -c .../polar_cases.json --op-type custom --msprof
```

结果在 `result.csv`（每用例 pass/fail）。`expect_func` 路径已按 `/home/ma-user/work/case_910b/Polar/aot/polar_golden.py:polar` 写死——若上传路径不同需同步改 `polar_cases.json`。

## 注意

- input/angle/out 的 name 与顺序须与 `Polar.json`、OpDef 一致（input, angle, out）——已对齐
- `Test_same_16M` 用例专门用于让官方工具**独立复核**之前 standalone 测出的 16M 非确定性问题（见项目记忆 ⚠️ 未决风险）
- 角度取 `[-3.14,3.14]`（单周期，干净查正确性）；abs 取 `[-10,10]`（含负值，验证负 abs 翻号合法）
