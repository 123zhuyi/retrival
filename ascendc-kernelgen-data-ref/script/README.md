# 快速上手 · 从零部署到跑通

> **一句话**：加 `--proxy-port=8799` 参数跑批跑脚本，自动完成 ① 批量生成算子 ② 采集轨迹 ③ 清洗成训练数据。
> 不加这个参数 = 普通批跑（不采集轨迹）。

---

## 第一步：git clone

```bash
git clone -b br_data_distribution_update https://gitcode.com/sijiaali/ascendc-kernelgen-data.git
cd ascendc-kernelgen-data
```

> **注意**：批跑脚本会调 `claude -p` 生成算子，claude 需要加载 ascendc skill。如果你的 skill 是项目级安装的（`.claude/skills/`），必须在这个项目目录下跑批跑脚本；如果是全局安装的（`~/.claude/skills/`），任何目录都行。

clone 下来后 `script/` 目录长这样：

```
script/
├── run_benchmark_ascendc.sh     ← 批跑入口（唯一入口，用户只跑这个）
└── proxy/                        ← 所有支撑脚本（不用手动跑）
    ├── trace_proxy.sh              proxy 启停（含 disown 修复）
    ├── clean_pipeline.sh            一键清洗（shell 串联 5 步，不依赖 claude）
    ├── clean/                       4 个清洗 python 脚本
    │   ├── export_training.py
    │   ├── prepare_for_training.py
    │   ├── batch_evaluate.py
    │   └── check_dataset.py
    └── claude-code-proxy/           proxy 程序
        ├── server.py                   FastAPI 主程序
        ├── trace_db.py                 实时分库引擎
        ├── .env.example                配置模板
        └── static/trace.html           轨迹查看页
```

---

## 第二步：配置（首次一次性改完）

### 配置 1：proxy 的 `.env`

```bash
cd script/proxy/claude-code-proxy/
cp .env.example .env
```

编辑 `.env`，填你的真实上游和 key：

```bash
ANTHROPIC_BASE_URL="https://api.deepseek.com"   # 你的真实 LLM 上游
ANTHROPIC_API_KEY="sk-xxx"                       # 你的真实 key
PREFERRED_PROVIDER="anthropic"                   # 透传模式（不改写请求）
```

### 配置 2：claude 的 `settings.json`

批跑脚本已自动通过 `--settings` 参数注入 proxy 地址，**不需要在 settings.json 中配 `ANTHROPIC_BASE_URL`**。只保留模型和其他配置即可：

```json
{
  "env": {
    "ANTHROPIC_MODEL": "glm-5.2",
    "ANTHROPIC_AUTH_TOKEN": "sk-1234"
  }
}
```

> `ANTHROPIC_BASE_URL` 由脚本自动设为 `http://127.0.0.1:<proxy-port>`，无需手写。

### 配置 3：CANN 环境

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

---

## 第三步：跑命令 ⭐

```bash
cd ascendc-kernelgen-data

bash script/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 \
    --ids "3,4" \
    --npu-list "4,5" \
    --output ./output/level_1 \
    --proxy-port=8799
```

**参数说明**：

| 参数 | 必填 | 含义 |
|------|:---:|------|
| `--benchmark-dir` | ✅ | 算子描述文件根目录 |
| `--level` | ✅ | Level 编号（1/2/3/4） |
| `--ids "3,4"` 或 `--range 1-30` | ✅ | 哪些算子 |
| `--npu-list "4,5"` 或 `--npu 0` | ✅ | 哪些卡（多卡并行 / 单卡） |
| `--output` | ✅ | 算子产出放哪 |
| `--proxy-port=8799` | ⭐ | **加了 = 启用轨迹采集 + 自动清洗；不加 = 普通批跑** |

---

## 多窗口并行（多个 API Key）

当算子量大、一个 key 不够用时，开多个窗口，每个窗口用自己的 key 和模型跑一部分算子：

**窗口 1**（Kimi key）：
```bash
cd ascendc-kernelgen-data
bash script/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 --ids "1-30" --npu-list "0,1,2,3" \
    --output ./output/level_1 \
    --proxy-port=8799
```

**窗口 2**（GLM key）：
```bash
cd ascendc-kernelgen-data
# 修改 script/proxy/claude-code-proxy/.env 中的 key 和 ANTHROPIC_BASE_URL
# 然后：
bash script/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 --ids "31-60" --npu-list "4,5,6,7" \
    --output ./output/level_1 \
    --proxy-port=8800
```

**关键点**：
- 每个窗口 `--proxy-port` 必须不同（如 8799、8800、8801）
- 起批跑前先改 proxy `.env` 里的 key，改完再起（已跑的 proxy 不受影响）
- 算子 id 不能重叠（窗口1 跑 1-30，窗口2 跑 31-60）
- `--output` 同目录或不同目录都行；同目录算子自动合并，不同目录各自独立
- 脚本自动通过 `--settings` 注入对应端口，无需手动改 settings.json

---

## 第四步：看产出（3 类）

### ① 算子 kernel 代码 → `--output` 目录

```
output/level_1/
├── 3_Add/
│   ├── model_new_ascendc.py    ← 生成的 AscendC kernel
│   ├── design/ docs/           （设计 / 开发文档）
│   └── perf_report.md          （性能报告）
├── 4_Abs/
└── batch_report.md             （批量执行报告）
```

### ② 算子轨迹 db（每算子独立） → proxy 目录下

```
script/proxy/claude-code-proxy/cc_traces/batch_20260721_145351/
├── 3_Add.db         ← 算子 3 的完整 API 轨迹
├── 4_Abs.db         ← 算子 4 的完整 API 轨迹
└── trace.db         ← fallback（非算子请求，可忽略）
```

> 每次批跑生成独立目录（`batch_年月日_时分秒`），**重跑不覆盖**。

### ③ SFT 训练集（最终目标） → batch 目录里的 `cleaned/`

```
script/proxy/claude-code-proxy/cc_traces/batch_20260721_145351/cleaned/
├── check_report.json （检测报告：多少 PASS / FAIL）
├── eval_report.json  （NPU 评测报告）
├── pass/             ← ⭐ 通过的样本（SFT 训练数据）
└── fail/             （失败的样本）
```

---

## 跑通后的终端输出

```
[trace] proxy ready (port=8799)
[14:53:54] 所有 worker 已启动，准备 wait
[15:35:07] [NPU 12] ✅ 算子 4: 4_Abs.py 完成 (2473s)
[15:43:47] [NPU 11] ✅ 算子 3: 3_Add.py 完成 (2993s)
[15:43:47] ✅ wait 返回
[15:43:47] 清洗使用 NPU: 11,12
[clean] 清洗启动（2 个算子，NPU=11,12，jobs=4）
[clean] ① export_training...        ✅ 2/2
[clean] ② prepare...                ✅
[clean] ③ batch_evaluate...         ✅ PASS=1 FAIL=0 ERROR=1
[clean] ④ check_dataset...          ✅ pass=0, fail=12
[clean] 清洗完成：PASS=0, FAIL=12（合格数据在 cleaned/pass/）
[15:44:07] 批跑完成
```

> 清洗的详细日志在 `cleaned/clean.log`（python 完整输出）。

---

## 跑不通怎么办

| 问题 | 原因 | 解决 |
|------|------|------|
| `proxy 启动失败` | uvicorn 没装或 .env 没配 | `pip install uvicorn fastapi litellm` + 检查 `.env` |
| 端口占用 | 8799 被占 | 换端口：`--proxy-port=8800` |
| 卡在 `准备 wait` | 某个算子还没跑完 | 正常等（看 ✅ 日志判断进度） |
| `wait 返回` 后卡住 | 清洗阶段报错 | 看 `cleaned/clean.log` |
| 清洗 0 PASS | 算子精度不达标或 JSON 格式不对 | 看 `cleaned/eval_report.json` 和 `check_report.json` |

### 不重跑算子，只重新清洗

算子轨迹 db 已经在 batch 目录里了，直接调清洗脚本：

```bash
export EVAL_NPU_LIST="4,5"
bash script/proxy/clean_pipeline.sh \
    script/proxy/claude-code-proxy/cc_traces/batch_xxxx/ \
    ./output/level_1 \
    script/proxy/claude-code-proxy/cc_traces/batch_xxxx/cleaned/
```

---

## 附：冒烟测试

跑 2 个小算子验证全流程：

```bash
bash script/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 --ids "3,4" --npu-list "4,5" \
    --output ./smoke_test \
    --proxy-port=8799
```

跑通后应该看到：算子目录 ✓ → 算子 db ✓ → `cleaned/pass/` 有合格样本 ✓。
