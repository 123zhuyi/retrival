# Round Contract

## 校验规则

每轮完成后必须满足以下条件才能通过合同校验：

### 文件完整性

- `summary.json` 必须存在且包含所有必需字段
- `{op_name}_generated.py` 必须存在且大小 > 100 字节
- `output/iter_*/verify/verify_result.json` 必须存在（Phase 3 验证结果）
- 若 `phase4_entered=true`：`output/opt_iter_*/verify/verify_result_optimized.json` 必须存在
- 若 `phase4_entered=true`：`output/attempts.md` 必须存在且包含 ≥ 26 条优化点记录

### 精度要求

- Phase 3 `verify_result.json` 中 `passed_cases == total_cases > 0`
- Phase 4 优化侧 `verify_result_optimized.json` 中 `passed_cases == total_cases > 0`

### 优化真实性

- 至少有一个 `opt_iter_N/optimized_code.py` 与 `output/generated_code.py` 有字节级差异
- `output/attempts.md` 中至少包含 1 条"已尝试"或"已命中"记录

### 无条件检查

- 若 Phase 3 验证通过（passed == total），则 `phase4_entered` 必须为 `true`，不受 `success` 标志位影响
- `plateau_review` 轮次必须产出 `analysis.md`，且禁止存在 `opt_iter_*` 目录

### 轮次过渡

- 过渡到下一轮的**唯一入口**是 `transition_next_round.py` 脚本
- 禁止手动写入 `.triton-agent/state-{op_name}-{algorithm}-{run_tag}.json`、`round_index.json`、`opt-note.md` 或复制 baseline 代码

## 校验失败处理

| 失败类型 | 处理方式 |
|---------|---------|
| 文件缺失 | 修复后重跑 submit_round.py |
| 精度未通过 | 返回 Phase 3/4 修复 |
| 优化未真实执行 | 返回 Phase 4 至少执行一次真实的代码变更 + verify + benchmark |
| plateau_review 违规（有 Phase 4 产物） | 清理 opt_iter_* 和 phase4_entered 标记后重跑 |
| 无条件检查失败 | 根据具体失败原因返回对应 Phase |
