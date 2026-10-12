# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MCTS 树核心实现（适配 triton-optimizer-multi-agent 版本）.

与原版的区别：
1. 超参数（C_UCT、C_PW、ALPHA 等）作为实例属性，可在构造时通过 params 覆盖。
2. select() 只考虑 ACTIVE 子节点，忽略 PENDING 子节点，避免选中尚未评估的 round。
3. create_child() 支持传入自定义 node_id，使新节点可直接命名为 opt-round-{N}。
4. 移除与原 MCTS Agent Loop 主循环/子 agent 启动器相关的依赖。
"""

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

from .config import (
    C_UCT,
    C_PW,
    ALPHA,
    FAILURE_THRESHOLD,
    REWARD_OUTPUT_ERROR,
    REWARD_COMPILE_ERROR,
    reward_success,
)


TREE_STATE_VERSION = "1.1.0"


class NodeStatus:
    PENDING = "pending"
    ACTIVE = "active"
    PRUNED = "pruned"
    EVALUATED = "evaluated"


class ErrorType:
    NONE = "none"
    OUTPUT_ERROR = "output_error"
    COMPILE_ERROR = "compile_error"


@dataclass
class Node:
    """MCTS 树节点."""
    node_id: str
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    depth: int = 0
    visit_count: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    total_failures: int = 0
    status: str = NodeStatus.PENDING
    error_type: str = ErrorType.NONE
    code_path: str = ""
    verify_result_path: str = ""
    perf_result_path: str = ""
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(**data)


@dataclass
class SearchMemory:
    """单次任务的 Search Memory，记录历史失败与已探索方向."""
    failures: List[Dict[str, Any]] = field(default_factory=list)
    explored_directions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchMemory":
        return cls(
            failures=deepcopy(data.get("failures", [])),
            explored_directions=deepcopy(data.get("explored_directions", [])),
        )

    def record_failure(self, error_type: str, parent_info: str, error_msg: str, direction: str = "") -> None:
        """记录一次失败；若同类失败已存在则增加计数."""
        error_msg = error_msg[:1000]
        parent_info = parent_info[:500]
        for f in self.failures:
            if f["error_type"] == error_type and f.get("parent_info", "") == parent_info:
                f["count"] = f.get("count", 1) + 1
                if error_msg and error_msg not in f.get("error_msg", ""):
                    f["error_msg"] = f.get("error_msg", "") + "\n" + error_msg
                    f["error_msg"] = f["error_msg"][-1500:]
                return
        self.failures.append({
            "error_type": error_type,
            "parent_info": parent_info,
            "error_msg": error_msg,
            "count": 1,
        })
        if direction and direction not in self.explored_directions:
            self.explored_directions.append(direction)

    def summary(self) -> str:
        """生成给 sub agent 的文本摘要."""
        lines = ["Search Memory Summary:"]
        if not self.failures:
            lines.append("- No historical failures recorded.")
        else:
            lines.append("- Historical failures to avoid:")
            for f in self.failures:
                lines.append(
                    f"  [{f['error_type']}] count={f.get('count', 1)}: {f.get('error_msg', '')[:200]}"
                )
        if self.explored_directions:
            lines.append("- Explored directions:")
            for d in self.explored_directions:
                lines.append(f"  - {d}")
        return "\n".join(lines)


@dataclass
class MCTSTree:
    """MCTS 树容器."""
    version: str
    task_id: str
    task_goal: str
    output_dir: str
    root_id: str
    nodes: Dict[str, Node]
    search_memory: SearchMemory
    next_node_id: int
    c_uct: float = field(default=C_UCT)
    c_pw: float = field(default=C_PW)
    alpha: float = field(default=ALPHA)
    failure_threshold: int = field(default=FAILURE_THRESHOLD)
    reward_output_error: float = field(default=REWARD_OUTPUT_ERROR)
    reward_compile_error: float = field(default=REWARD_COMPILE_ERROR)
    reward_success_fn: Callable[[float], float] = field(default=reward_success)
    pending_expansion: Optional[Dict[str, Any]] = field(default=None)
    round_dir_map: Dict[str, str] = field(default_factory=dict)
    tree_handle: str = ""

    @classmethod
    def create(
        cls,
        task_id: str,
        task_goal: str,
        output_dir: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> "MCTSTree":
        """创建一棵新树."""
        os.makedirs(output_dir, exist_ok=True)
        root = Node(
            node_id="root",
            parent_id=None,
            depth=0,
            status=NodeStatus.ACTIVE,
            description="virtual root",
        )
        p = params or {}
        tree = cls(
            version=TREE_STATE_VERSION,
            task_id=task_id,
            task_goal=task_goal,
            output_dir=os.path.abspath(output_dir),
            root_id="root",
            nodes={"root": root},
            search_memory=SearchMemory(),
            next_node_id=1,
            c_uct=float(p.get("c_uct", C_UCT)),
            c_pw=float(p.get("c_pw", C_PW)),
            alpha=float(p.get("alpha", ALPHA)),
            failure_threshold=int(p.get("failure_threshold", FAILURE_THRESHOLD)),
            reward_output_error=float(p.get("reward_output_error", REWARD_OUTPUT_ERROR)),
            reward_compile_error=float(p.get("reward_compile_error", REWARD_COMPILE_ERROR)),
            reward_success_fn=p.get("reward_success_fn", reward_success),
            pending_expansion=None,
            round_dir_map={},
            tree_handle="",
        )
        tree._update_tree_handle()
        tree.save()
        return tree

    def _update_tree_handle(self) -> None:
        self.tree_handle = os.path.join(self.output_dir, f"{self.task_id}_mcts_tree.json")

    def _generate_node_id(self) -> str:
        nid = f"node_{self.next_node_id}"
        self.next_node_id += 1
        return nid

    def _max_children(self, visit_count: int) -> int:
        """Progressive widening 上限."""
        if visit_count <= 0:
            return 1
        return max(1, int(math.floor(self.c_pw * (visit_count ** self.alpha))))

    def _uct_existing_child(self, child: Node, parent_visits: int) -> float:
        """已有子节点的 UCT 值."""
        if child.visit_count == 0:
            return child.mean_reward + self.c_uct * math.sqrt(math.log(max(1, parent_visits)))
        exploitation = child.mean_reward
        exploration = self.c_uct * math.sqrt(math.log(parent_visits) / child.visit_count)
        return exploitation + exploration

    def _uct_new_node(self, parent: Node) -> float:
        """新建节点的 UCT 值：假设访问次数为 1，继承父节点平均奖励."""
        parent_visits = max(1, parent.visit_count)
        exploitation = parent.mean_reward
        exploration = self.c_uct * math.sqrt(math.log(parent_visits) / 1.0)
        return exploitation + exploration

    def select(self, node_id: str = "root") -> Dict[str, Any]:
        """从指定节点出发，按 UCT 选择至扩展点.

        与原版不同：只考虑 ACTIVE 子节点，避免选中尚未评估的 PENDING round。
        """
        node = self.nodes.get(node_id)
        if node is None:
            return {"action": "none", "node_id": node_id, "parent_id": None,
                    "is_new_node": False, "depth": 0,
                    "context_for_model": "Error: node not found."}

        if node.status == NodeStatus.PRUNED:
            return {"action": "none", "node_id": node_id, "parent_id": node.parent_id,
                    "is_new_node": False, "depth": node.depth,
                    "context_for_model": "Error: node is pruned."}

        # 仅 ACTIVE 节点可被视为已探索的父节点。
        active_children = [self.nodes[cid] for cid in node.children
                           if self.nodes.get(cid, Node("", "")).status == NodeStatus.ACTIVE]
        max_children = self._max_children(node.visit_count)
        can_create = len(active_children) < max_children

        best_value = float("-inf")
        best_child: Optional[Node] = None
        best_is_new = False

        for child in active_children:
            uct = self._uct_existing_child(child, max(1, node.visit_count))
            if uct > best_value:
                best_value = uct
                best_child = child
                best_is_new = False

        if can_create:
            uct_new = self._uct_new_node(node)
            if uct_new > best_value:
                best_value = uct_new
                best_child = None
                best_is_new = True

        if best_child is None and not best_is_new:
            return {"action": "none", "node_id": node_id, "parent_id": node.parent_id,
                    "is_new_node": False, "depth": node.depth,
                    "context_for_model": "No active children and cannot expand further."}

        if best_is_new:
            context = self._build_context(node)
            return {"action": "create_child", "node_id": node_id, "parent_id": node.parent_id,
                    "is_new_node": True, "depth": node.depth,
                    "context_for_model": context}

        return self.select(best_child.node_id)

    def _build_context(self, parent: Node) -> str:
        """为 sub agent 构建上下文文本（保留，便于调试和可视化）."""
        lines = [
            f"Task Goal: {self.task_goal}",
            f"Parent Node ID: {parent.node_id}",
            f"Parent Depth: {parent.depth}",
            f"Parent Description: {parent.description or 'root (no implementation)'}",
            f"Parent Mean Reward: {parent.mean_reward:.4f}",
            f"Parent Visit Count: {parent.visit_count}",
        ]
        if parent.code_path:
            lines.append(f"Parent Code Path: {parent.code_path}")
        lines.append("")
        lines.append(self.search_memory.summary())
        return "\n".join(lines)

    def create_child(
        self,
        parent_id: str,
        parent_info: str,
        code_attempt: str = "",
        node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """在 parent_id 下创建一个待评估子节点.

        Args:
            node_id: 若提供则使用该 id（如 opt-round-2），否则自动生成 node_X。
        """
        parent = self.nodes.get(parent_id)
        if parent is None:
            raise ValueError(f"Parent node {parent_id} not found")
        if parent.status == NodeStatus.PRUNED:
            raise ValueError(f"Parent node {parent_id} is pruned")

        if node_id is None:
            child_id = self._generate_node_id()
        else:
            child_id = node_id
            # 保持 next_node_id 始终大于所有已用数字 id，避免未来自动生成时冲突。
            if child_id.startswith("node_"):
                try:
                    idx = int(child_id.split("_", 1)[1])
                    self.next_node_id = max(self.next_node_id, idx + 1)
                except ValueError:
                    pass

        child = Node(
            node_id=child_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            status=NodeStatus.PENDING,
            description=parent_info,
            code_path=code_attempt,
        )
        parent.children.append(child_id)
        self.nodes[child_id] = child
        self.save()
        return {
            "node_id": child_id,
            "parent_id": parent_id,
            "depth": child.depth,
        }

    def evaluate(self, node_id: str, verify_result_path: str, perf_result_path: Optional[str] = None) -> Dict[str, Any]:
        """评估节点、计算奖励、回传至祖先、必要时剪枝."""
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        reward, error_type, speedup = self._compute_reward(verify_result_path, perf_result_path)
        node.verify_result_path = verify_result_path
        node.perf_result_path = perf_result_path or ""
        node.error_type = error_type

        if error_type != ErrorType.NONE:
            node.total_failures += 1

        self._backpropagate(node_id, reward)

        if node.total_failures >= self.failure_threshold:
            self._prune_node(node_id)

        if error_type == ErrorType.NONE and node.description:
            direction = node.description.strip()
            if direction and direction not in self.search_memory.explored_directions:
                self.search_memory.explored_directions.append(direction)

        self.save()
        return {
            "node_id": node_id,
            "reward": reward,
            "status": node.status,
            "visit_count": node.visit_count,
            "mean_reward": node.mean_reward,
            "total_failures": node.total_failures,
            "error_type": error_type,
            "speedup": speedup,
        }

    def _compute_reward(self, verify_result_path: str, perf_result_path: Optional[str]) -> tuple[float, str, Optional[float]]:
        """计算单个节点的奖励."""
        if not verify_result_path or not os.path.exists(verify_result_path):
            return float(self.reward_compile_error), ErrorType.COMPILE_ERROR, None

        try:
            with open(verify_result_path, "r", encoding="utf-8") as f:
                verify = json.load(f)
        except Exception:
            return float(self.reward_compile_error), ErrorType.COMPILE_ERROR, None

        total_cases = verify.get("total_cases", 0)
        passed_cases = verify.get("passed_cases", 0)
        failures = verify.get("failures", [])

        if total_cases > 0 and passed_cases == total_cases:
            speedup = self._read_speedup(perf_result_path)
            if speedup is None or speedup <= 0:
                return float(self.reward_output_error), ErrorType.OUTPUT_ERROR, speedup
            return self.reward_success_fn(speedup), ErrorType.NONE, speedup

        if failures:
            compile_keywords = ("CompilationError", "CompileError", "SyntaxError", "ImportError",
                                "ModuleNotFoundError", "IndentationError", "TypeError", "AttributeError")
            for failure in failures:
                et = failure.get("error_type", "")
                em = failure.get("error_msg", "")
                if et.startswith(compile_keywords) or any(kw in em for kw in compile_keywords):
                    return float(self.reward_compile_error), ErrorType.COMPILE_ERROR, None

        return float(self.reward_output_error), ErrorType.OUTPUT_ERROR, None

    def _read_speedup(self, perf_result_path: Optional[str]) -> Optional[float]:
        if not perf_result_path or not os.path.exists(perf_result_path):
            return None
        try:
            with open(perf_result_path, "r", encoding="utf-8") as f:
                perf = json.load(f)
            speedup = perf.get("speedup_vs_torch")
            if speedup is None or not isinstance(speedup, (int, float)):
                return None
            return float(speedup)
        except Exception:
            return None

    def _backpropagate(self, node_id: str, reward: float) -> None:
        """奖励与访问次数沿祖先链累加."""
        current_id: Optional[str] = node_id
        while current_id is not None:
            node = self.nodes.get(current_id)
            if node is None:
                break
            node.visit_count += 1
            node.total_reward += reward
            node.mean_reward = node.total_reward / node.visit_count
            if node.status == NodeStatus.PENDING:
                node.status = NodeStatus.EVALUATED
            if node.status != NodeStatus.PRUNED:
                node.status = NodeStatus.ACTIVE
            current_id = node.parent_id

    def _prune_node(self, node_id: str) -> None:
        """标记节点为 pruned，并递归剪枝其所有后代."""
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.status = NodeStatus.PRUNED
        for child_id in node.children:
            self._prune_node(child_id)

    def get_path_to_node(self, node_id: str) -> List[Dict[str, Any]]:
        """返回从 root 到指定节点的路径信息列表."""
        path: List[Dict[str, Any]] = []
        current_id: Optional[str] = node_id
        while current_id is not None:
            node = self.nodes.get(current_id)
            if node is None:
                break
            path.append(self._node_info(node))
            current_id = node.parent_id
        return list(reversed(path))

    def get_best_path(self) -> Dict[str, Any]:
        """返回从 root 到叶子平均奖励最高的路径."""
        def dfs(node_id: str) -> tuple[List[Dict[str, Any]], float, int]:
            node = self.nodes.get(node_id)
            if node is None:
                return [], float("-inf"), 0
            active_children = [cid for cid in node.children
                               if self.nodes[cid].status != NodeStatus.PRUNED]
            if not active_children:
                return [self._node_info(node)], node.mean_reward, node.depth
            best_sub_path, best_sub_reward, best_depth = [], float("-inf"), node.depth
            for cid in active_children:
                sub_path, sub_reward, sub_depth = dfs(cid)
                if sub_reward > best_sub_reward:
                    best_sub_path, best_sub_reward, best_depth = sub_path, sub_reward, sub_depth
            return [self._node_info(node)] + best_sub_path, best_sub_reward, best_depth

        path, reward, depth = dfs(self.root_id)
        return {
            "path": path,
            "total_reward": reward,
            "depth": depth,
        }

    def _node_info(self, node: Node) -> Dict[str, Any]:
        info = {
            "node_id": node.node_id,
            "depth": node.depth,
            "visit_count": node.visit_count,
            "mean_reward": node.mean_reward,
            "status": node.status,
            "error_type": node.error_type,
            "description": node.description,
        }
        if node.node_id in self.round_dir_map:
            info["round_dir"] = self.round_dir_map[node.node_id]
        return info

    def get_search_memory_summary(self) -> str:
        return self.search_memory.summary()

    def record_search_memory(self, error_type: str, parent_info: str, error_msg: str, direction: str = "") -> None:
        self.search_memory.record_failure(error_type, parent_info, error_msg, direction)
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        # reward_success_fn 不可序列化，不写入 JSON。
        return {
            "version": self.version,
            "task_id": self.task_id,
            "task_goal": self.task_goal,
            "output_dir": self.output_dir,
            "root_id": self.root_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "search_memory": self.search_memory.to_dict(),
            "next_node_id": self.next_node_id,
            "c_uct": self.c_uct,
            "c_pw": self.c_pw,
            "alpha": self.alpha,
            "failure_threshold": self.failure_threshold,
            "reward_output_error": self.reward_output_error,
            "reward_compile_error": self.reward_compile_error,
            "pending_expansion": self.pending_expansion,
            "round_dir_map": self.round_dir_map,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSTree":
        nodes = {nid: Node.from_dict(nd) for nid, nd in data.get("nodes", {}).items()}
        tree = cls(
            version=data.get("version", TREE_STATE_VERSION),
            task_id=data["task_id"],
            task_goal=data["task_goal"],
            output_dir=data["output_dir"],
            root_id=data.get("root_id", "root"),
            nodes=nodes,
            search_memory=SearchMemory.from_dict(data.get("search_memory", {})),
            next_node_id=data.get("next_node_id", len(nodes) + 1),
            c_uct=float(data.get("c_uct", C_UCT)),
            c_pw=float(data.get("c_pw", C_PW)),
            alpha=float(data.get("alpha", ALPHA)),
            failure_threshold=int(data.get("failure_threshold", FAILURE_THRESHOLD)),
            reward_output_error=float(data.get("reward_output_error", REWARD_OUTPUT_ERROR)),
            reward_compile_error=float(data.get("reward_compile_error", REWARD_COMPILE_ERROR)),
            reward_success_fn=reward_success,
            pending_expansion=data.get("pending_expansion"),
            round_dir_map=dict(data.get("round_dir_map", {})),
        )
        tree._update_tree_handle()
        return tree

    def save(self) -> None:
        """持久化到 tree_handle."""
        if not self.tree_handle:
            self._update_tree_handle()
        os.makedirs(os.path.dirname(self.tree_handle), exist_ok=True)
        with open(self.tree_handle, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, tree_handle: str) -> "MCTSTree":
        with open(tree_handle, "r", encoding="utf-8") as f:
            data = json.load(f)
        tree = cls.from_dict(data)
        tree.tree_handle = os.path.abspath(tree_handle)
        return tree


if __name__ == "__main__":
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp()
    try:
        tree = MCTSTree.create("test", "softmax on ascend", tmpdir)
        sel = tree.select()
        assert sel["action"] == "create_child", sel
        child = tree.create_child("root", "baseline vectorized softmax")
        verify_path = os.path.join(tmpdir, "verify_result.json")
        perf_path = os.path.join(tmpdir, "perf_result.json")
        with open(verify_path, "w") as f:
            json.dump({"total_cases": 3, "passed_cases": 3, "failures": []}, f)
        with open(perf_path, "w") as f:
            json.dump({"speedup_vs_torch": 2.0}, f)
        ev = tree.evaluate(child["node_id"], verify_path, perf_path)
        print("evaluate:", ev)
        print("best path:", tree.get_best_path())
        print("tree file:", tree.tree_handle)
    finally:
        shutil.rmtree(tmpdir)
