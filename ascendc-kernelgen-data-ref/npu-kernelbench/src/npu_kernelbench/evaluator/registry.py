"""EvaluatorRegistry — priority-ordered evaluator resolution."""

from __future__ import annotations

from typing import Optional

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.evaluator.evaluator import Evaluator
from npu_kernelbench.evaluator.default import DefaultEvaluator


class EvaluatorRegistry:
    """Priority-ordered evaluator registry. First match wins.

    User-registered evaluators take priority over built-ins (prepended).
    DefaultEvaluator is always last as catch-all.
    """

    def __init__(self):
        self._evaluators: list[Evaluator] = []
        # Built-in evaluators in priority order
        self._register_builtin(DefaultEvaluator())

    def _register_builtin(self, evaluator: Evaluator) -> None:
        """Register a built-in evaluator (appended, lowest priority)."""
        self._evaluators.append(evaluator)

    def register(self, evaluator: Evaluator) -> None:
        """Register a user evaluator (prepended, highest priority)."""
        self._evaluators.insert(0, evaluator)

    def resolve(self, definition: Definition) -> Evaluator:
        """Find the first evaluator that claims this definition.

        Raises RuntimeError if no evaluator matches (should never happen
        since DefaultEvaluator is always the last entry).
        """
        for ev in self._evaluators:
            if ev.can_evaluate(definition):
                return ev
        raise RuntimeError(
            f"No evaluator registered for definition '{definition.name}' "
            f"(op_type={definition.op_type})"
        )


# Global singleton
_default_registry: Optional[EvaluatorRegistry] = None


def get_registry() -> EvaluatorRegistry:
    """Get or create the global EvaluatorRegistry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = EvaluatorRegistry()
    return _default_registry
