"""KernelWiki-aligned knowledge base and Agent adapter for Triton-Ascend."""

from .agent_bundle import AgentKnowledgeBundle, KernelWikiAgentAdapter
from .kernelwiki import KernelWikiSearch, SearchFilters

__all__ = ["AgentKnowledgeBundle", "KernelWikiAgentAdapter", "KernelWikiSearch", "SearchFilters"]
__version__ = "0.8.0"
