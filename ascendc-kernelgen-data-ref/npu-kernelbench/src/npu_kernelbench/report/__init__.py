"""Result reporting — formatting and scoring for evaluation results."""

from .formatter import format_traces_table
from .score import ScoreSummary, compute_score, format_score_summary
from .json_reporter import JsonReporter
from .html_reporter import HtmlReporter

__all__ = [
    "format_traces_table",
    "ScoreSummary",
    "compute_score",
    "format_score_summary",
    "JsonReporter",
    "HtmlReporter",
]
