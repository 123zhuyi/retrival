# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.template_scope — project template (*.claude/template/*.md) as a
parallel retrieval lane alongside runbooks/ops.

Template files are project-level distilled experience that grows after every
operator task (Phase 7 沉淀). This module indexes them on the fly (no build
step, no KB write access needed) by splitting each file into heading sections
("virtual cards") and scoring them with a small deterministic BM25-ish ranker.

Result payload mirrors retrieval.search._result_payload so downstream
consumers (b_context_*_template.json, get <path>, deep-read via local_path)
work unchanged. Card paths look like `template/{file}#{section}`; `local_path`
points at the real file so agents can Read it directly.
"""
from __future__ import annotations

import math
import os
import re

from retrieval.config import SCORE_PREC
from retrieval.relevance import annotate_relevance
from retrieval.tokenizer import tokenize

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*\S)\s*$", re.M)

TITLE_WEIGHT = 3.0
_BM25_K1 = 1.5
_BM25_B = 0.75


def find_template_dir(start=None):
    """Locate the nearest `.claude/template/` directory.

    Resolution order: $TRITON_TEMPLATE_DIR, then walk up from `start` (default
    CWD) looking for `.claude/template`. Returns None when not found.
    """
    env_dir = os.environ.get("TRITON_TEMPLATE_DIR")
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)
    current = os.path.abspath(start or os.getcwd())
    while True:
        candidate = os.path.join(current, ".claude", "template")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _split_sections(filename, text):
    """Split a template markdown file into section cards.

    Each card: (section_id, title, body). Content before the first heading
    becomes section `§0` (preamble) when non-trivial.
    """
    matches = list(HEADING_RE.finditer(text))
    sections = []
    if matches and matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if len(preamble) > 40:
            sections.append(("§0", "preamble", preamble))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(2).strip()
        body = text[match.end():end].strip()
        sections.append((f"§{i + 1}", title, body))
    if not sections:
        stripped = text.strip()
        if stripped:
            sections.append(("§0", os.path.basename(filename)[:-3], stripped))
    return sections


def load_template_cards(template_dir):
    """Parse every *.md under template_dir into virtual cards."""
    cards = []
    for name in sorted(os.listdir(template_dir)):
        if not name.endswith(".md"):
            continue
        file_path = os.path.join(template_dir, name)
        try:
            with open(file_path, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        category = name[:-3]
        for section_id, title, body in _split_sections(name, text):
            cards.append({
                "path": f"template/{name}#{section_id}",
                "doc_id": f"template/{name}#{section_id}",
                "local_path": file_path,
                "category": category,
                "section": section_id,
                "title": f"[{category}] {title}",
                "body": body,
            })
    return cards


def _snippet(body, terms, limit=240):
    """First paragraph containing any query term, else the leading excerpt."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    for paragraph in paragraphs:
        lowered = paragraph.lower()
        if any(term in lowered for term in terms):
            return paragraph[:limit]
    return (paragraphs[0] if paragraphs else "")[:limit]


def template_search(queries, *, k=10, template_dir=None):
    """Score template cards against queries; return (results, warnings)."""
    template_dir = template_dir or find_template_dir()
    if template_dir is None:
        return [], ["template scope requested but no .claude/template directory found"]
    cards = load_template_cards(template_dir)
    if not cards:
        return [], [f"template scope: no *.md cards under {template_dir}"]

    term_df = {}
    doc_terms = []
    for card in cards:
        title_tokens = tokenize(card["title"])
        body_tokens = tokenize(card["body"])
        doc_terms.append((title_tokens, body_tokens))
        for term in set(title_tokens) | set(body_tokens):
            term_df[term] = term_df.get(term, 0) + 1

    total = len(cards)
    avg_len = sum(len(body) for _, body in doc_terms) / max(total, 1) or 1.0
    scores = {}
    matched_terms = {}
    for query in queries or []:
        query_terms = set(tokenize(query))
        for index, card in enumerate(cards):
            title_tokens, body_tokens = doc_terms[index]
            body_tf = {}
            for token in body_tokens:
                body_tf[token] = body_tf.get(token, 0) + 1
            title_set = set(title_tokens)
            score = 0.0
            for term in query_terms:
                df = term_df.get(term, 0)
                idf = math.log((total - df + 0.5) / (df + 0.5) + 1.0)
                tf = body_tf.get(term, 0)
                denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * len(body_tokens) / avg_len)
                body_score = idf * tf / denom if denom else 0.0
                title_score = TITLE_WEIGHT * idf if term in title_set else 0.0
                if body_score or title_score:
                    score += body_score + title_score
                    matched_terms.setdefault(index, set()).add(term)
            if score > 0:
                scores[index] = scores.get(index, 0.0) + score

    ranked = sorted(scores, key=lambda i: (-round(scores[i], SCORE_PREC), cards[i]["path"]))
    results = []
    for index in ranked[:k]:
        card = cards[index]
        terms = {t.lower() for t in matched_terms.get(index, ())}
        results.append({
            "path": card["path"],
            "doc_id": card["doc_id"],
            "local_path": card["local_path"],
            "bundle": "template",
            "title": card["title"],
            "description": card["body"][:160].replace("\n", " "),
            "kind": "template_experience",
            "category": card["category"],
            "section": card["section"],
            "tags": [card["category"]],
            "platforms": [],
            "score": round(scores[index], SCORE_PREC),
            "snippet": _snippet(card["body"], terms),
            "matched": {"fields": ["title", "body"], "terms": sorted(terms), "queries": []},
            "variants": [],
        })
    return annotate_relevance(results, queries=queries), []


def is_template_scope(scope):
    """True when --scope/--dir designates the template lane."""
    if not scope:
        return False
    normalized = scope.strip().lstrip("/").lower()
    return normalized in ("template", "template/")
