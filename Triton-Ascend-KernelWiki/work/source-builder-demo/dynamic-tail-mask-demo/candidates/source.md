---
id: pr-source-builder-fixture-1200
title: 'Source Builder fixture: dynamic-extent tail mask'
repo: local/source-builder-demo
pr: 1200
author: demo-fixture
date: 2026-07-17
url: local://source-builder-demo/pr-1200
source_category: upstream-code
architectures:
- Ascend910B1
tags:
- tail_mask
- dynamic_extent
techniques:
- tail_mask
hardware_features:
- vector_core
- unified_buffer
kernel_types:
- math
captured_at: 2026-07-17
status: closed
fixture: true
---

# Dynamic-extent tail-mask source fixture

This synthetic PR fixture represents a change that intersects the logical
extent with the allocated tensor extent before loading and storing the
final tile. It exists only to demonstrate the Source Builder workflow.

The fixture reports compile and reference-correctness checks. It contains
no performance claim.
