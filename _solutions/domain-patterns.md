---
title: Domain Patterns
description: Applying proven solutions for recurring business problems
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/domain-patterns
problems:
- complex-and-obscure-logic
- poor-domain-model
- legacy-business-logic-extraction-difficulty
- suboptimal-solutions
- accumulation-of-workarounds
layout: solution
---

## How to Apply ◆

- Study domain-specific patterns relevant to the legacy system's industry (e.g., Martin Fowler's Analysis Patterns, enterprise integration patterns, accounting patterns).
- Identify where the legacy system has ad-hoc solutions for problems that have well-known domain patterns.
- Refactor legacy code to use established domain patterns incrementally, starting with the most problematic areas.
- Train the development team on domain patterns applicable to their system's business area.
- Use domain patterns as a shared vocabulary when discussing design decisions with the team.
- Document which domain patterns are used and where, creating a pattern map for the legacy system.

## Tradeoffs ⇄

**Benefits:**
- Replaces ad-hoc, home-grown solutions with proven approaches that are well-understood in the industry.
- Makes the codebase more familiar to new developers who know the standard patterns.
- Reduces the risk of subtle bugs that come from reinventing solutions to well-known problems.
- Provides a vocabulary for discussing recurring business concepts.

**Costs:**
- Finding the right pattern requires domain knowledge and pattern literacy.
- Force-fitting a pattern to a problem it does not match creates unnecessary complexity.
- Refactoring existing code to match a pattern requires effort and careful testing.
- Some domain patterns may not map cleanly to the legacy system's existing structure.

## How It Could Be

A legacy accounting system implements double-entry bookkeeping through scattered validation checks and reconciliation scripts rather than using the well-established accounting entry pattern. Discrepancies between accounts are a recurring problem, and debugging requires tracing through multiple code paths. The team refactors the core transaction handling to use the standard accounting entry pattern, where every financial event produces balanced debit and credit entries as an atomic operation. The pattern makes it structurally impossible to create unbalanced entries, eliminating an entire category of bugs. New developers with accounting domain knowledge immediately recognize the pattern and can work productively without studying the custom implementation that it replaced.
