---
title: Code Comments
description: Enhance code with meaningful comments and documentation blocks
category:
- Code
- Communication
quality_tactics_url: https://qualitytactics.de/en/maintainability/code-comments
problems:
- difficult-code-comprehension
- difficult-to-understand-code
- poor-documentation
- implicit-knowledge
- tacit-knowledge
- complex-and-obscure-logic
- knowledge-gaps
layout: solution
---

## How to Apply ◆

> In legacy systems, strategic code comments explain the "why" behind decisions that cannot be understood from the code alone, preserving institutional knowledge that would otherwise be lost.

- Focus comments on explaining why code exists and why it works the way it does, not what it does — the code itself should communicate the "what" through clear naming and structure.
- Document non-obvious business rules that are embedded in the code, especially when the rule contradicts what seems logical (e.g., "Discount is applied before tax for orders from Region 3 due to 2008 regulatory agreement with state of...").
- Add comments to workarounds and hacks explaining the underlying problem they address, the legacy constraint that prevents a proper fix, and any conditions under which the workaround could be removed.
- Use documentation blocks (Javadoc, JSDoc, docstrings) for public APIs and interfaces to explain contracts, preconditions, and edge case behavior.
- Add "WARNING" or "CAUTION" comments to code that has known fragile dependencies or non-obvious side effects that could trap future maintainers.
- During legacy code review or maintenance, add explanatory comments whenever you spend significant time understanding a piece of code — the next person will face the same struggle without them.

## Tradeoffs ⇄

> Comments preserve institutional knowledge but require discipline to maintain and can mislead when they become stale.

**Benefits:**

- Preserves the rationale behind legacy code decisions that cannot be inferred from the code itself, preventing future developers from inadvertently removing important behavior.
- Reduces the time developers spend reverse-engineering obscure legacy logic by providing context at the point of need.
- Documents workarounds and their prerequisites, making it possible to remove them when the underlying constraint is eventually resolved.
- Serves as a knowledge transfer mechanism when original developers leave, capturing insights that would otherwise be lost.

**Costs and Risks:**

- Comments that are not updated when code changes become misleading, creating false understanding that can lead to bugs.
- Excessive commenting of obvious code creates noise that makes truly important comments harder to find.
- Comments cannot be tested or compiled — there is no automated way to detect when a comment has become inaccurate.
- Relying on comments instead of improving code clarity through refactoring can perpetuate poor code quality.

## How It Could Be

> The following scenario shows how strategic comments preserve critical knowledge in legacy systems.

A telecommunications company's legacy billing system contained a method that calculated usage charges with what appeared to be an arbitrary 0.3% adjustment factor applied to calls exceeding 45 minutes. Three different developers had attempted to "fix" this apparent bug over the years, each time causing billing discrepancies that required manual corrections. When a senior developer finally traced the factor to a 2005 interconnect agreement with a partner carrier, they added a detailed comment explaining the regulatory origin of the adjustment, the specific agreement reference number, and the conditions under which it applied. The comment also noted that the agreement was set to expire in 2027, at which point the adjustment could be removed. This single comment prevented future "fix" attempts and provided the business context needed for the eventual modernization of the billing engine.
