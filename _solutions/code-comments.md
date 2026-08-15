---
title: Code Comments
description: Enhance code with meaningful comments and documentation blocks
category:
- Code
- Communication
problems:
- difficult-code-comprehension
- difficult-to-understand-code
- poor-documentation
- implicit-knowledge
- tacit-knowledge
- complex-and-obscure-logic
- knowledge-gaps
layout: solution
related_solutions:
- slug: code-conventions
  similarity: 0.75
- slug: code-review-process-reform
  similarity: 0.75
- slug: clean-code
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
- slug: documentation-as-code
  similarity: 0.7
- slug: architecture-documentation
  similarity: 0.7
---

## Description

Code comments are annotations embedded directly in source code that explain aspects of the code the code itself cannot express — most importantly the reasoning behind a decision, rather than a restatement of what the code visibly does. Used well, they capture the "why": the business rule that a piece of logic encodes, the constraint that forced a particular workaround, the historical context that makes an otherwise-illogical piece of code actually necessary. In legacy systems this function is disproportionately valuable, because the people who made the original decisions have frequently left the organization, no external documentation survives, and the code is the only remaining trace of institutional knowledge that would otherwise vanish entirely. A comment explaining that an odd calculation stems from a specific regulatory agreement, for instance, is often the only thing standing between that logic staying intact and a future maintainer "fixing" it into a production incident because it looked like an obvious bug. Comments work alongside clear naming and structure rather than replacing them — the "what" should ideally be legible from the code itself, leaving comments to carry only the context that cannot be inferred no matter how the code is written. Their central weakness is that they are never checked by a compiler or test suite, so a comment that is not kept in sync with the code it describes silently becomes actively misleading rather than merely unhelpful, which is a real risk in legacy code that gets modified without anyone revisiting its accompanying commentary.

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
