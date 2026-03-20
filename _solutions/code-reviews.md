---
title: Code Reviews
description: Conduct regular reviews of the source code by team members
category:
- Code
- Process
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/code-reviews
problems:
- inadequate-code-reviews
- insufficient-code-review
- superficial-code-reviews
- lower-code-quality
- inconsistent-coding-standards
- knowledge-silos
- high-bug-introduction-rate
- difficult-code-comprehension
layout: solution
---

## How to Apply ◆

- Establish code review as a mandatory step before merging any change to the main branch of the legacy codebase.
- Define review checklists that include legacy-specific concerns: proper handling of existing conventions, preservation of undocumented business logic, and adequate test coverage for changed code.
- Keep pull requests small and focused to enable thorough reviews; break large legacy refactoring efforts into reviewable increments.
- Rotate reviewers to spread knowledge of the legacy codebase across the team and prevent knowledge silos.
- Use code review as a teaching opportunity for developers unfamiliar with the legacy system's patterns and constraints.
- Set response time expectations (e.g., reviews completed within one business day) to prevent review bottlenecks.

## Tradeoffs ⇄

**Benefits:**
- Catches bugs and logic errors before they reach production, particularly important in legacy systems with limited test coverage.
- Distributes knowledge of the legacy codebase across team members, reducing bus-factor risk.
- Enforces consistency in coding standards and architectural patterns within the legacy system.
- Serves as a learning mechanism for developers new to the legacy codebase.

**Costs:**
- Adds time to the development workflow, which can be challenging under deadline pressure.
- Ineffective reviews (rubber-stamping) provide false confidence without catching issues.
- Can create bottlenecks if reviewer availability is limited.
- Interpersonal dynamics (nitpicking, conflicting opinions) can make reviews counterproductive.

## Examples

A legacy financial system has critical calculation logic that only two senior developers fully understand. The team institutes mandatory code reviews with a rotation policy ensuring every developer reviews code across different modules over time. Within six months, three additional developers gain sufficient understanding of the calculation engine to make changes confidently. Reviews also catch several instances where new developers inadvertently break undocumented business rules embedded in the legacy code. The review process becomes the primary mechanism for transferring institutional knowledge about the legacy system's quirks and conventions to newer team members.
