---
title: Code Metrics
description: Collecting and analyzing quantitative measures to evaluate code quality
category:
- Code
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/code-metrics
problems:
- high-technical-debt
- invisible-nature-of-technical-debt
- lower-code-quality
- difficult-code-comprehension
- complex-and-obscure-logic
- monolithic-functions-and-classes
- bloated-class
- quality-degradation
layout: solution
---

## How to Apply ◆

> In legacy systems, code metrics make the invisible visible — they quantify the technical debt and complexity that developers feel but cannot easily communicate to stakeholders.

- Integrate code metrics tools (SonarQube, CodeClimate, NDepend, or language-specific alternatives) into the CI pipeline to track metrics automatically on every build.
- Focus on a small set of actionable metrics: cyclomatic complexity, class and method length, coupling between components, and duplication percentage.
- Establish baselines for the legacy codebase's current metrics and set improvement targets that guide refactoring priorities.
- Use metrics to identify the worst hotspots — the classes and methods with the highest complexity and the most frequent changes — as priority targets for refactoring.
- Present metric trends to stakeholders to make technical debt visible and justify investment in code quality improvement.
- Combine code metrics with change frequency data to focus improvement efforts on code that is both complex and frequently modified, maximizing the return on refactoring investment.
- Set quality gates that prevent new code from introducing metrics regressions, ensuring that the codebase improves over time rather than continuing to degrade.

## Tradeoffs ⇄

> Code metrics provide objective quality indicators but can be gamed and must be interpreted in context.

**Benefits:**

- Makes code quality visible and measurable, enabling data-driven decisions about where to invest refactoring effort.
- Provides objective evidence for communicating technical debt to non-technical stakeholders who may not appreciate the problem otherwise.
- Identifies the most problematic areas of the codebase through quantitative analysis rather than anecdotal evidence.
- Tracks quality improvement over time, demonstrating the impact of refactoring investments.

**Costs and Risks:**

- Metrics can be gamed — developers can reduce cyclomatic complexity by splitting methods without improving actual comprehensibility.
- Overemphasis on metrics can lead to optimizing for numbers rather than genuine code quality.
- Setting metric thresholds too aggressively on a legacy codebase can overwhelm the team with violations that cannot all be addressed.
- Some important quality attributes (naming quality, design appropriateness, business alignment) cannot be captured by automated metrics.

## How It Could Be

> The following scenario demonstrates how code metrics guide legacy modernization priorities.

A manufacturing company's legacy ERP system had 2 million lines of code, and the development team knew it had quality problems but could not agree on where to focus improvement efforts. After integrating SonarQube, the team discovered that 15 classes (0.3% of the codebase) accounted for 40% of all complexity violations and 35% of production defects. These "toxic" classes — including a 12,000-line `OrderProcessor` and an 8,000-line `InventoryManager` — became the explicit targets of a six-month refactoring initiative. By tracking complexity metrics monthly, the team demonstrated to management that the average cyclomatic complexity of modified classes dropped from 45 to 12 over the initiative, and the defect rate in refactored areas declined proportionally. The metrics dashboard became a standing agenda item in management reviews, making technical debt a visible business concern rather than a developer complaint.
