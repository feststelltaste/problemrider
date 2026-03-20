---
title: Large Estimates for Small Changes
description: The team consistently provides large time estimates for seemingly small
  changes, indicating underlying code complexity and risk.
category:
- Code
- Process
related_problems:
- slug: fear-of-change
  similarity: 0.6
- slug: brittle-codebase
  similarity: 0.55
- slug: history-of-failed-changes
  similarity: 0.55
- slug: slow-feature-development
  similarity: 0.55
- slug: increased-cost-of-development
  similarity: 0.55
- slug: frequent-changes-to-requirements
  similarity: 0.55
solutions:
- architecture-roadmap
- regression-testing
layout: problem
---

## Description
When small, seemingly simple changes are consistently estimated to take a long time to implement, it is a strong indicator of underlying problems in the codebase. This phenomenon, often referred to as "high-cost-of-change," suggests that the system has become rigid and fragile. The development team is likely navigating a minefield of technical debt, where every modification carries the risk of unforeseen side effects. This problem can cripple a team's ability to respond to changing business needs and can be a major source of frustration for both developers and stakeholders.

## Indicators ⟡
- A simple bug fix is estimated to take days or weeks.
- Stakeholders are surprised by the high cost of minor feature requests.
- The team spends more time in meetings discussing the risks of a change than actually implementing it.
- There is a noticeable reluctance from the team to take on tasks that involve modifying existing code.

## Symptoms ▲

- [Stakeholder Frustration](stakeholder-frustration.md)
<br/>  Business stakeholders become frustrated when seemingly simple changes require disproportionate time and cost, eroding trust in the development team.
- [Slow Feature Development](slow-feature-development.md)
<br/>  When every small change requires significant effort, overall feature delivery pace drops dramatically.
- [Increased Cost of Development](increased-cost-of-development.md)
<br/>  The high effort required for even minor modifications directly drives up the total cost of development work.
- [Stakeholder Confidence Loss](stakeholder-confidence-loss.md)
<br/>  Repeated large estimates for small work erodes business confidence in the team's ability to deliver efficiently.
- [Planning Credibility Issues](planning-credibility-issues.md)
<br/>  When estimates seem disproportionate to the apparent work, stakeholders question the reliability of all future planning and estimates.
## Causes ▼

- [Brittle Codebase](brittle-codebase.md)
<br/>  A fragile codebase where changes risk breaking other parts forces teams to pad estimates to account for extensive testing and risk mitigation.
- [High Coupling and Low Cohesion](high-coupling-low-cohesion.md)
<br/>  Tightly coupled components mean that even small changes ripple across many parts of the system, legitimately requiring large effort.
- [Legacy Code Without Tests](legacy-code-without-tests.md)
<br/>  Without tests to verify changes, developers must manually verify that modifications don't break anything, significantly increasing estimated effort.
- [High Technical Debt](high-technical-debt.md)
<br/>  Accumulated shortcuts and design compromises make the codebase harder to work with, inflating the effort for even minor changes.
- [Complex and Obscure Logic](complex-and-obscure-logic.md)
<br/>  When code is hard to understand, developers need extra time to comprehend the system before making changes, driving up estimates.
## Detection Methods ○
- **Analyze Estimation Trends:** Track the estimates for tasks of similar complexity over time. A consistent increase in estimates is a red flag.
- **Compare Estimated vs. Actual Time:** If the actual time taken to complete tasks is consistently much higher than the estimates, it indicates that the team is struggling with unforeseen complexity.
- **Developer Feedback:** Ask developers why their estimates are so high. Their answers will often point to the root causes.
- **Code Complexity Metrics:** Use static analysis tools to measure code complexity. High complexity scores often correlate with high-cost-of-change.

## Examples
A product manager requests a small change to the user interface: adding a new field to a form. The development team estimates that this will take two weeks to implement. The product manager is shocked, as they expected it to be a simple, one-day task. The developers explain that the form is used in multiple places throughout the application, and the underlying data model is tightly coupled to other parts of the system. Any change to the form requires extensive testing to ensure that it doesn't break anything else. This is a classic example of how a brittle codebase can lead to large estimates for small changes.
