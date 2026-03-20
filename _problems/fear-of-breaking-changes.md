---
title: Fear of Breaking Changes
description: The team is reluctant to make changes to the codebase for fear of breaking
  existing functionality, which can lead to a stagnant and outdated system.
category:
- Code
- Culture
- Process
related_problems:
- slug: fear-of-change
  similarity: 0.85
- slug: resistance-to-change
  similarity: 0.7
- slug: fear-of-failure
  similarity: 0.7
- slug: history-of-failed-changes
  similarity: 0.65
- slug: maintenance-paralysis
  similarity: 0.65
- slug: refactoring-avoidance
  similarity: 0.65
solutions:
- blue-green-canary-deployments
- feature-flags
- strangler-fig-pattern
layout: problem
---

## Description
Fear of breaking changes is a common problem in software development. It is the fear that a change to the codebase will have unintended consequences and will break existing functionality. This fear can be paralyzing, and it can prevent a team from making necessary changes to the system. When a team is afraid to make changes, the system can become stagnant and outdated. This can lead to a number of problems, including a decline in user satisfaction, a loss of competitive advantage, and a great deal of frustration for the development team.

## Indicators ⟡
- The team is hesitant to make changes to the codebase.
- The team is not refactoring the code.
- The team is not keeping up with the latest technologies.
- The team is not innovating.

## Symptoms ▲

- [Refactoring Avoidance](refactoring-avoidance.md)
<br/>  When the team fears breaking changes, they actively avoid refactoring even when they know it is necessary.
- [Accumulation of Workarounds](accumulation-of-workarounds.md)
<br/>  Instead of modifying existing code, developers create workarounds to avoid touching risky areas, adding complexity.
- [Stagnant Architecture](stagnant-architecture.md)
<br/>  The system architecture fails to evolve because the team is too afraid to make the structural changes needed for improvement.
- [High Technical Debt](high-technical-debt.md)
<br/>  Avoiding necessary changes causes technical debt to accumulate as the codebase becomes increasingly outdated.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Fear of breaking changes slows development as teams take excessive precautions or implement features in roundabout ways.
- [System Stagnation](system-stagnation.md)
<br/>  The system remains unchanged and fails to evolve because the team avoids making modifications.
## Causes ▼

- [Legacy Code Without Tests](legacy-code-without-tests.md)
<br/>  Without automated tests, there is no safety net to verify that changes do not break existing functionality, making fear rational.
- [History of Failed Changes](history-of-failed-changes.md)
<br/>  Past experiences where changes caused production failures create lasting fear and reluctance to make future modifications.
- [Brittle Codebase](brittle-codebase.md)
<br/>  A fragile codebase where small changes frequently cause unexpected breakages gives the team legitimate reasons to fear modifications.
- [High Coupling and Low Cohesion](high-coupling-low-cohesion.md)
<br/>  Tightly coupled code means changes in one area frequently affect other areas, making it genuinely risky to modify the system.
- [Poor Test Coverage](poor-test-coverage.md)
<br/>  Without sufficient test coverage, developers cannot verify their changes are safe, reinforcing the fear of making modifications.
## Detection Methods ○
- **Code Churn:** Analyze the history of the codebase to see how often the code is being changed.
- **Technical Debt:** Track the amount of technical debt in the system.
- **Developer Surveys:** Ask developers about their feelings about making changes to the system.
- **Willingness to Experiment:** Is the team willing to experiment with new ideas and technologies?

## Examples
A company has a legacy system that is critical to its business. The system is old and fragile, and the team is afraid to make changes to it. As a result, the system is not being updated, and it is falling behind the competition. The company is losing market share, and they are at risk of going out of business. The team knows that they need to make changes to the system, but they are paralyzed by fear.
