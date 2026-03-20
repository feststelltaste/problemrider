---
title: High Technical Debt
description: Accumulation of design or implementation shortcuts that lead to increased
  costs and effort in the long run.
category:
- Code
- Process
related_problems:
- slug: increased-technical-shortcuts
  similarity: 0.75
- slug: invisible-nature-of-technical-debt
  similarity: 0.65
- slug: accumulation-of-workarounds
  similarity: 0.65
- slug: accumulated-decision-debt
  similarity: 0.65
- slug: test-debt
  similarity: 0.65
- slug: maintenance-overhead
  similarity: 0.6
solutions:
- technical-debt-backlog
layout: problem
---

## Description
High technical debt is the implied cost of rework caused by choosing an easy (limited) solution now instead of using a better approach that would take longer. This debt accumulates when organizations fail to allocate dedicated time, resources, or budget for improving existing code quality, addressing technical debt, or modernizing system architecture. This creates a cycle where technical debt accumulates faster than it can be addressed, eventually making the system increasingly difficult and expensive to maintain. Technical debt can be a major drag on productivity, and it can make it difficult and risky to add new features or make changes to the codebase.

## Indicators ⟡
- The team is constantly fixing bugs instead of building new features.
- It takes a long time to onboard new developers.
- The team is hesitant to refactor code.
- There is a lot of duplicated code.
- All development time is allocated to new features or bug fixes.
- Refactoring work is only done when absolutely necessary to complete other features.
- Technical debt items are identified but never prioritized in sprint planning.
- Developers express frustration about not having time to "clean up" code.

## Symptoms ▲

- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Accumulated shortcuts and code complexity make every change more expensive, increasing the overall cost of maintaining the system.
- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Technical debt slows feature development as developers must navigate complex, fragile code and work around existing issues.
- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  Complex, poorly structured code is more prone to bugs since changes have unpredictable side effects.
- [Difficult Developer Onboarding](difficult-developer-onboarding.md)
<br/>  A codebase burdened with technical debt is harder for new developers to understand due to inconsistencies, workarounds, and complexity.
- [Fear of Change](fear-of-change.md)
<br/>  High technical debt makes changes risky, causing developers and management to resist modifications to the system.
- [Inability to Innovate](inability-to-innovate.md)
<br/>  Teams spend so much effort managing debt-laden code that they have no capacity to explore new approaches or technologies.
- [Increased Cost of Development](increased-cost-of-development.md)
<br/>  Technical debt directly increases the cost of all development work as developers must navigate complexity, work aroun....
## Causes ▼

- [Refactoring Avoidance](refactoring-avoidance.md)
<br/>  When code is never improved or restructured, design shortcuts and complexity accumulate over time into significant debt.
- [Time Pressure](time-pressure.md)
<br/>  Tight deadlines push developers to take shortcuts and skip quality practices, directly creating technical debt.
- [Insufficient Code Review](insufficient-code-review.md)
<br/>  Without peer review, poor design decisions and implementation shortcuts enter the codebase unchallenged.
- [Insufficient Testing](insufficient-testing.md)
<br/>  Without tests, refactoring is risky, so debt-laden code remains untouched and accumulates further.
- [Implementation Starts Without Design](implementation-starts-without-design.md)
<br/>  Coding without upfront design leads to ad-hoc architecture and implementation shortcuts that become technical debt.
## Detection Methods ○

- **Codebase Metrics:** Monitor metrics like cyclomatic complexity, coupling, and code coverage. High values often indicate technical debt.
- **Bug Tracking Systems:** Analyze the types and frequency of bugs, especially those related to specific modules.
- **Developer Surveys/Interviews:** Ask developers about their pain points, areas of the codebase they avoid, and perceived technical debt.
- **Code Audits:** Conduct regular, systematic reviews of the codebase to identify areas of concern.
- **Retrospectives:** Discuss recurring issues and identify if they stem from technical debt.
- **Sprint Planning Analysis:** Track what percentage of sprint capacity is allocated to technical improvements.
- **Velocity Trends:** Track whether development velocity is declining over time due to increasing technical complexity.

## Examples
A legacy e-commerce platform has a highly coupled monolithic architecture. Adding a new payment gateway requires changes across multiple, seemingly unrelated modules, leading to weeks of development and several new bugs in production. In another case, a function that was originally designed for a simple task has been modified over time with numerous `if-else` statements and special cases, making it thousands of lines long and impossible to understand or test.

A software company has identified that their user authentication system is built on deprecated libraries with known security vulnerabilities. The development team estimates it would take three weeks to modernize the authentication system, significantly improving security and maintainability. However, the product roadmap is packed with new features for the next six months, and management refuses to allocate developer time for "infrastructure work" that doesn't directly provide customer value. Over the following year, the team spends an estimated eight weeks total working around limitations of the old authentication system, dealing with security patches, and troubleshooting integration issues that would have been eliminated by the modernization effort.

Another example involves an e-commerce platform where the product catalog module has grown into a 5,000-line monolithic class that takes hours to understand and test. Developers frequently estimate extra time for catalog-related features due to the complexity, but requests to refactor the module are always deferred in favor of adding new product features. Eventually, a critical bug in the catalog code takes two weeks to fix because of the complexity, costing more time than a proper refactoring would have required.
