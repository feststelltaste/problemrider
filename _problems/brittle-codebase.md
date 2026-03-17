---
title: Brittle Codebase
description: The existing code is difficult to modify without introducing new bugs,
  making maintenance and feature development risky.
category:
- Architecture
- Code
related_problems:
- slug: inconsistent-codebase
  similarity: 0.75
- slug: increasing-brittleness
  similarity: 0.7
- slug: difficult-code-comprehension
  similarity: 0.65
- slug: spaghetti-code
  similarity: 0.65
- slug: uncontrolled-codebase-growth
  similarity: 0.65
- slug: complex-and-obscure-logic
  similarity: 0.65
layout: problem
---

## Description
A brittle codebase is one that is difficult and risky to change. When a small change in one part of the codebase leads to unexpected failures in other parts, it is a sign of a brittle codebase. This is often caused by a lack of automated tests, a high degree of coupling between components, and a general lack of good design principles. A brittle codebase is a major source of technical debt, and it can significantly slow down the pace of development.

## Indicators ⟡
- Developers express fear or hesitation when asked to modify certain parts of the system.
- Estimates for small changes are consistently much larger than expected.
- The team avoids refactoring, even when they know it is needed.
- Onboarding new developers takes an unusually long time because the codebase is so hard to understand.

## Symptoms ▲

- [Regression Bugs](regression-bugs.md)
<br/>  Small code changes frequently introduce bugs in seemingly unrelated parts of the system due to hidden coupling.
- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  Developers become afraid to modify the codebase because even minor changes often cause unexpected failures.
- [Avoidance Behaviors](avoidance-behaviors.md)
<br/>  Developers avoid touching fragile code areas, leading to deferred maintenance and growing technical debt.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Development velocity decreases as developers spend excessive time working around brittle code to avoid breakage.
- [Accumulation of Workarounds](accumulation-of-workarounds.md)
<br/>  Rather than modifying brittle code directly, developers add workarounds that further increase complexity.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Maintaining a brittle codebase requires disproportionate effort as small changes demand extensive testing and fixing.
## Causes ▼

- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  High coupling between components means changes propagate unpredictably, making the codebase fragile.
- [Insufficient Testing](insufficient-testing.md)
<br/>  Without adequate test coverage, there is no safety net to catch regressions introduced by changes.
- [Spaghetti Code](spaghetti-code.md)
<br/>  Tangled, unstructured code with unclear control flow makes changes risky and unpredictable.
- [Refactoring Avoidance](refactoring-avoidance.md)
<br/>  Long-term avoidance of refactoring allows structural problems to accumulate, making the codebase increasingly brittle.
## Detection Methods ○

- **Code Coverage Tools:** Use tools to measure test coverage. Low coverage is a strong indicator of brittleness.
- **Static Analysis Tools:** Tools that measure code complexity (e.g., cyclomatic complexity), coupling, and other metrics can highlight problematic areas.
- **Bug Tracking Metrics:** Monitor the rate of regression bugs introduced after new features or changes.
- **Developer Surveys/Interviews:** Ask developers about their experience working with the codebase and their confidence in making changes.
- **Code Review Feedback:** Look for recurring comments about code being hard to understand or risky to change.

## Examples

A team needs to update a small piece of business logic in a legacy system. The change is estimated to take a few hours, but because the code is so tightly coupled and lacks tests, the team spends two weeks trying to implement the change and fix all the new bugs it introduces. For example, a function that calculates a user's discount also updates their loyalty points and sends an email. Changing the discount calculation logic unexpectedly breaks the email sending feature because the function has too many responsibilities. This problem is a hallmark of aging, unmaintained software systems. It often arises from a lack of discipline in software engineering practices, especially testing and design principles, over a long period.
