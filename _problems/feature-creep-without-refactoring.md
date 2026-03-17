---
title: Feature Creep Without Refactoring
description: The continuous addition of new features to a codebase without taking
  the time to refactor and improve the design.
category:
- Code
- Process
related_problems:
- slug: feature-creep
  similarity: 0.85
- slug: scope-creep
  similarity: 0.6
- slug: maintenance-paralysis
  similarity: 0.55
- slug: convenience-driven-development
  similarity: 0.55
- slug: feature-bloat
  similarity: 0.55
- slug: accumulation-of-workarounds
  similarity: 0.55
layout: problem
---

## Description
Feature creep without refactoring is the process of continuously adding new features to a codebase without taking the time to refactor and improve the design. This leads to a gradual degradation of the codebase, making it more and more difficult to maintain and extend. It is a common problem in software development, and it is often driven by a desire to deliver new features as quickly as possible.

## Indicators ⟡
- The codebase is becoming increasingly complex and difficult to understand.
- It is taking longer and longer to add new features.
- The number of bugs is increasing.
- Developers are becoming more and more frustrated with the codebase.

## Symptoms ▲

- [High Technical Debt](high-technical-debt.md)
<br/>  Adding features without refactoring directly accumulates design and implementation shortcuts that increase long-term costs.
- [Increasing Brittleness](increasing-brittleness.md)
<br/>  Each feature added without refactoring makes the codebase more fragile, as new code is layered onto an increasingly unstable foundation.
- [Slow Feature Development](slow-feature-development.md)
<br/>  The degrading codebase makes each subsequent feature harder and slower to implement as complexity grows.
- [Increased Bug Count](increased-bug-count.md)
<br/>  Without refactoring to maintain code quality, each new feature is more likely to introduce defects.
- [Difficult Code Comprehension](difficult-code-comprehension.md)
<br/>  The codebase becomes progressively harder to understand as features are added without improving the underlying design.
- [Spaghetti Code](spaghetti-code.md)
<br/>  Continuous feature additions without structural improvement create tangled, unstructured code that is nearly impossible to maintain.

## Causes ▼
- [Deadline Pressure](deadline-pressure.md)
<br/>  Pressure to deliver features quickly leaves no time allocated for refactoring or design improvement.
- [Short-Term Focus](short-term-focus.md)
<br/>  Management prioritizes immediate feature delivery over long-term code health, consistently deferring refactoring work.
- [Feature Factory](feature-factory.md)
<br/>  An organizational culture obsessed with feature output metrics discourages spending time on non-feature work like refactoring.
- [Invisible Nature of Technical Debt](invisible-nature-of-technical-debt.md)
<br/>  When technical debt is not visible to stakeholders, there is no support for allocating time to refactoring alongside feature development.

## Detection Methods ○
- **Code Metrics Tools:** Use tools to measure code complexity, class size, and other metrics.
- **Code Reviews:** Look for code that is difficult to understand and review.
- **Static Analysis Tools:** Use tools to identify code smells, such as large classes and long methods.

## Examples
A startup is building a new social media application. The team is under a lot of pressure to deliver new features as quickly as possible. They are constantly adding new features to the codebase without taking the time to refactor it. As a result, the codebase is becoming more and more complex and difficult to maintain. The team is starting to experience a slowdown in development velocity, and the number of bugs is increasing. If they don't start taking the time to refactor the code, they will eventually reach a point where it is impossible to add new features.
