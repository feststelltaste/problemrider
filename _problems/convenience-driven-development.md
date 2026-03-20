---
title: Convenience-Driven Development
description: A development practice where developers choose the easiest and most convenient
  solution, rather than the best solution.
category:
- Code
- Process
related_problems:
- slug: cv-driven-development
  similarity: 0.7
- slug: brittle-codebase
  similarity: 0.6
- slug: feature-creep-without-refactoring
  similarity: 0.55
- slug: difficult-code-comprehension
  similarity: 0.55
- slug: assumption-based-development
  similarity: 0.55
- slug: increased-technical-shortcuts
  similarity: 0.55
solutions:
- architecture-reviews
- clean-code
- separation-of-concerns
- solid-principles
layout: problem
---

## Description
Convenience-driven development is a development practice where developers choose the easiest and most convenient solution, rather than the best solution. This often leads to a gradual degradation of the codebase, as developers take shortcuts and make design decisions that are not in the best long-term interest of the project. Convenience-driven development is often a sign of a lack of experience or a lack of discipline on the part of the development team.

## Indicators ⟡
- The codebase is full of hacks and workarounds.
- The design of the codebase is inconsistent.
- There is a lot of duplicated code.
- The codebase is difficult to understand and maintain.

## Symptoms ▲

- [Code Duplication](code-duplication.md)
<br/>  Taking the convenient path often means copying existing code rather than investing time in creating reusable abstractions.
- [High Technical Debt](high-technical-debt.md)
<br/>  Consistently choosing the easiest solution over the best solution accumulates design shortcuts that become technical debt.
- [Inconsistent Codebase](inconsistent-codebase.md)
<br/>  When each developer takes their own convenient shortcuts, the codebase develops inconsistent patterns and design approaches.
- [Difficult Code Comprehension](difficult-code-comprehension.md)
<br/>  Convenient shortcuts like poor naming, missing abstractions, and ad hoc solutions make the codebase harder to understand over time.
- [Increasing Brittleness](increasing-brittleness.md)
<br/>  Shortcuts and quick fixes make the codebase increasingly fragile as they bypass proper design principles.
## Causes ▼

- [Time Pressure](time-pressure.md)
<br/>  Pressure to deliver quickly pushes developers toward the fastest solution rather than the best-designed one.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers lacking experience may not recognize that the convenient solution creates long-term problems, defaulting to what they know.
- [Short-Term Focus](short-term-focus.md)
<br/>  When management prioritizes immediate delivery over long-term code health, developers are incentivized to take convenient shortcuts.
- [Deadline Pressure](deadline-pressure.md)
<br/>  Intense deadline pressure leaves developers no time to pursue proper solutions, making convenience the only viable option.
## Detection Methods ○
- **Code Reviews:** Look for code that is poorly designed and difficult to understand.
- **Static Analysis Tools:** Use tools to identify code smells, such as duplicated code and large classes.
- **Developer Surveys:** Ask developers if they feel like they are able to write high-quality code.

## Examples
A developer needs to add a new feature to a legacy system. The developer is under pressure to deliver the feature as quickly as possible. The developer decides to copy and paste a large block of code from another part of the system, rather than taking the time to refactor the code and create a new, reusable component. This saves the developer a few hours of work in the short term, but it adds to the technical debt of the system and makes it more difficult to maintain in the long term.
