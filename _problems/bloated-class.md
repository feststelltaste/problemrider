---
title: Bloated Class
description: A class that has grown so large that it has become difficult to understand,
  maintain, and test.
category:
- Code
related_problems:
- slug: excessive-class-size
  similarity: 0.75
- slug: monolithic-functions-and-classes
  similarity: 0.6
- slug: uncontrolled-codebase-growth
  similarity: 0.55
- slug: over-reliance-on-utility-classes
  similarity: 0.55
- slug: god-object-anti-pattern
  similarity: 0.55
- slug: feature-creep
  similarity: 0.5
layout: problem
---

## Description
A bloated class is a class that has accumulated too many responsibilities over time. It often starts as a small, well-designed class, but as new features are added, it grows in size and complexity. This makes it difficult to understand, maintain, and test. Bloated classes are a common code smell and a sign of technical debt.

## Indicators ⟡
- A class with a large number of methods and properties.
- A class that is difficult to name because it does too many things.
- A class that is frequently modified by multiple developers for different reasons.
- A class that is difficult to test in isolation.

## Symptoms ▲

- [Difficult Code Comprehension](difficult-code-comprehension.md)
<br/>  Oversized classes with too many responsibilities become extremely hard for developers to understand and reason about.
- [Superficial Code Reviews](superficial-code-reviews.md)
<br/>  Changes to bloated classes require reviewers to understand extensive context, making reviews time-consuming.
- [Regression Bugs](regression-bugs.md)
<br/>  Modifying one part of a bloated class frequently breaks unrelated functionality within the same class.
- [Merge Conflicts](merge-conflicts.md)
<br/>  Multiple developers working on different features within the same bloated class frequently create merge conflicts.
## Causes ▼

- [Inconsistent Coding Standards](inconsistent-coding-standards.md)
<br/>  Without standards enforcing single responsibility and class size limits, classes grow unchecked over time.
- [Feature Creep](feature-creep.md)
<br/>  Continuously adding features without refactoring leads to responsibilities piling up in existing classes.
- [Refactoring Avoidance](refactoring-avoidance.md)
<br/>  Avoiding the effort of splitting classes into smaller, focused components allows bloating to continue unchecked.
- [Time Pressure](time-pressure.md)
<br/>  Under deadline pressure, developers add functionality to existing classes rather than properly designing new ones.
## Detection Methods ○
- **Code Metrics Tools:** Use tools to measure class size, number of methods, and cyclomatic complexity.
- **Code Reviews:** Look for classes that are difficult to understand and review.
- **Static Analysis Tools:** Use tools to identify code smells, such as large classes and long methods.

## Examples
A `User` class in a social media application that is responsible for everything from authentication and authorization to profile management, news feed generation, and sending notifications. The class has over 50 methods and 1000 lines of code. When a developer wants to make a change to the news feed generation logic, they have to be careful not to break the authentication logic. It is also very difficult to write unit tests for the class because it has so many dependencies. As a result, development is slow and error-prone.
