---
title: Tangled Cross-Cutting Concerns
description: A situation where cross-cutting concerns, such as logging, security,
  and transactions, are tightly coupled with the business logic.
category:
- Architecture
- Code
related_problems:
- slug: spaghetti-code
  similarity: 0.6
- slug: tight-coupling-issues
  similarity: 0.6
- slug: complex-and-obscure-logic
  similarity: 0.6
- slug: team-coordination-issues
  similarity: 0.6
- slug: circular-dependency-problems
  similarity: 0.55
- slug: poorly-defined-responsibilities
  similarity: 0.55
layout: problem
---

## Description
Tangled cross-cutting concerns is a situation where cross-cutting concerns, such as logging, security, and transactions, are tightly coupled with the business logic. This is a common problem in monolithic architectures, where there is no clear separation of concerns. Tangled cross-cutting concerns can lead to a number of problems, including code duplication, tight coupling issues, and difficult-to-test code.

## Indicators ⟡
- The same code for logging, security, or transactions is repeated in multiple places.
- It is not possible to change the implementation of a cross-cutting concern without affecting the business logic.
- It is not possible to test the business logic without also testing the cross-cutting concerns.
- The code is difficult to understand and maintain.

## Symptoms ▲

- [Code Duplication](code-duplication.md)
<br/>  Cross-cutting logic like logging and security gets copied into every component rather than being centralized.
- [Difficult to Test Code](difficult-to-test-code.md)
<br/>  Business logic intertwined with cross-cutting concerns cannot be tested in isolation without also exercising logging, security, etc.
- [Ripple Effect of Changes](ripple-effect-of-changes.md)
<br/>  Changing a cross-cutting concern like logging requires modifications across all components where it is embedded.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Maintaining cross-cutting logic scattered throughout the codebase requires disproportionate effort for any change.
- [Complex and Obscure Logic](complex-and-obscure-logic.md)
<br/>  Business logic becomes hard to understand when interleaved with transaction management, security checks, and logging code.

## Causes ▼
- [High Coupling and Low Cohesion](high-coupling-low-cohesion.md)
<br/>  Poor separation of concerns at the architectural level leads to cross-cutting logic being embedded directly in business components.
- [Feature Creep Without Refactoring](feature-creep-without-refactoring.md)
<br/>  Continuously adding features without restructuring leads to cross-cutting concerns being mixed into business logic incrementally.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers lacking experience with separation-of-concerns patterns embed cross-cutting logic directly into business code.

## Detection Methods ○
- **Code Reviews:** Look for code where cross-cutting concerns are mixed in with the business logic.
- **Static Analysis Tools:** Use tools to identify duplicated code and other code smells.
- **Architectural Diagrams:** Create a diagram of the system architecture to identify where the cross-cutting concerns are located.

## Examples
A company has a large, monolithic e-commerce application. The application has a number of different services, including a product catalog, a shopping cart, and a payment gateway. The code for logging, security, and transactions is duplicated in all of the services. This makes it difficult to change the implementation of a cross-cutting concern, and it also makes it difficult to test the business logic in isolation. As a result, the code is difficult to maintain, and the code quality is poor.
