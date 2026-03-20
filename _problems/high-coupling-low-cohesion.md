---
title: High Coupling and Low Cohesion
description: Software components are overly dependent on each other and perform too
  many unrelated functions, making the system difficult to change and understand.
category:
- Architecture
- Code
related_problems:
- slug: tight-coupling-issues
  similarity: 0.65
- slug: poor-encapsulation
  similarity: 0.6
- slug: poorly-defined-responsibilities
  similarity: 0.6
- slug: ripple-effect-of-changes
  similarity: 0.55
- slug: deployment-coupling
  similarity: 0.55
- slug: inconsistent-quality
  similarity: 0.55
solutions:
- architecture-reviews
- loose-coupling
- separation-of-concerns
- solid-principles
- abstraction
- architecture-conformity-analysis
- architecture-governance
- aspect-oriented-programming-aop
- bounded-contexts
- bridges
- bubble-context
- bulkhead
- event-driven-integration
- facades
- high-cohesion
- layered-architecture
- mediator
- microservices-architecture
- modulith
- object-relational-mapping-orm
layout: problem
---

## Description
High coupling and low cohesion are two of the most common design problems in software development. Coupling refers to the degree of interdependence between software modules, while cohesion refers to the degree to which the elements of a module belong together. A well-designed system should have low coupling and high cohesion. This makes the system easier to understand, maintain, and extend. A system with high coupling and low cohesion, on the other hand, is a nightmare to work with.

## Indicators ⟡
- A small change in one part of the system requires changes in many other seemingly unrelated parts.
- It is difficult to understand the purpose of a module or function without understanding many other parts of the system.
- Changes are prone to introducing new bugs due to unexpected side effects in tightly coupled components.
- Developers spend more time navigating dependencies and understanding complex interactions.

## Symptoms ▲

- [Ripple Effect of Changes](ripple-effect-of-changes.md)
<br/>  Small changes in one module require modifications across many other modules due to tight coupling between components.
- [Difficult to Test Code](difficult-to-test-code.md)
<br/>  Tightly coupled components cannot be tested in isolation because they depend heavily on other components.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Developers must understand and modify multiple interdependent components for even simple feature additions.
- [High Bug Introduction Rate](high-bug-introduction-rate.md)
<br/>  Changes in tightly coupled code have unintended effects in dependent components, frequently introducing new bugs.
- [Fear of Change](fear-of-change.md)
<br/>  The unpredictable cascading effects of changes in coupled code makes developers hesitant to modify the system.
- [Increased Cognitive Load](increased-cognitive-load.md)
<br/>  Understanding any single component requires understanding many other components it is coupled to, overwhelming developers.
- [Difficult Code Comprehension](difficult-code-comprehension.md)
<br/>  Highly coupled code with low cohesion requires understanding many interconnected modules to comprehend any single par....
## Causes ▼

- [Poorly Defined Responsibilities](poorly-defined-responsibilities.md)
<br/>  Without clear module responsibilities, functionality gets spread across multiple components creating unnecessary dependencies.
- [Feature Creep Without Refactoring](feature-creep-without-refactoring.md)
<br/>  Adding features without restructuring the codebase causes responsibilities to bleed across module boundaries over time.
- [Poor Encapsulation](poor-encapsulation.md)
<br/>  Exposing internal implementation details allows other modules to depend on them, creating tight coupling.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers unfamiliar with design principles create tightly coupled code that mixes unrelated concerns within single modules.
- [Refactoring Avoidance](refactoring-avoidance.md)
<br/>  Avoiding necessary refactoring allows coupling to accumulate as the system grows and evolves.
## Detection Methods ○

- **Code Metrics Tools:** Use static analysis tools that measure coupling (e.g., afferent/efferent coupling, CBO - Coupling Between Objects) and cohesion (e.g., LCOM - Lack of Cohesion in Methods).
- **Code Review:** Look for code that is hard to understand, has many dependencies, or performs multiple unrelated tasks.
- **Dependency Graphs:** Visualize the dependencies between modules or classes to identify highly coupled components.
- **Refactoring Challenges:** If refactoring a small part of the system proves to be extremely difficult or risky, it's a sign of high coupling.

## Examples
A legacy e-commerce system has a single `OrderProcessor` class that handles everything from validating customer data, calculating taxes, processing payments, updating inventory, and sending email notifications. A small change to the tax calculation logic requires understanding and potentially modifying the entire `OrderProcessor` class, risking unintended side effects on payment processing or email sending. In another case, a utility function `calculate_total` in a Python application directly accesses and modifies a global `database_connection` object and a global `logging_level` variable. This makes it impossible to test `calculate_total` in isolation without setting up a real database connection and affecting the global logging configuration. This problem is a fundamental design flaw that often accumulates over time in systems that lack continuous architectural oversight and refactoring. It is a major contributor to technical debt and makes legacy system modernization extremely challenging.
