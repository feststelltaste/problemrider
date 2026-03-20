---
title: Mediator
description: Decouple direct communication between components
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/compatibility/mediator
problems:
- high-coupling-low-cohesion
- tight-coupling-issues
- spaghetti-code
- circular-dependency-problems
- monolithic-architecture-constraints
- ripple-effect-of-changes
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify clusters of components that communicate directly with each other in complex, tangled ways
- Introduce a mediator object that encapsulates the interaction logic between these components
- Refactor components to communicate through the mediator rather than holding direct references to each other
- Use the mediator to manage coordination workflows that previously spanned multiple tightly coupled classes
- Keep the mediator focused on coordination logic; avoid turning it into a god object with business logic
- Introduce mediators incrementally, starting with the most tangled component clusters

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces the number of direct dependencies between components, simplifying the dependency graph
- Makes it easier to add, remove, or replace individual components without affecting others
- Centralizes coordination logic that was previously scattered and duplicated

**Costs and Risks:**
- The mediator can become a single point of complexity if it accumulates too much logic
- Adds a level of indirection that can make control flow harder to follow
- Overapplication creates unnecessary mediators for simple interactions
- The mediator must be carefully designed to avoid becoming a bottleneck

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy UI framework had 20 form components that directly referenced each other to coordinate validation, visibility, and data updates. Adding a new field required modifying up to 12 existing components. The team introduced a FormMediator that managed all inter-component communication through events. After the refactoring, adding a new field required implementing only the field itself and registering it with the mediator, reducing the effort from two days to two hours.
