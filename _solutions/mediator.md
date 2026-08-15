---
title: Mediator
description: Decouple direct communication between components
category:
- Architecture
- Code
problems:
- high-coupling-low-cohesion
- tight-coupling-issues
- spaghetti-code
- circular-dependency-problems
- monolithic-architecture-constraints
- ripple-effect-of-changes
layout: solution
related_solutions:
- slug: adapter
  similarity: 0.75
- slug: facades
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.75
- slug: abstraction
  similarity: 0.75
- slug: dependency-injection
  similarity: 0.7
- slug: protocol-abstraction
  similarity: 0.7
---

## Description

The mediator pattern introduces a dedicated object that encapsulates the communication and coordination logic between a set of components, so that those components no longer hold direct references to and call each other but instead interact exclusively through the mediator. Mechanically, this converts a dense, many-to-many web of direct dependencies into a simpler, star-shaped structure where every component depends only on the mediator, which then owns the responsibility for orchestrating how they work together. In legacy systems, clusters of classes frequently accrete direct references to one another over years of incremental feature additions, until a change to one component requires understanding and modifying a dozen others that all coordinate with it directly and in slightly different ways — a hallmark of the tangled, spaghetti-like coupling that makes legacy code disproportionately expensive to change. Introducing a mediator around such a cluster does not reduce the total amount of coordination logic in the system, but it consolidates and centralizes it, so that adding, removing, or replacing one component only requires updating its interaction with the mediator rather than every other component it used to reference directly. The risk to watch for, especially when retrofitting this pattern onto legacy code, is that the mediator itself can accumulate so much logic over time that it becomes a new god object and a new bottleneck, so it needs to be scoped narrowly to coordination and kept free of business logic that belongs elsewhere.

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
