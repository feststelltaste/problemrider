---
title: Abstraction
description: Decouple components through contracts so that implementations can vary
  independently
category:
- Architecture
- Code
problems:
- high-coupling-low-cohesion
- tight-coupling-issues
- ripple-effect-of-changes
- monolithic-architecture-constraints
- technology-lock-in
- vendor-lock-in
- difficult-code-reuse
- stagnant-architecture
- poor-encapsulation
layout: solution
related_solutions:
- slug: protocol-abstraction
  similarity: 0.85
- slug: abstraction-layers
  similarity: 0.8
- slug: database-abstraction
  similarity: 0.8
- slug: loose-coupling
  similarity: 0.8
- slug: bridges
  similarity: 0.75
- slug: facades
  similarity: 0.75
---

## Description

Abstraction is the general practice of defining explicit interfaces or contracts at the boundaries between components, so that each side depends only on the agreed contract rather than on the other side's concrete implementation details. Once such a contract is in place, either side can change internally — swap a data structure, replace a library, rewrite an algorithm — as long as it continues to honor the contract, which decouples the pace and risk of change on one side from the other. Legacy systems tend to accumulate the opposite condition over time: modules reach directly into each other's internals, business logic instantiates concrete vendor classes, and a change anywhere ripples unpredictably everywhere, because no stable boundary was ever established. Introducing abstraction at these boundaries is usually done incrementally, often as part of a Strangler Fig migration, by wrapping an existing component behind a newly defined interface before its internals are touched, which turns a monolith's rigid internal structure into a set of independently replaceable parts. This is also what makes many other structural remedies possible in the first place: dependency injection, mocking in tests, and vendor substitution all rely on a prior abstraction step to have somewhere to attach to. Because contracts are only useful if they remain stable, introducing abstraction prematurely — before it is clear where variation is actually needed — can add complexity without a corresponding benefit.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify tightly coupled boundaries in the legacy system and define explicit interfaces or contracts between them
- Introduce interface types or abstract base classes at module boundaries before changing implementations
- Replace direct class instantiation with dependency injection or factory patterns
- Extract platform-specific or vendor-specific code behind abstraction layers so alternatives can be swapped in
- Use the Strangler Fig approach to gradually wrap legacy components with clean abstractions
- Write integration tests against the contract rather than the implementation to validate substitutability

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables independent evolution of components, reducing the blast radius of changes
- Makes it possible to replace legacy implementations incrementally without big-bang rewrites
- Improves testability by allowing mock or stub implementations
- Reduces vendor lock-in by keeping implementation details behind stable contracts

**Costs and Risks:**
- Adds indirection that can make debugging and tracing harder in unfamiliar codebases
- Premature abstraction can create unnecessary complexity if the variation points never materialize
- Requires team discipline to keep contracts stable and well-documented
- Performance-sensitive paths may suffer from the overhead of additional layers

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company had its order processing system directly coupled to a specific message queue product. When the vendor raised prices significantly, switching was estimated at six months of work. By introducing a messaging abstraction layer over a three-month period, the team was able to swap the underlying broker in two weeks. The same abstraction later allowed them to run an in-memory implementation during integration tests, cutting test suite execution time by 60%.
