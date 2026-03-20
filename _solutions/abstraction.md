---
title: Abstraction
description: Decouple components through contracts so that implementations can vary independently
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/compatibility/abstraction
problems:
- high-coupling-low-cohesion
- tight-coupling-issues
- ripple-effect-of-changes
- monolithic-architecture-constraints
- technology-lock-in
- vendor-lock-in
- difficult-code-reuse
- stagnant-architecture
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company had its order processing system directly coupled to a specific message queue product. When the vendor raised prices significantly, switching was estimated at six months of work. By introducing a messaging abstraction layer over a three-month period, the team was able to swap the underlying broker in two weeks. The same abstraction later allowed them to run an in-memory implementation during integration tests, cutting test suite execution time by 60%.
