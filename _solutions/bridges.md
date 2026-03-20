---
title: Bridges
description: Let abstraction hierarchies and implementation hierarchies evolve independently
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/compatibility/bridges
problems:
- high-coupling-low-cohesion
- tight-coupling-issues
- monolithic-architecture-constraints
- difficult-code-reuse
- ripple-effect-of-changes
- technology-lock-in
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify places where class hierarchies mix abstraction concerns with implementation details (e.g., PlatformXRenderer, PlatformYRenderer)
- Separate the abstraction hierarchy from the implementation hierarchy by introducing a bridge interface between them
- Inject the implementation through the bridge at construction time rather than inheriting it
- Use this pattern when a legacy system needs to support multiple platforms, drivers, or rendering backends without duplicating logic
- Refactor incrementally by bridging one implementation variant at a time while keeping the legacy hierarchy functional

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Both abstraction and implementation can be extended independently without combinatorial explosion of classes
- Simplifies adding new platform or technology support to a legacy system
- Reduces code duplication across implementation variants

**Costs and Risks:**
- Adds structural complexity that may be excessive for systems with only one implementation variant
- Requires careful interface design at the bridge boundary
- Developers unfamiliar with the pattern may find the indirection confusing
- Retrofitting the pattern into a deeply entangled legacy hierarchy can be risky without good test coverage

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A manufacturing company had a reporting system with separate class hierarchies for each output format (PDF, Excel, CSV), each duplicating significant rendering logic. By introducing a bridge pattern that separated the report structure from the output rendering, the team reduced the codebase by 35% and was able to add a new HTML output format in two days instead of the three weeks it had previously taken to clone and modify an entire hierarchy.
