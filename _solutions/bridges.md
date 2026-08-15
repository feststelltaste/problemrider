---
title: Bridges
description: Let abstraction hierarchies and implementation hierarchies evolve independently
category:
- Architecture
- Code
problems:
- high-coupling-low-cohesion
- tight-coupling-issues
- monolithic-architecture-constraints
- difficult-code-reuse
- ripple-effect-of-changes
- technology-lock-in
layout: solution
related_solutions:
- slug: abstraction
  similarity: 0.75
- slug: abstraction-layers
  similarity: 0.75
- slug: facades
  similarity: 0.75
- slug: protocol-abstraction
  similarity: 0.75
- slug: adapter
  similarity: 0.75
- slug: database-abstraction
  similarity: 0.75
---

## Description

The bridge pattern separates an abstraction hierarchy (what a client thinks it is calling) from its implementation hierarchy (how that operation is actually carried out), connecting the two through an interface injected at construction time rather than through inheritance, so that either hierarchy can be extended without touching or duplicating the other. Its purpose is to avoid the combinatorial explosion that occurs when a system needs to support multiple variants along two independent dimensions — multiple output formats, multiple platforms, multiple drivers — and inheritance-based designs respond to that by creating one class per combination. Legacy systems frequently arrive at exactly this state organically: a class hierarchy created for one implementation variant is copied and modified for each new one that comes along, since introducing a proper abstraction boundary was more work than duplicating an existing class under deadline pressure. Retrofitting a bridge into such a hierarchy means identifying where abstraction and implementation concerns are mixed together, and incrementally extracting a bridge interface one implementation variant at a time while the legacy hierarchy continues to function unmodified for the rest. The benefit is a sharp reduction in duplicated logic and much cheaper support for new variants going forward, at the cost of an added layer of indirection that is only worthwhile once a system genuinely needs more than one implementation variant.

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
