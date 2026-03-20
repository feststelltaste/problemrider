---
title: Abstraction Layers
description: Encapsulating hardware-specific details through abstraction layers
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/portability/abstraction-layers
problems:
- tight-coupling-issues
- technology-lock-in
- vendor-lock-in
- difficult-code-reuse
- hidden-dependencies
- architectural-mismatch
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify platform-specific or vendor-specific dependencies in the legacy codebase that limit portability
- Define technology-neutral interfaces that capture the essential operations without exposing implementation details
- Implement concrete adapters for each target platform or technology behind the abstraction
- Use dependency injection to wire the appropriate implementation at runtime based on the deployment environment
- Migrate legacy code to depend on the abstraction interfaces rather than concrete implementations
- Start with the most painful coupling points and expand the abstraction layer incrementally
- Test each adapter independently and verify that behavior is consistent across implementations

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables migration between platforms, vendors, or technologies without rewriting business logic
- Improves testability by allowing mock or in-memory implementations
- Reduces the blast radius of technology changes to the adapter layer
- Promotes cleaner architecture by separating concerns

**Costs and Risks:**
- Abstraction layers add indirection that can obscure what is actually happening at runtime
- Designing the right abstraction level is difficult; too broad and it leaks, too narrow and it over-constrains
- Maintaining multiple adapter implementations increases the overall maintenance surface
- Premature abstraction can add unnecessary complexity when portability is not actually needed
- Performance-critical paths may suffer from the additional indirection

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A manufacturing company's legacy control system was tightly coupled to a specific PLC (programmable logic controller) vendor's proprietary SDK. When the vendor announced end-of-life for their product line, the team faced a complete rewrite. Instead, they introduced a hardware abstraction layer that defined generic interfaces for sensor reading, actuator control, and alarm management. They implemented adapters for both the existing vendor's SDK and the new vendor's API. This allowed them to migrate production lines incrementally, running both hardware platforms simultaneously during the transition, and the business logic remained completely unchanged throughout the process.
