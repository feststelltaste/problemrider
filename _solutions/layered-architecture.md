---
title: Layered Architecture
description: Divide software system into logical layers with clear responsibilities
category:
- Architecture
problems:
- spaghetti-code
- high-coupling-low-cohesion
- monolithic-architecture-constraints
- tangled-cross-cutting-concerns
- difficult-code-comprehension
- tight-coupling-issues
- ripple-effect-of-changes
- single-entry-point-design
layout: solution
related_solutions:
- slug: abstraction-layers
  similarity: 0.75
- slug: hexagonal-architecture
  similarity: 0.75
- slug: microservices-architecture
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.75
- slug: adapter
  similarity: 0.7
- slug: dependency-injection
  similarity: 0.7
---

## Description

Layered architecture organizes a system into a stack of horizontal layers — typically presentation, business logic, and data access — where each layer exposes a defined interface and depends only on the layer directly beneath it, never the reverse. The dependency rule is the mechanism that does the actual work: by forbidding a layer from reaching past its immediate neighbor, it prevents presentation code from touching the database directly or business rules from leaking into UI controllers, which is precisely the kind of entanglement that accumulates in unmanaged legacy code over time. In a legacy system context, layering is often less about designing from scratch and more about archaeology and extraction: identifying where SQL, validation logic, and rendering code have become interleaved in the same file or class, and carving out the responsibilities into their proper layer one violation at a time. Because each layer can be tested and modified independently once boundaries are established, the blast radius of a change shrinks from "anywhere in the codebase" to "within one layer," which directly counters the ripple-effect-of-changes and difficult-code-comprehension problems that plague tangled legacy code. Layering does not eliminate coupling — it organizes it — so its value in modernization comes from making the remaining dependencies visible, predictable, and enforceable rather than implicit and scattered.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define clear layers such as presentation, business logic, and data access, each with explicit responsibilities
- Establish a dependency rule: each layer may only depend on the layer directly below it
- Identify violations in the legacy code where presentation code directly accesses the database or business logic is embedded in UI controllers
- Refactor incrementally by extracting misplaced logic into the appropriate layer
- Use package or module naming conventions that reflect the layered structure
- Introduce interfaces at layer boundaries so implementations can be replaced independently
- Enforce layer boundaries through architectural fitness functions or static analysis tools

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides a well-understood structure that most developers can follow immediately
- Isolates changes to a single layer, reducing the blast radius of modifications
- Enables independent testing of each layer through well-defined interfaces
- Makes the codebase navigable by providing a predictable organization

**Costs and Risks:**
- Strict layering can lead to pass-through methods that add boilerplate without value
- May not fit well for cross-cutting concerns like logging, security, or transaction management
- Can become a straightjacket if enforced too rigidly, preventing pragmatic shortcuts
- Retrofitting layers onto deeply entangled legacy code requires substantial effort

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A government agency maintained a legacy case management system where JSP pages contained SQL queries, business validation, and HTML rendering in the same file. Modifying a business rule required editing presentation code, and database changes broke the UI in unpredictable ways. The team introduced a three-layer architecture, first extracting all SQL into a data access layer with repository classes, then moving validation and business rules into a service layer. The JSP pages were reduced to pure presentation concerns. This separation allowed the team to later replace the JSP frontend with a React application while keeping the service and data access layers unchanged.
