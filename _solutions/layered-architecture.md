---
title: Layered Architecture
description: Divide software system into logical layers with clear responsibilities
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/layered-architecture
problems:
- spaghetti-code
- high-coupling-low-cohesion
- monolithic-architecture-constraints
- tangled-cross-cutting-concerns
- difficult-code-comprehension
- tight-coupling-issues
- ripple-effect-of-changes
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A government agency maintained a legacy case management system where JSP pages contained SQL queries, business validation, and HTML rendering in the same file. Modifying a business rule required editing presentation code, and database changes broke the UI in unpredictable ways. The team introduced a three-layer architecture, first extracting all SQL into a data access layer with repository classes, then moving validation and business rules into a service layer. The JSP pages were reduced to pure presentation concerns. This separation allowed the team to later replace the JSP frontend with a React application while keeping the service and data access layers unchanged.
