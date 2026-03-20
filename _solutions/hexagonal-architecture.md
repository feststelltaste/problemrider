---
title: Hexagonal Architecture
description: Isolating business logic from infrastructure through ports and adapters
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/hexagonal-architecture
problems:
- tight-coupling-issues
- difficult-to-test-code
- legacy-business-logic-extraction-difficulty
- monolithic-architecture-constraints
- technology-lock-in
- vendor-dependency
- architectural-mismatch
- stagnant-architecture
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify the core business logic in the legacy system and separate it from infrastructure concerns such as databases, messaging, and UI
- Define ports (interfaces) that express what the business logic needs from the outside world and what it offers
- Create adapters that implement ports, bridging the gap between the domain and infrastructure technologies
- Start at the boundaries where coupling to infrastructure causes the most pain, such as database access layers
- Introduce the pattern incrementally, wrapping legacy infrastructure calls behind port interfaces one subsystem at a time
- Use dependency injection to wire adapters to ports so that test doubles can replace real infrastructure
- Ensure that no domain code imports infrastructure packages directly

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Business logic becomes testable in isolation without databases, networks, or external services
- Technology migrations become feasible because only adapters need replacement, not core logic
- Enforces clear architectural boundaries that prevent architectural erosion over time
- Enables parallel development: teams can work on adapters and domain independently

**Costs and Risks:**
- Introduces additional abstractions and indirection that increase the number of files and interfaces
- Requires discipline to maintain the boundary, especially under deadline pressure
- Retrofitting onto a deeply coupled legacy system can be a large upfront investment
- Over-engineering risk if applied to simple systems that do not benefit from the separation

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company had a monolithic order management system where business rules were interleaved with direct JDBC calls, SOAP client code, and Swing UI logic. Testing any rule required spinning up the full application with a live database. The team began extracting the pricing engine by defining a port interface for inventory lookups and another for tax calculations. Legacy JDBC queries were wrapped in adapter implementations, while tests used in-memory stubs. Within a few months, the pricing engine could be tested in milliseconds rather than minutes, and when the company later migrated from Oracle to PostgreSQL, only the adapter implementations needed changes.
