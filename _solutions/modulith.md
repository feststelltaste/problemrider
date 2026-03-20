---
title: Modulith
description: Structure system architecture into independent, interchangeable modules
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/modulith
problems:
- monolithic-architecture-constraints
- high-coupling-low-cohesion
- tight-coupling-issues
- stagnant-architecture
- ripple-effect-of-changes
- difficult-code-reuse
- deployment-coupling
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify logical module boundaries within the monolith based on domain capabilities
- Enforce module boundaries using language-level mechanisms such as packages, namespaces, or build modules
- Define explicit public APIs for each module and restrict access to internal implementation
- Use architectural fitness tests or tools like ArchUnit to detect and prevent boundary violations
- Communicate between modules through well-defined interfaces or events rather than direct internal calls
- Migrate the monolith incrementally, converting one tangled area at a time into a proper module
- Keep modules deployable as a single unit while maintaining the option to extract them as services later

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Achieves many benefits of microservices without the operational complexity of distributed systems
- Provides a natural stepping stone toward microservices if needed later
- Keeps the simplicity of a single deployment while enforcing clear boundaries
- Easier to introduce in legacy systems than a full microservice decomposition

**Costs and Risks:**
- Requires discipline to maintain module boundaries within a single codebase
- Without strict enforcement, boundaries erode over time under deadline pressure
- Does not provide independent scaling or deployment of individual modules
- Teams may treat it as a halfway measure and not invest enough in boundary enforcement

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A mid-size SaaS company had a monolithic Spring Boot application where all domain logic was interleaved across packages with no clear boundaries. They considered microservices but lacked the operational maturity. Instead, they restructured the application into a modulith using Spring Modulith, defining clear module boundaries for billing, user management, and reporting. Each module exposed a public API package and communicated via application events. ArchUnit tests prevented cross-module internal access. This gave teams clear ownership of modules and significantly reduced accidental coupling, while the system remained a single deployable artifact.
