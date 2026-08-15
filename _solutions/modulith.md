---
title: Modulith
description: Structure system architecture into independent, interchangeable modules
category:
- Architecture
problems:
- monolithic-architecture-constraints
- high-coupling-low-cohesion
- tight-coupling-issues
- stagnant-architecture
- ripple-effect-of-changes
- difficult-code-reuse
- deployment-coupling
layout: solution
related_solutions:
- slug: microservices-architecture
  similarity: 0.75
- slug: modularization-and-bounded-contexts
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.7
- slug: layered-architecture
  similarity: 0.7
- slug: hexagonal-architecture
  similarity: 0.65
- slug: containerization
  similarity: 0.65
---

## Description

A modulith keeps a system deployed as a single unit while enforcing hard internal boundaries between its logical modules — typically through language-level mechanisms such as packages or build modules, explicit public APIs for each module, and architectural fitness tests like ArchUnit that fail the build if code reaches across a boundary it should not. It achieves many of the coupling and clarity benefits associated with microservices — well-defined interfaces, restricted internal access, clear ownership of a bounded area of functionality — without introducing the network calls, independent deployments, and distributed-systems failure modes that come with actually splitting the system into separate services. In a legacy monolith, this addresses a very specific failure pattern: domain logic that has become interleaved across packages with no enforced boundaries, so that a change to one area silently ripples into others because nothing in the codebase prevents modules from reaching into each other's internals. Because a modulith remains a single deployable artifact, it is markedly easier to retrofit onto legacy code than a full microservices decomposition, making it a practical stepping stone for teams that recognize their monolith's coupling is a problem but do not yet have the operational maturity, or the clearly enough understood domain boundaries, to justify distributed services. Its principal risk is that, unlike a genuine service boundary enforced by network calls, the boundaries in a modulith are enforced only by discipline and tooling within a single codebase, so without consistently run fitness tests they tend to erode again under the same deadline pressure that caused the original entanglement.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A mid-size SaaS company had a monolithic Spring Boot application where all domain logic was interleaved across packages with no clear boundaries. They considered microservices but lacked the operational maturity. Instead, they restructured the application into a modulith using Spring Modulith, defining clear module boundaries for billing, user management, and reporting. Each module exposed a public API package and communicated via application events. ArchUnit tests prevented cross-module internal access. This gave teams clear ownership of modules and significantly reduced accidental coupling, while the system remained a single deployable artifact.
