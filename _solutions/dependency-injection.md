---
title: Dependency Injection
description: Manage and inject dependencies between components externally
category:
- Code
- Architecture
problems:
- tight-coupling-issues
- difficult-to-test-code
- hidden-dependencies
- high-coupling-low-cohesion
- difficult-code-reuse
- technology-lock-in
- global-state-and-side-effects
- improper-event-listener-management
- circular-dependency-problems
layout: solution
related_solutions:
- slug: abstracted-file-system-access
  similarity: 0.8
- slug: adapter
  similarity: 0.75
- slug: dependency-injection-container
  similarity: 0.75
- slug: integration-tests
  similarity: 0.75
- slug: database-abstraction
  similarity: 0.75
- slug: abstraction-layers
  similarity: 0.75
---

## Description

Dependency injection is the practice of supplying a component with the objects it depends on from the outside — typically through constructor parameters — rather than having the component construct or look up those dependencies itself using `new` calls or static factory methods. Making dependencies explicit in this way means a class's constructor signature becomes a complete, visible list of what it needs to function, and any of those dependencies can be swapped for an alternative implementation — a test double, a different environment-specific implementation, a cloud storage adapter in place of a local file system — without modifying the class itself. This is foundational to legacy modernization because code that creates its own dependencies internally is, by construction, resistant to unit testing: exercising a single class inevitably pulls in every concrete dependency it constructs, which is precisely why legacy codebases built on static helpers and direct instantiation typically have little to no automated test coverage. Adopting dependency injection in an existing system proceeds incrementally, extracting interfaces for the dependencies of the most testability-constrained classes first and refactoring their constructors to accept those interfaces as parameters, often introducing a DI container to manage the resulting object wiring once enough classes have been converted. Beyond enabling tests, making dependencies explicit routinely exposes structural problems that had been hidden inside implicit object graphs — circular dependencies and classes with an unreasonably large number of collaborators — surfacing design issues that the legacy code's implicit wiring had been quietly obscuring.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify classes in the legacy codebase that create their own dependencies internally using new operators or static factory calls
- Extract interfaces for key dependencies so implementations can be swapped without changing consumers
- Refactor constructors to accept dependencies as parameters rather than creating them internally
- Introduce a DI container (Spring, Guice, .NET DI, or a simple hand-rolled factory) to manage object creation and wiring
- Start with the most testability-constrained classes and expand DI adoption incrementally
- Use DI to inject environment-specific implementations (production database vs. test double, cloud storage vs. local file system)
- Avoid service locator anti-patterns that hide dependencies behind a global registry

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Makes dependencies explicit and visible in constructor signatures
- Enables unit testing by allowing mock or stub implementations to be injected
- Reduces coupling between components, making the codebase more modular and portable
- Facilitates swapping implementations for different environments or platforms
- Simplifies refactoring by isolating change to the injection configuration

**Costs and Risks:**
- DI containers add framework complexity and a learning curve for teams unfamiliar with the pattern
- Over-use of DI can make the application's runtime behavior hard to understand by obscuring which implementation is active
- Legacy code with deep static method chains or global state requires substantial refactoring to adopt DI
- Constructor parameter lists can become unwieldy if too many dependencies are injected (indicating the class needs decomposition)
- Runtime wiring errors may not be caught until the application starts, unlike compile-time dependencies

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy .NET application used static helper classes and direct instantiation throughout, making unit testing nearly impossible. The team needed to add tests before a critical modernization effort. They started by introducing constructor injection for the 30 most critical business logic classes, extracting interfaces for database access, email sending, and file operations. Using .NET's built-in DI container, they wired production implementations for runtime and injected mock implementations in tests. Within three months, test coverage on those 30 classes went from zero to 80 percent, and the team discovered four latent bugs during the process. The explicit dependency declarations also revealed several circular dependencies that had been invisible when dependencies were created internally.
