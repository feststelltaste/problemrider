---
title: Dependency Injection
description: Manage and inject dependencies between components externally
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/dependency-injection
problems:
- tight-coupling-issues
- difficult-to-test-code
- hidden-dependencies
- high-coupling-low-cohesion
- difficult-code-reuse
- technology-lock-in
- global-state-and-side-effects
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy .NET application used static helper classes and direct instantiation throughout, making unit testing nearly impossible. The team needed to add tests before a critical modernization effort. They started by introducing constructor injection for the 30 most critical business logic classes, extracting interfaces for database access, email sending, and file operations. Using .NET's built-in DI container, they wired production implementations for runtime and injected mock implementations in tests. Within three months, test coverage on those 30 classes went from zero to 80 percent, and the team discovered four latent bugs during the process. The explicit dependency declarations also revealed several circular dependencies that had been invisible when dependencies were created internally.
