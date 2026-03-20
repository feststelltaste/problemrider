---
title: Dependency Injection Container
description: Centralized management and provision of dependencies
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/dependency-injection-container
problems:
- tight-coupling-issues
- hidden-dependencies
- difficult-to-test-code
- high-coupling-low-cohesion
- global-state-and-side-effects
- god-object-anti-pattern
- maintenance-overhead
layout: solution
---

## How to Apply ◆

> In legacy systems, a DI container centralizes object creation and wiring that is otherwise scattered across factories, singletons, and static initializers, making the dependency graph explicit and manageable.

- Select a DI container appropriate for the legacy system's technology stack (Spring for Java, Autofac or Microsoft.Extensions.DependencyInjection for .NET, InversifyJS for TypeScript).
- Migrate object creation from scattered `new` calls, static factories, and service locators to container-managed registration, starting with the most testability-constrained components.
- Define component lifetimes (singleton, scoped, transient) explicitly in the container configuration to replace implicit lifecycle management in legacy code.
- Use the container to manage cross-cutting concerns (logging, caching, transaction management) through decorator or interceptor patterns rather than embedding them in business logic.
- Register legacy components in the container alongside new components to enable gradual migration without requiring everything to be refactored at once.
- Avoid the service locator anti-pattern where the container itself is passed around — inject specific dependencies through constructors instead.

## Tradeoffs ⇄

> A DI container simplifies dependency management and enables testability but adds framework complexity and can obscure runtime behavior.

**Benefits:**

- Centralizes dependency configuration in one place, making the system's dependency graph explicit and manageable.
- Enables swapping implementations for testing, migration, or environment-specific behavior without changing consumer code.
- Manages object lifetimes automatically, preventing resource leaks from improper lifecycle handling in legacy code.
- Supports incremental legacy modernization by allowing legacy and modern components to coexist in the same container.

**Costs and Risks:**

- Container configuration errors manifest at runtime rather than compile time, potentially causing failures that are difficult to diagnose.
- Large container configurations can become complex and difficult to understand, effectively replacing one form of hidden complexity with another.
- Teams unfamiliar with DI containers may misuse them, creating overly complex configurations or falling into the service locator anti-pattern.
- Container startup time can become significant in legacy systems with thousands of registered components.

## How It Could Be

> The following scenario shows how a DI container enables testability in a legacy application.

A healthcare company's legacy Java application had 800 classes that created their dependencies using `new` operators and static factory methods, making unit testing impossible without starting the entire application. The team introduced Spring's DI container incrementally: they started by registering the 50 most critical service classes, extracting interfaces for their database and external service dependencies. Within two months, those 50 classes could be tested in isolation using mock implementations injected by the container. The container also revealed dependency cycles that had been invisible — three service classes formed a circular dependency chain that the container refused to resolve, forcing the team to untangle the cycle. Over the following year, the team migrated all 800 classes to container management, and test coverage increased from 5% to 55%.
