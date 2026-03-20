---
title: Single Entry Point Design
description: A design where all requests to a system must go through a single object
  or component.
category:
- Architecture
related_problems:
- slug: god-object-anti-pattern
  similarity: 0.6
- slug: monolithic-functions-and-classes
  similarity: 0.6
- slug: maintenance-bottlenecks
  similarity: 0.55
- slug: monolithic-architecture-constraints
  similarity: 0.5
- slug: process-design-flaws
  similarity: 0.5
solutions:
- architecture-reviews
- separation-of-concerns
- solid-principles
layout: problem
---

## Description
A single entry point design is a design where all requests to a system must go through a single object or component. This can be a problem because it can lead to a god object anti-pattern, where the single entry point becomes responsible for too many things. It can also create a maintenance bottleneck, as all changes to the system must go through the single entry point.

## Indicators ⟡
- A single class or component that is responsible for handling all incoming requests.
- The single entry point is often very large and complex.
- It is difficult to make changes to the system without touching the single entry point.
- The single entry point is a common source of bugs.

## Symptoms ▲

- [God Object Anti-Pattern](god-object-anti-pattern.md)
<br/>  The single entry point accumulates responsibilities over time, becoming a god object that handles too many concerns.
- [Maintenance Bottlenecks](maintenance-bottlenecks.md)
<br/>  All changes must flow through the single entry point, creating a bottleneck where modifications queue up and slow down development.
- [Brittle Codebase](brittle-codebase.md)
<br/>  Changes to the single entry point risk breaking many unrelated features since all requests depend on it.
- [High Coupling and Low Cohesion](high-coupling-low-cohesion.md)
<br/>  All components become coupled through the single entry point, creating excessive interdependencies.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Adding new features requires modifying the single entry point, which is risky and time-consuming due to its complexity.
## Causes ▼

- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  Monolithic designs naturally funnel all requests through centralized components rather than distributing responsibility.
## Detection Methods ○
- **Code Reviews:** Look for single classes or components that are responsible for handling all incoming requests.
- **Static Analysis Tools:** Use tools to identify large classes and classes with a large number of dependencies.
- **Architectural Diagrams:** Create a diagram of the system architecture to identify single points of entry.

## Examples
A web application has a single `FrontController` servlet that is responsible for handling all incoming HTTP requests. The `FrontController` is responsible for routing requests to the appropriate handler, but it is also responsible for authentication, authorization, logging, and a number of other cross-cutting concerns. The `FrontController` is over 1000 lines of code and has dependencies on dozens of other classes. It is a major maintenance bottleneck, and it is a common source of bugs.
