---
title: Facades
description: Use facades to hide complex subsystems behind a simplified interface
category:
- Architecture
- Code
problems:
- monolithic-architecture-constraints
- difficult-code-comprehension
- high-coupling-low-cohesion
- spaghetti-code
- difficult-code-reuse
- poor-interfaces-between-applications
- poor-encapsulation
layout: solution
related_solutions:
- slug: adapter
  similarity: 0.8
- slug: pattern-language
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.75
- slug: mediator
  similarity: 0.75
- slug: abstraction
  similarity: 0.75
- slug: dependency-injection
  similarity: 0.75
---

## Description

A facade is a single, simplified interface placed in front of a complex subsystem, delegating every call through to the existing lower-level classes and functions without changing their internal structure, so that most consumers only ever need to learn the facade's handful of high-level methods rather than the subsystem's full surface area. Legacy subsystems frequently grow dozens of public classes over years of incremental extension, with no single obvious entry point for someone who just wants to accomplish a common task, which turns onboarding into an exercise in archaeology and makes every new integration a fresh occasion to misuse the subsystem's internals. Introducing a facade for the subsystem's most common use cases gives new consumers a stable, learnable contract while leaving advanced users free to reach past it into the subsystem directly when they genuinely need to, and because the facade sits at a clean boundary, it also becomes a convenient seam for later replacing the subsystem entirely without having to touch every caller. The risk is that, once everything is routed through it, the facade can turn into a bottleneck or start absorbing logic that really belongs in the subsystem itself, and maintaining both the facade and any direct access paths in parallel adds surface area that must be kept consistent.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify complex subsystems in the legacy codebase that consumers interact with through many low-level classes or functions
- Create a facade class or module that provides a simplified, high-level API for the most common use cases
- Delegate all calls from the facade to the existing subsystem classes without changing their internal structure
- Use facades as the entry point for new consumers while allowing advanced users to bypass the facade when needed
- Introduce facades incrementally, starting with the subsystems that have the most consumers or the steepest learning curves
- Write tests against the facade interface to establish a stable contract

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces the learning curve for developers working with complex legacy subsystems
- Creates a stable interface that shields consumers from internal subsystem changes
- Provides a natural seam for future refactoring or replacement of the underlying subsystem

**Costs and Risks:**
- The facade may become a bottleneck if all access is forced through it
- Risk of the facade accumulating logic that should live in the subsystem
- A poorly designed facade can oversimplify the interface, limiting necessary functionality
- Maintaining both the facade and direct access paths increases the surface area

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy accounting system had 47 public classes in its billing module, with no clear entry point. New developers spent an average of two weeks understanding how to use the module for common tasks. The team introduced a BillingFacade with five methods covering 90% of billing use cases. Onboarding time for new developers dropped from two weeks to two days, and the facade later served as the contract boundary when the team began replacing the underlying billing engine with a modern implementation.
