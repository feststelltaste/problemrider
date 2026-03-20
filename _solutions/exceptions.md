---
title: Exceptions
description: Using exceptions for signaling and handling error states
category:
- Code
quality_tactics_url: https://qualitytactics.de/en/reliability/exceptions
problems:
- inadequate-error-handling
- debugging-difficulties
- unpredictable-system-behavior
- silent-data-corruption
- cascade-failures
- difficult-code-comprehension
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Replace error codes, boolean return values, and silent failures with typed exceptions that clearly describe what went wrong
- Define a hierarchy of custom exception types that distinguishes between recoverable and unrecoverable errors
- Catch exceptions at appropriate boundaries (service layer, API boundary, batch job entry point) rather than at every method call
- Never swallow exceptions silently; always log, wrap, or rethrow with additional context
- Use exception metadata (error codes, affected entities, suggested actions) to provide actionable information to callers
- Establish team conventions for when to use checked vs. unchecked exceptions based on the language and framework
- Refactor legacy code that uses error codes or magic return values to throw exceptions instead, module by module

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Makes error states explicit and impossible to ignore, unlike return codes that can be silently discarded
- Separates error handling logic from normal flow, improving code readability
- Provides stack trace context that aids debugging and root cause analysis
- Enables centralized error handling at architectural boundaries
- Typed exceptions allow callers to handle different error conditions specifically

**Costs and Risks:**
- Exceptions can be expensive in some languages (e.g., JVM stack trace capture) when thrown frequently
- Overuse of exceptions for control flow makes code harder to follow and degrades performance
- Uncaught exceptions can crash the application if global handlers are not in place
- Migrating from error codes to exceptions in a legacy codebase requires careful testing to preserve behavior
- Teams may disagree on what conditions warrant exceptions vs. return values

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy C++ application used integer error codes returned from functions, with -1 meaning failure and various positive values indicating specific errors. Many call sites did not check return values, causing failures to propagate silently until they manifested as data corruption or crashes far from the original error. The team introduced a custom exception hierarchy with domain-specific types like InvalidOrderException and InsufficientInventoryException. They refactored the most critical modules first, wrapping legacy functions that returned error codes in adapter functions that threw exceptions. Within four months, the number of "mystery crashes" dropped by 70% because errors were now caught and handled explicitly near their source.
