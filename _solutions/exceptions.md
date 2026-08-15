---
title: Exceptions
description: Using exceptions for signaling and handling error states
category:
- Code
problems:
- inadequate-error-handling
- debugging-difficulties
- unpredictable-system-behavior
- silent-data-corruption
- cascade-failures
- difficult-code-comprehension
layout: solution
related_solutions:
- slug: error-handling
  similarity: 0.8
- slug: error-reporting-and-analysis
  similarity: 0.8
- slug: error-logging
  similarity: 0.75
- slug: pattern-language
  similarity: 0.7
- slug: error-logs
  similarity: 0.7
- slug: logging
  similarity: 0.7
---

## Description

Exceptions are a language-level mechanism for signaling that an operation could not complete as expected, propagating that failure up the call stack until code that knows how to handle it catches it, in contrast to encoding errors as integer return codes, boolean flags, or other values a caller can silently ignore. Many legacy codebases, especially those originating in C-style languages, rely on the latter approach, and because nothing forces a caller to check a return value, ignored error codes are a common route by which a failure at one point in the code silently becomes data corruption or a crash somewhere else entirely, with no direct link between cause and eventual symptom. Migrating such code to a typed exception hierarchy, and catching those exceptions only at well-defined boundaries such as the API layer or a batch job's entry point rather than around every call, makes failure states impossible to overlook and gives the team a stack trace and structured context to work with when something does go wrong. The transition has to be done carefully, though, since introducing exceptions into code that previously relied on error codes can change observable behavior if not tested thoroughly, and in performance-sensitive paths on some platforms the cost of throwing exceptions frequently is non-trivial enough that they should be reserved for genuinely exceptional conditions rather than routine control flow.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy C++ application used integer error codes returned from functions, with -1 meaning failure and various positive values indicating specific errors. Many call sites did not check return values, causing failures to propagate silently until they manifested as data corruption or crashes far from the original error. The team introduced a custom exception hierarchy with domain-specific types like InvalidOrderException and InsufficientInventoryException. They refactored the most critical modules first, wrapping legacy functions that returned error codes in adapter functions that threw exceptions. Within four months, the number of "mystery crashes" dropped by 70% because errors were now caught and handled explicitly near their source.
