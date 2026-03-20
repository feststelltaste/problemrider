---
title: Error Handling
description: Mechanisms for detecting, logging, and handling errors
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/error-handling
problems:
- inadequate-error-handling
- cascade-failures
- unpredictable-system-behavior
- debugging-difficulties
- silent-data-corruption
- increased-error-rates
- slow-incident-resolution
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Audit the codebase for swallowed exceptions, empty catch blocks, and generic error handlers that hide root causes
- Establish a consistent error handling strategy: fail fast for unrecoverable errors, retry with backoff for transient failures, and degrade gracefully for non-critical features
- Replace catch-all exception handlers with specific handlers that take appropriate action for each error type
- Add contextual information to error messages and log entries to make diagnosis faster
- Implement structured error responses for APIs that provide meaningful error codes, messages, and suggested actions
- Create centralized error handling middleware rather than scattering try-catch blocks throughout the codebase
- Add monitoring and alerting for error rates so trending issues are detected before they become outages

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Makes the system predictable by defining explicit behavior for each failure mode
- Improves debugging speed through contextual error information
- Prevents silent failures that lead to data corruption or inconsistent state
- Enables faster incident resolution through clear error signals
- Reduces cascading failures by containing errors at appropriate boundaries

**Costs and Risks:**
- Retrofitting error handling into legacy code is labor-intensive and risks changing behavior
- Overly aggressive error handling (failing fast everywhere) can reduce system availability
- Verbose error messages may inadvertently expose sensitive system details
- Consistent error handling requires team discipline and ongoing code review attention

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce system had a pattern of catching all exceptions with generic handlers that logged "An error occurred" and returned HTTP 500. When production issues arose, the team spent hours correlating vague log entries with user reports. A systematic audit found 340 generic catch blocks. The team replaced them with specific handlers over three months: validation errors returned 400 with field-level details, authentication errors returned 401 with clear messages, and unexpected errors included correlation IDs linking logs to user sessions. Mean time to diagnose production issues dropped from four hours to 30 minutes, and the number of support tickets categorized as "unknown error" decreased by 85%.
