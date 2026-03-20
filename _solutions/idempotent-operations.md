---
title: Idempotent Operations
description: Design operations so that repeated execution produces the same result as a single execution
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/compatibility/idempotent-operations
problems:
- cascade-failures
- inconsistent-behavior
- race-conditions
- microservice-communication-overhead
- integration-difficulties
- silent-data-corruption
- unpredictable-system-behavior
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Design API endpoints and message handlers so that processing the same request twice produces the same outcome
- Use idempotency keys (unique request identifiers) to detect and deduplicate repeated operations
- Store the result of each operation so that retries return the cached result instead of re-executing
- Make database operations idempotent by using upserts or conditional updates instead of blind inserts
- Design message consumers to handle redelivery gracefully by checking whether the work has already been done
- Audit existing legacy operations for non-idempotent behavior and prioritize fixing those on critical paths

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables safe retries across unreliable networks, improving system resilience
- Simplifies error recovery by allowing operations to be replayed without side effects
- Reduces the need for distributed transactions or complex compensation logic

**Costs and Risks:**
- Implementing idempotency requires additional state tracking (idempotency keys, result caches)
- Not all operations are naturally idempotent; forcing idempotency on complex workflows adds design complexity
- Idempotency key storage requires cleanup to avoid unbounded growth
- Caching results of operations increases storage requirements

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy payment system occasionally double-charged customers when network timeouts triggered automatic retries. The team added idempotency keys to the payment API: each payment request included a unique key, and the system stored the result of the first successful processing. Subsequent requests with the same key returned the cached result without re-executing the payment. Double-charge incidents dropped from several per week to zero, and the operations team no longer needed to manually reverse duplicate transactions.
