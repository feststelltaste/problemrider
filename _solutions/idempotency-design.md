---
title: Idempotency Design
description: Designing safely retryable operations without unintended side effects
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/reliability/idempotency-design
problems:
- cascade-failures
- silent-data-corruption
- unpredictable-system-behavior
- inadequate-error-handling
- data-migration-integrity-issues
- race-conditions
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify all operations in the legacy system that modify state and assess which can be safely retried
- Assign unique idempotency keys to requests so that duplicate submissions produce the same result
- Store the result of completed operations keyed by their idempotency token to return cached responses on retry
- Convert destructive operations (increment, append) to absolute operations (set to value) where possible
- Add deduplication checks at service entry points to detect and discard duplicate messages
- Design database operations using upsert semantics rather than blind inserts
- Document which API endpoints and message handlers are idempotent and which are not

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables safe retry logic that recovers from transient failures automatically
- Prevents duplicate transactions, charges, or data entries caused by network timeouts
- Simplifies error handling since callers can safely retry without fear of side effects
- Supports reliable message processing in distributed legacy systems

**Costs and Risks:**
- Requires additional storage for idempotency keys and cached results
- Retrofitting idempotency into existing operations requires careful analysis of side effects
- Key expiration policies must balance storage costs against retry window requirements
- Some operations are inherently non-idempotent and need alternative strategies

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A payment processing legacy system occasionally created duplicate charges when network timeouts caused the client to retry submissions. The team added idempotency keys to payment requests and stored completed transaction results in a deduplication table. When a retry arrived with the same key, the system returned the original result without processing the payment again. This eliminated duplicate charge complaints and allowed the team to add aggressive retry logic to the client, improving overall reliability without risking financial errors.
