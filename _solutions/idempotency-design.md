---
title: Idempotency Design
description: Designing safely retryable operations without unintended side effects
category:
- Architecture
- Code
problems:
- cascade-failures
- silent-data-corruption
- unpredictable-system-behavior
- inadequate-error-handling
- data-migration-integrity-issues
- race-conditions
- deadlock-conditions
layout: solution
related_solutions:
- slug: idempotent-operations
  similarity: 0.95
- slug: transactions
  similarity: 0.75
- slug: retry
  similarity: 0.75
- slug: saga-pattern
  similarity: 0.7
- slug: redundancy
  similarity: 0.65
- slug: batch-processing
  similarity: 0.65
---

## Description

Idempotency design is the practice of deliberately engineering an operation so that executing it multiple times has the same effect as executing it once, most commonly by assigning each request a unique idempotency key, storing the outcome of the first execution against that key, and returning the stored result on any subsequent retry rather than repeating the underlying side effect. As a design discipline, it is applied at the point where new or modified operations are built or retrofitted, deciding case by case which operations can be converted to absolute, upsert-style semantics and which are inherently destructive and need an alternative safeguard such as explicit deduplication. This is particularly consequential in legacy systems because their state-changing operations were frequently written as blind inserts or increments long before network retries, message redelivery, or distributed failure modes were treated as a standing design concern, leaving duplicate charges, duplicate records, or double-counted increments as a latent risk whenever a client times out and resubmits. Retrofitting idempotency into such an operation requires a careful audit of its side effects — deciding, for instance, whether a duplicate charge or a duplicate email is the more tolerable failure while the fix is incomplete — since not every legacy operation can be made idempotent cheaply or safely. Once idempotency keys and cached results are in place, callers gain the freedom to retry aggressively without fear of side effects, replacing brittle manual error handling and reversal procedures with a substantially simpler and more automatable recovery model.

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
