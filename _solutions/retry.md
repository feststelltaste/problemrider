---
title: Retry
description: Retrying failed operations to handle transient errors
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/retry
problems:
- service-timeouts
- cascade-failures
- inadequate-error-handling
- unpredictable-system-behavior
- external-service-delays
- increased-error-rates
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify operations in the legacy system that fail due to transient issues (network timeouts, temporary unavailability)
- Implement retry logic with exponential backoff and jitter to avoid thundering herd problems
- Set maximum retry counts to prevent infinite loops when failures are persistent rather than transient
- Classify errors as retryable (timeout, connection refused) versus non-retryable (authentication failure, validation error)
- Combine retries with circuit breakers to stop retrying when a dependency is clearly down
- Ensure operations are idempotent before adding retry logic to prevent duplicate side effects
- Log retry attempts with context to aid in identifying chronic transient failure sources

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Automatically recovers from transient failures without manual intervention
- Improves perceived reliability by masking temporary infrastructure issues
- Simple to implement and adds resilience to legacy integration points
- Reduces the frequency of user-visible errors and support tickets

**Costs and Risks:**
- Retries on non-idempotent operations can cause duplicate data or transactions
- Aggressive retry without backoff can amplify load on already stressed systems
- Retrying persistently failing operations wastes resources and delays error reporting
- Masking transient failures can hide systemic issues that need investigation

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy order management system frequently failed to communicate with an external shipping provider API due to brief network interruptions between data centers. Each failure required manual resubmission by customer service staff. By adding retry logic with exponential backoff (1s, 2s, 4s) and a maximum of three attempts, the system automatically recovered from over 98% of transient failures. The remaining 2% that exhausted retries were automatically queued for manual review, reducing customer service workload by 95% for shipping-related issues.
