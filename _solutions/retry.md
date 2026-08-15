---
title: Retry
description: Retrying failed operations to handle transient errors
category:
- Code
- Architecture
problems:
- service-timeouts
- cascade-failures
- inadequate-error-handling
- unpredictable-system-behavior
- external-service-delays
- increased-error-rates
- upstream-timeouts
- service-discovery-failures
layout: solution
related_solutions:
- slug: failover-mechanisms
  similarity: 0.8
- slug: rate-limiting
  similarity: 0.8
- slug: circuit-breaker
  similarity: 0.8
- slug: resilience
  similarity: 0.8
- slug: error-handling
  similarity: 0.75
- slug: chaos-engineering
  similarity: 0.75
---

## Description

Retry is the practice of automatically re-attempting an operation that failed due to a transient condition — a brief network interruption, a momentarily unavailable dependency, a temporary timeout — instead of surfacing the failure to the user or requiring manual intervention immediately. Effective retry logic distinguishes between errors worth retrying, such as connection timeouts, and errors that will never succeed no matter how many times they are repeated, such as authentication failures or validation errors, and it spaces repeated attempts using exponential backoff and jitter to avoid overwhelming an already struggling dependency. In legacy systems, integration points with external services or between internally decomposed components are frequently the least reliable part of the architecture, having been added incrementally over the years without the resilience patterns that would be considered standard in a system designed today; retry is one of the cheapest ways to close that gap, since it can usually be added around an existing call without modifying the underlying operation. It is especially effective at eliminating the class of failures that previously required a human to notice, diagnose, and manually resubmit a failed request, which in legacy operational environments often consumed disproportionate support effort for problems that resolved themselves within seconds. The technique does carry a specific hazard in legacy contexts, however: many older operations were never designed to be idempotent, so blindly retrying them can produce duplicate transactions or side effects, which is why retry must be paired with an explicit check — or a redesign — that makes the underlying operation safe to repeat.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy order management system frequently failed to communicate with an external shipping provider API due to brief network interruptions between data centers. Each failure required manual resubmission by customer service staff. By adding retry logic with exponential backoff (1s, 2s, 4s) and a maximum of three attempts, the system automatically recovered from over 98% of transient failures. The remaining 2% that exhausted retries were automatically queued for manual review, reducing customer service workload by 95% for shipping-related issues.
