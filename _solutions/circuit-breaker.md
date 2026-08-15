---
title: Circuit Breaker
description: Mechanism for error and overload protection in distributed systems
category:
- Architecture
problems:
- cascade-failures
- service-timeouts
- external-service-delays
- system-outages
- thread-pool-exhaustion
- upstream-timeouts
- single-points-of-failure
- service-discovery-failures
layout: solution
related_solutions:
- slug: retry
  similarity: 0.8
- slug: failover-mechanisms
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.75
- slug: error-handling
  similarity: 0.75
- slug: isolation-of-faulty-components
  similarity: 0.7
- slug: resilience
  similarity: 0.7
---

## Description

A circuit breaker is a protective wrapper placed around a call to an external service or dependency that monitors for repeated failures and, once a failure threshold is crossed, "opens" to stop further calls from being attempted at all, failing fast with a fallback response instead of continuing to hit a service that is known to be unhealthy. After a configured interval it moves to a "half-open" state that allows a small number of probe requests through to test whether the dependency has recovered, closing again if they succeed. This directly targets a common legacy system failure pattern in which synchronous calls to a struggling downstream service accumulate in a thread pool or connection pool as each caller blocks waiting for a timeout, eventually exhausting that resource and causing an outage in a component that itself has nothing wrong with it. Because many legacy systems were built with tightly coupled, synchronous integration points and no isolation between them, a single slow or failing dependency can otherwise cascade into a system-wide failure far larger than the original problem. By failing fast and substituting a defined fallback — cached data, a degraded response, or a clear error — the circuit breaker converts an unbounded, resource-consuming failure into a bounded, predictable one. Its effectiveness depends on designing sensible fallback behavior for each protected call and on tuning thresholds against the dependency's actual behavior, since a poorly configured breaker can either fail to trip in time or reject legitimate traffic during ordinary transient blips.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify all external service calls and inter-service communication points that could block or fail
- Wrap critical external calls with a circuit breaker library (e.g., Resilience4j, Polly, Hystrix)
- Configure failure thresholds that trigger the circuit to open, preventing further calls to the failing service
- Define fallback behavior for each circuit breaker: cached data, degraded functionality, or a user-friendly error message
- Set appropriate timeout windows for half-open states that allow periodic probing of the recovered service
- Add monitoring dashboards that show circuit breaker states and trip counts for operational visibility
- Tune circuit breaker parameters based on observed service behavior and SLA requirements

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Prevents cascading failures by stopping calls to failing downstream services
- Allows the system to degrade gracefully rather than failing completely
- Gives failing services time to recover without being overwhelmed by retry storms
- Improves system responsiveness by failing fast instead of waiting for timeouts

**Costs and Risks:**
- Fallback behavior must be carefully designed to avoid data inconsistencies
- Open circuits may reject legitimate requests during transient failures
- Adds complexity to the codebase and requires careful configuration tuning
- Circuit breaker state can mask underlying issues if monitoring is insufficient
- Half-open probing logic must be tested to ensure proper recovery detection

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy order processing system made synchronous calls to an inventory service, a payment gateway, and a shipping provider. When the shipping provider experienced an outage, the order service's thread pool filled with blocked requests waiting for the shipping API timeout, eventually making the entire order flow unresponsive. The team added Resilience4j circuit breakers around each external call. When the shipping circuit opened after five consecutive failures, orders were accepted with shipping scheduled for later processing rather than blocking the entire checkout. The circuit breaker's half-open state automatically detected when the shipping provider recovered and resumed normal operations without manual intervention.
