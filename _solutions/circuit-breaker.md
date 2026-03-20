---
title: Circuit Breaker
description: Mechanism for error and overload protection in distributed systems
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/circuit-breaker
problems:
- cascade-failures
- service-timeouts
- external-service-delays
- system-outages
- thread-pool-exhaustion
- upstream-timeouts
- single-points-of-failure
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy order processing system made synchronous calls to an inventory service, a payment gateway, and a shipping provider. When the shipping provider experienced an outage, the order service's thread pool filled with blocked requests waiting for the shipping API timeout, eventually making the entire order flow unresponsive. The team added Resilience4j circuit breakers around each external call. When the shipping circuit opened after five consecutive failures, orders were accepted with shipping scheduled for later processing rather than blocking the entire checkout. The circuit breaker's half-open state automatically detected when the shipping provider recovered and resumed normal operations without manual intervention.
