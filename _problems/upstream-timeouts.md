---
title: Upstream Timeouts
description: Services that consume an API fail because they do not receive a response
  within their configured timeout window.
category:
- Code
- Performance
related_problems:
- slug: service-timeouts
  similarity: 0.85
- slug: external-service-delays
  similarity: 0.65
- slug: high-connection-count
  similarity: 0.6
- slug: high-api-latency
  similarity: 0.6
- slug: increased-error-rates
  similarity: 0.55
- slug: misconfigured-connection-pools
  similarity: 0.55
solutions:
- event-driven-architecture
- circuit-breaker
- timeout-management
layout: problem
---

## Description
Upstream timeouts are a common issue in distributed systems where a service fails to get a response from another service (an "upstream" service) it depends on within a specified time limit. This isn't just a simple error; it's a failure of one part of the system to meet the performance expectations of another. These timeouts can cascade, causing failures in downstream services and ultimately impacting the end-user experience. Understanding and mitigating upstream timeouts is crucial for building resilient and reliable microservices architectures.

## Indicators ⟡
- You are seeing a high number of timeout errors in your logs.
- Your application is slow, and you suspect that it is due to a high number of timeouts.
- You are getting complaints from users about slow performance.
- Your monitoring system is firing alerts for timeout errors.

## Symptoms ▲

- [Cascade Failures](cascade-failures.md)
<br/>  Upstream timeouts propagate through service chains, causing downstream services to also fail or time out.
- [Increased Error Rates](increased-error-rates.md)
<br/>  Timeout errors directly increase the overall error rate of the system as requests fail without receiving responses.
- [User Frustration](user-frustration.md)
<br/>  End users experience slow responses or errors caused by upstream timeouts, leading to dissatisfaction.
- [High Connection Count](high-connection-count.md)
<br/>  Waiting connections accumulate when upstream services are slow, as calling services hold connections open until timeout.
## Causes ▼

- [High API Latency](high-api-latency.md)
<br/>  Slow API response times are the direct cause of upstream timeouts when responses exceed configured timeout windows.
- [External Service Delays](external-service-delays.md)
<br/>  Delays in external services that the API depends on propagate upward, causing upstream callers to time out.
- [Misconfigured Connection Pools](misconfigured-connection-pools.md)
<br/>  Incorrectly configured connection pools can exhaust connections and cause delays that trigger upstream timeouts.
- [Resource Contention](resource-contention.md)
<br/>  Resource contention in the upstream service causes it to process requests slowly, exceeding caller timeout thresholds.
## Detection Methods ○

- **Distributed Tracing:** Use distributed tracing to follow a request across multiple services and pinpoint where the timeout is occurring.
- **Log Analysis:** Centralized logging can be used to correlate timeout errors in one service with slow responses in another.
- **Metrics and Alerting:** Monitor timeout metrics in both the calling service and the API. Set up alerts for unusual spikes.
- **Chaos Engineering:** Intentionally inject delays into services to test how the system behaves and ensure that timeouts are handled gracefully.

## Examples
A `UserService` calls an `AuthService` to authenticate a user. The `AuthService` is experiencing high latency. The `UserService` has a 2-second timeout for the call to the `AuthService`. When the `AuthService` takes longer than 2 seconds to respond, the `UserService` times out and returns an error to the user. In another case, a data processing pipeline consists of several services that call each other in sequence. One of the services in the middle of the pipeline is slow. This causes all subsequent services in the pipeline to time out, even though they are not the root cause of the problem. This is a common problem in microservices architectures, where a single user request can trigger a cascade of calls to multiple services. A timeout in any one of these services can cause the entire request to fail.
