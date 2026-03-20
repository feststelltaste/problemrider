---
title: External Service Delays
description: An API depends on other services (third-party or internal) that are slow
  to respond, causing the API itself to be slow.
category:
- Code
- Performance
related_problems:
- slug: high-api-latency
  similarity: 0.75
- slug: slow-application-performance
  similarity: 0.65
- slug: upstream-timeouts
  similarity: 0.65
- slug: network-latency
  similarity: 0.6
- slug: service-timeouts
  similarity: 0.6
- slug: delayed-value-delivery
  similarity: 0.6
solutions:
- caching-strategy
- serialization-optimization
layout: problem
---

## Description
External service delays are a common problem in distributed systems, where services often depend on third-party APIs to fulfill requests. When an external service is slow to respond, it can have a cascading effect, causing delays in downstream services and a poor user experience. External service delays can be caused by a variety of factors, from network issues and a lack of proper caching to a problem with the third-party service itself. A robust monitoring and alerting system is essential for detecting and responding to external service delays in a timely manner.

## Indicators ⟡
- Your application is slow, but your servers are not under heavy load.
- You see a high number of timeout errors in your logs.
- Your application's performance is inconsistent.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [High API Latency](high-api-latency.md)
<br/>  When external services are slow, the API that depends on them inherits their latency, directly causing high API response times.
- [Slow Application Performance](slow-application-performance.md)
<br/>  External service delays propagate to the application layer, making user-facing features feel sluggish and unresponsive.
- [Upstream Timeouts](upstream-timeouts.md)
<br/>  When external services take too long to respond, upstream callers may exceed their configured timeout windows and fail.
- [Service Timeouts](service-timeouts.md)
<br/>  Slow external dependencies cause the dependent service itself to time out when it cannot complete requests within acceptable limits.
- [Cascade Failures](cascade-failures.md)
<br/>  A slow external service can cause thread pool exhaustion and resource starvation in the calling service, triggering cascading failures across the system.
- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Users experience slow or failing operations due to external service delays, leading to frustration and complaints.

## Causes ▼

- [Network Latency](network-latency.md)
<br/>  Network transmission delays between services directly contribute to slow external service response times.
- [Poor Caching Strategy](poor-caching-strategy.md)
<br/>  Without proper caching, every request hits the external service, amplifying the impact of any slowness rather than serving cached responses.
- [Microservice Communication Overhead](microservice-communication-overhead.md)
<br/>  Excessive inter-service communication in a microservice architecture multiplies the chances and impact of external service delays.
- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  Tight coupling to external services without proper circuit breakers or fallback mechanisms means the system cannot gracefully handle slow dependencies.
## Detection Methods ○

- **Distributed Tracing:** Use distributed tracing to follow a request from the API to the external service and identify where the time is being spent.
- **Metrics and Alerting:** Monitor the latency of calls to the external service. Set up alerts for when the latency exceeds a certain threshold.
- **Status Pages:** Check the status page of the external service to see if they are reporting any issues.
- **Service Level Agreements (SLAs):** If there is an SLA in place for the external service, monitor the service's performance against the SLA.

## Examples
An e-commerce application uses a third-party service to process payments. The payment service is slow, which causes the checkout process to be slow. In a microservice architecture, a single slow service can cause a cascading failure that affects the entire application. This is a common problem in modern applications, which are often built by composing together a variety of different services. While this approach has many benefits, it also introduces new challenges, such as the need to deal with external service delays.
