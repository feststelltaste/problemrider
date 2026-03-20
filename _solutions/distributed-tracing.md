---
title: Distributed Tracing
description: Tracking requests across microservice boundaries with their performance impact
category:
- Operations
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/distributed-tracing
problems:
- debugging-difficulties
- slow-incident-resolution
- monitoring-gaps
- microservice-communication-overhead
- cascade-failures
- slow-application-performance
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Instrument services with a tracing library (OpenTelemetry, Jaeger, Zipkin) that propagates trace context across service boundaries
- Inject trace IDs at the system entry point and propagate them through all downstream calls via headers
- Record spans for significant operations: HTTP calls, database queries, message queue interactions, and cache lookups
- Deploy a trace collection and visualization backend to store and query trace data
- Add tracing to legacy services incrementally, starting with the services involved in the most common or problematic request paths
- Use trace data to identify latency bottlenecks and optimize the critical path
- Set sampling rates appropriately to balance observability with storage and performance costs

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides end-to-end visibility into request flow across service boundaries
- Pinpoints which service or operation is responsible for latency in distributed systems
- Enables identification of cascading failure patterns and dependency bottlenecks
- Significantly reduces mean time to resolution for distributed system issues

**Costs and Risks:**
- Instrumentation adds small latency and resource overhead to every traced operation
- Trace storage can grow rapidly and become expensive at high traffic volumes
- Incomplete instrumentation (missing spans in some services) produces misleading traces
- Requires team education to interpret trace data effectively

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A company had partially decomposed a legacy monolith into eight microservices. When users reported intermittent slow responses, the team could not determine which service was responsible because each service's logs showed normal response times in isolation. After deploying OpenTelemetry across all services, traces revealed that a specific request path traversed six services sequentially, and the third service in the chain was making a synchronous database call that occasionally took 5 seconds due to lock contention. The trace visualization made the bottleneck immediately obvious, and the team resolved the issue by optimizing the database query and adding a circuit breaker.
