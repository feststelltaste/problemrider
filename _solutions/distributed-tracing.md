---
title: Distributed Tracing
description: Tracking requests across microservice boundaries with their performance
  impact
category:
- Operations
- Performance
problems:
- debugging-difficulties
- slow-incident-resolution
- monitoring-gaps
- microservice-communication-overhead
- cascade-failures
- slow-application-performance
layout: solution
related_solutions:
- slug: monitoring
  similarity: 0.75
- slug: continuous-performance-monitoring
  similarity: 0.75
- slug: observability-and-monitoring
  similarity: 0.75
- slug: logging
  similarity: 0.75
- slug: performance-measurements
  similarity: 0.75
- slug: service-level-indicators
  similarity: 0.75
---

## Description

Distributed tracing attaches a unique trace identifier to a request at the moment it enters the system and propagates that identifier through every downstream service call, database query, and message queue interaction the request touches, recording each step as a timed span that can later be reassembled into a single end-to-end picture of what happened and how long each part took. This directly addresses a blind spot created when a legacy monolith is decomposed into microservices: each individual service's logs may show perfectly normal response times in isolation, while the actual user-facing latency or failure is caused by an interaction between several services that no single service's logs can reveal. Legacy modernization efforts that proceed by incrementally carving services out of a monolith are especially prone to this problem, because the resulting system has distributed complexity without yet having distributed observability to match, leaving teams unable to say with confidence which of several services is responsible for a given slowdown. Instrumenting legacy services for tracing typically has to be done incrementally, starting with whichever request paths are most common or most frequently implicated in incidents, since retrofitting tracing across an entire system with mixed technologies at once is rarely practical. Once in place, trace data turns "the system feels slow somewhere" into a precise, visualized answer of exactly which service and which operation on the critical path is responsible, which substantially shortens investigation time compared to correlating separate log files by hand.

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
