---
title: Reactive Programming
description: Development of applications that react to events and process data flows
category:
- Architecture
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/reactive-programming
problems:
- slow-application-performance
- thread-pool-exhaustion
- scaling-inefficiencies
- high-connection-count
- imperative-data-fetching-logic
- cascade-failures
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify I/O-bound components where threads spend most of their time waiting (database calls, HTTP requests, file I/O)
- Introduce reactive libraries (RxJava, Project Reactor, RxJS) incrementally at integration boundaries rather than rewriting entire applications
- Convert blocking API calls to non-blocking reactive streams, starting with the most resource-constrained endpoints
- Use backpressure mechanisms to prevent fast producers from overwhelming slow consumers
- Refactor callback-heavy legacy code into composable reactive pipelines for better readability and error handling
- Train the team on reactive concepts before adoption, as the paradigm shift requires a different mental model

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Handles many more concurrent connections with fewer threads, improving resource efficiency
- Provides built-in backpressure handling for managing data flow rates
- Makes the system more resilient to slow downstream services through non-blocking I/O
- Enables event-driven architectures that scale naturally with load

**Costs and Risks:**
- Steep learning curve for teams accustomed to imperative, sequential programming
- Stack traces and debugging become significantly more complex with reactive pipelines
- Mixing reactive and blocking code can cause subtle performance issues and thread pool starvation
- Testing reactive code requires specialized patterns and tools
- Not all legacy libraries and frameworks support non-blocking operation, limiting adoption scope

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy API gateway processed requests using a thread-per-request model with a pool of 200 threads. As traffic grew, the pool was frequently exhausted during peak hours because most threads were blocked waiting for responses from downstream microservices. The team rewrote the gateway's request routing layer using Project Reactor, replacing blocking HTTP calls with non-blocking WebClient operations. The same server now handled 10 times the concurrent connections with 50 event loop threads, and the cascade failure problem disappeared because slow downstream services no longer consumed gateway threads.
