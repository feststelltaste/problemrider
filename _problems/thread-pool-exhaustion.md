---
title: Thread Pool Exhaustion
description: All available threads in the thread pool are consumed by long-running
  or blocked operations, preventing new tasks from being processed.
category:
- Code
- Performance
related_problems:
- slug: resource-allocation-failures
  similarity: 0.6
- slug: deadlock-conditions
  similarity: 0.6
- slug: resource-contention
  similarity: 0.55
- slug: unreleased-resources
  similarity: 0.55
- slug: high-client-side-resource-consumption
  similarity: 0.55
- slug: insufficient-worker-capacity
  similarity: 0.5
solutions:
- backpressure
- capacity-planning
- concurrency-control
- elastic-scaling
- resource-pooling
- asynchronous-operations
- asynchronous-processing
- bulkhead
- circuit-breaker
- reactive-programming
- timeout-management
layout: problem
---

## Description

Thread pool exhaustion occurs when all available threads in an application's thread pool are consumed by long-running, blocked, or stuck operations, leaving no threads available to process new incoming requests or tasks. This creates a situation where the application appears to hang or become unresponsive, even though the underlying system has available CPU and memory resources. Thread pool exhaustion is common in server applications and can cause complete service outages.

## Indicators ⟡

- Application stops responding to new requests while appearing to run normally
- Thread pool monitoring shows all threads in use with none available for new tasks
- New operations queue up indefinitely without being processed
- CPU usage may be low despite the application appearing busy
- Response times increase dramatically or operations timeout

## Symptoms ▲

- [Cascade Failures](cascade-failures.md)
<br/>  When one service exhausts its thread pool, dependent services also fail as their requests time out, causing cascading failures.
- [System Outages](system-outages.md)
<br/>  Complete thread pool exhaustion effectively causes service outages as the application becomes entirely unresponsive.
- [Unpredictable System Behavior](unpredictable-system-behavior.md)
<br/>  The system appears to hang or behave unpredictably with low CPU usage, making the root cause difficult to diagnose.
- [Slow Application Performance](slow-application-performance.md)
<br/>  Before complete exhaustion, partial thread pool depletion causes slow application performance as fewer threads are av....
## Causes ▼

- [Resource Contention](resource-contention.md)
<br/>  Competition for limited thread pool resources among different operations leads to exhaustion under load.
- [Deadlock Conditions](deadlock-conditions.md)
<br/>  Deadlocked threads permanently consume thread pool resources, gradually depleting the available pool.
- [Unreleased Resources](unreleased-resources.md)
<br/>  Threads that are not properly released after completing or timing out permanently reduce the available thread pool.
- [Service Timeouts](service-timeouts.md)
<br/>  Without proper timeout settings, threads block indefinitely waiting for slow or unresponsive external services.
## Detection Methods ○

- **Thread Pool Monitoring:** Monitor thread pool utilization, active threads, and queue depths
- **Thread Dump Analysis:** Analyze thread dumps to identify what threads are doing when exhaustion occurs
- **Application Performance Monitoring:** Track response times and throughput to identify thread pool bottlenecks
- **Resource Usage Monitoring:** Monitor CPU, memory, and I/O usage during thread pool exhaustion
- **Load Testing:** Test application under various load conditions to identify thread pool limits
- **Timeout Configuration Analysis:** Review timeout settings for operations that consume thread pool threads

## Examples

A web service processes file uploads by reading the entire file content into memory within the request thread. When users upload very large files, these operations take several minutes to complete, consuming request-handling threads for extended periods. During peak usage, all available request threads become occupied with file upload processing, preventing the server from handling any other HTTP requests including simple health checks. Another example involves an application that makes synchronous calls to external web services without timeout configuration. When the external services become slow or unresponsive, all thread pool threads become blocked waiting for responses that may never come, effectively freezing the entire application until the external services recover or connections timeout at the TCP level.
