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

- [Unpredictable System Behavior](unpredictable-system-behavior.md)
<br/>  The system appears to hang or behave unpredictably with low CPU usage, making the root cause difficult to diagnose.

## Causes ▼
- [Resource Contention](resource-contention.md)
<br/>  Competition for limited thread pool resources among different operations leads to exhaustion under load.
- [Deadlock Conditions](deadlock-conditions.md)
<br/>  Deadlocked threads permanently consume thread pool resources, gradually depleting the available pool.
- [Unreleased Resources](unreleased-resources.md)
<br/>  Threads that are not properly released after completing or timing out permanently reduce the available thread pool.
- [Lock Contention](lock-contention.md)
<br/>  Threads blocked waiting for contested locks remain occupied, eventually exhausting the available thread pool.
- [Resource Allocation Failures](resource-allocation-failures.md)
<br/>  Threads that are never returned to the pool due to improper resource management eventually exhaust all available threads.
- [Service Timeouts](service-timeouts.md)
<br/>  Threads waiting for timed-out downstream services remain blocked, gradually exhausting the thread pool and preventing new request processing.

## Detection Methods ○

- **Thread Pool Monitoring:** Monitor thread pool utilization, active threads, and queue depths
- **Thread Dump Analysis:** Analyze thread dumps to identify what threads are doing when exhaustion occurs
- **Application Performance Monitoring:** Track response times and throughput to identify thread pool bottlenecks
- **Resource Usage Monitoring:** Monitor CPU, memory, and I/O usage during thread pool exhaustion
- **Load Testing:** Test application under various load conditions to identify thread pool limits
- **Timeout Configuration Analysis:** Review timeout settings for operations that consume thread pool threads

## Examples

A web service processes file uploads by reading the entire file content into memory within the request thread. When users upload very large files, these operations take several minutes to complete, consuming request-handling threads for extended periods. During peak usage, all available request threads become occupied with file upload processing, preventing the server from handling any other HTTP requests including simple health checks. Another example involves an application that makes synchronous calls to external web services without timeout configuration. When the external services become slow or unresponsive, all thread pool threads become blocked waiting for responses that may never come, effectively freezing the entire application until the external services recover or connections timeout at the TCP level.
