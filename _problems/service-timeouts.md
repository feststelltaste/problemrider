---
title: Service Timeouts
description: Services fail to complete requests within an acceptable time limit, causing
  errors, cascading failures, and system instability.
category:
- Code
- Performance
related_problems:
- slug: upstream-timeouts
  similarity: 0.85
- slug: high-connection-count
  similarity: 0.6
- slug: external-service-delays
  similarity: 0.6
- slug: high-api-latency
  similarity: 0.6
- slug: network-latency
  similarity: 0.6
- slug: increased-error-rates
  similarity: 0.55
layout: problem
---

## Description
Service timeouts occur when a service fails to respond to a request within a specified time period. This is a common problem in distributed systems, where services often depend on each other to fulfill requests. Timeouts can be caused by a variety of factors, including network issues, high latency in a downstream service, or a service that is simply overloaded. Properly handling timeouts is crucial for building resilient and reliable systems.

## Indicators ⟡
- You are seeing a high number of timeout errors in your logs.
- Your application is slow, and you suspect that it is due to a high number of timeouts.
- You are getting complaints from users about slow performance.
- Your monitoring system is firing alerts for timeout errors.

## Symptoms ▲

- [Cascade Failures](cascade-failures.md)
<br/>  When one service times out, callers may also time out waiting for it, creating a chain reaction of failures across the system.
- [Increased Error Rates](increased-error-rates.md)
<br/>  Timeout errors contribute directly to elevated error rates across the system as requests fail to complete.
- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Users experience slow responses or error messages when services time out, leading to frustration and dissatisfaction.
- [Slow Application Performance](slow-application-performance.md)
<br/>  Requests waiting for timed-out services contribute to overall application slowness as threads and connections are held open.
- [Thread Pool Exhaustion](thread-pool-exhaustion.md)
<br/>  Threads waiting for timed-out downstream services remain blocked, gradually exhausting the thread pool and preventing new request processing.

## Causes ▼
- [Network Latency](network-latency.md)
<br/>  High network latency between services increases round-trip times, causing requests to exceed timeout thresholds.
- [External Service Delays](external-service-delays.md)
<br/>  Slow responses from external or third-party services propagate through the system as upstream services wait and eventually time out.
- [Database Query Performance Issues](database-query-performance-issues.md)
<br/>  Slow database queries in downstream services cause request processing to exceed timeout limits.
- [Resource Contention](resource-contention.md)
<br/>  Overloaded services competing for limited CPU, memory, or I/O resources process requests too slowly, causing timeouts.
- [Excessive Disk I/O](excessive-disk-io.md)
<br/>  Operations waiting for disk reads or writes may exceed timeout thresholds, causing service failures.
- [Garbage Collection Pressure](garbage-collection-pressure.md)
<br/>  Long GC pauses can cause requests to exceed timeout thresholds, resulting in failed service calls.
- [GraphQL Complexity Issues](graphql-complexity-issues.md)
<br/>  Complex queries that take too long to resolve exceed service timeout limits, causing request failures.
- [Growing Task Queues](growing-task-queues.md)
<br/>  Tasks waiting too long in queues exceed timeout thresholds before they can be processed.
- [High API Latency](high-api-latency.md)
<br/>  Downstream services that call the slow API exceed their timeout thresholds, causing cascading failures.
- [High Connection Count](high-connection-count.md)
<br/>  When the connection limit is reached, new connection attempts are rejected or queued, causing service timeouts.
- [Incorrect Max Connection Pool Size](incorrect-max-connection-pool-size.md)
<br/>  When the pool is too small, requests wait for available connections and eventually time out.
- [Increased Error Rates](increased-error-rates.md)
<br/>  Elevated error rates often accompany cascading failures that cause service timeouts across dependent systems.
- [Insufficient Worker Capacity](insufficient-worker-capacity.md)
<br/>  Long queue wait times cause dependent services to time out waiting for task completion.
- [Load Balancing Problems](load-balancing-problems.md)
<br/>  Overloaded instances from poor load distribution fail to respond within timeout thresholds, causing service failures.
- [Long-Running Database Transactions](long-running-database-transactions.md)
<br/>  Application requests waiting for database operations blocked by long-running transactions exceed timeout thresholds.
- [Long-Running Transactions](long-running-transactions.md)
<br/>  Operations blocked by long-running transaction locks can exceed application timeout thresholds, causing failures.
- [Memory Swapping](memory-swapping.md)
<br/>  The dramatic slowdown from swapping causes services to exceed their timeout thresholds, leading to cascading failures.
- [Microservice Communication Overhead](microservice-communication-overhead.md)
<br/>  Network latency accumulation across multiple service calls causes requests to exceed timeout thresholds.
- [Misconfigured Connection Pools](misconfigured-connection-pools.md)
<br/>  When connection pools are exhausted, new requests wait for available connections and eventually time out.
- [Service Discovery Failures](service-discovery-failures.md)
<br/>  Failed service discovery causes services to attempt connections to stale or invalid endpoints, resulting in connection timeouts.
- [Task Queues Backing Up](task-queues-backing-up.md)
<br/>  Operations waiting for queue processing may exceed timeout thresholds as queue depth grows.
- [Unreleased Resources](unreleased-resources.md)
<br/>  Resource exhaustion from unreleased connections and handles causes services to become unresponsive and time out.
- [Virtual Memory Thrashing](virtual-memory-thrashing.md)
<br/>  Applications become so slow during thrashing that they fail to respond within timeout windows.

## Detection Methods ○

- **Distributed Tracing:** Use tools like Jaeger or Zipkin to trace requests across service boundaries and identify which service call is timing out.
- **Log Analysis:** Aggregate and search logs from all services to find timeout error messages and correlate them with other events.
- **Monitoring and Alerting:** Set up alerts on timeout error rates (both client-side and server-side) to detect problems proactively.
- **Chaos Engineering:** Intentionally inject delays or failures into the system to test how it behaves and ensure that timeout and retry mechanisms work as expected.

## Examples
In a microservices-based ordering system, the `Order` service calls the `Payment` service. The `Payment` service is slow, so the `Order` service times out. The user is shown a generic error, but the payment may have actually succeeded, leading to a confusing user experience and inconsistent data. In another case, a web server has a default timeout of 30 seconds. A data-intensive reporting endpoint can sometimes take longer than 30 seconds to generate a report. Users who try to access this report frequently get a 504 Gateway Timeout error. This problem is especially common in complex, distributed systems where a single user request can involve communication between dozens of services. Without careful design of timeouts and retry logic, these systems can be very fragile.
