---
title: Concurrency
description: Simultaneous execution of multiple tasks within a process
category:
- Performance
- Code
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/concurrency/
problems:
- race-conditions
- synchronization-problems
- long-running-transactions
- thread-pool-exhaustion
- resource-contention
- memory-leaks
layout: solution
---

## How to Apply ◆

> Legacy systems often evolved in single-threaded or poorly coordinated multi-threaded environments. Introducing proper concurrency control means restructuring how shared resources are accessed, how work is distributed across threads or processes, and how the system behaves under concurrent load.

- Identify all shared mutable state in the application — global variables, caches, session stores, counters, and in-memory collections. Map which threads or processes access each piece of shared state and whether those accesses include writes. This audit is the foundation for all subsequent concurrency improvements.
- Introduce appropriate synchronization primitives for shared state access. Use mutexes or synchronized blocks for simple critical sections, read-write locks where reads vastly outnumber writes, and atomic operations for simple counters and flags. Prefer the narrowest possible lock scope to minimize contention.
- Replace coarse-grained locking with fine-grained or lock-free data structures where contention is high. For example, replace a single lock protecting an entire map with a concurrent hash map that locks individual segments, or use compare-and-swap operations for simple state transitions.
- Adopt immutable data patterns wherever possible. Immutable objects are inherently thread-safe and eliminate entire categories of race conditions. In legacy code, start by making value objects and configuration data immutable, then gradually extend the pattern to domain objects.
- Implement proper transaction scoping for database operations. Break long-running transactions into smaller, bounded units of work. Use optimistic concurrency control (version columns, ETags) instead of pessimistic locking where conflicts are rare, reducing lock hold times and deadlock risk.
- Introduce asynchronous processing for operations that do not require immediate results. Move long-running tasks to background workers or message queues, freeing request-handling threads to serve new requests and preventing thread pool exhaustion.
- Add timeout and circuit-breaker patterns to all blocking operations — database queries, external service calls, lock acquisitions. Without timeouts, a single slow dependency can consume all available threads and cascade into a full system outage.
- Use structured concurrency frameworks or patterns (such as Java's ExecutorService, Python's asyncio, or Go's goroutines with channels) to manage thread lifecycles explicitly rather than spawning ad-hoc threads throughout the codebase.
- Instrument concurrent code paths with metrics for lock wait times, thread pool utilization, queue depths, and contention rates. These metrics provide early warning of concurrency bottlenecks before they manifest as user-visible failures.

## Tradeoffs ⇄

> Concurrency control enables higher throughput and better resource utilization, but it introduces complexity in reasoning about program correctness and can create new categories of bugs if applied incorrectly.

**Benefits:**

- Increases throughput by allowing multiple operations to execute simultaneously, making better use of modern multi-core hardware that legacy systems were often not designed to exploit.
- Reduces response times by overlapping I/O-bound operations such as database queries and external service calls, rather than executing them sequentially.
- Prevents data corruption and inconsistent state caused by unsynchronized concurrent access to shared resources.
- Eliminates thread pool exhaustion by moving blocking work to background threads and keeping request-handling threads available for new requests.
- Reduces database lock contention and deadlock frequency by shortening transaction durations and using optimistic concurrency where appropriate.

**Costs and Risks:**

- Concurrency bugs — race conditions, deadlocks, livelocks — are among the hardest defects to reproduce, diagnose, and fix. Adding concurrency to legacy code without comprehensive testing can introduce subtle, intermittent failures.
- Over-synchronization (excessive locking) can reduce throughput below single-threaded performance, turning a concurrency improvement into a regression.
- Debugging and profiling concurrent code requires specialized tools and expertise that legacy teams may not possess, creating a knowledge gap that must be addressed through training.
- Lock-free and wait-free data structures offer better performance under contention but are significantly harder to implement correctly and verify for correctness.
- Transitioning from synchronous to asynchronous processing changes the programming model substantially, requiring rework of error handling, transaction management, and result propagation patterns.

## Examples

> The following scenarios illustrate how concurrency control addresses problems in legacy systems.

A payroll processing system originally designed to handle 500 employees grew to serve 15,000 employees. The nightly payroll run wrapped the entire batch in a single database transaction, holding table-level locks for over four hours and blocking the HR application from any writes during that window. The team restructured the batch into per-department transactions of 50–200 employees each, added optimistic locking via version columns on employee records, and processed departments concurrently using a thread pool of 8 workers. The total processing time dropped from four hours to 25 minutes, and the HR application remained fully operational throughout the run because no single transaction held locks for more than a few seconds.

A legacy document management system served concurrent users but protected its in-memory document index with a single global lock. Under peak load of 80 users, read operations for document searches blocked behind write operations that updated the index during uploads, causing search response times to spike to 12 seconds. The team replaced the global lock with a read-write lock, allowing unlimited concurrent readers while still serializing writes. Search response times under the same load dropped to 200ms because read operations no longer blocked each other, and only the infrequent write operations required exclusive access.

An insurance claims processing application made synchronous calls to three external validation services within the request thread. When any of these services experienced latency spikes, all request-handling threads became blocked, and the application stopped responding entirely. The team introduced asynchronous calls with a 3-second timeout and a circuit breaker that tripped after 5 consecutive failures. Claims that could not be validated within the timeout were queued for background retry rather than blocking the request thread. Thread pool utilization dropped from a constant 100% during external service slowdowns to under 30%, and the application remained responsive even when downstream services degraded.
