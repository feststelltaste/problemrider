---
title: Lock Contention
description: Multiple threads compete for the same locks, causing threads to block
  and reducing parallel execution efficiency.
category:
- Code
- Performance
related_problems:
- slug: race-conditions
  similarity: 0.65
- slug: resource-contention
  similarity: 0.65
- slug: deadlock-conditions
  similarity: 0.6
- slug: atomic-operation-overhead
  similarity: 0.55
- slug: false-sharing
  similarity: 0.55
- slug: memory-barrier-inefficiency
  similarity: 0.55
layout: problem
---

## Description

Lock contention occurs when multiple threads frequently compete for the same synchronization primitives (mutexes, locks, semaphores), causing threads to block while waiting for locks to become available. This reduces the effectiveness of parallel execution as threads spend time waiting rather than doing useful work, and can lead to performance degradation that's worse than single-threaded execution in severe cases.

## Indicators ⟡

- Multi-threaded applications perform worse than single-threaded equivalents
- Thread profiling shows significant time spent waiting for locks
- Lock acquisition times increase under higher thread counts
- System monitoring shows threads in blocked or waiting states
- CPU utilization is low despite high thread activity

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  Lock contention causes threads to block instead of doing useful work, directly degrading application response times and throughput.
- [Thread Pool Exhaustion](thread-pool-exhaustion.md)
<br/>  Threads blocked waiting for contested locks remain occupied, eventually exhausting the available thread pool.
- [Deadlock Conditions](deadlock-conditions.md)
<br/>  High lock contention increases the probability of circular lock dependencies, leading to deadlocks.
- [Scaling Inefficiencies](scaling-inefficiencies.md)
<br/>  Adding more threads or cores provides diminishing or negative returns when they all contend for the same locks.
- [Resource Contention](resource-contention.md)
<br/>  Lock contention is a direct form of resource contention where threads compete for synchronization primitives rather than doing productive work.
## Causes ▼

- [God Object Anti-Pattern](god-object-anti-pattern.md)
<br/>  A god object that centralizes state forces all threads to synchronize through a single lock protecting that shared state.
- [High Coupling and Low Cohesion](high-coupling-low-cohesion.md)
<br/>  Tightly coupled components that share mutable state require coarse-grained locking, increasing contention.
- [Global State and Side Effects](global-state-and-side-effects.md)
<br/>  Global mutable state requires locks for thread safety, and widely accessed globals become natural contention hotspots.
## Detection Methods ○

- **Lock Profiling:** Use profiling tools that can identify lock contention and wait times
- **Thread State Analysis:** Monitor thread states to identify blocking patterns
- **Performance Scaling Tests:** Test performance with varying thread counts to identify contention
- **Lock Instrumentation:** Add instrumentation to measure lock hold times and wait times
- **Concurrency Profilers:** Use specialized tools designed to detect synchronization bottlenecks
- **CPU Utilization Monitoring:** Analyze CPU usage patterns during high-contention scenarios

## Examples

A web server uses a single global lock to protect access to a shared cache that stores user session data. Under high load, hundreds of request threads compete for this lock, causing most threads to block while waiting for cache access. The result is that a 16-core server performs worse than a single-threaded version because threads spend more time waiting for the lock than processing requests. Another example involves a parallel data processing system where multiple worker threads process items from a shared queue protected by a single mutex. As the number of worker threads increases, performance decreases because threads spend most of their time waiting to access the queue rather than processing data items.
