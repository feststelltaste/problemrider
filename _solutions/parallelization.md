---
title: Parallelization
description: Simultaneous execution of multiple calculations or tasks
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/parallelization
problems:
- slow-application-performance
- bottleneck-formation
- scaling-inefficiencies
- long-build-and-test-times
- slow-database-queries
- insufficient-worker-capacity
- growing-task-queues
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Profile the application to identify CPU-bound or I/O-bound bottlenecks that could benefit from concurrent execution
- Decompose independent tasks (e.g., batch processing, report generation, data imports) into parallelizable units
- Use thread pools, worker processes, or async I/O frameworks appropriate to the language and runtime
- Ensure shared state is properly synchronized or eliminated to prevent race conditions and deadlocks
- Start with embarrassingly parallel workloads (e.g., processing independent records) before tackling interdependent tasks
- Parallelize build and test pipelines to reduce feedback loop times during development
- Monitor thread and process utilization to right-size the degree of parallelism

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Can provide near-linear speedups for workloads that decompose cleanly into independent units
- Makes better use of modern multi-core hardware that legacy single-threaded code underutilizes
- Reduces end-to-end processing time for batch jobs and data pipelines

**Costs and Risks:**
- Introduces concurrency bugs (race conditions, deadlocks) that are difficult to reproduce and debug
- Legacy code with global state or shared mutable data requires significant refactoring to parallelize safely
- Parallelism adds complexity to error handling, retry logic, and result aggregation
- Can increase resource contention (memory, database connections, I/O) if not properly managed
- Diminishing returns beyond a certain degree of parallelism due to synchronization overhead

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A financial institution's end-of-day batch processing took over six hours running sequentially through account reconciliation, interest calculation, and report generation. Analysis showed these three processes operated on independent data partitions. The team parallelized each process across account ranges using a worker pool, and also ran the three processes concurrently where data dependencies allowed. Total batch processing time dropped to 90 minutes, well within the overnight maintenance window, without any changes to the business logic itself.
