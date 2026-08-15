---
title: Batch Processing
description: Collecting and processing multiple jobs together
category:
- Performance
- Operations
problems:
- slow-application-performance
- high-number-of-database-queries
- high-database-resource-utilization
- growing-task-queues
- gradual-performance-degradation
- excessive-disk-io
- interrupt-overhead
- unoptimized-file-access
- long-running-database-transactions
- long-running-transactions
layout: solution
related_solutions:
- slug: distributed-processing
  similarity: 0.8
- slug: pipelining
  similarity: 0.8
- slug: parallelization
  similarity: 0.75
- slug: streaming
  similarity: 0.75
- slug: transactions
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
---

## Description

Batch processing groups many individual operations — database writes, API calls, file operations — into a single collected unit that is executed together, amortizing fixed per-operation costs like connection setup, transaction overhead, and network round trips across many items instead of paying them once per item. The mechanism trades latency for throughput: individual items wait until a batch fills or a time window elapses, but the aggregate cost of processing the whole set drops substantially compared to handling each one separately. This is a natural fit for legacy systems that were originally built to process one record at a time and have since been pushed far beyond the transaction volumes their per-item design was ever intended for, resulting in the database or downstream system spending most of its capacity on per-call overhead rather than actual work. Introducing batching does not usually require rearchitecting the legacy system's core logic — it requires identifying which individually-processed operations can be safely collected and reordered, and replacing single-row database calls with bulk equivalents the database already supports but the legacy code never used. The tradeoff is that a failure now affects an entire batch rather than a single item, so restartability and partial-retry handling become necessary design concerns that a purely per-item system never had to address.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify operations that process items individually but could be grouped: database inserts, API calls, file writes
- Collect items into batches of appropriate size based on memory constraints and processing time requirements
- Use bulk database operations (batch inserts, bulk updates) instead of individual row operations
- Implement batch windows for non-time-critical operations to process during off-peak hours
- Add monitoring to track batch sizes, processing times, and failure rates
- Design batch processes to be restartable from the point of failure rather than from the beginning
- Consider micro-batching for near-real-time requirements where full batch windows are too slow

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically reduces per-item overhead such as connection setup, transaction management, and network round trips
- Improves throughput by amortizing fixed costs across many items
- Reduces load on downstream systems by smoothing out request patterns
- Enables efficient use of bulk APIs and database operations

**Costs and Risks:**
- Introduces latency for individual items that must wait for the batch to fill
- Batch failures affect multiple items, requiring robust error handling and partial retry logic
- Batch size tuning requires experimentation to balance throughput and latency
- Legacy systems may not support bulk operations, requiring workarounds

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy inventory management system updated stock levels by executing individual UPDATE statements for each sales transaction, processing over 50,000 individual database calls during peak hours. The team introduced batch processing that collected stock updates into groups of 500 and executed them as bulk UPDATE statements every 5 seconds. Database load dropped by over 90%, and the freed resources allowed the system to handle growing transaction volumes without hardware upgrades. The slight delay in stock level updates was acceptable because the business already operated with a tolerance for minor inventory discrepancies.
