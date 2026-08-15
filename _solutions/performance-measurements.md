---
title: Performance Measurements
description: Continuous measurement and storage of performance metrics in production
category:
- Performance
- Operations
problems:
- gradual-performance-degradation
- monitoring-gaps
- slow-application-performance
- slow-incident-resolution
- quality-blind-spots
- capacity-mismatch
- alignment-and-padding-issues
- atomic-operation-overhead
- data-structure-cache-inefficiency
- dma-coherency-issues
- endianness-conversion-overhead
- false-sharing
- incorrect-index-type
- incorrect-max-connection-pool-size
- index-fragmentation
- inefficient-database-indexing
- interrupt-overhead
- lock-contention
- memory-barrier-inefficiency
- misconfigured-connection-pools
- poor-caching-strategy
- queries-that-prevent-index-usage
- unoptimized-file-access
- unused-indexes
- algorithmic-complexity-problems
- garbage-collection-pressure
- high-resource-utilization-on-client
- inefficient-code
- insufficient-worker-capacity
- long-running-database-transactions
- memory-fragmentation
- memory-swapping
- n-plus-one-query-problem
- virtual-memory-thrashing
- work-queue-buildup
- high-number-of-database-queries
- imperative-data-fetching-logic
- inefficient-frontend-code
- long-running-transactions
- rate-limiting-issues
- serialization-deserialization-bottlenecks
- task-queues-backing-up
layout: solution
related_solutions:
- slug: continuous-performance-monitoring
  similarity: 0.9
- slug: transparent-performance-metrics
  similarity: 0.85
- slug: monitoring
  similarity: 0.8
- slug: performance-budgets
  similarity: 0.8
- slug: service-level-indicators
  similarity: 0.75
- slug: monitoring-system-utilization
  similarity: 0.75
---

## Description

Performance measurement instruments a system's code paths to continuously collect and store timing and resource-usage metrics in production, rather than relying on isolated benchmarks or user complaints to reveal how the system is actually behaving. Legacy systems accumulate performance regressions silently over years of incremental changes, and without a historical record of percentile response times, resource utilization, and their correlation to specific deployments, degradation only becomes visible once it has already reached crisis levels. Capturing full distributions — p50, p95, p99 — rather than averages exposes tail-latency problems that averages hide entirely, and correlating that data with deployment events turns "the system got slower at some point" into "this specific change caused it," which is the difference between an investigation that takes minutes and one that takes weeks.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Instrument key code paths with timing metrics, starting with the most user-visible operations
- Deploy a metrics collection system (e.g., Prometheus, Datadog, StatsD) that stores time-series performance data
- Create dashboards that visualize performance trends over time, making degradation immediately visible
- Set up alerts for performance threshold violations so issues are detected before users report them
- Capture percentile distributions (p50, p95, p99) rather than just averages to understand the full performance picture
- Correlate performance metrics with deployment events to identify regressions introduced by specific changes
- Retain historical data long enough to observe seasonal patterns and long-term trends

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Makes gradual performance degradation visible before it reaches crisis levels
- Provides evidence-based data for prioritizing performance improvements
- Reduces mean time to resolution for performance incidents through faster root cause identification
- Creates accountability by linking performance changes to specific deployments

**Costs and Risks:**
- Instrumentation adds a small overhead to request processing
- Legacy systems without standardized instrumentation points require significant initial effort
- Metrics infrastructure requires its own maintenance, storage, and monitoring
- Too many metrics can create noise and alert fatigue
- Teams may over-index on measurable metrics while missing user-perceived issues

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A banking application experienced intermittent slowdowns that were reported by customers but could never be reproduced in testing. The team added distributed tracing and response time metrics to all API endpoints, storing the data in Prometheus with Grafana dashboards. Within two weeks, the dashboards revealed that the p99 response time for account balance queries spiked to 15 seconds every day between 2 and 3 PM, correlating with an automated reconciliation batch job that competed for database connections. This insight, invisible without continuous measurement, led to rescheduling the batch job to off-peak hours and implementing connection pool isolation.
