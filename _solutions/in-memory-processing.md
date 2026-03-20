---
title: In-Memory Processing
description: Keeping all data in main memory instead of on slow storage media
category:
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/in-memory-processing
problems:
- slow-application-performance
- slow-database-queries
- excessive-disk-io
- gradual-performance-degradation
- high-database-resource-utilization
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify workloads where disk I/O is the primary bottleneck: frequently accessed reference data, session stores, real-time analytics
- Move hot data into in-memory data structures or in-memory databases (Redis, Apache Ignite, Hazelcast)
- For relational workloads, consider in-memory table features offered by databases like SAP HANA, SQL Server, or PostgreSQL extensions
- Design data structures optimized for memory access patterns rather than disk-based layouts
- Implement persistence strategies (snapshots, write-ahead logs) to protect against data loss during failures
- Size memory allocation based on the working set plus growth projections, not just current data volume
- Monitor memory utilization and garbage collection overhead to avoid swapping and OOM situations

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates disk I/O latency, providing orders-of-magnitude faster data access
- Enables complex computations and queries that are impractical with disk-based storage
- Provides predictable, low-variance response times not subject to disk seek patterns
- Enables real-time analytics and processing that batch-oriented disk systems cannot support

**Costs and Risks:**
- Memory is significantly more expensive than disk storage, limiting the data volume that can be kept in-memory
- Data loss risk during failures unless persistence mechanisms are properly configured
- Garbage collection pauses in managed-memory runtimes can cause latency spikes
- Memory capacity limits require careful data lifecycle management

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy trading platform performed real-time risk calculations by querying a disk-based relational database for position data. During market opening, when thousands of positions changed simultaneously, the disk I/O became saturated and risk calculations lagged behind market movements by several minutes. The team migrated the position data to an in-memory data grid, loading the current day's positions into memory at startup and updating them via market data events. Risk calculations that previously took seconds per position now completed in microseconds, allowing the system to maintain real-time risk visibility even during the most volatile market conditions.
