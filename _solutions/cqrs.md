---
title: CQRS
description: Separating read and write models into independently optimized and scaled paths
category:
- Architecture
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/cqrs
problems:
- slow-database-queries
- scaling-inefficiencies
- database-query-performance-issues
- high-database-resource-utilization
- monolithic-architecture-constraints
- slow-response-times-for-lists
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify areas where read and write workloads have fundamentally different performance characteristics or scaling needs
- Separate the read model from the write model, starting with the most performance-critical queries
- Create denormalized read projections optimized for specific query patterns
- Use domain events to keep read models synchronized with the write model
- Start with a simple synchronous projection before introducing eventual consistency if needed
- Apply CQRS selectively to the parts of the system that benefit most, not as a system-wide pattern
- Implement compensating mechanisms for handling eventual consistency in the user experience

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Read and write models can be optimized independently for their specific access patterns
- Enables independent scaling of read-heavy and write-heavy components
- Read models can be denormalized and pre-computed for fast query responses
- Simplifies complex queries by designing read models specifically for each query need

**Costs and Risks:**
- Introduces eventual consistency between read and write models, which complicates the user experience
- Increases system complexity with separate models, projections, and synchronization logic
- Requires careful handling of projection failures and rebuild scenarios
- May be over-engineering for systems where read and write patterns are similar

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy accounting system used the same normalized relational schema for both transaction recording and report generation. As transaction volume grew, complex reporting queries began competing with transaction processing for database resources, causing both to slow down. The team introduced CQRS by creating a separate read database with denormalized views optimized for the most common reports. Domain events from the transactional database triggered updates to the read projections. Report generation times dropped from minutes to seconds, and transaction processing throughput doubled because the write database no longer handled expensive analytical queries.
