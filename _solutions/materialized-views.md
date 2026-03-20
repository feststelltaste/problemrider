---
title: Materialized Views
description: Optimize database query performance by storing query results
category:
- Database
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/materialized-views
problems:
- slow-database-queries
- database-query-performance-issues
- high-number-of-database-queries
- high-database-resource-utilization
- slow-response-times-for-lists
- gradual-performance-degradation
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify expensive, frequently executed queries that aggregate or join large tables and produce results that change infrequently
- Create materialized views that precompute and store the results of these queries
- Establish a refresh strategy (periodic, on-demand, or incremental) that balances data freshness against resource cost
- Redirect application queries to the materialized views instead of the base tables
- Add indexes on the materialized views to further accelerate downstream queries
- Monitor refresh times and storage consumption to ensure the materialized views remain sustainable as data grows
- Document the staleness tolerance for each view so the team understands the freshness guarantees

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically reduces query execution time for complex aggregations and joins
- Offloads work from the primary database during peak hours when refreshes are scheduled off-peak
- Can be introduced without modifying legacy application code if queries are redirected at the database layer
- Reduces contention on heavily queried tables

**Costs and Risks:**
- Materialized views consume additional storage and require maintenance of the refresh schedule
- Stale data between refreshes can cause incorrect results if the staleness tolerance is not well understood
- Refresh operations themselves can be resource-intensive and must be scheduled carefully
- Adding materialized views increases the surface area of schema changes during migrations

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An insurance company's legacy reporting dashboard joined five large tables to compute claim summaries, taking over 30 seconds per query during business hours. The team created a materialized view that precomputed the summary and refreshed it every 15 minutes. Dashboard response times dropped to under one second, and the database CPU utilization during peak hours fell by 40 percent. The 15-minute staleness was acceptable for the reporting use case, and the team documented this constraint so that real-time data needs would be routed to a different query path.
