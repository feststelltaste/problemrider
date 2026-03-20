---
title: Data Archiving
description: Offloading infrequently needed data to more cost-effective storage media
category:
- Database
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/data-archiving
problems:
- unbounded-data-growth
- gradual-performance-degradation
- slow-database-queries
- high-database-resource-utilization
- database-schema-design-problems
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Analyze data access patterns to identify data that is rarely queried after a certain age
- Define archival policies based on business requirements: regulatory retention periods, audit needs, and access frequency
- Implement automated archival processes that move data from hot storage to cold storage on a schedule
- Ensure archived data remains accessible for compliance and ad-hoc queries, even if access times are slower
- Test the archival and restoration processes regularly to verify that archived data can be recovered when needed
- Update application queries to filter by date ranges so they naturally operate on the active dataset
- Coordinate with stakeholders to define what constitutes "active" versus "archival" data for each domain

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces the active dataset size, improving query performance and backup times
- Lowers storage costs by moving infrequently accessed data to cheaper media
- Simplifies database maintenance tasks like index rebuilds and schema migrations
- Improves application performance by keeping working sets manageable

**Costs and Risks:**
- Archived data is slower to access, which may frustrate users needing historical information
- Archival processes add operational complexity and require monitoring
- Improper archival can violate regulatory retention requirements
- Application logic may need updates to query both active and archived data transparently

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy claims processing system for an insurance company had accumulated 12 years of claims data in a single database, totaling over 500 million records. Query performance had degraded to the point where even simple lookups took several seconds. The team implemented a data archiving strategy that moved claims older than three years to a separate archival database on cheaper storage. The active database shrank by 75%, and query performance returned to sub-second levels. For regulatory audits requiring historical data, a dedicated query interface accessed the archive with acceptable response times of a few seconds per query.
