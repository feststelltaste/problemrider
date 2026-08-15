---
title: Data Archiving
description: Offloading infrequently needed data to more cost-effective storage media
category:
- Database
- Performance
problems:
- unbounded-data-growth
- gradual-performance-degradation
- slow-database-queries
- high-database-resource-utilization
- database-schema-design-problems
- unbounded-data-structures
- inadequate-test-data-management
- retention-obligations-block-change
layout: solution
related_solutions:
- slug: data-partitioning
  similarity: 0.8
- slug: data-replication
  similarity: 0.8
- slug: redundant-data-storage
  similarity: 0.8
- slug: materialized-views
  similarity: 0.75
- slug: platform-independent-data-storage
  similarity: 0.75
- slug: compression
  similarity: 0.75
---

## Description

Data archiving moves data that is no longer actively needed — typically identified by age or declining access frequency — out of the primary, performance-critical storage tier and into cheaper, slower storage where it remains available but no longer burdens day-to-day operations. Unlike deletion, archiving preserves the data for compliance, audit, or occasional historical lookup, but relocates it so that the active dataset the application queries against stays small and fast. This distinction matters greatly in legacy systems, where retention obligations or simple institutional caution have left years or decades of transactional history sitting in the same tables that power daily operations, causing indexes to bloat, backups to take longer, and even routine lookups to slow down as the database engine works through data nobody is actually using anymore. A well-designed archival process is automated and reversible: it runs on a defined schedule against clear criteria, and it is paired with a restoration path so that archived records can still be produced when an audit or a customer inquiry requires them. Because application queries in legacy systems were often written without any date-bounding assumption, introducing archiving typically also requires updating those queries to explicitly target the active dataset, closing a gap that let unbounded growth accumulate unnoticed in the first place.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy claims processing system for an insurance company had accumulated 12 years of claims data in a single database, totaling over 500 million records. Query performance had degraded to the point where even simple lookups took several seconds. The team implemented a data archiving strategy that moved claims older than three years to a separate archival database on cheaper storage. The active database shrank by 75%, and query performance returned to sub-second levels. For regulatory audits requiring historical data, a dedicated query interface accessed the archive with acceptable response times of a few seconds per query.
