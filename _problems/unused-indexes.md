---
title: Unused Indexes
description: The database has indexes that are never used by any queries, which still
  consume storage space and add overhead to write operations.
category:
- Code
- Performance
related_problems:
- slug: inefficient-database-indexing
  similarity: 0.7
- slug: index-fragmentation
  similarity: 0.7
- slug: queries-that-prevent-index-usage
  similarity: 0.7
- slug: incorrect-index-type
  similarity: 0.7
- slug: database-query-performance-issues
  similarity: 0.65
- slug: high-number-of-database-queries
  similarity: 0.65
layout: problem
---

## Description
Unused indexes are database indexes that are not being utilized by any queries. While indexes are crucial for speeding up data retrieval, unused ones still consume disk space and, more importantly, add overhead to write operations (INSERT, UPDATE, DELETE) because the database must update the index every time the underlying data changes. Regularly identifying and removing unused indexes is a key part of database maintenance and performance tuning, as it reclaims storage and reduces unnecessary processing.

## Indicators ⟡
- The database is using a lot of disk space, even though the data set is small.
- Write operations are slow, even though the database is not under heavy load.
- Backups and restores are taking a long time.
- The database is using a lot of memory, even when it is not under heavy load.

## Symptoms ▲

- [Resource Waste](resource-waste.md)
<br/>  Unused indexes consume disk space, memory, and CPU cycles during write operations without providing any query benefit.
- [Slow Database Queries](slow-database-queries.md)
<br/>  The overhead of maintaining unused indexes slows down overall database operations, especially write-heavy workloads.
## Causes ▼

- [Database Schema Design Problems](database-schema-design-problems.md)
<br/>  Without regular database maintenance and auditing, unused indexes accumulate over time as queries change but indexes remain.
- [Monitoring Gaps](monitoring-gaps.md)
<br/>  Without monitoring of index usage statistics, teams have no visibility into which indexes are actually being used.
- [Poor Documentation](poor-documentation.md)
<br/>  Poor documentation of why indexes were created makes it difficult to determine if they are still needed.
## Detection Methods ○

- **Database Monitoring Tools:** Most modern database systems provide statistics on index usage (e.g., `pg_stat_user_indexes` in PostgreSQL, `sys.dm_db_index_usage_stats` in SQL Server, `information_schema.statistics` in MySQL combined with query logs).
- **Query Execution Plan Analysis:** Regularly analyze `EXPLAIN` plans for common queries to ensure they are using the most efficient indexes.
- **Automated Index Advisors:** Some database management systems or third-party tools offer automated index recommendations and usage analysis.
- **Periodic Database Audits:** Schedule regular reviews of database schema and index usage.

## Examples
A legacy e-commerce application has an `orders` table with an index on `customer_id` that was created years ago for a specific reporting query. That report has since been deprecated, but the index remains, adding overhead to every new order placed without providing any query performance benefit. In another case, a developer creates an index on `(column_A, column_B)` but later creates another index on `(column_A)`. The database's query optimizer might prefer the smaller `(column_A)` index for queries only involving `column_A`, rendering the composite index partially unused for those queries. Unused indexes are a form of technical debt in database management. While they don't directly cause application errors, they silently degrade write performance, waste resources, and complicate database maintenance, especially in large-scale systems.
