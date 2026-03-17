---
title: Queries That Prevent Index Usage
description: The way a query is written can prevent the database from using an available
  index, forcing slower full-table scans or less efficient index scans.
category:
- Performance
related_problems:
- slug: inefficient-database-indexing
  similarity: 0.75
- slug: incorrect-index-type
  similarity: 0.7
- slug: unused-indexes
  similarity: 0.7
- slug: database-query-performance-issues
  similarity: 0.65
- slug: slow-database-queries
  similarity: 0.6
- slug: index-fragmentation
  similarity: 0.55
layout: problem
---

## Description
Even when appropriate indexes exist, certain query patterns can prevent the database from using them effectively, leading to slow performance. This can happen when functions are applied to indexed columns, when data types don't match, or when the query optimizer is otherwise unable to see that an index could satisfy the query. Writing queries that are "index-friendly" is a crucial skill for developers working with databases, as it can have a dramatic impact on application performance.

## Indicators ⟡
- Queries are slow, even though they are using an index.
- The database is not using an index that you expect it to use.
- The database is using a full table scan, even though an index is available.
- The database is using a less efficient index than you expect it to use.

## Symptoms ▲


- [Slow Database Queries](slow-database-queries.md)
<br/>  Queries that bypass indexes force full table scans, directly causing slow query execution times.
- [Database Query Performance Issues](database-query-performance-issues.md)
<br/>  Non-index-friendly query patterns create performance bottlenecks in database operations.
- [Gradual Performance Degradation](gradual-performance-degradation.md)
<br/>  As tables grow, queries that cannot use indexes degrade progressively because full scans take longer.

## Causes ▼
- [Insufficient Code Review](insufficient-code-review.md)
<br/>  Code reviews that don't evaluate query performance miss patterns that prevent index usage.
- [Inefficient Database Indexing](inefficient-database-indexing.md)
<br/>  Poorly designed indexes may not match query patterns, compounding the effect of index-unfriendly queries.

## Detection Methods ○

- **Query Execution Plan Analysis:** This is the primary method. Always use `EXPLAIN` or `EXPLAIN ANALYZE` to understand how the database is executing your queries. Look for `Seq Scan` or `Full Table Scan` on large tables where an index is expected.
- **Database Slow Query Logs:** Configure your database to log slow queries and regularly review these logs.
- **Automated Query Performance Tools:** Many APM tools or database monitoring solutions can identify inefficient queries and suggest improvements.
- **Code Review:** Developers should be aware of common patterns that prevent index usage during code reviews.

## Examples
A user search feature queries a `users` table with `WHERE LOWER(email) = 'john.doe@example.com'`. Even though `email` is indexed, the `LOWER()` function prevents the index from being used, leading to a full table scan. Rewriting it as `WHERE email ILIKE 'john.doe@example.com'` (if case-insensitive search is needed and supported by the database) or ensuring the application handles case sensitivity before the query can fix this. In another case, a report query uses `WHERE product_code LIKE '%ABC%'`. An index on `product_code` exists, but the leading wildcard prevents its use. If the search pattern is always a suffix, a reverse index could be used, or the query rewritten if possible. This problem highlights the importance of understanding how database optimizers work and writing queries that allow them to leverage existing indexes effectively. It's a common source of performance bottlenecks, especially in applications with complex reporting or search functionalities.
