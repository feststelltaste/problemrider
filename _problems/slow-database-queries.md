---
title: Slow Database Queries
description: Application performance degrades due to inefficient data retrieval from
  the database.
category:
- Performance
related_problems:
- slug: database-query-performance-issues
  similarity: 0.8
- slug: high-number-of-database-queries
  similarity: 0.75
- slug: slow-application-performance
  similarity: 0.7
- slug: high-database-resource-utilization
  similarity: 0.7
- slug: imperative-data-fetching-logic
  similarity: 0.7
- slug: lazy-loading
  similarity: 0.7
layout: problem
---

## Description
Slow database queries are a primary cause of poor application performance. When a query takes too long to execute, it can hold up application threads, block other queries, and lead to a frustrating user experience. These slow queries are often the result of inefficient query design, missing or improper indexes, or a database schema that is not optimized for the types of queries being run. Identifying and optimizing slow queries is a critical task in a healthy and performant application.

## Indicators ⟡
- The application is slow, and you suspect that it is due to slow database queries.
- You are seeing a high number of slow queries in your database logs.
- The database is using a lot of CPU or memory.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  Slow queries directly cause the application to respond slowly to user requests.
- [Slow Response Times for Lists](slow-response-times-for-lists.md)
<br/>  List pages are especially affected by slow queries because they execute multiple queries or process large result sets.
- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  Inefficient queries consume excessive CPU, memory, and I/O on the database server.
- [High API Latency](high-api-latency.md)
<br/>  API endpoints that depend on database queries inherit the slowness, increasing overall API response times.

## Causes ▼
- [High Number of Database Queries](high-number-of-database-queries.md)
<br/>  N+1 query patterns and excessive query counts compound into significant performance problems.
- [Imperative Data Fetching Logic](imperative-data-fetching-logic.md)
<br/>  Manually constructed data fetching logic often produces inefficient query patterns instead of leveraging optimized database operations.
- [Lazy Loading](lazy-loading.md)
<br/>  Lazy loading triggers additional database queries on demand, leading to unpredictable and often excessive query execution.
- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  When the database is under heavy resource load, query execution times increase significantly as CPU and memory contention delays processing.
- [Index Fragmentation](index-fragmentation.md)
<br/>  Fragmented indexes require more I/O operations to traverse, directly causing queries to execute more slowly.
- [Inefficient Database Indexing](inefficient-database-indexing.md)
<br/>  Missing or inappropriate indexes force full-table scans, directly causing slow query execution times.
- [Long-Running Database Transactions](long-running-database-transactions.md)
<br/>  Other queries are forced to wait for locks held by long-running transactions, increasing their execution time.
- [Queries That Prevent Index Usage](queries-that-prevent-index-usage.md)
<br/>  Queries that bypass indexes force full table scans, directly causing slow query execution times.

## Detection Methods ○

- **Database query logging:** Enable logging of slow queries in the database configuration.
- **Application performance monitoring (APM) tools:** Use tools like New Relic, Datadog, or Prometheus to monitor query performance and identify bottlenecks.
- **Database-specific tools:** Use tools like `EXPLAIN` in PostgreSQL or `EXPLAIN PLAN` in Oracle to analyze query execution plans.
- **Code reviews:** Look for common anti-patterns like N+1 queries or inefficient query logic.
- **Load testing:** Simulate high traffic to identify queries that do not scale well.

## Examples
A web application's user profile page takes a long time to load. Upon investigation, it is discovered that the page is making a separate database query for each of the user's friends to retrieve their profile pictures. In another case, a reporting dashboard that aggregates data from multiple tables is timing out because the queries are not using the correct indexes. This problem is common in applications that have a large amount of data or complex data models. It is often exacerbated by a lack of database expertise on the development team.
