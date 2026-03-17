---
title: High Database Resource Utilization
description: The database server consistently operates with high CPU or memory usage,
  risking instability and slowing down all dependent services.
category:
- Code
- Performance
related_problems:
- slug: high-resource-utilization-on-client
  similarity: 0.75
- slug: high-client-side-resource-consumption
  similarity: 0.7
- slug: slow-database-queries
  similarity: 0.7
- slug: high-number-of-database-queries
  similarity: 0.7
- slug: high-connection-count
  similarity: 0.7
- slug: database-query-performance-issues
  similarity: 0.7
layout: problem
---

## Description
High database resource utilization can be a major cause of poor application performance and stability. This can be caused by a variety of factors, from inefficient queries and a lack of proper indexing to a high number of connections and long-running transactions. When the database is under stress, it can lead to a degradation in performance, timeouts, and even a complete failure of the system. A robust monitoring and alerting system is essential for detecting and responding to high database resource utilization in a timely manner.

## Indicators ⟡
- Your database server is constantly running at high CPU or memory usage.
- You are seeing a high number of slow queries in your database logs.
- Your application is slow, and you suspect that it is due to a high number of database connections.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [Slow Database Queries](slow-database-queries.md)
<br/>  When the database is under heavy resource load, query execution times increase significantly as CPU and memory contention delays processing.
- [Slow Application Performance](slow-application-performance.md)
<br/>  High database resource usage directly degrades application response times since most operations depend on database interactions.
- [High API Latency](high-api-latency.md)
<br/>  API endpoints that depend on database queries experience increased latency when the database server is resource-constrained.
- [System Outages](system-outages.md)
<br/>  Database instability from sustained high resource usage can lead to crashes and complete service outages.
- [Resource Contention](resource-contention.md)
<br/>  High database resource utilization creates contention where multiple queries compete for limited CPU and memory resources.
## Causes ▼

- [High Number of Database Queries](high-number-of-database-queries.md)
<br/>  A large volume of queries per request multiplies the load on database CPU and memory resources.
- [Slow Database Queries](slow-database-queries.md)
<br/>  Inefficient queries that run for extended periods consume database resources disproportionately and hold locks longer.
- [Inefficient Database Indexing](inefficient-database-indexing.md)
<br/>  Missing or poorly designed indexes force the database to perform full table scans, consuming excessive CPU and I/O.
- [High Connection Count](high-connection-count.md)
<br/>  Too many open database connections consume memory and CPU resources on the database server.
## Detection Methods ○

- **Database Monitoring Tools:** Use specialized database monitoring tools (e.g., pgAdmin for PostgreSQL, MySQL Workbench, or third-party tools like Percona Monitoring and Management) to inspect resource usage, running queries, and configuration.
- **Cloud Provider Metrics:** If using a managed database service (like AWS RDS or Google Cloud SQL), use the cloud provider's monitoring dashboards to track CPU, memory, and I/O metrics.
- **Query Analysis:** Use the database's `EXPLAIN` or `EXPLAIN ANALYZE` commands to inspect the execution plans of slow or frequent queries and identify inefficiencies.
- **System Performance Utilities:** Use standard Linux/Windows command-line tools (`top`, `htop`, `iostat`, `vmstat`) on the database server to get a real-time view of resource consumption.

## Examples
A company's main application becomes slow every day at noon. An investigation reveals that a daily report, which runs a series of complex, unoptimized queries, is kicking off at this time and consuming all available database CPU. In another case, a web application using a connection pool is misconfigured to open far more connections than the database is tuned for. Over time, the database's memory usage climbs until it becomes unstable, even though the query workload itself is not particularly high. This is a critical issue in legacy systems where the database has been in use for many years. Over time, data volume grows, query patterns change, and indexes that were once effective may no longer be optimal, leading to a gradual increase in resource utilization.
