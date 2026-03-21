---
title: High Connection Count
description: A large number of open database connections, even if idle, can consume
  significant memory resources and lead to connection rejections.
category:
- Code
- Performance
related_problems:
- slug: incorrect-max-connection-pool-size
  similarity: 0.8
- slug: misconfigured-connection-pools
  similarity: 0.7
- slug: high-database-resource-utilization
  similarity: 0.7
- slug: high-number-of-database-queries
  similarity: 0.7
- slug: database-connection-leaks
  similarity: 0.65
- slug: service-timeouts
  similarity: 0.6
solutions:
- connection-pooling
- resource-pooling
- timeout-management
- capacity-planning
- elastic-scaling
- backpressure
- resource-usage-optimization
layout: problem
---

## Description
A high connection count occurs when a database is overwhelmed by a large number of open connections, both active and idle. Each connection consumes memory and other resources on the database server, and exceeding the configured limit can lead to connection rejections and application failures. This problem is often a symptom of misconfigured connection pooling, inefficient application code that fails to release connections, or sudden spikes in traffic. Properly managing connections is crucial for maintaining the stability and performance of any database-driven application.

## Indicators ⟡
- You are seeing a high number of connections in your database monitoring tools.
- Your application is slow, and you suspect that it is due to a high number of database connections.
- You are getting complaints from users about slow performance.
- You are seeing a high number of timeout errors in your logs.

## Symptoms ▲

- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  Each open connection consumes memory and CPU on the database server, driving up overall resource utilization.
- [Service Timeouts](service-timeouts.md)
<br/>  When the connection limit is reached, new connection attempts are rejected or queued, causing service timeouts.
- [Slow Application Performance](slow-application-performance.md)
<br/>  Resource contention from too many connections degrades database response times, slowing the entire application.
- [Increased Error Rates](increased-error-rates.md)
<br/>  Connection rejections when limits are reached cause application errors and failed requests.
- [Cascade Failures](cascade-failures.md)
<br/>  Database connection exhaustion causes failures that cascade to all services depending on that database.
## Causes ▼

- [Misconfigured Connection Pools](misconfigured-connection-pools.md)
<br/>  Improperly configured connection pool settings allow too many connections to be created or kept idle.
- [Database Connection Leaks](database-connection-leaks.md)
<br/>  Connections that are opened but never properly closed accumulate over time, steadily increasing the connection count.
- [Incorrect Max Connection Pool Size](incorrect-max-connection-pool-size.md)
<br/>  Setting the maximum pool size too high allows each application instance to hold more connections than the database can efficiently handle.
- [Resource Allocation Failures](resource-allocation-failures.md)
<br/>  Code that fails to properly release database connections after use causes connections to accumulate without being returned to the pool.
## Detection Methods ○

- **Database Monitoring Tools:** Use database-specific tools (e.g., `SHOW STATUS` in MySQL, `pg_stat_activity` in PostgreSQL) to monitor the number of active and idle connections.
- **Application Metrics:** Monitor connection pool metrics within the application (e.g., active connections, idle connections, wait times).
- **System Monitoring:** Observe the database server's memory usage and process count.
- **Log Analysis:** Look for database error logs indicating connection rejections.

## Examples
A web application experiences intermittent "Too many connections" errors during peak traffic. Investigation reveals that the application's connection pool is configured with a very high `max_idle_connections` setting, causing thousands of idle connections to accumulate on the database server. In another case, a batch job runs every hour and opens a new database connection for each record it processes, without closing them. Over time, this leads to a gradual increase in connection count until the database hits its limit. This problem is common in applications that are not designed with connection management in mind, or where default connection pool settings are used without proper tuning for the specific workload. It can be particularly problematic in microservices architectures where many services might independently open connections to the same database.
