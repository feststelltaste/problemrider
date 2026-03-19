---
title: Incorrect Max Connection Pool Size
description: The maximum number of connections in a database connection pool is set
  incorrectly, leading to either wasted resources or connection exhaustion.
category:
- Code
- Performance
related_problems:
- slug: misconfigured-connection-pools
  similarity: 0.85
- slug: high-connection-count
  similarity: 0.8
- slug: database-connection-leaks
  similarity: 0.65
- slug: high-number-of-database-queries
  similarity: 0.6
- slug: high-database-resource-utilization
  similarity: 0.6
- slug: database-query-performance-issues
  similarity: 0.6
layout: problem
---

## Description
Setting the maximum size of a connection pool is a delicate balancing act. If the size is too small, the application may be starved for connections, leading to timeouts and poor performance. If the size is too large, it can overwhelm the database with too many connections, leading to a degradation in performance and stability. The optimal size for a connection pool depends on a variety of factors, including the number of application instances, the number of threads in each instance, and the capacity of the database.

## Indicators ⟡
- You are seeing a high number of connection errors in your logs.
- Your application is slow, and you suspect that it is due to a high number of database connections.
- You are getting complaints from users about slow performance.
- You are seeing a high number of timeout errors in your logs.

## Symptoms ▲

- [Service Timeouts](service-timeouts.md)
<br/>  When the pool is too small, requests wait for available connections and eventually time out.
- [High Connection Count](high-connection-count.md)
<br/>  An oversized pool creates unnecessarily many connections to the database, wasting resources.
- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  Too many connections from an oversized pool consume database memory and CPU resources.
- [Increased Error Rates](increased-error-rates.md)
<br/>  Connection exhaustion from an undersized pool or database rejection from an oversized pool both produce application errors.
- [Misconfigured Connection Pools](misconfigured-connection-pools.md)
<br/>  An incorrectly sized connection pool is a key contributor to overall connection pool misconfiguration problems.
## Causes ▼

- [Incomplete Knowledge](incomplete-knowledge.md)
<br/>  Developers may not understand the relationship between application concurrency, database capacity, and optimal pool sizing.
- [Inadequate Configuration Management](inadequate-configuration-management.md)
<br/>  Poor configuration management means pool sizes are not properly tuned or tracked across environments.
## Detection Methods ○

- **Application Metrics:** Monitor connection pool metrics (e.g., active connections, idle connections, wait times, connection acquisition rates, pool size) provided by the application framework or a monitoring agent.
- **Database Monitoring Tools:** Observe the number of active and idle connections on the database server and compare it to the `max_connections` setting.
- **Log Analysis:** Look for connection-related errors in application and database logs.
- **Load Testing:** Systematically increase load while monitoring connection pool and database metrics to find the optimal `max_pool_size`.

## Examples
A web application is deployed with a default connection pool size of 10. During a marketing campaign, the number of concurrent users spikes to 100. The application starts throwing "Connection pool exhausted" errors because it cannot acquire enough database connections to serve all requests. In another case, a microservice is configured with a `max_pool_size` of 200, but the database it connects to only allows a maximum of 100 connections. This leads to intermittent connection failures and wasted application resources trying to open connections that the database will reject. Proper configuration of database connection pools is crucial for the performance and stability of any application that interacts with a relational database. It requires understanding both the application's concurrency needs and the database's capacity.
