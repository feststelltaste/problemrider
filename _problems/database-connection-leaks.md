---
title: Database Connection Leaks
description: Database connections are opened but not properly closed, leading to connection
  pool exhaustion and application failures.
category:
- Code
- Database
- Performance
related_problems:
- slug: misconfigured-connection-pools
  similarity: 0.7
- slug: high-connection-count
  similarity: 0.65
- slug: incorrect-max-connection-pool-size
  similarity: 0.65
- slug: database-query-performance-issues
  similarity: 0.6
- slug: resource-allocation-failures
  similarity: 0.6
- slug: long-running-transactions
  similarity: 0.6
solutions:
- query-optimization-process
- connection-pooling
layout: problem
---

## Description

Database connection leaks occur when applications open database connections but fail to properly close them when they are no longer needed. This leads to the gradual depletion of the connection pool, eventually causing new database operations to fail when no connections are available. Connection leaks are particularly problematic in high-traffic applications and can cause complete service outages that require application restarts to resolve.

## Indicators ⟡

- Application fails to execute database queries with "connection pool exhausted" errors
- Database monitoring shows steadily increasing number of active connections
- Application performance degrades over time as available connections diminish
- Database operations timeout or fail after the application has been running for a period
- Connection pool metrics show high utilization with low throughput

## Symptoms ▲

- [System Outages](system-outages.md)
<br/>  Connection pool exhaustion from leaked connections causes complete application failures requiring restarts to restore service.
- [Slow Application Performance](slow-application-performance.md)
<br/>  As available connections diminish, database operations queue up and timeout, making the application progressively slower.
- [Gradual Performance Degradation](gradual-performance-degradation.md)
<br/>  Connection leaks cause performance to slowly worsen over time as the connection pool is gradually depleted.
- [High Connection Count](high-connection-count.md)
<br/>  Leaked connections accumulate as open but unused connections, driving up the total connection count on the database server.
- [Resource Allocation Failures](resource-allocation-failures.md)
<br/>  When the connection pool is exhausted by leaked connections, new database operations fail because no resources can be allocated.
## Causes ▼

- [Inadequate Error Handling](inadequate-error-handling.md)
<br/>  Connections opened in try blocks but not properly closed in exception paths leak when errors occur during database operations.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers unfamiliar with connection lifecycle management fail to use try-with-resources patterns or proper cleanup logic.
- [Legacy Code Without Tests](legacy-code-without-tests.md)
<br/>  Without tests that exercise error paths and long-running scenarios, connection leak patterns go undetected until production.
- [Insufficient Testing](insufficient-testing.md)
<br/>  Connection leaks typically only manifest under sustained load or error conditions that are not covered by superficial testing.
## Detection Methods ○

- **Connection Pool Monitoring:** Monitor database connection pool usage, active connections, and pool exhaustion events
- **Database Connection Tracking:** Track database connection lifecycle from creation to closure
- **Application Performance Monitoring:** Monitor database operation response times and failure rates
- **Resource Leak Detection:** Use profiling tools to identify unreleased database connections
- **Load Testing:** Run sustained load tests to identify connection leak patterns
- **Database Server Monitoring:** Monitor active connections at the database server level

## Examples

A web application opens database connections in a try block to execute queries but only closes them in the main execution path, not in the exception handling paths. When database queries fail due to temporary network issues, the connections remain open and are never returned to the pool. After several hours of intermittent database errors, the connection pool is exhausted and the application can no longer serve any requests that require database access. Another example involves a batch processing system that opens database connections inside loops but closes them outside the loop. When the loop processes thousands of records, thousands of connections are opened but only one is closed, quickly exhausting the connection pool and causing the batch process to fail.
