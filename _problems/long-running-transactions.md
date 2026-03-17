---
title: Long-Running Transactions
description: Database transactions that remain open for a long time can hold locks,
  consume resources, and block other operations.
category:
- Code
- Database
- Performance
related_problems:
- slug: long-running-database-transactions
  similarity: 0.95
- slug: database-connection-leaks
  similarity: 0.6
- slug: high-database-resource-utilization
  similarity: 0.6
- slug: high-connection-count
  similarity: 0.6
- slug: unreleased-resources
  similarity: 0.55
- slug: slow-database-queries
  similarity: 0.55
layout: problem
---

## Description
Long-running transactions are database transactions that remain open for an extended period of time. This can be caused by a variety of factors, from inefficient queries and a lack of proper indexing to application logic that holds transactions open while performing other tasks. Long-running transactions can cause a number of problems, including holding locks on database resources, preventing other queries from executing, and increasing the risk of deadlocks. They are a common source of performance and stability issues in database-driven applications.

## Indicators ⟡
- The database is slow, even when there are no obvious signs of high CPU or memory usage.
- You are seeing a high number of deadlocks in your database logs.
- You are getting complaints from users about slow performance.
- You are seeing a high number of timeout errors in your logs.

## Symptoms ▲

- [Lock Contention](lock-contention.md)
<br/>  Transactions holding locks for extended periods cause other operations to block, waiting for those locks to be released.
- [Deadlock Conditions](deadlock-conditions.md)
<br/>  The longer transactions hold locks, the greater the chance of circular dependencies forming between concurrent transactions.
- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  Long-running transactions consume database connections, memory, and transaction log space over extended periods.
- [Database Connection Leaks](database-connection-leaks.md)
<br/>  Transactions that run for extended periods tie up connection pool resources, and abandoned transactions may leak connections entirely.
- [Service Timeouts](service-timeouts.md)
<br/>  Operations blocked by long-running transaction locks can exceed application timeout thresholds, causing failures.
## Causes ▼

- [Slow Database Queries](slow-database-queries.md)
<br/>  Slow queries within a transaction directly extend its duration, keeping locks held longer.
- [Inefficient Database Indexing](inefficient-database-indexing.md)
<br/>  Missing indexes cause queries to perform full table scans within transactions, greatly extending transaction duration.
- [External Service Delays](external-service-delays.md)
<br/>  Application logic that calls slow external services while holding an open transaction extends its lifetime.
- [Incorrect Max Connection Pool Size](incorrect-max-connection-pool-size.md)
<br/>  Undersized connection pools under load can cause transactions to queue, and when finally executed, their effective duration spans the wait time.
## Detection Methods ○

- **Database Monitoring Tools:** Use database-specific commands (e.g., `pg_stat_activity` in PostgreSQL, `SHOW PROCESSLIST` in MySQL) to identify active transactions and their duration.
- **Transaction Log Monitoring:** Monitor the size and growth rate of transaction logs.
- **Lock Monitoring:** Use database tools to identify currently held locks and which transactions are holding them.
- **Application Logging:** Add logging to the application to track the start and end times of transactions.

## Examples
An e-commerce application has a checkout process that starts a database transaction at the beginning of the process. If the user abandons the checkout halfway through, the transaction remains open until the session times out, holding locks on inventory tables and preventing other users from purchasing those items. In another case, a nightly batch job for data synchronization wraps its entire operation in a single transaction. If the job processes millions of records, this single transaction can run for hours, consuming significant resources and potentially blocking other database operations. This problem is often a result of insufficient understanding of database transaction semantics or poor application design. It can lead to severe performance bottlenecks and data consistency issues, especially in high-concurrency environments.
