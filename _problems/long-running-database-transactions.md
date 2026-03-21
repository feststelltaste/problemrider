---
title: Long-Running Database Transactions
description: Database transactions remain open for extended periods, holding locks
  and consuming resources, which can block other operations.
category:
- Code
- Performance
related_problems:
- slug: long-running-transactions
  similarity: 0.95
- slug: high-connection-count
  similarity: 0.55
- slug: high-database-resource-utilization
  similarity: 0.55
- slug: slow-database-queries
  similarity: 0.55
- slug: database-connection-leaks
  similarity: 0.55
- slug: resource-contention
  similarity: 0.55
solutions:
- query-optimization-process
- transactions
- evolutionary-database-design
layout: problem
---

## Description
Long-running database transactions are a specific type of long-running transaction that occurs at the database level. These transactions can be particularly problematic, as they can hold locks on database resources for an extended period of time, preventing other queries from executing and potentially leading to deadlocks. They are often caused by inefficient queries, a lack of proper indexing, or application logic that holds transactions open while performing other tasks. Minimizing the duration of database transactions is a key principle of good database design.

## Indicators ⟡
- The database is slow, even when there are no obvious signs of high CPU or memory usage.
- You are seeing a high number of deadlocks in your database logs.
- You are getting complaints from users about slow performance.
- You are seeing a high number of timeout errors in your logs.

## Symptoms ▲

- [Lock Contention](lock-contention.md)
<br/>  Long-held database locks block other queries trying to access the same rows or tables, creating contention.
- [Deadlock Conditions](deadlock-conditions.md)
<br/>  Transactions holding locks for extended periods increase the window for circular lock dependencies to form.
- [Slow Database Queries](slow-database-queries.md)
<br/>  Other queries are forced to wait for locks held by long-running transactions, increasing their execution time.
- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  Long-running transactions consume connection slots, memory, and transaction log space for extended periods.
- [Service Timeouts](service-timeouts.md)
<br/>  Application requests waiting for database operations blocked by long-running transactions exceed timeout thresholds.
- [Database Connection Leaks](database-connection-leaks.md)
<br/>  Long-running database transactions tie up connections for extended periods, and abandoned transactions can leak conne....
## Causes ▼

- [Inefficient Database Indexing](inefficient-database-indexing.md)
<br/>  Missing or poor indexes cause queries within transactions to take much longer, extending transaction duration.
- [External Service Delays](external-service-delays.md)
<br/>  Calling external services while a database transaction is open means the transaction waits for slow external responses.
- [Database Schema Design Problems](database-schema-design-problems.md)
<br/>  Poor schema design can lead to excessive locking scope or require complex multi-table operations that extend transaction duration.
## Detection Methods ○

- **Database Monitoring Tools:** Use database-specific tools (e.g., `pg_stat_activity` in PostgreSQL, `SHOW PROCESSLIST` in MySQL, `sys.dm_tran_active_transactions` in SQL Server) to identify active transactions, their duration, and what they are waiting on.
- **Transaction Log Monitoring:** Monitor the size and growth rate of the database transaction logs.
- **Application Logging:** Add logging to the application to track the start and end times of database transactions.
- **Alerting:** Set up alerts for transactions that exceed a certain duration.

## Examples
An e-commerce application processes an order. It starts a database transaction, updates the inventory, then calls a third-party payment gateway. If the payment gateway is slow, the database transaction remains open, holding a lock on the inventory table. This blocks other users from placing orders for the same product. In another case, a batch job that imports millions of records into a database wraps the entire import in a single transaction. If the import fails halfway through, the transaction is rolled back, but the rollback itself takes hours, during which time the database is heavily impacted. This problem is particularly critical in high-concurrency systems where even short-lived locks can have a significant impact. It often requires careful design of transaction boundaries and asynchronous processing for long-running tasks.
