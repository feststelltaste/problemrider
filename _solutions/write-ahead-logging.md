---
title: Write-Ahead Logging
description: Recording changes in a durable append-only log before applying them
category:
- Database
- Architecture
problems:
- silent-data-corruption
- data-migration-integrity-issues
- cascade-failures
- system-outages
- debugging-difficulties
- missing-rollback-strategy
- long-running-database-transactions
- inconsistent-behavior
layout: solution
related_solutions:
- slug: transactions
  similarity: 0.75
- slug: backup-and-recovery
  similarity: 0.7
- slug: logging
  similarity: 0.7
- slug: regular-backups
  similarity: 0.7
- slug: audit-trail-management
  similarity: 0.7
- slug: timestamping
  similarity: 0.7
---

## Description

Write-ahead logging is a durability technique in which every intended change to a data store is first recorded in a sequential, append-only log and only afterward applied to the actual data structures it describes. The log entry captures enough information — the operation, its parameters, and a sequence number — to either replay the change forward or, if needed, reconstruct the state that preceded it, and it is only considered complete once flushed to durable storage. This ordering is the essential property: because the log is written and made durable before the corresponding in-place modification is confirmed, a crash occurring mid-write leaves behind a trail that a recovery process can use to finish or undo the interrupted operation, rather than leaving the data store in an ambiguous, partially-applied state. In legacy systems, this matters because many older data-handling components update records directly and assume writes either fully succeed or the system never crashes mid-operation — an assumption that regularly fails during unplanned outages, power loss, or abrupt process termination, and one of the more common sources of silent data corruption in aging systems. Write-ahead logging also extends naturally to legacy data migration efforts: because it produces a durable, ordered record of every change, it lets a migration process resume from the exact point of interruption instead of requiring a full re-comparison of source and target datasets. Most mature databases and messaging systems already implement this technique internally, so applying it in legacy modernization work is often a matter of enabling and correctly configuring existing WAL mechanisms rather than building new logging infrastructure from scratch.

## How to Apply ◆

> Legacy systems often modify data in place without any recovery mechanism, meaning that a crash during a write operation can leave data in a corrupted, partially-updated state. Write-ahead logging ensures that all changes are recorded in a durable log before they are applied, enabling reliable recovery after failures.

- Identify critical data modification paths in the legacy system where a failure during the write process would leave data in an inconsistent or unrecoverable state. These are the highest-priority candidates for write-ahead logging.
- Implement an append-only log that records each intended change before it is applied to the primary data store. Each log entry should include a unique sequence number, the operation details, a timestamp, and enough information to both replay the operation and reverse it if needed.
- Ensure the log is written to durable storage (flushed to disk) before acknowledging the operation to the caller. Without durability guarantees, the log cannot serve its recovery purpose.
- Implement a recovery procedure that replays uncommitted log entries after a crash. On startup, the system reads the log, identifies operations that were logged but not confirmed as applied, and replays them to bring the data store to a consistent state.
- Use checkpointing to periodically mark a point in the log where all prior operations have been successfully applied. This limits the number of log entries that must be replayed during recovery and allows old log segments to be archived or deleted.
- Apply write-ahead logging to data migration operations in legacy systems, where a failure midway through migration can leave data split across old and new stores in an inconsistent state. The log provides a mechanism to resume migration from the last successful checkpoint.
- Consider using existing WAL implementations (database transaction logs, Apache Kafka, event sourcing frameworks) rather than building a custom solution, as correct implementation of crash-safe logging requires careful handling of edge cases.

## Tradeoffs ⇄

> Write-ahead logging provides crash recovery and data consistency guarantees by ensuring that no change is lost even during unexpected failures, but it adds write amplification and storage overhead.

**Benefits:**

- Prevents data corruption from partial writes by ensuring that either the complete operation can be recovered from the log or rolled back to the previous consistent state.
- Enables point-in-time recovery by replaying the log to any desired position, which is invaluable during data migration and system modernization.
- Provides a complete audit trail of all data modifications, supporting debugging, compliance, and root cause analysis.
- Supports replication and synchronization by streaming log entries to secondary systems, enabling gradual migration from legacy to modern data stores.

**Costs and Risks:**

- Write amplification: every data modification results in at least two writes (one to the log, one to the data store), which can impact performance on I/O-constrained legacy systems.
- Log storage grows continuously and requires management — archiving, compression, and cleanup policies to prevent disk space exhaustion.
- Implementing crash-safe logging correctly is complex and subtle; bugs in the logging or recovery logic can cause the very data corruption it is designed to prevent.
- Recovery replay time after a crash depends on the amount of uncompacted log, which can delay system restart if checkpointing is infrequent.

## How It Could Be

> The following scenarios illustrate how write-ahead logging prevents data loss and corruption in legacy systems.

A legacy inventory management system updates stock levels by directly modifying rows in a database table. During a server crash caused by a power failure, several UPDATE statements execute partially — the database commits some changes but loses others that were in the buffer pool but not yet flushed to disk. The result is inventory counts that are internally inconsistent: some items show negative stock levels, and the total inventory value no longer matches the sum of individual items. The team implements write-ahead logging by routing all inventory changes through a durable log (using PostgreSQL's built-in WAL with synchronous commits enabled) and verifying that all changes are fully logged before returning success to the application. After the next unplanned server restart, the database recovers automatically by replaying the WAL, and inventory counts are perfectly consistent. The team also implements periodic checkpointing at 15-minute intervals, limiting recovery time to under 2 minutes.

A legacy data warehouse undergoes a major migration from an on-premises Oracle database to a cloud-based PostgreSQL instance. The migration must happen incrementally while both systems remain operational. The team implements a change data capture log that records every write to the Oracle database as an append-only entry. A migration worker continuously reads from this log and applies changes to the PostgreSQL instance. When a network disruption interrupts the migration worker for 45 minutes, it resumes from its last recorded log position and applies all missed changes without any data loss. Without the write-ahead log, the team would have needed to perform a full comparison of both databases to identify and reconcile the missed changes — a process that previously took 8 hours and introduced its own risk of errors.
