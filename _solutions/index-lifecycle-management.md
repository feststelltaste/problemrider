---
title: Index Lifecycle Management
description: Treat database indexes as maintained assets — reviewed against actual query patterns, measured for use, and removed when they stop earning their cost.
category:
- Database
- Performance
- Operations
problems:
- inefficient-database-indexing
- unused-indexes
- incorrect-index-type
- index-fragmentation
- queries-that-prevent-index-usage
- high-number-of-database-queries
- slow-database-queries
- n-plus-one-query-problem
- long-running-database-transactions
- slow-application-performance
- imperative-data-fetching-logic
- lock-contention
- long-running-transactions
- poor-caching-strategy
- entity-attribute-value-overuse
layout: solution
---

## Description

Index lifecycle management is the practice of treating indexes as assets with an ongoing cost rather than as one-time additions: reviewed periodically against the queries actually being run, measured for whether anything uses them, maintained against fragmentation, and removed when they no longer earn their keep. Indexes accumulate in legacy databases in a characteristic way. Each was added to fix a specific slow query, often years ago, by someone who has left, and none was ever removed. The result is a table with fourteen indexes, several redundant, a few unused since a query was rewritten in 2016, and every write paying the cost of maintaining all of them. Meanwhile the queries that are slow today are slow because their access pattern was never indexed for. The problem is not a lack of indexing skill but the absence of any process that revisits decisions.

## How to Apply ◆

> A legacy database's index set is a historical record of past performance incidents, not a design — which is why it usually both over-indexes and under-indexes at the same time.

- **Inventory the current indexes** with their size, and get usage statistics from the database — every mainstream engine tracks how often each index is used by the planner. Indexes with zero reads over a full business cycle are pure write overhead.
- **Start from the queries, not the tables.** Capture the actual workload from the slow query log or the statement statistics view, ranked by total time consumed rather than by individual duration. A query taking 40 milliseconds and running two million times a day matters more than one taking eight seconds nightly.
- **Look for redundancy**: an index on `(a)` is redundant when `(a, b)` exists. Composite index prefixes are the most common source of unnecessary indexes in old schemas, because each was added by someone who did not check what already existed.
- **Check that the column order matches the query patterns.** A composite index is usable only for queries that constrain its leading columns. An index whose order was chosen for a query that no longer exists is often the reason a current query cannot use it.
- **Identify queries that defeat their indexes** — a function applied to the indexed column, a type mismatch forcing conversion, a leading wildcard, an `OR` across columns. These usually need the query changed rather than another index added, and adding one instead is how index sets grow.
- **Choose the index type deliberately** where the engine offers several. Partial indexes for queries that always filter on the same condition, and covering indexes for hot read paths, frequently outperform adding another plain index and cost less to maintain.
- **Schedule maintenance** for fragmentation and statistics freshness according to what the engine requires. Stale statistics cause the planner to make bad choices even when the indexes are correct, and this is a common cause of a query that was fast last month and is slow now.
- **Remove unused indexes in a revertable way**: make them invisible to the planner if the engine supports it, or drop them during a window where re-creation is feasible, and monitor before finishing. Dropping an index used only by a quarterly report is a mistake discovered a quarter later.
- **Review on a cadence** — quarterly, or after any significant change in query patterns — and record the reason for each index. An index whose purpose is documented can be evaluated later; one without a reason will never be removed.
- **Verify against realistic data volume.** Index behavior depends on cardinality and distribution, so a decision validated against a small test dataset tells you almost nothing about production.

## Tradeoffs ⇄

> Managed indexes speed up reads and reduce write overhead, but every change carries risk on a live system and the review consumes specialist time.

**Benefits:**

- Read performance improves where it matters, because the indexes are derived from the current workload rather than from historical incidents.
- Write performance improves and storage drops when redundant and unused indexes are removed — often a substantial effect on tables with many indexes.
- Slow queries that no index will fix are identified as query problems, which is the correct diagnosis and prevents further index accumulation.
- Backup, restore, and migration times shorten, which matters directly for maintenance windows and disaster recovery.
- The documented rationale makes future review possible, breaking the cycle in which indexes are only ever added.

**Costs and Risks:**

- Dropping an index that is used rarely but critically degrades that path severely, and the discovery may be a quarter away.
- Index changes on large live tables can be expensive or lock-heavy depending on the engine, requiring maintenance windows that are hard to obtain.
- Usage statistics reset on restart in some engines and reflect only the observation period, so a short window produces misleading conclusions.
- The work requires database expertise that many teams maintaining legacy systems no longer have in-house.
- Adding indexes to fix reads shifts cost to writes, and on write-heavy tables the net effect can be negative in ways that are not obvious until load increases.

## How It Could Be

A team maintaining an order management database investigated why nightly batch processing had grown from 90 minutes to over four hours across two years. The obvious suspects were data volume and query plans. An index inventory found 142 indexes across the 30 largest tables, of which usage statistics over a full quarter showed 31 had never been read and a further 18 were redundant prefixes of composite indexes. The order line table alone carried 11 indexes, adding roughly 40 percent overhead to every insert — and the batch job inserted several million rows nightly. Removing the 31 unused and 18 redundant indexes, in two stages with monitoring between, brought the batch window back to 105 minutes. No query was made slower.

The same review changed how the team handled a chronically slow customer search. Their previous three attempts had each added an index, none of which helped. Examining the query showed it applied `UPPER()` to the surname column, which made every index on that column unusable — the planner had been doing a full scan regardless of how many indexes existed. Adding a case-insensitive expression index brought the query from 3.2 seconds to 15 milliseconds, and let them remove two of the three indexes that had been added in previous attempts. The team adopted a rule from this: no index is added to fix a slow query until someone has read the execution plan and confirmed the query can use one.
