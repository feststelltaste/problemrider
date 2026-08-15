---
title: Data Integrity
description: Mechanisms to ensure data accuracy, consistency, and reliability
category:
- Database
problems:
- silent-data-corruption
- data-migration-integrity-issues
- cross-system-data-synchronization-problems
- database-schema-design-problems
- inconsistent-behavior
- unbounded-data-growth
- cache-invalidation-problems
- dma-coherency-issues
- synchronization-problems
layout: solution
related_solutions:
- slug: continuous-data-verification
  similarity: 0.8
- slug: checksums
  similarity: 0.8
- slug: data-deduplication
  similarity: 0.8
- slug: fault-tolerant-data-structures
  similarity: 0.75
- slug: error-correction-codes
  similarity: 0.75
- slug: monitoring-system-integrity
  similarity: 0.75
---

## Description

Data integrity comprises the constraints and mechanisms — foreign keys, uniqueness and check constraints, transactional atomicity, and complementary application-level validation — that keep stored data accurate, internally consistent, and free of contradictions such as orphaned references or impossible values. The core mechanism is defense at multiple levels: database-level constraints act as a hard backstop that rejects invalid states regardless of which application code path attempted to create them, while application-level validation catches problems earlier and provides better error messages, and neither substitutes for the other. In legacy systems, integrity constraints are frequently missing entirely because they were never added during initial development or were deliberately relaxed to work around some now-forgotten obstacle, and the result, after years of operation, is an accumulation of silently corrupted data: contacts pointing at deleted companies, duplicate entities, and inconsistencies that only surface when someone tries to build a reliable report or migration on top of the data. Restoring integrity to such a system is necessarily incremental, since constraints cannot simply be turned on over data that already violates them — existing violations first have to be found and resolved, often through purpose-built cleanup scripts, before the constraint that would have prevented them can be safely enabled. Once in place, these constraints convert data quality problems from a recurring investigative burden into build-time or transaction-time failures that surface immediately, at the moment the bad state would otherwise have been created.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Audit existing database schemas for missing constraints (foreign keys, unique constraints, check constraints, not-null)
- Add database-level constraints incrementally, starting with the most critical business entities
- Implement application-level validation as a complement to database constraints, not a replacement
- Use transactions appropriately to ensure atomicity of multi-step data operations
- Add referential integrity constraints between related tables that may have been omitted in the original design
- Implement data quality monitoring that continuously checks for orphaned records, duplicates, and constraint violations
- Create data repair scripts for known integrity issues and run them as part of regular maintenance

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Prevents corrupt or inconsistent data from entering the system at the database level
- Reduces the need for expensive data cleanup and reconciliation processes
- Increases trust in data for reporting, analytics, and downstream integrations
- Makes implicit data rules explicit and enforceable

**Costs and Risks:**
- Adding constraints to legacy databases with existing bad data requires data cleanup first
- Strict constraints may break legacy code that relied on lax validation
- Foreign key constraints can impact write performance on high-throughput tables
- Retroactively enforcing integrity on historical data can be extremely time-consuming

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy CRM system had no foreign key constraints in its database. Over years of operation, orphaned records accumulated: contacts referenced deleted companies, activities linked to non-existent opportunities, and duplicate records proliferated. The team began by profiling the data to quantify integrity violations, finding over 50,000 orphaned records across 12 tables. They wrote cleanup scripts to resolve existing violations, then added foreign key constraints with cascading rules appropriate to each relationship. Application code that had silently created orphaned records began throwing errors, which were fixed one by one. After six months, data quality issues reported by sales staff dropped from weekly occurrences to near zero.
