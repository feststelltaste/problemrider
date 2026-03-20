---
title: Data Integrity
description: Mechanisms to ensure data accuracy, consistency, and reliability
category:
- Database
quality_tactics_url: https://qualitytactics.de/en/reliability/data-integrity
problems:
- silent-data-corruption
- data-migration-integrity-issues
- cross-system-data-synchronization-problems
- database-schema-design-problems
- inconsistent-behavior
- unbounded-data-growth
layout: solution
---

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
