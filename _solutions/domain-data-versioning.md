---
title: Domain Data Versioning
description: Track and restore changes to domain-specific data
category:
- Database
problems:
- silent-data-corruption
- data-migration-integrity-issues
- insufficient-audit-logging
- schema-evolution-paralysis
- debugging-difficulties
layout: solution
related_solutions:
- slug: timestamping
  similarity: 0.65
- slug: versioning-scheme
  similarity: 0.65
- slug: evolutionary-database-design
  similarity: 0.6
- slug: data-integrity
  similarity: 0.6
- slug: continuous-data-verification
  similarity: 0.6
- slug: write-ahead-logging
  similarity: 0.6
---

## How to Apply ◆

- Implement temporal tables or audit tables that record every change to critical domain entities along with timestamps, users, and change reasons.
- Add versioning to domain objects so that the current state and full history of each entity are available.
- Use event sourcing for critical business entities where the ability to reconstruct state at any point in time is valuable.
- Build tools for comparing entity versions and identifying when and why data changed, supporting root cause analysis.
- Ensure that data versioning covers migrations and bulk updates, not just individual record changes.
- Define retention policies for historical data versions to manage storage growth.

## Tradeoffs ⇄

**Benefits:**
- Enables auditing and compliance by providing a complete history of data changes.
- Supports debugging by allowing reconstruction of the system's state at any past point in time.
- Provides a safety net for data corrections: incorrect changes can be identified and reverted.
- Facilitates data migration validation by comparing source and target versions.

**Costs:**
- Storing every version of every domain entity significantly increases storage requirements.
- Adds write overhead to every data modification operation.
- Querying historical data adds complexity to the data access layer.
- Retrofitting versioning into a legacy system with no existing audit trail requires schema changes and migration.

## How It Could Be

A legacy contract management system has no audit trail, making it impossible to determine when or why a contract's terms were modified. After a dispute where a customer claims their pricing was changed without authorization, the team adds domain data versioning using temporal tables. Every contract modification is now recorded with a timestamp, the user who made the change, and the previous values. When a similar dispute arises six months later, the team can show the exact history of changes, who authorized them, and when they occurred. The versioning system also proves invaluable during a data migration, where the team uses version histories to verify that the migration preserved all contract terms correctly.
