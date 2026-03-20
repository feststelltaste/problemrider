---
title: Backward-Compatible Schema Migrations
description: Consider backward compatibility in database schemas and migrations
category:
- Database
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/backward-compatible-schema-migrations
problems:
- database-schema-design-problems
- data-migration-complexities
- data-migration-integrity-issues
- schema-evolution-paralysis
- deployment-risk
- breaking-changes
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Use expand-and-contract migrations: add the new column or table first, then migrate data, then remove the old structure
- Never rename or drop columns in a single deployment; use a multi-step process across releases
- Make new columns nullable or provide defaults so the old application version can still write to the database
- Run schema migrations in a separate deployment step before application code changes
- Test migrations against a production-size dataset copy to catch performance and compatibility issues
- Maintain a migration compatibility matrix showing which application versions work with which schema versions

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables zero-downtime deployments by decoupling schema changes from application releases
- Allows rollback of application code without rolling back the database
- Reduces the risk of data loss during schema evolution

**Costs and Risks:**
- Multi-step migrations take longer and require coordination across multiple releases
- Temporary duplication of columns or tables increases storage and query complexity
- Teams must track which migration phase each environment is in
- Complex migrations may require backfill jobs that run against large datasets

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A healthcare platform needed to split a single address text column into structured fields (street, city, postal code) across a database with 40 million patient records. Using the expand-and-contract pattern, the team first added the new columns as nullable, deployed a background job to parse and backfill existing addresses, updated the application to write to both old and new columns, and finally removed the old column two releases later. The entire migration completed with zero downtime and no data loss.
