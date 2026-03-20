---
title: Platform-Independent Data Storage
description: Choose database systems and storage solutions that are available on various platforms
category:
- Database
- Architecture
quality_tactics_url: https://qualitytactics.de/en/portability/platform-independent-data-storage
problems:
- technology-lock-in
- vendor-lock-in
- vendor-dependency-entrapment
- database-schema-design-problems
- data-migration-complexities
- data-migration-integrity-issues
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Evaluate current database dependencies and identify vendor-specific features such as proprietary SQL extensions, stored procedures, or data types
- Select database systems that are available across all target platforms (e.g., PostgreSQL, MySQL, SQLite, MongoDB)
- Introduce a data access abstraction layer that isolates application code from database-specific APIs
- Replace vendor-specific SQL syntax with ANSI SQL or use an ORM to generate compatible queries
- Migrate stored procedures and database-side business logic into the application layer where possible
- Use standardized data export formats (CSV, JSON, Parquet) for data interchange between systems
- Test data operations on all target platforms as part of the CI/CD pipeline

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables database migration without rewriting application code
- Reduces dependency on a single database vendor's pricing and licensing model
- Supports hybrid deployment scenarios with different databases per environment
- Facilitates disaster recovery by allowing failover to alternative database platforms

**Costs and Risks:**
- Avoiding vendor-specific features may sacrifice performance optimizations unique to a particular database
- Data migration between different database systems carries integrity and compatibility risks
- Maintaining compatibility across multiple databases increases testing complexity
- Some legacy applications have deep dependencies on specific database features that are costly to abstract

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An insurance company had a legacy claims processing system built on Oracle Database, using over 500 PL/SQL stored procedures and Oracle-specific features like materialized views and Oracle Text for full-text search. Annual licensing costs exceeded $800,000. The team began migrating to PostgreSQL by first introducing an application-layer data access module that abstracted database calls. They replaced PL/SQL procedures with application-side logic over eight months and swapped Oracle Text for Elasticsearch. The migration reduced licensing costs by 90% and gave the team freedom to deploy on any cloud provider's managed PostgreSQL offering.
