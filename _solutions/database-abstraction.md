---
title: Database Abstraction
description: Implementing database accesses through an abstracted layer
category:
- Database
- Architecture
quality_tactics_url: https://qualitytactics.de/en/portability/database-abstraction
problems:
- technology-lock-in
- vendor-lock-in
- tight-coupling-issues
- data-migration-complexities
- database-schema-design-problems
- difficult-to-test-code
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Introduce an ORM or data access layer (e.g., Hibernate, Entity Framework, SQLAlchemy) between business logic and raw SQL
- Encapsulate all database access behind repository interfaces that hide the underlying database technology
- Replace database-specific SQL syntax (stored procedures, proprietary functions) with portable equivalents where possible
- Isolate unavoidably database-specific operations into clearly marked adapter modules
- Use database migration tools that generate portable DDL rather than hand-written database-specific scripts
- Implement the repository pattern with in-memory implementations for unit testing
- Gradually migrate raw SQL queries to the abstraction layer, prioritizing the most frequently modified code paths

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables migration between database vendors without rewriting business logic
- Improves testability by allowing database logic to be tested with in-memory implementations
- Centralizes query optimization and caching concerns in one layer
- Reduces the spread of SQL throughout the codebase, improving maintainability

**Costs and Risks:**
- ORM abstractions can generate inefficient queries that perform worse than hand-written SQL
- Complex legacy queries using vendor-specific features may not map cleanly to the abstraction
- The abstraction layer itself introduces a learning curve and potential bugs
- Performance-critical operations may need to bypass the abstraction, creating inconsistency
- Migrating a large legacy codebase with thousands of raw SQL statements is a multi-year effort

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy Java application contained over 2,000 Oracle-specific SQL queries scattered across its codebase, including PL/SQL stored procedure calls and Oracle-specific date functions. When the company decided to migrate to PostgreSQL to reduce licensing costs, every query needed modification. The team introduced JPA repositories and gradually migrated queries to JPQL over 18 months. They isolated the 50 queries that genuinely required database-specific features into adapter classes with both Oracle and PostgreSQL implementations. This approach allowed them to run both databases in parallel during the migration, with the adapter selection controlled by configuration, and ultimately completed the migration without any business logic changes.
