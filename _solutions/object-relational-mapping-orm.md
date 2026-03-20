---
title: Object-Relational Mapping (ORM)
description: Abstracting database interactions through objects
category:
- Code
- Database
quality_tactics_url: https://qualitytactics.de/en/portability/object-relational-mapping-orm
problems:
- technology-lock-in
- vendor-lock-in
- database-query-performance-issues
- database-schema-design-problems
- difficult-code-comprehension
- high-coupling-low-cohesion
- n-plus-one-query-problem
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Inventory all raw SQL queries and database access patterns in the legacy codebase to understand the scope of migration
- Choose an ORM framework appropriate to the language and ecosystem (e.g., Hibernate, Entity Framework, SQLAlchemy)
- Start by mapping the most stable domain entities to ORM models while keeping complex queries as native SQL initially
- Introduce a repository or data access layer that encapsulates ORM usage behind clean interfaces
- Migrate raw SQL incrementally, replacing hand-written queries with ORM equivalents module by module
- Configure lazy loading, eager fetching, and query optimization settings to avoid common performance pitfalls
- Retain the option to use native SQL for performance-critical queries where the ORM abstraction introduces unacceptable overhead

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Decouples application code from specific database dialects, making database migration feasible
- Reduces boilerplate data access code and eliminates many classes of SQL injection vulnerabilities
- Improves developer productivity by working with domain objects rather than result sets
- Simplifies unit testing through easier mocking of data access layers

**Costs and Risks:**
- ORM-generated queries can be inefficient, especially for complex joins or bulk operations
- The N+1 query problem can silently degrade performance if loading strategies are not configured carefully
- Adds a layer of abstraction that can obscure what is actually happening at the database level
- Legacy schemas with unconventional structures may not map cleanly to ORM models
- Learning curve for teams unfamiliar with ORM concepts and configuration

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A manufacturing company had a legacy inventory system with over 2,000 raw SQL queries written for Oracle. When the company decided to migrate to PostgreSQL to reduce licensing costs, the prospect of rewriting every query was daunting. The team introduced SQLAlchemy as an ORM layer, starting with the 50 most-used entity types. Over four months, they migrated 80% of queries to ORM-managed operations. The remaining 20%, which involved complex reporting and bulk operations, were kept as native SQL but centralized in a repository layer with database-dialect-aware query builders. The migration to PostgreSQL, originally estimated at two years, was completed in eight months.
