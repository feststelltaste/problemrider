---
title: Prepared Statements
description: Use parameterized queries to prevent SQL injection
category:
- Security
- Database
quality_tactics_url: https://qualitytactics.de/en/security/prepared-statements
problems:
- sql-injection-vulnerabilities
- buffer-overflow-vulnerabilities
- inadequate-error-handling
- poor-documentation
- legacy-code-without-tests
- insufficient-testing
- insecure-data-transmission
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Audit the codebase for raw SQL string concatenation or interpolation patterns and catalog every query construction site
- Replace string-concatenated queries with parameterized prepared statements using your language's database driver API
- Introduce an ORM or query builder layer that enforces parameterized queries by default for new code paths
- Add static analysis rules to flag raw SQL concatenation in code reviews and CI pipelines
- Create reusable data access functions or repository classes that encapsulate prepared statement usage
- Update legacy stored procedures that dynamically build SQL with `EXEC` or `sp_executesql` to use proper parameterization
- Write integration tests that attempt SQL injection payloads to verify prepared statements are correctly applied

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates the most common and dangerous class of injection vulnerabilities
- Improves query plan caching and database performance through statement reuse
- Simplifies code by separating query structure from data values
- Reduces risk of silent data corruption caused by malformed input

**Costs and Risks:**
- Migrating large legacy codebases with thousands of raw queries requires significant effort
- Some dynamic queries (e.g., variable column names or table names) cannot be fully parameterized and need allow-listing
- Developers unfamiliar with prepared statements may introduce workarounds that bypass protections
- ORM adoption may introduce its own complexity and performance overhead in certain scenarios

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A financial services company discovered during a penetration test that its legacy PHP application contained over 300 instances of direct SQL string concatenation. The team systematically replaced these with PDO prepared statements over a six-week sprint, starting with the most critical payment processing queries. After migration, a follow-up penetration test confirmed zero SQL injection findings, and the database team observed a 15% improvement in query response times due to better query plan caching.
