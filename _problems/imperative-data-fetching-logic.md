---
title: Imperative Data Fetching Logic
description: The application code is written in a way that fetches data in a loop,
  rather than using a more efficient, declarative approach, leading to performance
  problems.
category:
- Architecture
- Database
- Performance
related_problems:
- slug: lazy-loading
  similarity: 0.75
- slug: high-number-of-database-queries
  similarity: 0.7
- slug: slow-database-queries
  similarity: 0.7
- slug: n-plus-one-query-problem
  similarity: 0.7
- slug: inefficient-code
  similarity: 0.65
- slug: inefficient-database-indexing
  similarity: 0.6
layout: problem
---

## Description
Imperative data fetching logic is a common performance problem in database-driven applications. It occurs when the application code is written in a way that fetches data in a loop, rather than using a more efficient, declarative approach. This can lead to a number of problems, including the N+1 query problem, slow application performance, and a high number of database queries. Imperative data fetching logic is often a sign of a lack of experience with declarative programming or a lack of a clear data fetching strategy.

## Indicators ⟡
- The application code contains loops that fetch data from the database.
- The application is making a large number of small, fast queries instead of a few larger, slower queries.
- The application is slow, even though the database server is not under heavy load.
- The database logs are full of similar-looking queries.

## Symptoms ▲

- [High Number of Database Queries](high-number-of-database-queries.md)
<br/>  Fetching data in loops generates individual queries for each iteration, resulting in an excessive number of database calls.
- [N+1 Query Problem](n-plus-one-query-problem.md)
<br/>  Imperative fetching patterns that load related data item-by-item are the direct implementation cause of N+1 query issues.
- [Slow Application Performance](slow-application-performance.md)
<br/>  The accumulated latency of many sequential database round-trips significantly degrades application response times.
- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  The excessive query volume from imperative fetching increases CPU and memory usage on the database server.
## Causes ▼

- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers unfamiliar with declarative data access patterns and ORM best practices default to imperative loop-based fetching.
- [Insufficient Code Review](insufficient-code-review.md)
<br/>  Without code review, inefficient data fetching patterns go undetected and become established in the codebase.
- [Inconsistent Coding Standards](inconsistent-coding-standards.md)
<br/>  Without established data access patterns and standards, developers implement fetching logic inconsistently and inefficiently.
## Detection Methods ○
- **Code Review:** During code reviews, specifically look for loops that contain database queries.
- **Application Performance Monitoring (APM):** APM tools can often detect and flag the N+1 query problem, which is a common symptom of imperative data fetching logic.
- **SQL Logging:** Enable SQL logging in your application or database and inspect the logs for a large number of similar-looking queries.
- **Static Analysis:** Use static analysis tools to identify loops that contain database queries.

## Examples
A web application has a page that displays a list of products and their prices. The application first executes a query to get the list of products. Then, for each product, it executes another query to get the price. This is an example of imperative data fetching logic. The problem could be solved by using a single query that joins the products and prices tables. This would be a more efficient and declarative way to fetch the data.
