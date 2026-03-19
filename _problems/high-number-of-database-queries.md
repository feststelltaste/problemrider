---
title: High Number of Database Queries
description: A single user request triggers an unexpectedly large number of database
  queries, leading to performance degradation and increased database load.
category:
- Database
- Performance
related_problems:
- slug: slow-database-queries
  similarity: 0.75
- slug: database-query-performance-issues
  similarity: 0.75
- slug: n-plus-one-query-problem
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.7
- slug: high-database-resource-utilization
  similarity: 0.7
- slug: imperative-data-fetching-logic
  similarity: 0.7
layout: problem
---

## Description
A high number of database queries is a common performance problem in database-driven applications. It occurs when a single user request triggers an unexpectedly large number of database queries. This can happen for a variety of reasons, such as the N+1 query problem, a lack of caching, or a poorly designed data access layer. A high number of database queries can lead to a number of problems, including slow application performance, high database resource utilization, and a poor user experience.

## Indicators ⟡
- The application is slow, even though the database server is not under heavy load.
- The database logs are full of similar-looking queries.
- The application is making a lot of small, fast queries instead of a few larger, slower queries.
- The application is not using a caching layer.

## Symptoms ▲

- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  A large number of queries per request increases CPU, memory, and I/O load on the database server.
- [Slow Application Performance](slow-application-performance.md)
<br/>  The cumulative latency of many database round-trips per request directly slows down application response times.
- [High API Latency](high-api-latency.md)
<br/>  API endpoints that trigger excessive database queries experience increased response times due to accumulated query overhead.
## Causes ▼

- [N+1 Query Problem](n-plus-one-query-problem.md)
<br/>  Loading related entities one at a time in a loop generates an additional query for each item, multiplying the total query count.
- [Imperative Data Fetching Logic](imperative-data-fetching-logic.md)
<br/>  Fetching data in loops rather than using batch or declarative approaches generates excessive individual queries.
- [Lazy Loading](lazy-loading.md)
<br/>  Lazy-loaded relationships trigger additional queries when accessed, often unexpectedly multiplying query counts.
- [Poor Caching Strategy](poor-caching-strategy.md)
<br/>  Without caching, the same data is repeatedly fetched from the database instead of being served from memory.
## Detection Methods ○
- **Application Performance Monitoring (APM):** APM tools can often detect and flag a high number of database queries.
- **SQL Logging:** Enable SQL logging in your application or database and inspect the logs for a large number of queries being executed in a short period of time.
- **Code Review:** During code reviews, specifically look for code that is making a large number of database queries.
- **Load Testing:** Use load testing to see how the application behaves under heavy load.

## Examples
A web application has a page that displays a list of products. For each product, it also displays the name of the category that the product belongs to. The application first executes a query to get the list of products. Then, for each product, it executes another query to get the name of the category. This results in a large number of database queries, which makes the page slow to load. The problem could be solved by using a single query that joins the products and categories tables.
