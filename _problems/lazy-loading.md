---
title: Lazy Loading
description: The use of lazy loading in an ORM framework leads to a large number of
  unnecessary database queries, which can significantly degrade application performance.
category:
- Code
- Database
- Performance
related_problems:
- slug: imperative-data-fetching-logic
  similarity: 0.75
- slug: high-number-of-database-queries
  similarity: 0.7
- slug: n-plus-one-query-problem
  similarity: 0.7
- slug: slow-database-queries
  similarity: 0.7
- slug: inefficient-database-indexing
  similarity: 0.7
- slug: incorrect-index-type
  similarity: 0.65
solutions:
- caching-strategy
- efficient-algorithms
- lazy-evaluation
layout: problem
---

## Description
Lazy loading is a design pattern that is used to defer the initialization of an object until it is actually needed. This can be a useful pattern in some cases, but it can also lead to performance problems. In the context of an object-relational mapping (ORM) framework, lazy loading can lead to the N+1 query problem. This is because the ORM will execute a separate query for each object that is lazily loaded. This can result in a large number of unnecessary database queries, which can significantly degrade application performance.

## Indicators ⟡
- The application is making a large number of small, fast queries instead of a few larger, slower queries.
- The application is slow, even though the database server is not under heavy load.
- The database logs are full of similar-looking queries.
- The application is using an ORM framework, and you are not sure if it is configured correctly.

## Symptoms ▲

- [N+1 Query Problem](n-plus-one-query-problem.md)
<br/>  Lazy loading directly causes the N+1 query pattern where each lazily loaded relationship triggers a separate database query.
- [High Number of Database Queries](high-number-of-database-queries.md)
<br/>  Each lazily loaded association generates additional queries, multiplying the total number of database calls per request.
- [Slow Database Queries](slow-database-queries.md)
<br/>  The cumulative effect of many lazy-loaded queries degrades overall database performance and response times.
- [Slow Application Performance](slow-application-performance.md)
<br/>  The excessive number of database round-trips caused by lazy loading makes the application feel sluggish to users.
- [High Database Resource Utilization](high-database-resource-utilization.md)
<br/>  The flood of small queries from lazy loading consumes excessive database CPU and connection resources.
## Causes ▼

- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers unfamiliar with ORM behavior may use default lazy loading settings without understanding the performance implications.
- [Convenience-Driven Development](convenience-driven-development.md)
<br/>  Lazy loading is often the default and most convenient ORM option, and developers choose it without evaluating the performance trade-offs.
## Detection Methods ○
- **Application Performance Monitoring (APM):** APM tools can often detect and flag the N+1 query problem, which is a common symptom of lazy loading.
- **SQL Logging:** Enable SQL logging in your application or database and inspect the logs for a large number of similar-looking queries.
- **Code Review:** During code reviews, specifically look for code that is using lazy loading.
- **ORM Profiling:** Some ORM frameworks provide tools for profiling the performance of your queries.

## Examples
A web application is using an ORM framework to fetch data from the database. The application has a page that displays a list of users and their posts. The application is using lazy loading to fetch the posts for each user. This means that the application first executes a query to get the list of users. Then, for each user, it executes another query to get their posts. This results in a large number of unnecessary database queries, which makes the page slow to load. The problem could be solved by using eager loading to fetch the users and their posts in a single query.
