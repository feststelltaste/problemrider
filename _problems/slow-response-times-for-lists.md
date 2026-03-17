---
title: Slow Response Times for Lists
description: Web pages or API endpoints that display lists of items are significantly
  slower to load than those that display single items, often due to inefficient data
  fetching.
category:
- Data
- Performance
related_problems:
- slug: slow-application-performance
  similarity: 0.65
- slug: high-api-latency
  similarity: 0.6
- slug: database-query-performance-issues
  similarity: 0.6
- slug: network-latency
  similarity: 0.6
- slug: slow-database-queries
  similarity: 0.6
- slug: external-service-delays
  similarity: 0.6
layout: problem
---

## Description
Slow response times for lists is a common performance problem in web applications. It occurs when a page or API endpoint that displays a list of items is significantly slower to load than one that displays a single item. This is often a sign of an inefficient data fetching strategy, such as the N+1 query problem. Slow response times for lists can have a major impact on the user experience, and they can be a major source of frustration for users.

## Indicators ⟡
- A page that displays a list of items takes a long time to load.
- The application is making a large number of database queries when it is loading a list of items.
- The application is not using pagination to limit the number of items that are displayed on a single page.
- The application is not using a caching layer.

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  Slow list pages are a visible component of overall application sluggishness perceived by users.
- [Poor User Experience (UX) Design](poor-user-experience-ux-design.md)
<br/>  Users frequently interact with list views, so their slowness directly degrades the overall user experience.
- [High API Latency](high-api-latency.md)
<br/>  List endpoints that fetch large amounts of data contribute significantly to API latency.
## Causes ▼

- [High Number of Database Queries](high-number-of-database-queries.md)
<br/>  N+1 query patterns cause list pages to execute one query per item, dramatically increasing load times.
- [Slow Database Queries](slow-database-queries.md)
<br/>  Inefficient queries that scan large tables without proper indexes are especially impactful when loading lists.
- [Lazy Loading](lazy-loading.md)
<br/>  Lazy loading related data for each item in a list triggers many additional queries, multiplying response time.
- [Imperative Data Fetching Logic](imperative-data-fetching-logic.md)
<br/>  Manually coded data fetching often fails to batch or optimize queries for list operations.
## Detection Methods ○
- **Application Performance Monitoring (APM):** APM tools can often detect and flag slow response times for lists.
- **Browser Developer Tools:** Use the browser developer tools to see how long it takes to load a page.
- **Load Testing:** Use load testing to see how the application behaves under heavy load.
- **Code Review:** During code reviews, specifically look for code that is fetching a list of items from the database.

## Examples
A web application has a page that displays a list of products. The page is very slow to load. The reason for this is that the application is not using pagination, and it is trying to load all of the products in the database at once. The problem could be solved by using pagination to limit the number of products that are displayed on a single page.
