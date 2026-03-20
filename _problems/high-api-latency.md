---
title: High API Latency
description: The time it takes for an API to respond to a request is excessively long,
  leading to poor application performance and a negative user experience.
category:
- Performance
related_problems:
- slug: network-latency
  similarity: 0.8
- slug: external-service-delays
  similarity: 0.75
- slug: slow-application-performance
  similarity: 0.7
- slug: slow-response-times-for-lists
  similarity: 0.6
- slug: high-resource-utilization-on-client
  similarity: 0.6
- slug: excessive-disk-io
  similarity: 0.6
solutions:
- api-first-design
- caching-strategy
- contract-testing
- serialization-optimization
- api-calls-optimization
- api-gateway
- api-security
- load-balancing
- optimistic-ui-updates
- predictive-loading
- predictive-prefetching
- rate-limiting
layout: problem
---

## Description
High API latency is a common problem in distributed systems, where services often depend on each other to fulfill requests. When an API takes a long time to respond, it can have a cascading effect, causing delays in downstream services and a poor user experience. High API latency can be caused by a variety of factors, from inefficient code and slow database queries to network issues and a lack of proper caching. A systematic approach to performance analysis is required to identify and address the root causes of high API latency.

## Indicators ⟡
- Your application is slow, but your servers are not under heavy load.
- You see a high number of timeout errors in your logs.
- Your application's performance is inconsistent.
- You are getting complaints from users about slow performance.

## Symptoms ▲

- [Slow Application Performance](slow-application-performance.md)
<br/>  User-facing features that depend on API calls feel sluggish when the underlying API responses are slow.
- [Service Timeouts](service-timeouts.md)
<br/>  Downstream services that call the slow API exceed their timeout thresholds, causing cascading failures.
- [Negative User Feedback](negative-user-feedback.md)
<br/>  Users directly experience slow page loads and unresponsive features caused by high API response times.
- [User Frustration](user-frustration.md)
<br/>  Consistently slow API responses lead to poor user experience and growing dissatisfaction.
- [Cascade Failures](cascade-failures.md)
<br/>  In distributed systems, high latency in one API cascades to all dependent services, causing widespread slowdowns.
## Causes ▼

- [Database Query Performance Issues](database-query-performance-issues.md)
<br/>  Slow database queries are a primary contributor to API latency, especially for data-heavy endpoints.
- [N+1 Query Problem](n-plus-one-query-problem.md)
<br/>  APIs that trigger N+1 database queries for related data multiply database round-trips and dramatically increase response times.
- [External Service Delays](external-service-delays.md)
<br/>  APIs that depend on slow external services inherit those delays in their own response times.
- [Poor Caching Strategy](poor-caching-strategy.md)
<br/>  Fetching data from the source on every request instead of caching adds unnecessary overhead to API response times.
- [Network Latency](network-latency.md)
<br/>  Network transmission delays between API components and data sources directly increase API response times.
## Detection Methods ○

- **Application Performance Monitoring (APM):** Use APM tools to trace requests, measure the duration of each operation (e.g., database calls, external service calls), and pinpoint the exact source of the delay.
- **Logging:** Add detailed logging to track the time taken at different stages of the request lifecycle.
- **Metrics and Alerting:** Monitor key metrics like p95/p99 response times and set up alerts to be notified of performance degradations.
- **Load Testing:** Use load testing tools to simulate traffic and identify how latency is affected by concurrent users.

## Examples
An e-commerce site's "product details" API endpoint becomes progressively slower as the number of products grows. Investigation with an APM tool reveals that the endpoint is making a slow, unindexed query to fetch product reviews. In another case, a mobile application's startup time is poor because it makes multiple blocking API calls to fetch initial configuration data. The latency of these calls, especially on slower mobile networks, adds up significantly. This is a common problem in distributed systems and microservices architectures where a single user action can trigger a chain of API calls across multiple services.
