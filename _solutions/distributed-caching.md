---
title: Distributed Caching
description: Caching frequently needed data on multiple computers
category:
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/distributed-caching
problems:
- slow-application-performance
- poor-caching-strategy
- cache-invalidation-problems
- high-database-resource-utilization
- scaling-inefficiencies
- slow-database-queries
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify data that is read frequently but changes infrequently: reference data, session state, computed results
- Deploy a distributed cache (Redis, Memcached, Hazelcast) accessible to all application instances
- Implement cache-aside pattern: check the cache before querying the database, populate it on cache miss
- Define appropriate TTL (time-to-live) values based on how stale the data can be for each use case
- Implement cache invalidation strategies that match the consistency requirements: event-driven, TTL-based, or write-through
- Monitor cache hit rates and eviction rates to tune cache size and TTL configurations
- Add circuit breakers so the application degrades gracefully if the cache becomes unavailable

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces database load by serving frequently accessed data from memory
- Provides consistent performance across multiple application instances through a shared cache
- Enables horizontal scaling of the application tier without proportionally increasing database load
- Dramatically improves response times for cached data

**Costs and Risks:**
- Cache invalidation is notoriously difficult and can lead to stale data being served
- Adds infrastructure dependency: cache failures can cascade to the database if not handled
- Memory costs for caching large datasets can be significant
- Distributed cache adds network latency compared to local in-process caches

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy travel booking platform had product catalog queries that hit the database for every search request, with the same destination data being fetched thousands of times per hour. As traffic grew, the database became a bottleneck during peak booking seasons. The team deployed a Redis cluster and implemented a cache-aside pattern for destination data, hotel details, and pricing tiers with a 15-minute TTL. Cache hit rates exceeded 95% for catalog queries, reducing database query volume by an order of magnitude. Peak-season performance issues disappeared, and the team was able to defer a costly database hardware upgrade.
