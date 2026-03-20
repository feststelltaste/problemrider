---
title: Caching
description: Caching frequently needed data
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/caching/
problems:
- poor-caching-strategy
- cache-invalidation-problems
- data-structure-cache-inefficiency
- network-latency
- external-service-delays
- high-api-latency
- excessive-disk-io
- unoptimized-file-access
- lazy-loading
- memory-swapping
- serialization-deserialization-bottlenecks
- microservice-communication-overhead
layout: solution
---

## How to Apply ◆

> Legacy systems typically evolved without a deliberate caching strategy. Data is fetched from databases, filesystems, or external services on every request because the original workload never demanded otherwise. Introducing caching requires understanding the system's data access patterns and adding cache layers where they deliver the most benefit with acceptable staleness risk.

- Profile the application to identify the highest-frequency, highest-cost data access operations. In legacy systems, these are often database queries for reference data (product catalogs, configuration tables, user roles) that change infrequently but are read on every request. Prioritize caching these operations first for the greatest return on effort.
- Introduce an application-level cache (such as Ehcache, Caffeine, or a simple in-memory dictionary) for data that is read far more often than it is written. In legacy systems where adding an external cache like Redis or Memcached requires infrastructure changes that may be blocked by organizational constraints, an in-process cache can be deployed with minimal disruption.
- Add HTTP caching headers (Cache-Control, ETag, Last-Modified) to API responses that serve relatively stable data. Many legacy systems omit these headers entirely, forcing clients and intermediary proxies to re-fetch unchanged data on every request. This is especially effective for reducing network latency for geographically distributed users.
- Cache responses from external services and third-party APIs that the legacy system depends on. External service delays are a common problem in aging systems that accumulated integrations over the years. Even short-lived caching (30-60 seconds) can dramatically reduce the impact of slow or unreliable external dependencies, preventing cascading failures.
- Implement a distributed cache (Redis, Memcached, or Hazelcast) when the legacy system runs on multiple application server instances. Without a shared cache, each instance maintains its own cache, leading to inconsistent behavior between nodes and wasted memory storing duplicate data. A distributed cache also survives application restarts, which is valuable for systems that require frequent redeployments during modernization.
- Replace repeated file reads with in-memory caches for configuration files, templates, and lookup data stored on the filesystem. Legacy systems often read the same files from disk on every request because the code predates buffered I/O abstractions or because developers were unaware of the performance cost. Caching file contents in memory eliminates excessive disk I/O for these access patterns.
- Cache serialized representations of frequently requested objects to avoid repeated serialization/deserialization overhead. In legacy systems that use verbose formats like XML or SOAP, caching the serialized payload rather than re-serializing from domain objects on each request can reclaim significant CPU and memory resources.
- Design cache keys carefully to balance granularity and hit rate. Keys that are too specific (including timestamps or user IDs for shared data) produce poor hit rates, while keys that are too broad serve stale or incorrect data. Audit existing data access patterns to determine which parameters genuinely vary the response.
- Implement explicit cache invalidation tied to data modification paths. In legacy systems, data is often modified through multiple entry points (admin interfaces, batch jobs, direct database updates, integration APIs), and missing even one invalidation path causes stale data. Map all write paths for cached data and add invalidation hooks to each one.
- Add cache monitoring from the start: track hit rates, miss rates, eviction counts, and cache size. In legacy systems without observability infrastructure, even simple log-based metrics help detect whether the cache is effective or has become a source of stale data. Set alerts for hit rate drops that indicate invalidation problems or access pattern changes.

## Tradeoffs ⇄

> Caching is one of the most effective performance optimizations for legacy systems, but it introduces a layer of state management that must be carefully controlled to avoid data consistency problems and operational complexity.

**Benefits:**

- Reduces database load by serving repeated queries from memory, often eliminating 60-90% of database round-trips in legacy systems where the same reference data is fetched on every request.
- Lowers API latency significantly by avoiding the cost of network calls, database queries, and serialization for data that has not changed since the last request.
- Absorbs the impact of slow or unreliable external service dependencies, allowing the application to continue serving cached data even when third-party services are degraded or temporarily unavailable.
- Reduces excessive disk I/O by keeping frequently accessed file data and configuration in memory, which is particularly valuable for legacy systems running on aging hardware with slow storage.
- Decreases serialization/deserialization overhead by caching pre-serialized payloads, reclaiming CPU and memory that would otherwise be spent repeatedly converting the same objects.
- Improves user experience without requiring changes to business logic or database schemas, making it one of the lowest-risk performance improvements available for legacy systems under modernization constraints.
- Mitigates the N+1 query problem and lazy loading overhead by caching the results of queries that ORM frameworks would otherwise execute repeatedly, reducing the number of database round-trips without refactoring the data access layer.

**Costs and Risks:**

- Stale data is the primary risk: cached data that is not properly invalidated causes users to see outdated information, which may lead to incorrect business decisions, security vulnerabilities, or data corruption in downstream processes.
- Cache invalidation is genuinely difficult in legacy systems where data is modified through many paths (batch jobs, admin tools, direct SQL, integration APIs) and no single code path controls all writes. Missing one invalidation path creates intermittent bugs that are extremely hard to reproduce.
- Memory consumption increases, which can trigger memory swapping on systems that are already near their physical memory limits. In legacy environments running on constrained hardware, an improperly sized cache can make performance worse rather than better.
- Adds operational complexity that legacy teams may not be prepared for: cache infrastructure must be monitored, configured per environment, and maintained alongside the existing system. Distributed caches introduce additional network dependencies and potential failure modes.
- Caching masks underlying performance problems rather than fixing them. Developers may accept a cache as a permanent solution and never address the root cause (inefficient queries, poor data models, excessive serialization), creating technical debt that compounds over time.
- Debugging becomes harder because application behavior depends on cache state. Issues that only manifest with specific cache contents or timing are difficult to reproduce in development environments where caches are frequently cleared.

## Examples

> The following scenarios illustrate how caching addresses performance problems commonly found in legacy systems.

A 15-year-old insurance claims processing system queries a reference table of policy types, coverage rules, and regulatory codes on every claim submission. The table contains 2,000 rows and changes only during quarterly regulatory updates, yet the system executes the same query 50,000 times per day. Each query takes 15ms including network round-trip to the database. The team introduces an in-process Caffeine cache with a 4-hour TTL and an event-driven invalidation hook triggered by the quarterly import job. Database queries for reference data drop to fewer than 10 per day, claim processing latency decreases by 35%, and database CPU utilization falls by 20%, freeing capacity for actual transactional queries. The total implementation effort is two days because the cache layer wraps the existing data access methods without modifying business logic.

A legacy order management system integrates with five external services for tax calculation, shipping rates, inventory verification, fraud screening, and payment processing. During peak periods, external service delays cause checkout times to exceed 8 seconds, with tax and shipping rate lookups contributing 3 seconds combined. The team adds a Redis cache for tax rates (keyed by jurisdiction and product category, 1-hour TTL) and shipping rates (keyed by origin, destination, and weight bracket, 30-minute TTL). Cache hit rates reach 85% for tax lookups and 70% for shipping, reducing average checkout time from 8 seconds to 3.5 seconds. When the tax service experiences a 20-minute outage during a holiday sale, the application continues serving cached tax rates without any customer-facing impact.

A monolithic Java ERP system generates PDF reports by reading XML templates from the filesystem, parsing them, and merging them with database query results. Each report reads and parses the same 50 template files, and during month-end reporting, the server processes 500 reports in a batch. Profiling reveals that 40% of batch processing time is spent on repeated file I/O and XML parsing of unchanged templates. The team caches parsed template objects in a WeakHashMap with file modification timestamps as invalidation keys. Batch processing time drops from 4 hours to 2.5 hours, disk I/O during reporting decreases by 75%, and the approach requires no changes to the template files or the report generation logic itself.
