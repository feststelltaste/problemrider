---
title: Lazy Loading
description: Delayed loading of data and resources until the moment of actual use
category:
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/lazy-loading
problems:
- slow-application-performance
- high-client-side-resource-consumption
- memory-leaks
- slow-response-times-for-lists
- excessive-object-allocation
- gradual-performance-degradation
- high-resource-utilization-on-client
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Profile the application to identify resources loaded eagerly that are rarely or never used in typical user flows
- Replace eager initialization of heavyweight objects with lazy proxies or factory methods that defer creation
- Implement lazy loading for UI components by splitting bundles and loading them on demand
- Convert database queries that fetch entire object graphs into queries that load associations only when accessed
- Use framework-specific lazy loading features (e.g., ORM lazy associations, React.lazy, dynamic imports) where available
- Add monitoring to track actual resource usage patterns and validate that deferred resources are loaded when genuinely needed
- Ensure error handling covers cases where lazy-loaded resources become unavailable at the time of actual use

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces initial load time and memory footprint, improving perceived performance
- Lowers resource consumption for features that users may never access in a given session
- Allows legacy systems to handle larger datasets without requiring infrastructure upgrades
- Improves startup time for monolithic applications with many subsystems

**Costs and Risks:**
- Introduces latency at the point of first access, which can surprise users if not handled with loading indicators
- Adds complexity to initialization logic and can create hard-to-debug ordering issues
- May cause N+1 query problems in ORMs if lazy associations are accessed in loops
- Complicates testing because behavior depends on when resources are actually loaded

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy enterprise resource planning system loaded all reference data tables into memory at startup, causing a 45-second boot time and consuming over 2 GB of RAM. By converting the reference data loaders to lazy initialization, the team reduced startup time to under 8 seconds and cut baseline memory usage in half. Rarely accessed modules such as archival reporting and audit history were loaded only when users navigated to those sections, which also reduced the blast radius of bugs in those subsystems during normal operations.
