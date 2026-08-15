---
title: Lazy Loading
description: Delayed loading of data and resources until the moment of actual use
category:
- Performance
problems:
- slow-application-performance
- high-client-side-resource-consumption
- memory-leaks
- slow-response-times-for-lists
- excessive-object-allocation
- gradual-performance-degradation
- high-resource-utilization-on-client
- inefficient-frontend-code
layout: solution
related_solutions:
- slug: lazy-evaluation
  similarity: 0.95
- slug: predictive-loading
  similarity: 0.8
- slug: progressive-loading
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: connection-pooling
  similarity: 0.8
- slug: code-splitting
  similarity: 0.8
---

## Description

Lazy loading delays the retrieval or initialization of a resource — a UI component, a database association, an image, or a data page — until the point where it is genuinely required by the user's current interaction, instead of fetching everything a screen or object might ever need at construction time. It is implemented through mechanisms such as bundle splitting and dynamic imports on the frontend, lazy-initialized associations in an ORM, or virtual scrolling and pagination for large lists, all of which share the same underlying idea of substituting a deferred reference for immediate materialization. Legacy applications commonly grew their eager-loading habits organically: a screen that once showed a handful of records now renders thousands, or a startup routine that once initialized a few modules now boots dozens of subsystems nobody remembers is unused, and because nothing forced a reconsideration of that loading strategy, resource consumption crept upward year after year. Introducing lazy loading into such a system directly targets the slow startup times, bloated memory footprints, and sluggish list rendering that accumulate this way, without requiring the surrounding legacy code to be rewritten — the loading boundary can usually be inserted at the point of access rather than throughout the codebase. Because the deferred cost still has to be paid eventually, often at an unpredictable moment visible to the end user, lazy loading in legacy contexts needs to be paired with clear loading indicators and safeguards against patterns like N+1 queries, where a lazy association accessed inside a loop silently multiplies the number of deferred fetches.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy enterprise resource planning system loaded all reference data tables into memory at startup, causing a 45-second boot time and consuming over 2 GB of RAM. By converting the reference data loaders to lazy initialization, the team reduced startup time to under 8 seconds and cut baseline memory usage in half. Rarely accessed modules such as archival reporting and audit history were loaded only when users navigated to those sections, which also reduced the blast radius of bugs in those subsystems during normal operations.
