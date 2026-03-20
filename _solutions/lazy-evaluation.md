---
title: Lazy Evaluation
description: Load and process data only when needed
category:
- Performance
- Code
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/lazy-evaluation
problems:
- slow-application-performance
- excessive-object-allocation
- high-client-side-resource-consumption
- memory-leaks
- gradual-performance-degradation
- lazy-loading
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify eagerly loaded data that is frequently unused: preloaded collections, joined relationships, computed fields
- Replace eager initialization with lazy proxies or supplier patterns that defer computation until first access
- Implement lazy loading for ORM relationships that are not always needed in every use case
- Use generators or streams instead of materializing entire collections into memory for processing
- Apply pagination and virtual scrolling on the frontend rather than loading entire datasets
- Be cautious of the N+1 problem: use batch-fetching or explicit eager loading where lazy loading causes excessive queries
- Profile to verify that lazy evaluation actually improves performance in each specific case

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces startup time and memory consumption by deferring work until it is actually needed
- Eliminates computation and data loading for code paths that are never executed
- Improves perceived performance by spreading out initialization costs over time
- Enables working with datasets larger than available memory through streaming

**Costs and Risks:**
- Can shift latency to unexpected moments, causing user-visible delays on first access
- Lazy-loaded ORM relationships can trigger N+1 query problems if not carefully managed
- Debugging becomes harder because initialization happens at unpredictable times
- Thread safety of lazy initialization requires careful implementation in concurrent environments

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy content management system eagerly loaded all metadata, related documents, and access control lists for every document when a folder listing was displayed. A folder with 200 documents triggered over 1,000 database queries and loaded several hundred megabytes of data into memory, even though users only interacted with a few documents at a time. The team changed the folder listing to load only document titles and dates, with metadata and relationships loaded lazily when a user clicked on a specific document. Folder listing response time dropped from 8 seconds to 300 milliseconds, and server memory usage during folder browsing decreased by over 80%.
