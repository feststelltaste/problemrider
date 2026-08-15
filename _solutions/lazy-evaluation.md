---
title: Lazy Evaluation
description: Load and process data only when needed
category:
- Performance
- Code
problems:
- slow-application-performance
- excessive-object-allocation
- high-client-side-resource-consumption
- memory-leaks
- gradual-performance-degradation
- lazy-loading
layout: solution
related_solutions:
- slug: lazy-loading
  similarity: 0.95
- slug: predictive-loading
  similarity: 0.8
- slug: progressive-loading
  similarity: 0.75
- slug: distributed-caching
  similarity: 0.75
- slug: connection-pooling
  similarity: 0.75
- slug: parallelization
  similarity: 0.75
---

## Description

Lazy evaluation defers the computation or loading of a value until the moment it is actually needed, rather than computing it eagerly as soon as it is declared or constructed. The mechanism typically takes the form of a proxy, a supplier or thunk, a generator, or a lazily-initialized ORM association that intercepts the first access and only then performs the expensive work — database query, object construction, or computation — caching or discarding the result afterward as appropriate. Legacy systems frequently default to eager initialization because it is simpler to reason about at write time: entire object graphs, collections, or configuration trees are loaded up front regardless of whether a given code path will ever use them, which becomes increasingly expensive as the system's data volume grows over the years while the eager-loading code itself is never revisited. Applying lazy evaluation to such code shifts cost from "always, whether needed or not" to "only when actually used," which is especially effective in legacy systems where a large fraction of preloaded data serves rarely exercised features or edge cases. The tradeoff that matters most in a legacy context is that lazy evaluation trades predictable, front-loaded latency for latency that appears unpredictably at first access, which can surface as new, hard-to-diagnose slowdowns unless the team accounts for it explicitly, particularly around the N+1 query problem in lazily-loaded ORM relationships.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy content management system eagerly loaded all metadata, related documents, and access control lists for every document when a folder listing was displayed. A folder with 200 documents triggered over 1,000 database queries and loaded several hundred megabytes of data into memory, even though users only interacted with a few documents at a time. The team changed the folder listing to load only document titles and dates, with metadata and relationships loaded lazily when a user clicked on a specific document. Folder listing response time dropped from 8 seconds to 300 milliseconds, and server memory usage during folder browsing decreased by over 80%.
