---
title: Pagination
description: Loading large outputs of data into smaller, manageable chunks
category:
- Performance
- Code
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/pagination
problems:
- slow-response-times-for-lists
- high-client-side-resource-consumption
- slow-database-queries
- memory-leaks
- unbounded-data-growth
- high-number-of-database-queries
- slow-application-performance
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify endpoints and screens that return unbounded result sets, particularly those that have grown beyond their original expected size
- Choose a pagination strategy: offset-based for simplicity, cursor-based for consistency with large or frequently changing datasets
- Add LIMIT and OFFSET (or keyset-based) clauses to the underlying database queries
- Implement pagination parameters in the API layer with sensible defaults and maximum page sizes
- Update the frontend to display paging controls or infinite scroll with progressive loading
- Ensure sort order is deterministic to prevent items from appearing on multiple pages or being skipped
- Retrofit existing API consumers gradually by making pagination optional with a default limit

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Prevents memory exhaustion on both server and client when datasets grow beyond original expectations
- Reduces database query time and network transfer size
- Improves perceived performance by displaying initial results quickly
- Limits blast radius of slow queries to smaller result sets

**Costs and Risks:**
- Offset-based pagination becomes slow at high offsets on large tables
- Adding pagination to existing APIs can break consumers that expect complete result sets
- Cursor-based pagination is more complex to implement and requires stable sort keys
- Users may not find specific items easily if search or filtering is not also improved
- Legacy reports that depend on processing all records at once require rework

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A customer support application originally designed for a few hundred tickets per agent now served teams managing tens of thousands of open cases. The ticket list endpoint returned all tickets in a single response, causing browser tabs to crash and API timeouts during peak hours. The team added cursor-based pagination using the ticket creation timestamp as the cursor, with a default page size of 50. Combined with server-side filtering, this reduced the average API response time from 12 seconds to 200 milliseconds and eliminated the browser memory issues entirely.
