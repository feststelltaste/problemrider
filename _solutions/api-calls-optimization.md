---
title: API Calls Optimization
description: Designing API calls efficiently
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/api-calls-optimization
problems:
- high-api-latency
- high-number-of-database-queries
- n-plus-one-query-problem
- slow-application-performance
- high-client-side-resource-consumption
- rest-api-design-issues
- network-latency
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify chatty API patterns where clients make multiple round trips for data that could be fetched in a single call
- Consolidate related endpoints into coarser-grained operations that return all needed data at once
- Implement pagination for endpoints returning large collections to avoid transferring unnecessary data
- Use field selection or sparse fieldsets so clients request only the data they need
- Replace sequential API calls with batch endpoints that process multiple operations in a single request
- Add response compression and use ETags or conditional requests to reduce redundant data transfer
- Profile API usage patterns to identify the most impactful optimization targets

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces network round trips, directly improving response times and throughput
- Lowers server load by consolidating multiple operations into fewer, more efficient calls
- Decreases bandwidth consumption, especially important for mobile clients on constrained networks
- Improves user experience through faster page loads and interactions

**Costs and Risks:**
- Coarser-grained APIs can become overly complex and harder to maintain
- Batch endpoints may increase individual request processing time even as they reduce total round trips
- Over-optimization can reduce API flexibility, making it harder for new consumers to use the API
- Requires coordination between frontend and backend teams to agree on optimal API contracts

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce platform's product detail page required 12 separate API calls to load: one for product data, one for pricing, one for inventory, one for reviews, and several more for recommendations and related products. Each call added network latency, and on mobile connections the page took over eight seconds to render. The team consolidated these into two calls: a primary product endpoint that included pricing, inventory, and basic review summary, and a secondary endpoint for recommendations that loaded asynchronously. Page load time dropped to under two seconds, and backend server CPU utilization decreased by roughly 30% due to fewer requests to process.
