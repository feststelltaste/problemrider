---
title: API Calls Optimization
description: Designing API calls efficiently
category:
- Performance
- Architecture
problems:
- high-api-latency
- high-number-of-database-queries
- n-plus-one-query-problem
- slow-application-performance
- high-client-side-resource-consumption
- rest-api-design-issues
- network-latency
layout: solution
related_solutions:
- slug: image-and-asset-optimization
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.7
- slug: performance-optimization
  similarity: 0.7
- slug: pagination
  similarity: 0.7
- slug: api-first-design
  similarity: 0.7
- slug: connection-pooling
  similarity: 0.7
---

## Description

API calls optimization is the practice of reducing the number, size, and latency cost of the network requests a client must make to accomplish a task, typically by consolidating chatty sequences of fine-grained calls into fewer, coarser-grained ones, adding pagination and field selection, and batching related operations into a single round trip. Legacy APIs frequently exhibit chatty designs because each endpoint was added independently over time to satisfy a specific screen or integration need, without anyone stepping back to consider how many round trips a typical client workflow actually requires; the result is pages that issue a dozen or more sequential calls to render, each adding its own network latency on top of the others. This problem compounds on high-latency or bandwidth-constrained connections, such as mobile networks, where every additional round trip is felt directly by the end user as slower page loads and higher server load from processing many small requests instead of fewer larger ones. Optimizing these calls means analyzing actual usage patterns to find the highest-impact consolidation opportunities, then redesigning the API surface — without necessarily touching the underlying legacy business logic — so that a client can retrieve what it needs in one or two calls instead of many. The approach directly improves response time, throughput, and bandwidth consumption, but coarser-grained endpoints are inherently less flexible and can become complex to maintain, so the redesign requires close coordination between the teams that consume and produce the API to avoid simply moving the chattiness problem into an overly rigid contract.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce platform's product detail page required 12 separate API calls to load: one for product data, one for pricing, one for inventory, one for reviews, and several more for recommendations and related products. Each call added network latency, and on mobile connections the page took over eight seconds to render. The team consolidated these into two calls: a primary product endpoint that included pricing, inventory, and basic review summary, and a secondary endpoint for recommendations that loaded asynchronously. Page load time dropped to under two seconds, and backend server CPU utilization decreased by roughly 30% due to fewer requests to process.
