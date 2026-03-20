---
title: Approximation Methods
description: Use of heuristics and estimations instead of exact calculations
category:
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/approximation-methods
problems:
- algorithmic-complexity-problems
- slow-application-performance
- gradual-performance-degradation
- slow-database-queries
- high-database-resource-utilization
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify computations where approximate results are acceptable: analytics dashboards, search relevance, recommendation engines
- Replace exact counting with probabilistic data structures like HyperLogLog for cardinality estimation
- Use sampling techniques for large dataset analytics rather than processing every record
- Implement Bloom filters for membership tests where false positives are tolerable
- Replace exact distance calculations with bounding box checks or spatial hashing for geospatial queries
- Set acceptable error margins with stakeholders before implementing approximations
- Monitor approximation accuracy in production to ensure it stays within acceptable bounds

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically reduces computation time for operations that would otherwise be prohibitively expensive
- Enables real-time responses for queries that exact methods cannot answer quickly enough
- Reduces memory and storage requirements compared to maintaining exact data structures
- Allows systems to scale to data volumes that exact approaches cannot handle

**Costs and Risks:**
- Results are inherently imprecise, which may not be acceptable for financial or regulatory reporting
- Error bounds must be understood and communicated to consumers of the data
- Debugging issues caused by approximation errors can be subtle and difficult
- Some approximation techniques require specialized knowledge to implement correctly

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy analytics platform for a media company computed exact unique visitor counts by maintaining large hash sets in memory for each content item. As the site grew, memory consumption became unsustainable and query times degraded. The team replaced exact counting with HyperLogLog, which reduced memory usage per counter from megabytes to a few kilobytes while maintaining accuracy within 2%. The dashboard response time improved from 30 seconds to under one second, and stakeholders confirmed that the slight imprecision was acceptable for editorial decision-making.
