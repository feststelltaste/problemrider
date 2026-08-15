---
title: Sampling
description: Using a representative subset of data for analysis or testing
category:
- Performance
- Testing
problems:
- unbounded-data-growth
- slow-database-queries
- high-database-resource-utilization
- slow-application-performance
- inadequate-test-data-management
- excessive-logging
layout: solution
related_solutions:
- slug: distributed-processing
  similarity: 0.75
- slug: data-archiving
  similarity: 0.7
- slug: logging
  similarity: 0.7
- slug: data-replication
  similarity: 0.7
- slug: compression
  similarity: 0.7
- slug: data-partitioning
  similarity: 0.7
---

## Description

Sampling processes and analyzes a representative subset of data rather than the complete dataset, using a strategy — random, stratified, or reservoir sampling among others — chosen to match the statistical requirements of the task, and applied at the point of data collection so that only the necessary subset is ever gathered in the first place. This is particularly effective for workloads such as monitoring, trend analysis, and testing, where processing every single data point adds cost without adding proportional insight, and where a well-chosen sample yields conclusions statistically indistinguishable from analyzing the full dataset. Legacy systems frequently accumulate monitoring, logging, and tracing data at a volume that was never anticipated when the original collection mechanism was designed, and by the time this becomes a problem, the exhaustive collection habit is often too deeply embedded in the system's operational tooling to simply switch off; sampling offers a way to reduce that volume dramatically without abandoning observability altogether. It is especially useful when combined with stratification that guarantees full capture of the rarest and most important events — such as capturing 100 percent of error traces while sampling only a small fraction of successful ones — so that the exact cases most valuable for debugging are never the ones lost to the reduction. Because sampled results are approximations rather than exact figures, the methodology and its confidence intervals need to be documented and periodically validated against full-data analysis, so that consumers of the sampled data understand its limitations rather than mistaking it for a complete record.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify workloads where processing 100 percent of data is unnecessary: analytics, monitoring, trend detection, testing
- Choose an appropriate sampling strategy (random, stratified, reservoir) based on the statistical requirements
- Implement sampling at the data collection point rather than collecting everything and filtering later
- Use stratified sampling when different data segments have varying importance or variance
- Apply sampling to distributed tracing and logging to reduce storage costs while maintaining diagnostic capability
- Validate that sampled results remain statistically representative by periodically comparing against full-data analysis
- Document the sampling methodology and confidence intervals so consumers understand the data's limitations

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically reduces processing time, storage costs, and infrastructure requirements
- Makes real-time analysis feasible for datasets too large to process exhaustively
- Reduces log storage costs while retaining sufficient data for troubleshooting
- Enables faster testing cycles by working with manageable data subsets

**Costs and Risks:**
- Rare events may be missed if the sample size is too small or sampling is not stratified
- Results are approximate and may not satisfy audit or compliance requirements
- Incorrect sampling methodology can introduce systematic bias
- Teams may not understand the limitations of sampled data and treat it as exact
- Debugging specific production issues is harder when the relevant trace was not sampled

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy monitoring system collected and stored every single request trace, consuming 2 TB of storage daily and making trace search prohibitively slow. The team implemented adaptive sampling that captured 100 percent of error traces and 1 percent of successful traces, with stratified sampling ensuring that every endpoint was represented regardless of traffic volume. This reduced storage to 50 GB per day and made trace search responsive, while the 100 percent error capture ensured that no debugging-critical data was lost. Monthly statistical comparisons confirmed that the sampled latency distributions remained within 2 percent of the true values.
