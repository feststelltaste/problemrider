---
title: Sampling
description: Using a representative subset of data for analysis or testing
category:
- Performance
- Testing
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/sampling
problems:
- unbounded-data-growth
- slow-database-queries
- high-database-resource-utilization
- slow-application-performance
- inadequate-test-data-management
- excessive-logging
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy monitoring system collected and stored every single request trace, consuming 2 TB of storage daily and making trace search prohibitively slow. The team implemented adaptive sampling that captured 100 percent of error traces and 1 percent of successful traces, with stratified sampling ensuring that every endpoint was represented regardless of traffic volume. This reduced storage to 50 GB per day and made trace search responsive, while the 100 percent error capture ensured that no debugging-critical data was lost. Monthly statistical comparisons confirmed that the sampled latency distributions remained within 2 percent of the true values.
