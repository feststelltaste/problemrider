---
title: Data Aggregation
description: Summarize fine-grained data into more compact units
category:
- Performance
- Database
problems:
- slow-database-queries
- unbounded-data-growth
- high-database-resource-utilization
- gradual-performance-degradation
- database-query-performance-issues
layout: solution
related_solutions:
- slug: materialized-views
  similarity: 0.75
- slug: data-partitioning
  similarity: 0.7
- slug: data-archiving
  similarity: 0.7
- slug: denormalization
  similarity: 0.7
- slug: data-replication
  similarity: 0.65
- slug: sampling
  similarity: 0.65
---

## Description

Data aggregation summarizes fine-grained records into compact, pre-computed units — hourly, daily, or monthly totals, averages, or counts — so that queries needing an overview no longer have to scan and recompute over the full volume of raw data each time they run. The mechanism separates the concern of collecting detailed data from the concern of answering summary questions about it: detailed rows keep accumulating at the source, while a separate aggregation process periodically or incrementally rolls them up into secondary tables or materialized views tailored to known reporting needs. In legacy systems, this matters because years of uncontrolled data growth combined with reporting queries that were designed for a much smaller dataset routinely turn dashboards and periodic reports into multi-minute operations that compete with transactional workloads for the same database resources. Aggregation lets a legacy system keep its detailed history intact for audit and drill-down purposes while giving reporting consumers a dataset that is orders of magnitude smaller and cheaper to query. Because the aggregates are derived rather than authoritative, they also provide a natural boundary at which fine-grained data can later be archived or discarded without affecting the summaries that most consumers actually rely on.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify queries that scan large volumes of fine-grained data to produce summary results
- Create pre-aggregated tables or materialized views for common reporting time periods (hourly, daily, monthly)
- Implement incremental aggregation that processes only new data rather than recalculating from scratch
- Schedule aggregation jobs during off-peak hours to minimize impact on transactional workloads
- Define retention policies: keep fine-grained data for a limited period and aggregated data for longer
- Use the aggregated data for dashboards and reports while preserving detailed data for drill-down when needed

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically improves query performance for summary and reporting use cases
- Reduces the volume of data that must be scanned for analytics queries
- Enables fast dashboard rendering even over large historical datasets
- Reduces storage growth when combined with archival of fine-grained data

**Costs and Risks:**
- Aggregated data loses detail, making ad-hoc investigations of individual records harder
- Aggregation logic must be maintained and kept consistent with the source data model
- Errors in aggregation can go undetected and produce misleading reports
- Changing aggregation dimensions after the fact requires reprocessing historical data

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy IoT monitoring platform stored individual sensor readings every second, accumulating billions of rows per month. Dashboard queries that computed hourly averages over the past year took over two minutes, making the application nearly unusable for operations teams. The team introduced an aggregation pipeline that computed hourly and daily summaries as new data arrived. Dashboard queries now read from the aggregated tables, returning results in under one second. The raw second-level data was retained for 90 days for detailed troubleshooting, while aggregated data was kept indefinitely, reducing the active dataset size by over 95%.
