---
title: Denormalization
description: Introducing controlled redundancy in database schemas for faster reads
category:
- Database
- Performance
problems:
- slow-database-queries
- database-query-performance-issues
- high-number-of-database-queries
- n-plus-one-query-problem
- slow-response-times-for-lists
- lazy-loading
layout: solution
related_solutions:
- slug: materialized-views
  similarity: 0.85
- slug: data-replication
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: read-replicas
  similarity: 0.8
- slug: data-partitioning
  similarity: 0.8
- slug: nosql-databases
  similarity: 0.75
---

## Description

Denormalization deliberately introduces controlled redundancy into a database schema — duplicating or pre-computing values that would otherwise require an expensive join or aggregation at query time — trading write-time complexity and extra storage for dramatically faster reads. In practice, this means adding computed or cached columns for frequently needed derived values directly onto the tables consumers query, or maintaining separate summary tables, and then keeping those denormalized values synchronized with their source of truth through triggers, application-level hooks, or event handlers rather than trusting them to stay correct on their own. Legacy systems accumulate exactly the conditions that make this worthwhile: schemas normalized decades ago for data integrity reasons now serve read-heavy access patterns that require joining across many tables just to render a single page, and under real production load those joins routinely turn what should be a fast lookup into a multi-second query. Applying denormalization selectively, starting with read-heavy and write-light areas, lets a team eliminate the worst-offending queries without restructuring the schema wholesale, while documenting which source tables remain authoritative keeps the redundancy from turning into an unmanageable web of competing "truths." Because every denormalized value can drift from its source over time due to a missed update path or a bug in a synchronization hook, the pattern requires ongoing reconciliation checks to catch and correct inconsistencies before they are mistaken for authoritative data.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify the most expensive queries that join multiple tables and analyze whether pre-joining data would eliminate the bottleneck
- Add computed or cached columns that store frequently needed derived values (e.g., order totals, display names)
- Create summary tables that duplicate aggregated data for fast retrieval
- Implement triggers, application-level hooks, or event handlers to keep denormalized data synchronized with source data
- Document every denormalization decision including which source tables are authoritative
- Start with read-heavy, write-light areas where the synchronization overhead is minimal
- Monitor for data inconsistencies between normalized and denormalized copies

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates expensive joins and aggregations at query time by pre-computing results
- Dramatically improves read performance for complex queries
- Reduces database load by avoiding repeated computation of the same derived data
- Can be applied selectively without restructuring the entire schema

**Costs and Risks:**
- Introduces data redundancy that must be kept in sync, risking inconsistencies
- Write operations become more complex and potentially slower due to synchronization overhead
- Storage requirements increase due to duplicated data
- Schema complexity grows with additional columns and tables

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce platform had a product listing page that required joining seven tables to display each product with its category name, average rating, current price, and stock status. Under load, this query took over two seconds for a page of 50 products. The team added denormalized columns directly to the product table: `category_name`, `avg_rating`, `current_price`, and `stock_status`. Application-level event listeners updated these columns whenever the source data changed. The product listing query became a single-table scan that returned in under 50 milliseconds. The team added a nightly reconciliation job to detect and correct any drift between the denormalized columns and their source tables.
