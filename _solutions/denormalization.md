---
title: Denormalization
description: Introducing controlled redundancy in database schemas for faster reads
category:
- Database
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/denormalization
problems:
- slow-database-queries
- database-query-performance-issues
- high-number-of-database-queries
- n-plus-one-query-problem
- slow-response-times-for-lists
layout: solution
---

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
