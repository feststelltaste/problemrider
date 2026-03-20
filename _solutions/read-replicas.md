---
title: Read Replicas
description: Distributing query load across read-only database replicas away from the primary
category:
- Database
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/read-replicas
problems:
- slow-database-queries
- high-database-resource-utilization
- scaling-inefficiencies
- database-query-performance-issues
- bottleneck-formation
- single-points-of-failure
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Analyze the read/write ratio of database traffic to determine how much load can be offloaded to replicas
- Set up one or more read replicas using the database engine's built-in replication features
- Modify the data access layer to route read queries to replicas and write queries to the primary
- Use a connection proxy or middleware to handle read/write splitting transparently if the legacy application cannot be easily modified
- Account for replication lag in application logic, ensuring that operations requiring read-your-writes consistency use the primary
- Monitor replication lag and replica health continuously, with alerts for unacceptable delays
- Start with reporting and analytics queries on replicas before moving transactional read traffic

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces load on the primary database, improving write performance and overall stability
- Provides a scalable path for read-heavy workloads without application redesign
- Read replicas can serve as warm standbys for disaster recovery
- Enables running expensive reports and analytics without impacting production performance

**Costs and Risks:**
- Replication lag means replicas may serve slightly stale data
- Legacy applications with tightly coupled read-after-write patterns require careful refactoring
- Each replica adds infrastructure and operational costs
- Failover logic between primary and replicas adds complexity
- Schema changes must be coordinated across primary and all replicas

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce platform's single PostgreSQL instance served both transactional traffic and business intelligence queries. During sales events, analytics queries from the BI team caused lock contention that slowed checkout operations. The team provisioned two read replicas: one dedicated to the BI tools and another for the product catalog's read-heavy API endpoints. A connection proxy transparently routed queries based on the originating application. This reduced primary database CPU utilization by 60 percent during peak events and completely eliminated the interference between analytics and transaction processing.
