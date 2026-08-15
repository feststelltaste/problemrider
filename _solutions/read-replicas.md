---
title: Read Replicas
description: Distributing query load across read-only database replicas away from
  the primary
category:
- Database
- Performance
problems:
- slow-database-queries
- high-database-resource-utilization
- scaling-inefficiencies
- database-query-performance-issues
- bottleneck-formation
- single-points-of-failure
- lock-contention
layout: solution
related_solutions:
- slug: data-replication
  similarity: 0.9
- slug: denormalization
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: data-partitioning
  similarity: 0.75
- slug: materialized-views
  similarity: 0.75
- slug: load-balancing
  similarity: 0.75
---

## Description

Read replicas are read-only copies of a primary database, kept current through the database engine's built-in replication, to which an application's read queries are routed while writes continue to go to the primary — either through changes to the data access layer or transparently via a connection proxy for legacy applications that cannot easily be modified. This is a common and comparatively low-disruption way to scale a legacy system's database tier because it requires no change to the schema or the fundamental data model, only a routing decision about which queries go where, which makes it feasible even for systems whose core logic is too risky or poorly understood to refactor directly. It is particularly effective in legacy systems that grew a single database instance to serve both transactional application traffic and heavier analytical or reporting queries side by side, since those reporting queries are exactly the kind of read-heavy, latency-tolerant workload that can be moved off the primary entirely, eliminating the lock contention they cause against transactional writes. The unavoidable consequence of asynchronous replication is replication lag, meaning replicas may briefly serve stale data, so any legacy workflow that depends on immediately reading its own just-written value must be identified and explicitly routed to the primary rather than a replica. Beyond that consistency caveat, each replica adds ongoing infrastructure and operational cost, and schema changes must now be coordinated and applied consistently across the primary and every replica rather than a single database instance.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce platform's single PostgreSQL instance served both transactional traffic and business intelligence queries. During sales events, analytics queries from the BI team caused lock contention that slowed checkout operations. The team provisioned two read replicas: one dedicated to the BI tools and another for the product catalog's read-heavy API endpoints. A connection proxy transparently routed queries based on the originating application. This reduced primary database CPU utilization by 60 percent during peak events and completely eliminated the interference between analytics and transaction processing.
