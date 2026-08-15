---
title: Data Replication
description: Creating and synchronizing copies of data across multiple systems
category:
- Database
- Architecture
problems:
- single-points-of-failure
- system-outages
- cross-system-data-synchronization-problems
- slow-database-queries
- high-database-resource-utilization
- scaling-inefficiencies
layout: solution
related_solutions:
- slug: read-replicas
  similarity: 0.9
- slug: redundant-data-storage
  similarity: 0.85
- slug: distributed-caching
  similarity: 0.8
- slug: denormalization
  similarity: 0.8
- slug: distributed-processing
  similarity: 0.8
- slug: redundancy
  similarity: 0.8
---

## Description

Data replication creates and continuously synchronizes copies of a dataset across multiple systems or nodes, using synchronous or asynchronous mechanisms depending on how strict the required consistency is, so that the same data is available from more than one location for reading, failover, or geographic distribution. In practice this usually means designating a primary system of record and one or more replicas that receive a continuous stream of changes — either through native database replication or through change data capture that observes the primary without requiring modifications to it — with monitoring in place to detect replication lag or synchronization failures. For legacy systems, replication addresses two distinct pain points at once: a single database instance that has to serve both transactional and reporting workloads suffers lock contention and slowdowns when the two compete for the same resources, and a single database instance with no standby is also a single point of failure that turns any hardware issue into an extended outage. Directing read-heavy reporting traffic to replicas relieves the primary, while a geographically separated replica doubles as a disaster recovery target that can be promoted if the primary becomes unavailable. The tradeoff inherent to replication is that copies are not instantaneously consistent — replication lag can produce stale reads, and any configuration allowing writes to more than one copy introduces conflicts that must be resolved by an explicit strategy rather than left to chance.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Assess the legacy database's replication capabilities and determine whether synchronous or asynchronous replication is appropriate
- Set up read replicas to offload reporting and analytics queries from the primary database
- Configure replication monitoring to detect lag, conflicts, and synchronization failures
- Define a clear consistency model (eventual, strong, or session consistency) based on business requirements
- Implement failover procedures that promote a replica to primary when the primary becomes unavailable
- Use change data capture (CDC) to replicate data to downstream systems without modifying the legacy application
- Test failover and recovery procedures regularly to ensure they work when needed

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates the database as a single point of failure through redundancy
- Improves read performance by distributing queries across replicas
- Enables geographic distribution of data for latency reduction
- Supports disaster recovery with off-site data copies

**Costs and Risks:**
- Replication lag can cause stale reads and temporary inconsistencies
- Write conflicts in multi-primary configurations require conflict resolution strategies
- Increases storage and infrastructure costs with each additional replica
- Monitoring and managing replication health adds operational complexity
- Schema changes must be coordinated carefully across all replicas

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy inventory management system ran on a single PostgreSQL database that was both the transactional store and the source for reporting queries. Heavy reporting queries during business hours caused lock contention that slowed order processing. The team set up two asynchronous read replicas and routed all reporting queries to them using a connection routing layer. Transaction processing latency improved by 40% during peak hours. Additionally, one replica was placed in a secondary data center, providing a warm standby for disaster recovery. When the primary database experienced a hardware failure six months later, the team failed over to the standby with only three minutes of data loss, compared to what would have been hours of downtime without replication.
