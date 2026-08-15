---
title: Redundancy
description: Multiple instances of critical components or systems
category:
- Architecture
- Operations
problems:
- single-points-of-failure
- system-outages
- cascade-failures
- capacity-mismatch
- deployment-risk
- high-maintenance-costs
layout: solution
related_solutions:
- slug: redundant-data-storage
  similarity: 0.85
- slug: failover-cluster
  similarity: 0.8
- slug: high-availability-architectures
  similarity: 0.8
- slug: failover-mechanisms
  similarity: 0.8
- slug: data-replication
  similarity: 0.8
- slug: resilience
  similarity: 0.8
---

## Description

Redundancy is the deliberate duplication of critical components, services, or infrastructure so that the failure of one instance does not take down the system as a whole. Instead of relying on a single application server, database, or network path, redundant architectures run multiple equivalent instances in parallel, with a failover or load-balancing mechanism directing traffic away from any instance that becomes unavailable. The concept applies at every layer — hardware, network, data, and application — and can be implemented as active-active configurations that share load continuously or active-passive configurations that keep a standby ready to take over. In legacy systems, redundancy is often the fastest way to eliminate single points of failure that were never questioned when the system was small and its uptime requirements were modest, because it can frequently be layered onto an existing architecture without rewriting the application logic itself. It matters especially during modernization because a legacy system undergoing incremental change is more exposed to outages than a stable one, and redundancy provides the safety margin needed to keep the business running while the underlying architecture is reshaped. The tradeoff is that redundancy trades capital and operational cost for reduced risk, and its protection is only as good as the diversity and testing of the redundant paths themselves.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify all single points of failure in the legacy system architecture and prioritize them by business impact
- Deploy redundant instances of critical application components behind load balancers
- Implement database replication with automatic failover for data persistence layers
- Ensure redundant components are deployed across different failure domains (racks, zones, regions)
- Test that redundant components can actually take over load by regularly simulating primary failures
- Avoid common-mode failures by using diverse implementations or configurations where practical
- Monitor all redundant instances to ensure standby components remain healthy and ready

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates single points of failure that are common in legacy architectures
- Enables zero-downtime maintenance and upgrades
- Increases overall system capacity through active-active configurations
- Provides insurance against hardware failures and infrastructure issues

**Costs and Risks:**
- Doubles or triples infrastructure costs for redundant components
- State synchronization between redundant instances adds complexity
- Redundant components that are never tested may fail when actually needed
- Legacy applications may not support multi-instance deployment without modification

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A municipal government ran its citizen services portal on a single legacy application server and a single database server. A hard drive failure on the database server caused a three-day outage while data was restored from tape backups. After this incident, the team deployed redundant database servers with synchronous replication, redundant application servers behind a load balancer, and redundant network paths. The investment increased infrastructure costs by 120%, but the next hardware failure was handled transparently with automatic failover and zero citizen-facing impact.
