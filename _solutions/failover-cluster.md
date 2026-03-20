---
title: Failover Cluster
description: Maintaining servers or systems as a functional group redundantly
category:
- Architecture
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/failover-cluster
problems:
- single-points-of-failure
- system-outages
- cascade-failures
- slow-incident-resolution
- capacity-mismatch
- high-maintenance-costs
- deployment-risk
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Inventory all single-point-of-failure components in the legacy system and prioritize them by business criticality
- Introduce active-passive or active-active clustering for the most critical services first
- Configure shared storage or replicated data stores so that failover nodes have access to current state
- Set up automatic health checks and failover triggers with appropriate timeout thresholds
- Test failover scenarios regularly in staging environments that mirror production topology
- Document the failover process in runbooks so on-call staff can intervene when automatic failover does not engage
- Gradually extend clustering to secondary services as the team gains operational confidence

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates single points of failure for critical legacy services
- Reduces unplanned downtime and its associated business impact
- Enables maintenance windows without full service interruption
- Provides a foundation for future high-availability improvements

**Costs and Risks:**
- Increased infrastructure cost for redundant hardware or cloud instances
- Operational complexity grows with cluster management, quorum rules, and split-brain prevention
- Legacy applications may require modifications to support session sharing or stateless operation
- Failover testing requires careful planning to avoid accidental production outages

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A retail company ran its order processing system on a single legacy application server. Every hardware failure or OS patch required a full maintenance window, costing hours of lost revenue. By introducing a two-node active-passive failover cluster with shared database storage, the team reduced unplanned downtime by over 90%. The passive node automatically assumed traffic within seconds of detecting a heartbeat loss, and planned maintenance could proceed by gracefully failing over before applying patches.
