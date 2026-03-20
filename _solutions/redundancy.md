---
title: Redundancy
description: Multiple instances of critical components or systems
category:
- Architecture
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/redundancy
problems:
- single-points-of-failure
- system-outages
- cascade-failures
- capacity-mismatch
- deployment-risk
- high-maintenance-costs
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A municipal government ran its citizen services portal on a single legacy application server and a single database server. A hard drive failure on the database server caused a three-day outage while data was restored from tape backups. After this incident, the team deployed redundant database servers with synchronous replication, redundant application servers behind a load balancer, and redundant network paths. The investment increased infrastructure costs by 120%, but the next hardware failure was handled transparently with automatic failover and zero citizen-facing impact.
