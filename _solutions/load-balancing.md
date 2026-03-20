---
title: Load Balancing
description: Distributing workload across multiple resources
category:
- Architecture
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/load-balancing
problems:
- load-balancing-problems
- capacity-mismatch
- single-points-of-failure
- slow-application-performance
- scaling-inefficiencies
- system-outages
- high-api-latency
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Deploy a load balancer in front of legacy application instances to distribute incoming requests
- Choose an appropriate balancing algorithm (round-robin, least connections, weighted) based on workload characteristics
- Configure health checks so the load balancer routes traffic only to healthy instances
- Refactor legacy applications to be stateless or use external session stores to enable proper load distribution
- Implement sticky sessions as a transitional measure for stateful legacy applications that cannot be immediately refactored
- Plan for load balancer redundancy to avoid introducing a new single point of failure
- Use load balancing metrics to identify capacity bottlenecks and plan scaling decisions

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Improves system availability by distributing load across multiple instances
- Enables horizontal scaling of legacy application tiers
- Provides a natural integration point for health checking and traffic management
- Supports rolling deployments and canary releases

**Costs and Risks:**
- Stateful legacy applications may require session affinity, reducing balancing effectiveness
- Adds network latency and a potential point of failure if not properly redundant
- Configuration complexity increases with SSL termination, routing rules, and rate limiting
- Load balancer misconfiguration can cause uneven distribution or dropped connections

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An online education platform ran its legacy course management system on a single server that regularly became unresponsive during enrollment periods. By deploying a load balancer with three application instances behind it, the team distributed enrollment traffic across all nodes. They used external session storage to avoid sticky sessions and configured health checks that removed unresponsive instances from rotation. Peak enrollment traffic was handled smoothly, and the team could perform rolling deployments without any downtime.
