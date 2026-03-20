---
title: Horizontal Scaling
description: Increasing performance by adding additional components
category:
- Performance
- Operations
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/horizontal-scaling
problems:
- scaling-inefficiencies
- capacity-mismatch
- single-points-of-failure
- slow-application-performance
- load-balancing-problems
- monolithic-architecture-constraints
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Make the application stateless so any instance can handle any request: externalize sessions, caches, and file storage
- Identify and eliminate instance-specific state such as local file caches, in-memory session stores, and instance-bound scheduled tasks
- Deploy a load balancer to distribute traffic across multiple application instances
- Implement health checks so the load balancer can detect and route around unhealthy instances
- Use auto-scaling policies based on metrics (CPU, request count, queue depth) to add capacity dynamically
- Test the application under load with multiple instances to verify correct behavior without shared mutable state
- Address database scaling separately: read replicas, connection pooling, or sharding as needed

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides near-linear capacity increase by adding more instances
- Improves availability: individual instance failures do not take down the entire system
- Enables cost-efficient scaling by adding capacity only when demand requires it
- Uses commodity hardware rather than requiring increasingly expensive vertical upgrades

**Costs and Risks:**
- Requires applications to be stateless, which legacy systems often are not
- Database and shared resources can become bottlenecks that limit horizontal scaling benefits
- Adds infrastructure complexity: load balancers, service discovery, instance management
- Distributed coordination problems (cache coherence, leader election) increase with instance count

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy web application for a university registration system ran on a single large server. During enrollment periods, the server was overwhelmed by traffic spikes, causing outages at the worst possible time. The application stored session data in server memory, which prevented running multiple instances. The team externalized session storage to Redis, moved uploaded files to object storage, and deployed the application behind a load balancer with three instances. During the next enrollment period, auto-scaling added two more instances to handle the peak, and the system remained responsive throughout. After the peak, instances scaled back down to reduce costs.
