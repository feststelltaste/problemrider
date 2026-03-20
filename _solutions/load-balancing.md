---
title: Load Balancing
description: Distribution of the load across multiple parallel processing units
category:
- Performance
- Operations
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/load-balancing
problems:
- slow-application-performance
- single-points-of-failure
- scaling-inefficiencies
- capacity-mismatch
- bottleneck-formation
- system-outages
- high-connection-count
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Assess current traffic patterns and identify components that are overloaded or represent single points of failure
- Place a reverse proxy or load balancer in front of legacy application servers to distribute incoming requests
- Start with simple round-robin or least-connections algorithms before adopting more complex strategies
- Configure health checks so that unhealthy legacy instances are automatically removed from the pool
- Use session affinity (sticky sessions) if the legacy application relies on server-side session state, while planning to externalize sessions
- Introduce load balancing incrementally, starting with stateless services before tackling stateful ones
- Monitor request distribution to ensure load is actually balanced and no single node is consistently overloaded

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates single points of failure and improves system availability
- Enables horizontal scaling without rewriting legacy components
- Provides a foundation for rolling deployments and zero-downtime upgrades
- Distributes peak load across instances, reducing response times during traffic spikes

**Costs and Risks:**
- Legacy applications with in-memory state or local file storage require additional work to become load-balancer-friendly
- Adds network hops and infrastructure complexity that must be monitored and maintained
- Misconfigured load balancing can cause uneven distribution, making problems worse
- Session affinity can negate the benefits of balancing if most users hit the same node
- Debugging becomes harder when requests are spread across multiple instances

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A retail company ran a monolithic Java application on a single application server that buckled during seasonal sales events. The team placed an Nginx load balancer in front of three identical application instances and externalized session data to Redis. This allowed the system to handle three times the previous peak traffic without code changes. The health check mechanism also improved reliability by automatically routing around an instance that experienced memory exhaustion, buying the team time to investigate rather than suffering a full outage.
