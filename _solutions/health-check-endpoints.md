---
title: Health Check Endpoints
description: Exposing standardized health check APIs for load balancer and orchestrator monitoring
category:
- Operations
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/health-check-endpoints
problems:
- monitoring-gaps
- slow-incident-resolution
- system-outages
- single-points-of-failure
- poor-operational-concept
- service-discovery-failures
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Add lightweight HTTP endpoints to legacy services that report readiness and liveness status
- Include dependency checks (database connectivity, downstream service availability) in health responses
- Standardize the response format across all services so monitoring tools can parse them uniformly
- Configure load balancers and orchestrators to use these endpoints for routing and restart decisions
- Implement shallow checks for liveness (is the process running) and deep checks for readiness (can it serve requests)
- Avoid expensive operations in health checks that could themselves degrade system performance
- Add versioning information to health responses to aid in deployment verification

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides immediate visibility into service health without manual investigation
- Enables automated traffic routing away from unhealthy instances
- Supports zero-downtime deployments by signaling readiness before accepting traffic
- Standardizes health reporting across heterogeneous legacy components

**Costs and Risks:**
- Health endpoints can become stale or misleading if they do not check meaningful conditions
- Deep health checks that verify dependencies can create cascading failures if a dependency is slow
- Exposing health endpoints without authentication can leak internal system information
- Adding endpoints to legacy applications may require framework modifications

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A media company operated several legacy Java services behind a load balancer that relied solely on TCP port checks. Services frequently entered states where the port was open but the application was deadlocked or had lost its database connection. By adding standardized health check endpoints that verified thread pool availability and database connectivity, the load balancer could automatically remove unhealthy instances from rotation. This reduced user-facing errors by 60% and gave the operations team clear diagnostic information when investigating incidents.
