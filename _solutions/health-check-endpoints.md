---
title: Health Check Endpoints
description: Exposing standardized health check APIs for load balancer and orchestrator
  monitoring
category:
- Operations
- Architecture
problems:
- monitoring-gaps
- slow-incident-resolution
- system-outages
- single-points-of-failure
- poor-operational-concept
- service-discovery-failures
- load-balancing-problems
layout: solution
related_solutions:
- slug: ping
  similarity: 0.75
- slug: heartbeat
  similarity: 0.7
- slug: status-monitoring
  similarity: 0.7
- slug: self-test
  similarity: 0.7
- slug: self-monitoring-and-diagnosis
  similarity: 0.7
- slug: monitoring
  similarity: 0.7
---

## Description

A health check endpoint is a lightweight, standardized HTTP interface that a service exposes so that load balancers, orchestrators, and monitoring tools can query its status programmatically instead of inferring it indirectly from things like an open TCP port. The pattern distinguishes between liveness checks, which answer only whether the process is running, and readiness checks, which verify that the service can actually handle a request — including its critical dependencies such as database connectivity or downstream availability. This distinction matters a great deal for legacy services, which frequently enter states where the process is technically alive but functionally stuck — deadlocked, out of database connections, or waiting on an unresponsive dependency — a condition that a simple port check cannot detect but a well-designed readiness probe can. Retrofitting health endpoints onto legacy components gives infrastructure the information it needs to automatically route traffic away from unhealthy instances and to sequence deployments safely, capabilities that legacy systems built before this pattern was standard often lack entirely. Because health checks are the input that automated orchestration acts on, a check that reports too little (bare liveness) provides false confidence, while one that checks too much (expensive downstream calls) risks becoming a performance liability or a cause of cascading failure in its own right — so scoping what each check actually verifies is itself a central design decision.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A media company operated several legacy Java services behind a load balancer that relied solely on TCP port checks. Services frequently entered states where the port was open but the application was deadlocked or had lost its database connection. By adding standardized health check endpoints that verified thread pool availability and database connectivity, the load balancer could automatically remove unhealthy instances from rotation. This reduced user-facing errors by 60% and gave the operations team clear diagnostic information when investigating incidents.
