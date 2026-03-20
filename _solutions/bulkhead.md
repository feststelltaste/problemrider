---
title: Bulkhead
description: Dividing a system into isolated areas to limit fault propagation
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/bulkhead
problems:
- cascade-failures
- single-points-of-failure
- monolithic-architecture-constraints
- system-outages
- resource-contention
- thread-pool-exhaustion
- high-coupling-low-cohesion
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify critical and non-critical system functions and separate their resource pools (thread pools, connection pools, memory)
- Isolate external service calls into dedicated thread pools or process boundaries so a slow dependency cannot starve the entire system
- Use separate database connection pools for different modules to prevent one module's queries from exhausting shared connections
- Deploy critical components on separate infrastructure so that resource-intensive batch jobs cannot impact real-time operations
- Implement request classification to route high-priority traffic through dedicated bulkhead partitions
- Add monitoring and alerting for each bulkhead partition to detect when one is approaching capacity

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Contains failures to a single partition, preventing cascading outages across the system
- Ensures critical functions remain available even when non-critical components fail
- Provides clearer resource utilization visibility per system function
- Enables independent scaling of different system partitions

**Costs and Risks:**
- Increases overall resource consumption since each partition needs its own reserved capacity
- Adds configuration complexity for managing multiple pools and partition boundaries
- Under-provisioned partitions may throttle legitimate traffic while other partitions sit idle
- Requires careful analysis to draw partition boundaries at the right places

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An online retail platform experienced total outages whenever its recommendation engine became slow due to third-party API timeouts. The recommendation service shared a thread pool with the checkout flow, so when recommendation threads blocked, checkout requests queued up and the entire site became unresponsive. The team introduced separate thread pools for checkout, recommendations, and inventory operations. When the recommendation API slowed down, only recommendations degraded while checkout continued to process orders normally. This single change eliminated the most common cause of their site-wide outages.
