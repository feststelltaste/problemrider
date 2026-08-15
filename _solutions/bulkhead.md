---
title: Bulkhead
description: Dividing a system into isolated areas to limit fault propagation
category:
- Architecture
problems:
- cascade-failures
- single-points-of-failure
- monolithic-architecture-constraints
- system-outages
- resource-contention
- thread-pool-exhaustion
- high-coupling-low-cohesion
- upstream-timeouts
layout: solution
related_solutions:
- slug: fault-containment
  similarity: 0.8
- slug: isolation-of-faulty-components
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.7
- slug: circuit-breaker
  similarity: 0.7
- slug: resilience
  similarity: 0.65
- slug: backpressure
  similarity: 0.65
---

## Description

The bulkhead pattern divides a system's resource pools — threads, database connections, memory — into separate, isolated partitions assigned to different functions, so that exhaustion or failure in one partition cannot consume the capacity that another partition needs to keep functioning. The mechanism is named after ship design for exactly this reason: a compartment that floods should not sink the whole vessel, and a slow or failing dependency should not, by consuming a shared thread pool, take down unrelated functionality that happens to share the same process. Legacy systems are particularly prone to the failure this pattern prevents because they were frequently built as monoliths where all functionality quietly shares one thread pool or one connection pool by default, with no one having deliberately decided that recommendation-engine calls and checkout-processing calls should be able to starve each other. Introducing bulkheads means identifying which functions are critical and which are not, and giving each its own reserved capacity — separate thread pools, separate connection pools, sometimes separate infrastructure entirely — so that a slow third-party API called by a non-critical feature degrades only that feature rather than cascading into a site-wide outage. The tradeoff is that reserved-but-unused capacity in an under-loaded partition is wasted resource that a fully shared pool would have put to use, so bulkhead boundaries need to be sized deliberately rather than applied uniformly everywhere.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An online retail platform experienced total outages whenever its recommendation engine became slow due to third-party API timeouts. The recommendation service shared a thread pool with the checkout flow, so when recommendation threads blocked, checkout requests queued up and the entire site became unresponsive. The team introduced separate thread pools for checkout, recommendations, and inventory operations. When the recommendation API slowed down, only recommendations degraded while checkout continued to process orders normally. This single change eliminated the most common cause of their site-wide outages.
