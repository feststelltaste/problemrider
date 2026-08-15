---
title: Serverless Computing
description: Execution of code without managing the underlying infrastructure
category:
- Operations
- Architecture
problems:
- scaling-inefficiencies
- operational-overhead
- complex-deployment-process
- capacity-mismatch
- high-maintenance-costs
- poor-system-environment
layout: solution
related_solutions:
- slug: cloud-native-development
  similarity: 0.75
- slug: containerization
  similarity: 0.75
- slug: distributed-processing
  similarity: 0.75
- slug: microservices-architecture
  similarity: 0.7
- slug: elastic-resource-utilization
  similarity: 0.7
- slug: load-balancing
  similarity: 0.7
---

## Description

Serverless computing runs code in response to events without the caller managing any underlying server, scaling automatically with demand and billing only for actual execution time rather than for provisioned capacity that mostly sits idle. For legacy modernization, the practical value is usually not migrating the whole system but offloading specific, well-bounded workloads — image processing, report generation, webhook handling — out of a monolith and onto serverless functions, routed selectively through an API gateway while the bulk of the legacy application keeps running unchanged. This targeted extraction relieves exactly the kind of resource contention that occurs when a CPU-intensive, on-demand task inside a monolith degrades performance for every other concurrent user, without requiring the rest of the legacy system to be touched at all. The tradeoffs are real, though: cold-start latency can be a poor fit for latency-sensitive operations, state has to be externalized since functions are stateless, and the approach introduces a new form of vendor lock-in to whichever cloud platform hosts the functions.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify workloads suitable for serverless: event-driven tasks, periodic batch jobs, webhook handlers, API endpoints with variable traffic
- Start by offloading peripheral functions (image processing, email sending, report generation) from the legacy monolith to serverless functions
- Use API gateways to route specific endpoints to serverless functions while the rest of the traffic continues to the legacy application
- Refactor stateful operations to externalize state to managed services (databases, caches, queues) since serverless functions are stateless
- Implement appropriate timeout and retry strategies given the execution limits of serverless platforms
- Monitor cold start latency and optimize function size to minimize startup delays

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates server management and infrastructure provisioning for migrated workloads
- Automatic scaling handles traffic spikes without capacity planning
- Pay-per-use pricing reduces costs for intermittent or variable workloads
- Enables rapid deployment of new features without infrastructure changes

**Costs and Risks:**
- Cold start latency can be problematic for latency-sensitive operations
- Vendor lock-in to a specific cloud provider's serverless platform
- Execution time limits and memory constraints may not suit all workloads
- Debugging and monitoring serverless functions requires different tools and approaches
- State management complexity increases when functions must be stateless
- Cost can exceed traditional hosting for consistently high-traffic workloads

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy document management system generated PDF reports on demand, a CPU-intensive operation that caused performance issues for other users when multiple reports were requested simultaneously. The team extracted the PDF generation logic into AWS Lambda functions triggered by an SQS queue. The legacy application simply placed report requests on the queue and polled for completion. This eliminated the impact on the main application's performance, automatically scaled during month-end reporting peaks, and reduced infrastructure costs by 70 percent since PDF generation only consumed resources when actually running.
