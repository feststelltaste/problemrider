---
title: Elastic Resource Utilization
description: Automatic adjustment of resources based on current load
category:
- Operations
- Performance
quality_tactics_url: https://qualitytactics.de/en/reliability/elastic-resource-utilization
problems:
- scaling-inefficiencies
- capacity-mismatch
- slow-application-performance
- system-outages
- resource-contention
- high-database-resource-utilization
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Instrument the application to expose metrics that drive scaling decisions (CPU, memory, request queue depth, response time)
- Containerize the legacy application or deploy it behind a load balancer that supports dynamic backend registration
- Configure auto-scaling policies based on observed traffic patterns and performance thresholds
- Define minimum and maximum resource boundaries to prevent runaway scaling and control costs
- Implement health checks that auto-scaling systems use to determine instance readiness before routing traffic
- Design the application to be stateless or use externalized session storage so instances can be added and removed freely
- Test scaling behavior under load to verify that scale-out and scale-in work correctly without dropping requests

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Handles traffic spikes automatically without manual intervention or over-provisioning
- Reduces costs during low-traffic periods by scaling down unused resources
- Improves system reliability by distributing load across multiple instances
- Eliminates capacity planning guesswork for variable workloads

**Costs and Risks:**
- Legacy applications with stateful designs require refactoring before they can scale horizontally
- Auto-scaling lag can cause brief performance degradation during sudden traffic spikes
- Misconfigured scaling policies can lead to excessive costs or insufficient resources
- Cold start times for new instances may be too slow for latency-sensitive applications
- Increased complexity in monitoring and troubleshooting distributed instances

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A ticket sales platform experienced extreme traffic spikes during popular event releases, with load increasing 50x within minutes. The legacy monolith was deployed on fixed-size hardware that could not handle these peaks, resulting in outages during the most critical business moments. The team containerized the application with Docker, externalized session state to Redis, and deployed to Kubernetes with horizontal pod autoscaling based on request queue depth. During the next major ticket release, the system automatically scaled from 4 to 60 pods within three minutes, handled the peak traffic without degradation, and scaled back down within an hour. Infrastructure costs actually decreased because they no longer needed to maintain peak-capacity hardware 24/7.
