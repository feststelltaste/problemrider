---
title: Rolling Updates
description: Stepwise updating of servers or instances
category:
- Operations
problems:
- deployment-risk
- system-outages
- large-risky-releases
- complex-deployment-process
- deployment-coupling
- release-instability
layout: solution
related_solutions:
- slug: rollback-mechanisms
  similarity: 0.75
- slug: canary-releases
  similarity: 0.75
- slug: regular-maintenance-and-updates
  similarity: 0.75
- slug: restore-points
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.7
- slug: blue-green-canary-deployments
  similarity: 0.7
---

## Description

Rolling updates deploy a new version of a system incrementally, instance by instance or in small batches, rather than replacing an entire fleet of servers simultaneously, with health checks validating each updated instance before the rollout proceeds to the next one. Because old and new versions of the application run side by side for the duration of the rollout, this approach both eliminates the downtime that a big-bang deployment would otherwise require and limits the blast radius of a bad release to whatever fraction of the fleet has been updated when a problem is first detected. For legacy systems, which are often deployed as a fixed set of long-lived servers rather than dynamically scaled infrastructure, rolling updates offer a way to reduce deployment risk without first re-architecting the system into something more cloud-native, since the technique operates at the deployment layer rather than requiring changes to the application itself. The approach does impose one significant precondition on the application, however: it must tolerate running two versions concurrently, which means database schema changes and any shared state must remain compatible across both versions for the duration of the rollout — a constraint that legacy applications with tightly coupled schemas or singleton in-memory state may not satisfy without additional work. Where that precondition can be met, rolling updates turn what was previously a scheduled maintenance window with full-fleet exposure into an incremental, self-checking process that halts automatically the moment early batches show signs of trouble.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Configure deployment tooling to update instances one at a time or in small batches rather than all at once
- Implement health checks that validate each updated instance before proceeding to the next
- Define automatic rollback triggers that halt the rolling update if error rates exceed thresholds
- Ensure the legacy application supports running old and new versions simultaneously during the transition
- Handle database schema changes carefully to maintain compatibility with both versions during the update window
- Use connection draining to gracefully remove instances from load balancer rotation before updating them
- Monitor key metrics during the rollout and pause if anomalies are detected

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates deployment downtime by maintaining service availability throughout the update
- Limits blast radius since only a subset of instances run the new version at any point
- Enables early detection of issues before they affect all instances
- Provides natural checkpoints for automated rollback decisions

**Costs and Risks:**
- Both old and new versions must coexist, requiring backward-compatible changes
- Rolling updates take longer than full-fleet deployments
- Debugging issues during mixed-version states can be challenging
- Legacy applications with shared state or singleton patterns may not support gradual rollout

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A media streaming company deployed its legacy content delivery application across 12 servers, previously updating all servers simultaneously during a 30-minute maintenance window. By implementing rolling updates that updated two servers at a time with health check validation between batches, the team eliminated planned downtime entirely. When a deployment introduced a memory leak, it was detected during the update of the first batch through health check failures, and only two of twelve servers were affected. The deployment was automatically halted and rolled back on those two servers, preventing any user-visible impact.
