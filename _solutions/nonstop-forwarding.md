---
title: Nonstop Forwarding
description: Continuous request forwarding despite failures or errors
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/nonstop-forwarding
problems:
- cascade-failures
- system-outages
- service-timeouts
- single-points-of-failure
- unpredictable-system-behavior
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Separate the control plane (routing decisions) from the data plane (request forwarding) in legacy network and service architectures
- Configure forwarding components to continue routing traffic using last-known-good routing tables during control plane failures
- Implement graceful restart capabilities so that components can restart their control logic without interrupting data flow
- Use persistent forwarding state that survives process restarts or failover events
- Test nonstop forwarding scenarios by simulating control plane failures while measuring data plane impact
- Apply this pattern at service mesh or API gateway layers to protect legacy backend services

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Maintains request flow continuity during management or control plane disruptions
- Reduces the blast radius of failures in routing and orchestration components
- Enables zero-downtime upgrades of routing infrastructure
- Prevents cascading timeouts when control components restart

**Costs and Risks:**
- Stale routing information during extended control plane outages can direct traffic to removed or unhealthy nodes
- Increases complexity of the forwarding layer
- Debugging routing issues becomes harder when forwarding operates independently of control
- Not all legacy architectures can cleanly separate control and data planes

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A telecommunications company's legacy API gateway required periodic restarts for configuration updates, causing brief traffic interruptions that triggered timeout errors in downstream systems. By redesigning the gateway to separate its routing configuration management from its request forwarding engine, updates could be applied through graceful restarts where the forwarding engine continued processing requests using cached routing rules while the control plane reloaded. This eliminated the 10-15 second traffic interruptions that had been causing cascading failures in latency-sensitive legacy backend services.
