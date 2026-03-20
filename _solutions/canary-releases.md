---
title: Canary Releases
description: Gradual introduction of changes for a limited user group to minimize risk
category:
- Operations
- Process
quality_tactics_url: https://qualitytactics.de/en/reliability/canary-releases
problems:
- deployment-risk
- large-risky-releases
- frequent-hotfixes-and-rollbacks
- release-instability
- release-anxiety
- fear-of-change
- high-defect-rate-in-production
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Set up infrastructure for routing a configurable percentage of traffic to the new version alongside the stable version
- Define key health metrics (error rate, latency, business metrics) that will be monitored during the canary phase
- Start with a small percentage (1-5%) of traffic directed to the canary and increase gradually based on metric thresholds
- Implement automated rollback triggers that revert traffic to the stable version if health metrics deteriorate
- Use feature flags in combination with canary routing to control which features are active for canary users
- Establish a minimum observation window before each traffic increase to detect slow-developing issues
- Ensure logging and monitoring distinguish between canary and stable traffic for accurate comparison

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Limits the blast radius of defective releases to a small subset of users
- Provides real production validation before full rollout
- Enables data-driven rollout decisions based on actual metrics rather than assumptions
- Reduces the pressure and anxiety associated with big-bang releases
- Allows quick rollback by simply redirecting traffic away from the canary

**Costs and Risks:**
- Requires infrastructure capable of running two versions simultaneously and splitting traffic
- Database schema changes must be backward compatible to support both versions concurrently
- Monitoring and metric comparison infrastructure must be in place before canary releases are useful
- Small canary populations may not surface issues that only appear at scale
- Adds operational complexity to the deployment process

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A travel booking platform had a history of disruptive production incidents after releases, leading the team to release only once per quarter with extensive manual testing. The team implemented canary releases using their load balancer's weighted routing capability. New versions were initially exposed to 2% of traffic with automated health checks monitoring booking completion rates and error rates. If metrics held stable for two hours, traffic was gradually increased to 10%, 50%, and then 100%. This approach caught a critical payment integration bug during a canary phase, affecting only 2% of users instead of the entire customer base. Release frequency increased from quarterly to weekly.
