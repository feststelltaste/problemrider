---
title: Chaos Engineering
description: Intentional introduction of disruptions to test system resilience
category:
- Operations
- Testing
quality_tactics_url: https://qualitytactics.de/en/reliability/chaos-engineering
problems:
- cascade-failures
- single-points-of-failure
- system-outages
- unpredictable-system-behavior
- slow-incident-resolution
- monitoring-gaps
- fear-of-change
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Start with a hypothesis about what should happen when a specific failure occurs (e.g., "the system should fail over to the backup database within 30 seconds")
- Begin chaos experiments in non-production environments to build confidence and identify obvious gaps
- Introduce controlled failures such as killing processes, injecting network latency, filling disks, or disabling dependencies
- Use established tools like Chaos Monkey, Gremlin, or Litmus to manage experiments safely
- Implement an abort mechanism that can stop the experiment immediately if impact exceeds acceptable thresholds
- Run experiments during business hours with the team present so issues can be observed and addressed in real time
- Document findings from each experiment and track remediation of discovered weaknesses

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reveals hidden failure modes and single points of failure before they cause production incidents
- Builds team confidence in the system's resilience through empirical validation
- Improves incident response skills by exposing teams to controlled failure scenarios
- Identifies monitoring and alerting gaps that would otherwise go unnoticed
- Drives architectural improvements based on observed weaknesses

**Costs and Risks:**
- Poorly controlled experiments can cause real production outages
- Requires mature monitoring and observability to detect the impact of injected faults
- Teams may resist the practice due to fear of causing incidents
- Legacy systems without proper failover mechanisms may fail catastrophically during experiments
- Requires organizational buy-in since experiments temporarily degrade system behavior

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce platform experienced unexplained outages during peak traffic events. The team suspected various single points of failure but had no way to validate their resilience assumptions. They started chaos engineering in their staging environment by systematically killing individual services and observing system behavior. The first experiment revealed that the session management service had no failover, causing complete checkout failure when it went down. After fixing this, they progressed to network partition experiments that uncovered a database connection retry bug dormant for three years. Over six months, the team resolved 14 critical resilience issues and reduced unplanned downtime by 60%.
