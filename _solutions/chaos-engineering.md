---
title: Chaos Engineering
description: Intentional introduction of disruptions to test system resilience
category:
- Operations
- Testing
problems:
- cascade-failures
- single-points-of-failure
- system-outages
- unpredictable-system-behavior
- slow-incident-resolution
- monitoring-gaps
- fear-of-change
layout: solution
related_solutions:
- slug: resilience
  similarity: 0.85
- slug: stress-testing
  similarity: 0.85
- slug: incident-management
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
- slug: site-reliability-engineering-sre
  similarity: 0.8
- slug: error-budgets
  similarity: 0.8
---

## Description

Chaos engineering is the discipline of deliberately injecting controlled failures into a system — killing processes, degrading network connections, exhausting resources, disabling dependencies — in order to empirically validate whether the system's resilience assumptions actually hold, rather than trusting that they do because the architecture was designed that way. Each experiment starts from an explicit hypothesis about expected behavior under a specific fault, and the experiment either confirms that hypothesis or exposes a gap between the assumed and actual failure behavior. This is especially important in legacy systems, where failover logic, retry behavior, and single points of failure have often accumulated undocumented and untested over many years, so that nobody in the current team can say with confidence what actually happens when a given dependency goes down. Rather than waiting for a real production incident to reveal these gaps at the worst possible time, chaos engineering surfaces them under controlled conditions, with the team present, monitoring active, and an abort mechanism ready to halt the experiment if the blast radius grows too large. The practice depends on the system already having reasonably mature observability, since without it the impact of an injected fault cannot be reliably measured or contained, which is often the harder prerequisite in legacy environments that predate proper monitoring. Over time, running these experiments systematically shifts a team's relationship to failure from fear and avoidance toward confidence grounded in evidence, which is often what actually enables faster and more frequent legacy system changes.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce platform experienced unexplained outages during peak traffic events. The team suspected various single points of failure but had no way to validate their resilience assumptions. They started chaos engineering in their staging environment by systematically killing individual services and observing system behavior. The first experiment revealed that the session management service had no failover, causing complete checkout failure when it went down. After fixing this, they progressed to network partition experiments that uncovered a database connection retry bug dormant for three years. Over six months, the team resolved 14 critical resilience issues and reduced unplanned downtime by 60%.
