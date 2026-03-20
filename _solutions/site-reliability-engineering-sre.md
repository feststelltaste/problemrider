---
title: Site Reliability Engineering (SRE)
description: Applying principles for stable system operations
category:
- Operations
- Process
quality_tactics_url: https://qualitytactics.de/en/reliability/site-reliability-engineering-sre
problems:
- system-outages
- constant-firefighting
- slow-incident-resolution
- monitoring-gaps
- deployment-risk
- operational-overhead
- poor-operational-concept
- high-maintenance-costs
- cascade-failures
- developer-frustration-and-burnout
layout: solution
---

## How to Apply ◆

> Legacy systems frequently lack operational discipline, leading to chronic firefighting and unpredictable reliability. SRE principles bring engineering rigor to operations, treating operational work as a software problem that can be measured, automated, and systematically improved.

- Establish error budgets tied to Service Level Objectives for each critical service. When the error budget is exhausted, shift engineering effort from feature development to reliability work. This creates a self-regulating mechanism that prevents reliability from being perpetually deprioritized.
- Implement a blameless postmortem process for all significant incidents. Document the timeline, root cause, contributing factors, and concrete action items. Focus on systemic improvements rather than individual blame to encourage honest reporting and learning.
- Automate toil — repetitive, manual operational tasks that scale linearly with system size. In legacy systems, this often means automating deployment procedures, log analysis, capacity checks, and routine maintenance tasks that consume operator time without adding lasting value.
- Introduce on-call rotations with clear escalation paths and runbooks for known failure modes. Legacy systems often rely on a single expert who handles all incidents; distributing this responsibility reduces single points of failure in the team.
- Apply the principle of reducing mean time to recovery (MTTR) rather than pursuing zero failures. For legacy systems that cannot be easily redesigned, fast detection and recovery are more achievable and more impactful than preventing all failures.
- Measure and cap the percentage of engineering time spent on operational toil (SRE recommends a 50% cap). If operational work exceeds this threshold, it signals that the system requires structural improvements, not more firefighters.
- Implement progressive rollout strategies (canary deployments, feature flags) to reduce the blast radius of changes in systems where changes are inherently risky.

## Tradeoffs ⇄

> SRE practices transform operations from a reactive cost center into a proactive engineering discipline, but they require organizational commitment and cultural change.

**Benefits:**

- Reduces chronic firefighting by establishing clear policies for when reliability work takes priority over feature development.
- Distributes operational knowledge across teams through runbooks and on-call rotations, reducing dependency on individual experts.
- Provides measurable criteria for system health through SLOs and error budgets, making reliability discussions objective rather than political.
- Systematically eliminates toil through automation, freeing engineering capacity for higher-value work over time.
- Improves incident response through blameless postmortems and shared learning, reducing repeat incidents.

**Costs and Risks:**

- Requires significant cultural change, especially in organizations where operations and development are separate functions with different incentives.
- Error budget policies can create friction when feature deadlines conflict with reliability priorities, requiring strong management support.
- On-call rotations for legacy systems can be burdensome if the system has many failure modes and few runbooks, leading to burnout during the transition period.
- Automation of legacy operational tasks may require substantial investment in tooling and infrastructure that the legacy system was not designed to support.

## How It Could Be

> The following scenarios illustrate how SRE principles stabilize legacy system operations.

A logistics company operates a legacy shipment tracking system that experiences frequent outages, each requiring the same senior engineer to diagnose and fix. The company introduces SRE practices by first documenting the top 10 failure modes in runbooks, then establishing an on-call rotation among four engineers. They define an SLO of 99.9% availability for the tracking API and measure it weekly. Within three months, the runbooks enable junior engineers to resolve 80% of incidents without escalation, the senior engineer's on-call burden drops from every night to one week in four, and the team identifies two systemic issues that account for 60% of all incidents. Fixing these two issues with targeted automation reduces monthly incident count from 15 to 4.

A healthcare platform runs a legacy patient records system where deployments are performed manually by a single operations engineer over a weekend, resulting in extended downtime and frequent rollback-requiring errors. The SRE team automates the deployment pipeline, introduces canary deployments that roll out changes to 5% of traffic first, and establishes automated health checks that trigger automatic rollback if error rates exceed baseline by more than 2%. Deployment frequency increases from monthly to weekly, deployment-related incidents drop by 90%, and the operations engineer redirects their weekend time to building monitoring infrastructure that further improves system reliability.
