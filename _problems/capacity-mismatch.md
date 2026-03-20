---
title: Capacity Mismatch
description: Available capacity at different stages of the development process doesn't
  match demand patterns, creating bottlenecks and underutilization.
category:
- Performance
- Process
related_problems:
- slug: bottleneck-formation
  similarity: 0.6
- slug: organizational-structure-mismatch
  similarity: 0.6
- slug: staff-availability-issues
  similarity: 0.6
- slug: work-queue-buildup
  similarity: 0.6
- slug: maintenance-bottlenecks
  similarity: 0.6
- slug: resource-waste
  similarity: 0.6
solutions:
- capacity-planning
- elastic-scaling
layout: problem
---

## Description

Capacity mismatch occurs when the available capacity at different stages of the development process is not aligned with actual demand patterns. This can manifest as some process stages being overwhelmed while others have excess capacity, creating an inefficient flow where resources are wasted while bottlenecks form elsewhere. Effective capacity matching requires understanding demand patterns and allocating resources accordingly.

## Indicators ⟡

- Some team members or process stages are consistently overloaded while others have spare capacity
- Work queues build up at certain stages while other stages remain idle
- Process throughput is limited by capacity constraints at specific stages
- Resource utilization varies dramatically across different roles or process steps
- Adding resources to the team doesn't improve overall throughput

## Symptoms ▲

- [Bottleneck Formation](bottleneck-formation.md)
<br/>  Stages with insufficient capacity become bottlenecks that constrain the throughput of the entire development process.
- [Resource Waste](resource-waste.md)
<br/>  Team members at stages with excess capacity sit idle while constrained stages are overloaded.
- [Work Queue Buildup](work-queue-buildup.md)
<br/>  Work accumulates at under-capacity stages, creating growing queues that delay delivery.
- [Developer Frustration and Burnout](developer-frustration-and-burnout.md)
<br/>  Developers become frustrated when they are either overloaded or idle due to misaligned capacity across stages.
- [Cascade Delays](cascade-delays.md)
<br/>  Capacity-constrained stages cause delays that propagate to downstream work and dependent projects.
## Causes ▼

- [Poor Planning](poor-planning.md)
<br/>  Failure to analyze demand patterns leads to capacity allocation that doesn't match actual workflow needs.
- [Knowledge Silos](knowledge-silos.md)
<br/>  Specialized knowledge concentrated in few people creates capacity constraints at stages requiring that expertise.
- [Organizational Structure Mismatch](organizational-structure-mismatch.md)
<br/>  Team structures that don't align with process demands create inherent capacity imbalances across workflow stages.
## Detection Methods ○

- **Capacity Utilization Analysis:** Monitor utilization rates across different roles and process stages
- **Throughput Analysis:** Measure actual throughput versus theoretical capacity at each stage
- **Queue Length Monitoring:** Track where work accumulates and where capacity sits idle
- **Resource Allocation Review:** Assess whether resource distribution matches actual demand patterns
- **Flow Efficiency Measurement:** Calculate overall process efficiency considering capacity constraints

## Examples

A software development team has five developers but only one person qualified to do database work. The team consistently has database-related tasks backing up in a queue while other developers wait for database changes to be completed before they can proceed with their work. Despite having adequate overall development capacity, the team's throughput is limited by the single database specialist. Another example involves a testing team where automated test creation capacity far exceeds manual testing capacity, but the process requires manual verification of all automated test results. The automated testing team becomes idle waiting for manual testers to catch up, while the manual testing team is overwhelmed with verification work.
