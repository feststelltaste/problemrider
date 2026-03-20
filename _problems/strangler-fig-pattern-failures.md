---
title: Strangler Fig Pattern Failures
description: Incremental modernization using the strangler fig pattern stalls due
  to complex interdependencies and data consistency challenges
category:
- Architecture
- Code
- Operations
related_problems:
- slug: modernization-strategy-paralysis
  similarity: 0.65
- slug: second-system-effect
  similarity: 0.55
- slug: cascade-failures
  similarity: 0.55
- slug: modernization-roi-justification-failure
  similarity: 0.55
- slug: stagnant-architecture
  similarity: 0.55
- slug: maintenance-paralysis
  similarity: 0.55
solutions:
- strangler-fig-pattern
- walking-skeleton
layout: problem
---

## Description

Strangler fig pattern failures occur when attempts to gradually replace legacy system components with modern alternatives stall or fail due to underestimated complexity in system boundaries, data consistency requirements, and interdependencies. The strangler fig pattern, intended to enable low-risk incremental modernization, becomes a source of increased complexity and technical debt when the "strangling" process cannot be completed, leaving organizations with hybrid systems that are more complex than either the original legacy system or a complete replacement would have been.

## Indicators ⟡

- Strangler fig implementation projects that consistently miss deadlines and milestones
- Difficulty identifying clean boundaries between legacy and new system components
- Data synchronization complexity between legacy and new components that exceeds expectations
- New system components that require increasingly complex integration with remaining legacy parts
- Performance degradation as requests flow through both legacy and new system layers
- Team estimates for completing the "strangling" process that keep extending
- Growing operational complexity from managing both legacy and new system components simultaneously

## Symptoms ▲

- [Delayed Project Timelines](delayed-project-timelines.md)
<br/>  The stalled strangler fig migration causes the modernization project to miss deadlines repeatedly as complexity escalates.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Managing both legacy and new components simultaneously doubles operational overhead and maintenance effort.
- [System Outages](system-outages.md)
<br/>  Data synchronization failures and performance issues in the hybrid system cause service interruptions.
- [Budget Overruns](budget-overruns.md)
<br/>  The unexpected complexity of completing the migration causes costs to significantly exceed original estimates.
- [Stakeholder Confidence Loss](stakeholder-confidence-loss.md)
<br/>  Repeated delays and escalating costs in the modernization effort erode stakeholder trust in the technical approach.
## Causes ▼

- [Hidden Dependencies](hidden-dependencies.md)
<br/>  Undiscovered dependencies between legacy components make it impossible to cleanly separate and replace individual parts.
- [High Coupling and Low Cohesion](high-coupling-low-cohesion.md)
<br/>  Tightly coupled legacy components resist clean boundary identification needed for incremental replacement.
- [Cross-System Data Synchronization Problems](cross-system-data-synchronization-problems.md)
<br/>  Data consistency challenges between legacy and modern components undermine the incremental migration approach.
- [Complex Domain Model](complex-domain-model.md)
<br/>  Inherently complex business domains make it difficult to identify clean boundaries for incremental replacement.
## Detection Methods ○

- Track progress metrics for strangler fig implementation against original timeline estimates
- Monitor data consistency issues and synchronization failures between system components
- Measure system complexity metrics before and during the strangling process
- Assess team confidence levels and estimate accuracy for completing remaining modernization work
- Analyze performance impacts and operational overhead of the hybrid system state
- Review technical debt accumulation in integration and synchronization code
- Survey development teams about challenges and blockers in continuing the modernization
- Evaluate whether the current hybrid system provides better value than the original legacy system

## Examples

A retail company begins modernizing their inventory management system using the strangler fig pattern, starting with the product catalog component. The new catalog service works well initially, but as they attempt to replace the pricing engine, they discover that pricing logic is deeply intertwined with inventory allocation, order processing, and promotional systems. Maintaining data consistency between the new catalog, legacy pricing, and various downstream systems requires complex real-time synchronization that frequently fails during peak traffic. Each additional component replacement exposes new dependencies that weren't apparent in the original system analysis. After 18 months, the team has replaced 40% of the legacy system but estimates that completing the modernization will take another 3 years due to increasing complexity. The hybrid system now requires more operational overhead than the original legacy system, performs worse during peak loads, and has introduced data consistency bugs that didn't exist before. The organization faces the difficult choice of abandoning the modernization effort or committing to years more work with uncertain outcomes.
