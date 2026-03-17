---
title: Extended Cycle Times
description: The time from when work begins until it's completed and delivered becomes
  much longer than the actual work time required.
category:
- Process
related_problems:
- slug: extended-review-cycles
  similarity: 0.7
- slug: extended-research-time
  similarity: 0.65
- slug: delayed-project-timelines
  similarity: 0.65
- slug: long-release-cycles
  similarity: 0.65
- slug: increased-time-to-market
  similarity: 0.65
- slug: uneven-work-flow
  similarity: 0.6
layout: problem
---

## Description

Extended cycle times occur when the total time from work initiation to completion is significantly longer than the actual time spent working on the task. This indicates that work spends more time waiting in queues, blocked by dependencies, or stalled in processes than being actively worked on. Extended cycle times reduce responsiveness to business needs and indicate inefficiencies in the development process.

## Indicators ⟡

- Total time from task start to completion is many times longer than actual work time
- Work items spend more time "in progress" than being actively worked on
- Tasks remain in the same status for extended periods without progress
- Small changes take weeks or months to complete despite requiring hours of work
- Team can identify significant waiting periods in their work process

## Symptoms ▲

- [Delayed Value Delivery](delayed-value-delivery.md)
<br/>  When cycle times are extended, users wait much longer to receive features and fixes, reducing the product's responsiveness to needs.
- [Increased Time to Market](increased-time-to-market.md)
<br/>  Extended cycle times directly translate to longer time-to-market for new features and products.
- [Stakeholder Frustration](stakeholder-frustration.md)
<br/>  Stakeholders become frustrated when small changes take weeks or months to reach production despite requiring only hours of work.
- [Missed Deadlines](missed-deadlines.md)
<br/>  Extended waiting times throughout the process cause tasks to consistently miss their estimated completion dates.
- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Long cycle times reduce the team's apparent velocity since work items sit in queues rather than being completed.
- [Competitive Disadvantage](competitive-disadvantage.md)
<br/>  Slow delivery cycles prevent the team from responding quickly to market changes and user needs, giving competitors an edge.

## Causes ▼
- [Extended Review Cycles](extended-review-cycles.md)
<br/>  Multiple rounds of code review feedback and revision add significant waiting time to the overall cycle.
- [Approval Dependencies](approval-dependencies.md)
<br/>  Required approvals from specific individuals create queues and delays that extend the total cycle time.
- [Review Bottlenecks](review-bottlenecks.md)
<br/>  The code review process becoming a bottleneck adds waiting time that inflates overall cycle times.
- [Long Release Cycles](long-release-cycles.md)
<br/>  Infrequent release windows mean completed work sits waiting for the next deployment opportunity.
- [Complex Deployment Process](complex-deployment-process.md)
<br/>  Manual and complicated deployment processes add significant time between code completion and production delivery.
- [Extended Research Time](extended-research-time.md)
<br/>  Research overhead adds substantial time to the overall cycle, as tasks take much longer than the actual implementation work.
- [Work Queue Buildup](work-queue-buildup.md)
<br/>  Tasks spend more time waiting in queues than being actively worked on, dramatically increasing total cycle time.

## Detection Methods ○

- **Cycle Time Measurement:** Track total time from work start to completion
- **Flow Efficiency Analysis:** Calculate ratio of work time to total cycle time
- **Wait Time Tracking:** Identify how much time tasks spend waiting versus being worked on
- **Process Step Analysis:** Measure time spent at each stage of the development process
- **Comparative Analysis:** Compare cycle times for similar work items to identify patterns

## Examples

A simple bug fix that requires 2 hours of development time takes 6 weeks to reach production due to extended code review queues, deployment approval processes, and monthly release cycles. The actual work is completed quickly, but the fix spends most of its time waiting in various queues and approval processes. Another example involves a feature request that takes 3 months from approval to delivery, despite only requiring 1 week of actual development work. The extended cycle time is caused by waiting for design approval, development queue backlogs, testing bottlenecks, and deployment scheduling constraints.
