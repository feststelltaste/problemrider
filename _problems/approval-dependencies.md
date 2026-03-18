---
title: Approval Dependencies
description: Work progress is frequently blocked by the need for approvals from specific
  individuals, creating bottlenecks and delays.
category:
- Dependencies
- Process
- Process
related_problems:
- slug: work-blocking
  similarity: 0.75
- slug: delayed-decision-making
  similarity: 0.7
- slug: rushed-approvals
  similarity: 0.6
- slug: decision-avoidance
  similarity: 0.6
- slug: decision-paralysis
  similarity: 0.6
- slug: cascade-delays
  similarity: 0.6
layout: problem
---

## Description

Approval dependencies occur when work cannot proceed without explicit approval from specific individuals, creating bottlenecks and delays in the development process. This problem is particularly acute when approvals are required for routine decisions, when approval authorities are frequently unavailable, or when approval processes are unnecessarily complex. These dependencies can bring entire teams to a standstill while waiting for authorization.

## Indicators ⟡

- Work frequently stops while waiting for approvals from specific people
- Approval requests accumulate faster than they can be processed
- Simple decisions require approval from senior management
- Team members spend significant time seeking approvals rather than doing productive work
- Approval processes vary inconsistently based on who is available

## Symptoms ▲

- [Delayed Project Timelines](delayed-project-timelines.md)
<br/>  Work blocked by pending approvals directly delays project timelines as tasks cannot proceed.
- [Cascade Delays](cascade-delays.md)
<br/>  A single blocked approval can delay downstream tasks, creating cascading delays throughout the project.
- [Reduced Team Productivity](reduced-team-productivity.md)
<br/>  Team members idle while waiting for approvals, directly reducing overall team productivity.
- [Developer Frustration and Burnout](developer-frustration-and-burnout.md)
<br/>  Repeatedly being blocked by approval processes frustrates developers and contributes to burnout.
- [Context Switching Overhead](context-switching-overhead.md)
<br/>  Developers forced to switch to other tasks while waiting for approvals lose focus and efficiency.
- [Rushed Approvals](rushed-approvals.md)
<br/>  Backlog pressure from accumulated approval requests leads approvers to rush through decisions.
- [Bottleneck Formation](bottleneck-formation.md)
<br/>  Concentrating approval authority in a few individuals creates structural bottlenecks that block work.
## Causes ▼

- [Micromanagement Culture](micromanagement-culture.md)
<br/>  A culture of micromanagement requires approvals for routine decisions that teams should be empowered to make independently.
- [Poorly Defined Responsibilities](poorly-defined-responsibilities.md)
<br/>  Unclear responsibility boundaries mean decisions get escalated to approvers unnecessarily.
- [Fear of Failure](fear-of-failure.md)
<br/>  Organizations fearful of mistakes create excessive approval requirements as a risk mitigation strategy.
## Detection Methods ○

- **Approval Queue Tracking:** Monitor how many approval requests are pending and for how long
- **Work Blocking Analysis:** Track how often work is blocked waiting for approvals
- **Approval Response Time:** Measure how long it takes to get approvals for different types of decisions
- **Decision Type Analysis:** Categorize what types of decisions require approval versus those that don't
- **Team Productivity Impact:** Assess how approval dependencies affect overall team productivity

## Examples

A development team must get approval from their director for any database schema changes, even minor ones like adding an index or renaming a column. The director is frequently in meetings or traveling, so schema change requests often wait 1-2 weeks for approval while development work is blocked. Simple performance optimizations that could be implemented in an hour instead take weeks to complete because of the approval bottleneck. Another example involves a team where any deployment to production requires approval from both the security team and the operations team, but there's no coordination between these approvals, so deployments often get approved by one team but delayed by the other, creating unpredictable deployment schedules and forcing developers to maintain multiple versions of their changes.
