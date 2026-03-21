---
title: Work Blocking
description: Development tasks cannot proceed without pending approvals, creating
  bottlenecks and delays in the development process.
category:
- Management
- Process
related_problems:
- slug: approval-dependencies
  similarity: 0.75
- slug: work-queue-buildup
  similarity: 0.65
- slug: delayed-decision-making
  similarity: 0.65
- slug: bottleneck-formation
  similarity: 0.65
- slug: decision-paralysis
  similarity: 0.65
- slug: decision-avoidance
  similarity: 0.6
solutions:
- team-autonomy-and-empowerment
- sustainable-pace-practices
layout: problem
---

## Description

Work blocking occurs when development tasks cannot move forward because they require approvals, decisions, or inputs that are delayed or unavailable. This creates a bottleneck effect where developers and teams sit idle or switch to less productive work while waiting for permission to proceed. Work blocking often indicates over-centralized decision-making, unclear authority structures, or processes that prioritize control over productivity.

## Indicators ⟡

- Developers frequently report being "blocked" on tasks during stand-up meetings
- Tasks remain in "waiting for approval" status for extended periods
- Team velocity decreases due to context switching while waiting for decisions
- Developers work on lower-priority tasks while higher-priority work is blocked
- Multiple team members are dependent on the same person or process for approvals

## Symptoms ▲

- [Context Switching Overhead](context-switching-overhead.md)
<br/>  When developers are blocked waiting for approvals, they switch to lower-priority tasks, incurring cognitive overhead from frequent context changes.
- [Delayed Project Timelines](delayed-project-timelines.md)
<br/>  Tasks stuck in blocked status directly delay project milestones and delivery schedules.
- [Work Queue Buildup](work-queue-buildup.md)
<br/>  Blocked work items accumulate in queues, creating backlogs at approval and decision points.
- [Developer Frustration and Burnout](developer-frustration-and-burnout.md)
<br/>  Developers waiting idly for approvals on important work experience frustration and decreased morale over time.
- [Reduced Team Productivity](reduced-team-productivity.md)
<br/>  Time spent waiting for approvals rather than producing code directly reduces team output.
- [Workaround Culture](workaround-culture.md)
<br/>  When proper changes are blocked by approval processes, developers resort to workarounds that bypass the blocking process.
- [Wasted Development Effort](wasted-development-effort.md)
<br/>  When developers are blocked, they switch to lower-priority tasks or do work that may be invalidated once the blocking....
## Causes ▼

- [Approval Dependencies](approval-dependencies.md)
<br/>  Work blocking is directly caused by processes that require specific individuals to approve before work can proceed.
- [Bottleneck Formation](bottleneck-formation.md)
<br/>  Centralized decision-making or scarce reviewer availability creates bottlenecks that block work items.
- [Decision Avoidance](decision-avoidance.md)
<br/>  When decision-makers avoid or defer decisions, work that depends on those decisions remains blocked.
- [Micromanagement Culture](micromanagement-culture.md)
<br/>  Excessive management oversight requiring approval for routine decisions creates unnecessary blocking of development tasks.
## Detection Methods ○

- **Blocking Time Tracking:** Monitor how much time tasks spend in blocked status
- **Approval Queue Analysis:** Track the volume and processing time of different types of approval requests
- **Developer Surveys:** Ask team members about their experience with approvals and decision-making autonomy
- **Stand-up Meeting Analysis:** Count frequency of "blocked" status reports and reasons
- **Decision Authority Mapping:** Identify decision types that require approval vs. those that can be made independently
- **Flow Efficiency Measurement:** Calculate percentage of time work items are actively progressing vs. waiting

## Examples

A development team needs approval from the architecture committee for any database schema changes. The committee meets once per week, and decisions often require additional documentation or clarification, leading to multi-week delays for simple changes like adding an index. Developers end up working on less important tasks while critical performance improvements are blocked. Another example involves a mobile app team that must get UI design approval from a design director who travels frequently. Simple layout adjustments that could be implemented in hours instead wait weeks for approval, forcing developers to work around incomplete designs or delay feature releases.
