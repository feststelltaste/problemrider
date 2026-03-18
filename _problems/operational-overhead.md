---
title: Operational Overhead
description: A significant amount of time and resources are spent on emergency response
  and firefighting, rather than on planned development and innovation.
category:
- Code
- Process
related_problems:
- slug: maintenance-overhead
  similarity: 0.7
- slug: budget-overruns
  similarity: 0.65
- slug: context-switching-overhead
  similarity: 0.65
- slug: constant-firefighting
  similarity: 0.65
- slug: development-disruption
  similarity: 0.6
- slug: poor-operational-concept
  similarity: 0.6
layout: problem
---

## Description
Operational overhead is the indirect cost of running a software system. This includes the cost of things like monitoring, logging, alerting, and on-call support. When operational overhead is high, it can be a major drain on the resources of a company. It can also be a major source of stress and frustration for the development team. High operational overhead is often a symptom of a complex and unstable system. It is a sign that the team is spending too much time on reactive work and not enough time on proactive work.

## Indicators ⟡
- The on-call team is constantly being paged.
- The development team is spending a lot of time on operational tasks.
- The cost of monitoring and logging is high.
- There is a general sense of chaos and urgency in the team's daily work.

## Symptoms ▲

- [Reduced Team Productivity](reduced-team-productivity.md)
<br/>  Time spent on operational tasks like monitoring, incident response, and firefighting directly reduces time available for productive development work.
- [Developer Frustration and Burnout](developer-frustration-and-burnout.md)
<br/>  Constant operational demands and firefighting create stress and frustration, leading to burnout among team members.
- [Inability to Innovate](inability-to-innovate.md)
<br/>  When teams are consumed by operational tasks, they have no capacity to explore improvements or innovative approaches.
- [Budget Overruns](budget-overruns.md)
<br/>  High operational overhead consumes resources that were budgeted for development, leading to cost overruns.
- [Delayed Value Delivery](delayed-value-delivery.md)
<br/>  Operational demands divert the team from planned feature work, delaying delivery of new value to users.
## Causes ▼

- [Poor Operational Concept](poor-operational-concept.md)
<br/>  Lack of planning for monitoring, maintenance, and support creates reactive operational patterns that consume excessive resources.
- [Monitoring Gaps](monitoring-gaps.md)
<br/>  Insufficient monitoring means issues are detected late, requiring more effort to diagnose and resolve, increasing operational overhead.
- [High Technical Debt](high-technical-debt.md)
<br/>  Accumulated technical debt creates a fragile system that generates frequent production issues, increasing operational burden.
- [System Outages](system-outages.md)
<br/>  Frequent system outages require emergency response and incident management, directly driving operational overhead.
- [Constant Firefighting](constant-firefighting.md)
<br/>  Constant firefighting is a direct driver of operational overhead, consuming team resources on reactive work rather th....
## Detection Methods ○
- **On-Call Load:** Track the number of pages that the on-call team receives.
- **Time Spent on Operational Tasks:** Track the amount of time that the development team spends on operational tasks.
- **Cost of Monitoring and Logging:** Track the cost of your monitoring and logging tools.
- **Mean Time to Resolution (MTTR):** Measure the average time it takes to resolve a production issue.

## Examples
A company is running a large, distributed system. The system is complex and difficult to understand. The on-call team is constantly being paged to deal with production issues. The development team is spending a lot of time on operational tasks, such as debugging production issues and adding more logging. As a result, the team is making very little progress on new features, and the cost of running the system is high.
