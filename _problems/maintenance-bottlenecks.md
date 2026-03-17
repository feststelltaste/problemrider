---
title: Maintenance Bottlenecks
description: A situation where a small number of developers are the only ones who
  can make changes to a critical part of the system.
category:
- Code
- Process
- Team
related_problems:
- slug: bottleneck-formation
  similarity: 0.75
- slug: maintenance-paralysis
  similarity: 0.7
- slug: legacy-skill-shortage
  similarity: 0.65
- slug: single-points-of-failure
  similarity: 0.65
- slug: review-bottlenecks
  similarity: 0.65
- slug: maintenance-overhead
  similarity: 0.65
layout: problem
---

## Description
A maintenance bottleneck occurs when a small number of developers, or even a single developer, are the only ones who have the knowledge and expertise to maintain a critical part of the system. This creates a single point of failure and can significantly slow down the pace of development. It also puts a great deal of stress on the developers who are the bottlenecks.

## Indicators ⟡
- A small number of developers are consistently assigned to work on a specific part of the system.
- Other developers are hesitant to make changes to that part of the system.
- The developers who are the bottlenecks are often overloaded with work.
- There is a lack of documentation for that part of the system.

## Symptoms ▲

- [Slow Development Velocity](slow-development-velocity.md)
<br/>  When only a few people can modify critical system parts, work queues up and overall development speed drops.
- [Increased Stress and Burnout](increased-stress-and-burnout.md)
<br/>  The few developers who are bottlenecks become overloaded with work, leading to stress and eventual burnout.
- [Single Points of Failure](single-points-of-failure.md)
<br/>  When only one or two people can maintain a critical system, their unavailability creates a direct single point of failure.
- [Delayed Issue Resolution](delayed-issue-resolution.md)
<br/>  Bug fixes and improvements are delayed because they must wait for the limited bottleneck developers to be available.
- [Work Blocking](work-blocking.md)
<br/>  Other team members are blocked from making progress on tasks that touch the bottlenecked system components.
## Causes ▼

- [Knowledge Silos](knowledge-silos.md)
<br/>  Knowledge concentrated in a few individuals about critical system parts creates the bottleneck condition.
- [Implicit Knowledge](implicit-knowledge.md)
<br/>  When system knowledge exists only in developers' heads rather than documentation, new developers cannot contribute to those areas.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  When a system uses obsolete technologies, few developers have the required skills to maintain it.
- [Difficult Code Comprehension](difficult-code-comprehension.md)
<br/>  Complex, poorly documented code discourages other developers from learning and working on the system.
## Detection Methods ○
- **Bus Factor Analysis:** Identify the key developers who are the only ones who know how to work on a critical part of the system.
- **Code Ownership Analysis:** Use tools to identify the developers who have made the most changes to a specific part of the system.
- **Developer Surveys:** Ask developers if they feel like there are any parts of the system that they are afraid to change.

## Examples
A company has a legacy billing system that was written by a single developer who has since left the company. Now, only one other developer on the team understands how the system works. This developer is constantly being pulled away from their other work to fix bugs and make changes to the billing system. The other developers on the team are afraid to touch the billing system because they don't understand it and they are afraid of breaking it. As a result, the billing system has become a major bottleneck for the company.
