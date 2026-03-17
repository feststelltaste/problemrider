---
title: Team Coordination Issues
description: A situation where multiple developers or teams have difficulty working
  together on the same codebase.
category:
- Process
- Team
related_problems:
- slug: team-dysfunction
  similarity: 0.75
- slug: poor-teamwork
  similarity: 0.7
- slug: team-confusion
  similarity: 0.7
- slug: inconsistent-codebase
  similarity: 0.65
- slug: team-silos
  similarity: 0.65
- slug: duplicated-effort
  similarity: 0.65
layout: problem
---

## Description
Team coordination issues arise when multiple developers or teams have to work on the same codebase and have difficulty coordinating their work. This can lead to merge conflicts, duplicated effort, and a general slowdown in the pace of development. Team coordination issues are often a sign of a monolithic architecture, where everything is tightly coupled and it is difficult to work on different parts of the system in isolation.

## Indicators ⟡
- Frequent merge conflicts.
- Developers are often blocked waiting for other developers to finish their work.
- There is a lot of duplicated effort.
- It is difficult to get a clear picture of the overall status of the project.

## Symptoms ▲

- [Duplicated Effort](duplicated-effort.md)
<br/>  Without coordination, multiple developers independently solve the same problems, wasting effort.
- [Merge Conflicts](merge-conflicts.md)
<br/>  Poor coordination leads developers to make conflicting changes to the same code areas, resulting in frequent merge conflicts.
- [Reduced Team Productivity](reduced-team-productivity.md)
<br/>  Developers blocked waiting for others and time spent resolving conflicts directly reduces team output.
- [Inconsistent Codebase](inconsistent-codebase.md)
<br/>  Uncoordinated development leads to different approaches and patterns being used for similar problems across the codebase.
- [Delayed Value Delivery](delayed-value-delivery.md)
<br/>  Coordination overhead, blocking dependencies, and conflict resolution slow down the overall delivery pace.
## Causes ▼

- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  A monolithic architecture forces all teams to work in the same codebase, making coordination essential but difficult.
- [Tight Coupling Issues](tight-coupling-issues.md)
<br/>  When components are tightly coupled, changes in one area frequently affect others, requiring careful coordination between developers.
- [Team Silos](team-silos.md)
<br/>  Teams working in isolation lack awareness of what others are doing, making coordination on shared codebases difficult.
- [Unclear Documentation Ownership](unclear-documentation-ownership.md)
<br/>  When code ownership boundaries are undefined, developers don't know who to coordinate with about changes.
## Detection Methods ○
- **Version Control Metrics:** Use tools to measure the number of merge conflicts and the amount of time that developers spend resolving them.
- **Developer Surveys:** Ask developers if they feel like they are able to work effectively with other developers on the team.
- **Project Management Metrics:** Track the amount of time that developers spend waiting for other developers to finish their work.

## Examples
A company has a large, monolithic e-commerce application. The front-end team and the back-end team are constantly stepping on each other's toes. The front-end team wants to make changes to the UI, but they have to wait for the back-end team to make changes to the API. The back-end team is busy working on other features, so the front-end team is often blocked. This leads to a lot of frustration and a slowdown in the pace of development.
