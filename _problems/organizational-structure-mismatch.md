---
title: Organizational Structure Mismatch
description: A situation where the structure of the organization does not match the
  architecture of the system.
category:
- Architecture
- Process
- Team
related_problems:
- slug: architectural-mismatch
  similarity: 0.75
- slug: team-coordination-issues
  similarity: 0.65
- slug: capacity-mismatch
  similarity: 0.6
- slug: team-dysfunction
  similarity: 0.6
- slug: scaling-inefficiencies
  similarity: 0.55
- slug: inadequate-mentoring-structure
  similarity: 0.55
layout: problem
---

## Description
An organizational structure mismatch is a situation where the structure of the organization does not match the architecture of the system. This is a common problem in companies that have a monolithic architecture but are organized into small, autonomous teams. An organizational structure mismatch can lead to a number of problems, including team coordination issues, communication breakdowns, and a slowdown in development velocity.

## Indicators ⟡
- The teams are organized around features, but the architecture is monolithic.
- The teams are constantly stepping on each other's toes.
- There is a lot of duplicated effort.
- It is difficult to get a clear picture of the overall status of the project.

## Symptoms ▲

- [Team Coordination Issues](team-coordination-issues.md)
<br/>  When organizational boundaries don't align with system architecture, teams must constantly coordinate across mismatched boundaries.
- [Merge Conflicts](merge-conflicts.md)
<br/>  Multiple teams working on the same monolithic codebase due to structural mismatch leads to frequent version control conflicts.
- [Duplicated Effort](duplicated-effort.md)
<br/>  Teams working in misaligned structures unknowingly duplicate work because ownership boundaries are unclear.
- [Communication Breakdown](communication-breakdown.md)
<br/>  Misalignment between organizational and system structure creates unclear communication channels, leading to information loss and miscoordination.
- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Teams stepping on each other's toes and excessive cross-team coordination slow down the overall pace of development.

## Causes ▼
- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  A monolithic architecture forces multiple autonomous teams to work on the same codebase, creating the mismatch between team structure and system boundaries.
- [Stagnant Architecture](stagnant-architecture.md)
<br/>  When the system architecture doesn't evolve alongside organizational changes, the mismatch between structure and architecture grows.
- [Rapid Team Growth](rapid-team-growth.md)
<br/>  Rapid expansion of teams without corresponding architectural changes creates misalignment between organizational and system structure.

## Detection Methods ○
- **Architectural Diagrams:** Create a diagram of the system architecture to identify how the system is structured.
- **Organizational Charts:** Create a chart of the organization to identify how the teams are structured.
- **Developer Surveys:** Ask developers if they feel like they are able to work effectively with other teams.

## Examples
A company has a large, monolithic e-commerce application. The company is organized into a number of small, autonomous teams. Each team is responsible for a different feature of the application. The teams are constantly stepping on each other's toes because they are all working on the same codebase. This leads to a lot of frustration and a slowdown in the pace of development.
