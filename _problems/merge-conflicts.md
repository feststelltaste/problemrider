---
title: Merge Conflicts
description: Multiple developers frequently modify the same large functions or files,
  creating version control conflicts that slow development.
category:
- Code
- Process
- Team
related_problems:
- slug: team-coordination-issues
  similarity: 0.65
- slug: conflicting-reviewer-opinions
  similarity: 0.6
- slug: duplicated-work
  similarity: 0.6
- slug: long-lived-feature-branches
  similarity: 0.55
- slug: duplicated-effort
  similarity: 0.55
- slug: duplicated-research-effort
  similarity: 0.55
layout: problem
---

## Description

Merge conflicts occur when multiple developers simultaneously modify the same portions of code, creating situations where version control systems cannot automatically reconcile the changes. While occasional conflicts are normal in collaborative development, frequent merge conflicts indicate underlying structural problems with the codebase or development process. These conflicts not only slow down individual developers but also create bottlenecks in the integration process and increase the risk of introducing bugs when resolving conflicts manually.

## Indicators ⟡
- Developers regularly encounter conflicts when pulling or merging changes
- The same files or functions are modified by multiple team members in most commits
- Resolving merge conflicts takes significant time and effort
- Code integration is delayed due to complex conflict resolution
- Developers express frustration about constantly fighting merge conflicts

## Symptoms ▲

- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Time spent resolving merge conflicts reduces the time available for actual feature development, slowing overall velocity.
- [Developer Frustration and Burnout](developer-frustration-and-burnout.md)
<br/>  Constantly fighting merge conflicts is tedious and frustrating, contributing to developer dissatisfaction.
- [Delayed Project Timelines](delayed-project-timelines.md)
<br/>  Complex merge conflicts create integration bottlenecks that delay feature delivery and project completion.

## Causes ▼
- [Long-Lived Feature Branches](long-lived-feature-branches.md)
<br/>  Branches that diverge from main for extended periods accumulate more differences, making conflicts more likely and complex.
- [Team Coordination Issues](team-coordination-issues.md)
<br/>  Poor coordination between team members leads to overlapping work on the same code areas without awareness.
- [Bloated Class](bloated-class.md)
<br/>  Multiple developers working on different features within the same bloated class frequently create merge conflicts.
- [Communication Breakdown](communication-breakdown.md)
<br/>  Teams unaware of each other's work modify the same code areas, creating frequent version control conflicts.
- [God Object Anti-Pattern](god-object-anti-pattern.md)
<br/>  Multiple developers frequently need to modify the same god object for different reasons, causing constant version control conflicts.
- [Large Feature Scope](large-feature-scope.md)
<br/>  Long-lived branches created by large features accumulate merge conflicts as the main branch evolves independently.
- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  Multiple teams working in the same codebase frequently encounter merge conflicts when modifying shared code.
- [Monolithic Functions and Classes](monolithic-functions-and-classes.md)
<br/>  Large functions and classes modified by multiple developers for different features frequently result in merge conflicts.
- [Organizational Structure Mismatch](organizational-structure-mismatch.md)
<br/>  Multiple teams working on the same monolithic codebase due to structural mismatch leads to frequent version control conflicts.

## Detection Methods ○
- **Version Control Analytics:** Monitor merge conflict frequency and resolution time through git statistics
- **Hotspot Analysis:** Identify files and functions that are modified most frequently across different branches
- **Conflict Resolution Time Tracking:** Measure time spent resolving conflicts versus time spent on actual development
- **Developer Feedback:** Survey team members about their experience with merge conflicts and integration challenges
- **Code Ownership Analysis:** Identify areas where multiple developers regularly make changes simultaneously

## Examples

A web application has a central `UserService` class that handles user authentication, profile management, permissions, notifications, and activity logging. Three developers working on different features all need to modify this class simultaneously - one adding social login, another implementing user preferences, and a third adding audit logging. Every pull request touching this class creates merge conflicts that require careful manual resolution, and the team spends hours each week dealing with conflicts in this single file. Another example involves a configuration management system where all application settings are stored in a single large JSON configuration file. As different team members add new features requiring configuration options, they constantly conflict when trying to add their settings to the same file, requiring manual merging that sometimes results in malformed JSON or missing configuration values.
