---
title: Procrastination on Complex Tasks
description: Difficult or cognitively demanding work is consistently postponed in
  favor of easier, more immediately gratifying tasks.
category:
- Culture
- Process
related_problems:
- slug: avoidance-behaviors
  similarity: 0.8
- slug: decision-avoidance
  similarity: 0.6
- slug: cognitive-overload
  similarity: 0.6
- slug: accumulation-of-workarounds
  similarity: 0.55
- slug: difficult-code-comprehension
  similarity: 0.55
- slug: delayed-issue-resolution
  similarity: 0.55
solutions:
- iterative-development
layout: problem
---

## Description

Procrastination on complex tasks occurs when developers consistently delay or avoid starting difficult, cognitively demanding, or uncertain work in favor of easier, more immediately satisfying activities. This behavior often stems from psychological factors such as fear of failure, perfectionism, or cognitive overload. While some level of task preference is natural, systematic procrastination on complex work can lead to accumulation of difficult problems and increased technical debt.

## Indicators ⟡

- Difficult tasks remain unstarted while easier tasks are completed quickly
- Team members find reasons to work on other activities when assigned complex problems
- Complex features consistently slip to later sprints or iterations
- Developers express anxiety or stress when discussing challenging work
- Simple bugs get fixed immediately while complex issues remain in the backlog

## Symptoms ▲

- [Accumulation of Workarounds](accumulation-of-workarounds.md)
<br/>  Instead of tackling the hard fix, developers create workarounds that add complexity to the system.
- [Delayed Issue Resolution](delayed-issue-resolution.md)
<br/>  Complex issues sit unaddressed in the backlog for months because they are consistently deferred.
- [Increasing Brittleness](increasing-brittleness.md)
<br/>  Deferred architectural work makes the system more fragile as changes accumulate around problematic areas.
- [Time Pressure](time-pressure.md)
<br/>  Postponed complex work eventually becomes urgent, creating last-minute deadline pressure.
## Causes ▼

- [Cognitive Overload](cognitive-overload.md)
<br/>  Mental exhaustion from system complexity makes developers avoid additional cognitive burden of hard tasks.
- [Difficult Code Comprehension](difficult-code-comprehension.md)
<br/>  When code is hard to understand, the perceived difficulty of complex tasks increases, encouraging avoidance.
## Detection Methods ○

- **Task Start Delay Analysis:** Measure time between task assignment and when work actually begins
- **Complexity vs Completion Time:** Compare task complexity ratings with actual completion patterns
- **Backlog Age by Complexity:** Track how long complex versus simple tasks remain in the backlog
- **Developer Feedback Surveys:** Ask about factors that influence task selection and avoidance
- **Sprint Planning Behavior:** Observe which tasks are volunteered for versus avoided during planning

## Examples

A development team has three architectural refactoring tasks in their backlog that have been there for four months, while dozens of smaller feature additions and bug fixes have been completed during the same period. Team members consistently volunteer for the smaller tasks during sprint planning and find reasons why the refactoring work "isn't quite ready" or "needs more analysis." The avoided refactoring becomes increasingly urgent as the system becomes harder to maintain, but by the time it must be addressed, the work has become even more complex and risky due to changes made around the problematic areas. Another example involves a developer who needs to implement a complex algorithm for data processing but keeps finding other tasks to work on first - updating documentation, fixing minor UI issues, optimizing database queries. The algorithm implementation remains untouched for weeks while the developer stays busy with less challenging work, eventually causing the feature to miss its deadline and requiring emergency weekend work to complete.
