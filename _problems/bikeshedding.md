---
title: Bikeshedding
description: Reviewers focus on trivial issues like whitespace and variable names
  instead of more important issues like logic and design.
category:
- Process
- Team
related_problems:
- slug: incomplete-knowledge
  similarity: 0.6
- slug: nitpicking-culture
  similarity: 0.6
- slug: team-coordination-issues
  similarity: 0.55
- slug: time-pressure
  similarity: 0.55
- slug: unproductive-meetings
  similarity: 0.55
- slug: development-disruption
  similarity: 0.55
layout: problem
---

## Description
Bikeshedding, also known as Parkinson's law of triviality, is a phenomenon where a disproportionate amount of time and energy is spent on trivial and insignificant details, while more important and complex issues are neglected. This often occurs in meetings where participants avoid challenging topics and instead focus on easy-to-understand but ultimately unimportant details. Bikeshedding is a major time-waster, and it can be a sign of a dysfunctional team culture.

## Symptoms ▲

- [Unproductive Meetings](unproductive-meetings.md)
<br/>  Meetings devolve into debates over trivial topics while important design and logic issues go unaddressed.
- [Missed Deadlines](missed-deadlines.md)
<br/>  Time wasted on trivial review discussions delays feature delivery and causes schedule slips.
- [Development Disruption](development-disruption.md)
<br/>  Excessive back-and-forth on trivial review comments disrupts developers' focus and workflow.

## Causes ▼
- [Incomplete Knowledge](incomplete-knowledge.md)
<br/>  Reviewers who lack deep understanding of the code's logic focus on surface-level issues they can easily evaluate.
- [Avoidance Behaviors](avoidance-behaviors.md)
<br/>  Reviewers avoid the cognitive effort of evaluating complex logic and instead focus on easy-to-judge trivial matters.

## Detection Methods ○

- **Code Review Metrics:** Analyze the types of comments made in pull requests (e.g., ratio of stylistic comments to logical/design comments).
- **Developer Surveys:** Ask developers about their perception of code review effectiveness and common feedback types.
- **Retrospectives:** Discuss code review processes and identify recurring frustrations or inefficiencies.
- **Reviewer Training:** Observe if training on effective code review practices improves the quality of feedback.

## Examples

- **Scenario:** A developer submits a pull request that introduces a new, complex algorithm. The review discussion spans days, with 80% of the comments debating whether to use single or double quotes for strings, while a critical edge case in the algorithm goes unnoticed.
- **Specific Instance:** A team implements a new feature, and during the code review, a senior developer spends an hour debating the naming convention for a private helper function, even though the project has a linter that could enforce such rules automatically.
- **Context:** This problem often arises when teams lack clear processes, automated tooling, or sufficient training for code reviews. It can significantly hinder development velocity and prevent the team from focusing on what truly matters for code quality and project success.
