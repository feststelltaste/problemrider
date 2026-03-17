---
title: Review Bottlenecks
description: The code review process becomes a significant bottleneck, delaying the
  delivery of new features and bug fixes.
category:
- Process
- Team
related_problems:
- slug: code-review-inefficiency
  similarity: 0.75
- slug: inadequate-code-reviews
  similarity: 0.75
- slug: review-process-breakdown
  similarity: 0.7
- slug: insufficient-code-review
  similarity: 0.7
- slug: team-members-not-engaged-in-review-process
  similarity: 0.7
- slug: maintenance-bottlenecks
  similarity: 0.65
layout: problem
---

## Description
Review bottlenecks occur when the code review process consistently slows down the development cycle. This can happen for a variety of reasons, such as having too few reviewers, large and complex pull requests, or a culture where reviews are not prioritized. When code reviews become a bottleneck, it can lead to frustration among developers, delayed releases, and a decrease in overall development velocity.

## Indicators ⟡
- Pull requests are sitting for a long time without being reviewed.
- Developers are frequently context-switching while waiting for reviews.
- The team has a low deployment frequency.
- There is a lot of pressure to approve pull requests quickly, even if they are not ready.

## Symptoms ▲

- [Slow Development Velocity](slow-development-velocity.md)
<br/>  When code reviews block the development pipeline, overall team velocity drops as developers wait for approvals.
- [Rushed Approvals](rushed-approvals.md)
<br/>  Pressure to clear the review backlog leads reviewers to approve changes hastily without thorough examination.
- [Review Process Avoidance](review-process-avoidance.md)
<br/>  Frustration with long review wait times motivates developers to find ways to bypass the review process entirely.
- [Delayed Value Delivery](delayed-value-delivery.md)
<br/>  Features and fixes that are ready but stuck in review queues cannot reach users, delaying value delivery.
- [Author Frustration](author-frustration.md)
<br/>  Developers become frustrated when their completed work sits idle in review queues for extended periods.

## Causes ▼
- [Reduced Review Participation](reduced-review-participation.md)
<br/>  When few team members participate in reviews, all review work falls on a small number of people, creating a bottleneck.
- [Large Pull Requests](large-pull-requests.md)
<br/>  Large pull requests take much longer to review thoroughly, consuming more reviewer time and creating backlogs.
- [Approval Dependencies](approval-dependencies.md)
<br/>  Requirements for specific individuals to approve changes create bottlenecks when those individuals are unavailable.
- [Time Pressure](time-pressure.md)
<br/>  Under time pressure, developers prioritize their own tasks over reviewing others' code, allowing review queues to grow.
- [Code Review Inefficiency](code-review-inefficiency.md)
<br/>  Slow and cumbersome review processes create bottlenecks that delay feature delivery.
- [Flaky Tests](flaky-tests.md)
<br/>  CI pipelines blocked by flaky test failures delay code review and merge processes.
- [Perfectionist Review Culture](perfectionist-review-culture.md)
<br/>  Perfectionist reviews take so long that the review process becomes a major bottleneck in the development pipeline.
- [Rapid Team Growth](rapid-team-growth.md)
<br/>  Code review queues become overwhelmed when too many new developers submit code with insufficient senior reviewers available.
- [Reviewer Anxiety](reviewer-anxiety.md)
<br/>  When anxious reviewers take excessively long on simple reviews or avoid reviewing entirely, review throughput drops and creates bottlenecks.
- [Team Members Not Engaged in Review Process](team-members-not-engaged-in-review-process.md)
<br/>  When only a few team members actively review code, pull requests queue up waiting for their attention.

## Detection Methods ○
- **Pull Request Lead Time:** Track the time it takes from when a pull request is created to when it is merged.
- **Reviewer Load:** Analyze the number of pull requests that are assigned to each reviewer.
- **Developer Surveys:** Ask developers about their experience with the code review process and whether they feel that it is a bottleneck.

## Examples
A team has a rule that all pull requests must be reviewed by two people. However, there are only two senior developers on the team who are qualified to review code. As a result, pull requests are often sitting for days or even weeks before they are reviewed. This is causing a lot of frustration among the junior developers, who are not able to get their code merged in a timely manner. In another example, a team has a culture where code reviews are not prioritized. Developers are expected to complete their own work before they review the code of others. This is leading to a situation where pull requests are often sitting for a long time before they are reviewed, which is slowing down the entire development process.
