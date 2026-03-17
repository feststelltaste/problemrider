---
title: Reduced Code Submission Frequency
description: Developers batch multiple changes together or delay submissions to avoid
  frequent code review cycles, reducing feedback quality and integration frequency.
category:
- Process
- Team
related_problems:
- slug: reduced-review-participation
  similarity: 0.6
- slug: inadequate-code-reviews
  similarity: 0.6
- slug: review-bottlenecks
  similarity: 0.6
- slug: large-pull-requests
  similarity: 0.6
- slug: fear-of-change
  similarity: 0.6
- slug: review-process-avoidance
  similarity: 0.6
layout: problem
---

## Description

Reduced code submission frequency occurs when developers intentionally batch multiple changes together or delay submitting code for review to avoid the overhead and frustration of frequent review cycles. While this might seem efficient from an individual perspective, it leads to larger, more complex changes that are harder to review effectively, increases integration risks, and reduces the collaborative benefits of frequent feedback.

## Indicators ⟡

- Developers submit large pull requests containing multiple unrelated changes
- Days or weeks pass between code submissions from active developers
- Team members mention waiting to "finish everything" before submitting for review
- Pull request sizes are consistently larger than team guidelines recommend
- Developers express reluctance to submit work-in-progress or incremental changes

## Symptoms ▲

- [Large Pull Requests](large-pull-requests.md)
<br/>  Batching multiple changes together to avoid frequent review cycles directly produces oversized pull requests.
- [Inadequate Code Reviews](inadequate-code-reviews.md)
<br/>  Large batched submissions are harder to review thoroughly, reducing the quality and effectiveness of code reviews.
- [Integration Difficulties](integration-difficulties.md)
<br/>  Infrequent code submissions increase the likelihood of merge conflicts and integration issues with teammates' work.
- [No Continuous Feedback Loop](no-continuous-feedback-loop.md)
<br/>  Less frequent submissions mean developers get feedback later, when design decisions are harder to change.
- [Regression Bugs](regression-bugs.md)
<br/>  Large, complex changes submitted infrequently are more likely to introduce regressions that are difficult to isolate.
## Causes ▼

- [Review Bottlenecks](review-bottlenecks.md)
<br/>  Slow review processes discourage frequent submissions as developers avoid waiting repeatedly for reviews.
- [Review Process Avoidance](review-process-avoidance.md)
<br/>  Frustration with the review process leads developers to minimize their exposure to it by batching changes.
- [Fear of Change](fear-of-change.md)
<br/>  Anxiety about submitting code that might be criticized causes developers to delay submissions until they feel everything is perfect.
- [Code Review Inefficiency](code-review-inefficiency.md)
<br/>  Inefficient review processes that focus on trivial issues discourage developers from submitting frequently.
## Detection Methods ○

- **Submission Frequency Tracking:** Monitor how often individual developers submit code for review
- **Pull Request Size Analysis:** Track the size and complexity of code submissions over time
- **Developer Behavior Surveys:** Collect feedback on reasons for batching changes or delaying submissions
- **Integration Frequency Measurement:** Assess how often code is integrated into main branches
- **Collaboration Pattern Analysis:** Evaluate whether reduced submissions correlate with decreased team collaboration

## Examples

A developer working on a new feature becomes frustrated after their first small pull request goes through four rounds of review with extensive style debates. For their next change, they decide to implement the entire feature, write all tests, update documentation, and handle three related bug fixes before submitting anything for review. The resulting 800-line pull request is difficult for reviewers to analyze comprehensively, contains multiple unrelated changes that should be evaluated separately, and takes two weeks to review instead of the few days each individual change would have required. Another example involves a team member who stops submitting daily progress because previous reviews focused heavily on minor formatting issues. They begin working for a week at a time before submitting, creating integration conflicts with teammates' work and making it harder for the team to provide early feedback on design decisions that are difficult to change later in the development process.
