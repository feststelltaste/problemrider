---
title: Inadequate Code Reviews
description: Code reviews are not consistently performed, are rushed, superficial,
  or fail to identify critical issues, leading to lower code quality and increased
  risk.
category:
- Code
- Process
related_problems:
- slug: review-process-breakdown
  similarity: 0.9
- slug: insufficient-code-review
  similarity: 0.85
- slug: superficial-code-reviews
  similarity: 0.8
- slug: code-review-inefficiency
  similarity: 0.75
- slug: team-members-not-engaged-in-review-process
  similarity: 0.75
- slug: review-bottlenecks
  similarity: 0.75
solutions:
- code-review-process-reform
layout: problem
---

## Description
Inadequate code reviews are a major contributor to poor software quality. This encompasses both superficial reviews that provide little meaningful feedback and inconsistent review practices. When code reviews are rushed, superficial, or performed by inexperienced reviewers, they are unlikely to catch bugs, design flaws, or deviations from best practices. Superficial reviews often focus on minor stylistic issues rather than critical logic or design flaws, providing little more than "looks good to me" approvals without thorough examination. This can lead to a gradual degradation of the codebase, as technical debt and potential issues are allowed to accumulate. A healthy code review culture is one where reviews are thorough, thoughtful, and performed by a diverse group of reviewers with shared responsibility for code quality.

## Indicators ⟡
- Code reviews are often a bottleneck in the development process.
- The same types of bugs are repeatedly found in production.
- Developers are not learning from each other through code reviews.
- There is a lot of debate about style and other trivial issues in code reviews.

## Symptoms ▲

- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  When code reviews fail to catch bugs and design flaws, more defects escape to production environments.
- [Inconsistent Codebase](inconsistent-codebase.md)
<br/>  Without thorough reviews enforcing standards, different coding styles and patterns proliferate unchecked across the codebase.
- [High Technical Debt](high-technical-debt.md)
<br/>  Superficial reviews allow shortcuts and poor design decisions to accumulate, increasing technical debt over time.
- [Regression Bugs](regression-bugs.md)
<br/>  Reviews that miss side effects and coupling issues allow changes that break existing functionality.
- [Lower Code Quality](lower-code-quality.md)
<br/>  Without meaningful review feedback, code quality steadily degrades as poor patterns go unchallenged.
- [Limited Team Learning](limited-team-learning.md)
<br/>  Superficial reviews eliminate the knowledge-sharing benefit of code reviews, reducing team learning opportunities.
- [Increased Risk of Bugs](increased-risk-of-bugs.md)
<br/>  When code reviews fail to catch issues, the risk of bugs reaching production increases directly.
- [Inadequate Initial Reviews](inadequate-initial-reviews.md)
<br/>  When code reviews are generally inadequate, first-round reviews become superficial, missing critical issues that must be caught in later rounds.
## Causes ▼

- [Time Pressure](time-pressure.md)
<br/>  Deadline pressure forces reviewers to rush through reviews, resulting in superficial examination of code changes.
- [Large Pull Requests](large-pull-requests.md)
<br/>  Oversized pull requests overwhelm reviewers, making thorough examination impractical and leading to superficial reviews.
- [Reviewer Inexperience](reviewer-inexperience.md)
<br/>  Inexperienced reviewers lack the expertise to identify deeper design, logic, or security issues during reviews.
- [Overworked Teams](overworked-teams.md)
<br/>  When teams are overloaded, code reviews are deprioritized and rushed to keep up with delivery demands.
## Detection Methods ○

- **Track Bug Density:** A high number of bugs in a particular module or feature may indicate that the code was not reviewed properly.
- **Analyze Code Review Comments:** Look for patterns in the comments to see if reviewers are focusing on the right things. Periodically review a sample of code review comments to assess their depth and focus.
- **Post-Mortems/Retrospectives:** When bugs escape to production, analyze whether they could have been caught in code review and why they weren't.
- **Developer Surveys:** Ask developers for their feedback on the code review process and about the quality of feedback they receive during reviews.
- **Code Quality Metrics:** Monitor metrics like bug density, technical debt, and code complexity, which can indirectly indicate review effectiveness.
- **Use Static Analysis Tools:** These tools can automatically identify many common issues, freeing up reviewers to focus on more important things.

## Examples
A junior developer submits a pull request with a significant performance issue. The reviewer, who is under pressure to meet a deadline, approves the pull request without noticing the issue. The performance issue is later discovered in production. A developer submits a pull request that introduces an N+1 query performance bottleneck. The code review focuses solely on whether the variable names adhere to the team's convention and the placement of curly braces, completely missing the performance issue.

In another case, a team has a rule that all pull requests must be reviewed by at least two people. However, in practice, the same two senior developers are always assigned to do the reviews, and they are often too busy to provide meaningful feedback. A security vulnerability is introduced in a new feature, but the code review only contains comments about code formatting, and the security flaw is only discovered much later during a penetration test. This problem is common in teams that are growing quickly, have high turnover, or are under pressure to deliver features quickly, or where the importance of code reviews as a quality gate and knowledge-sharing mechanism is not fully understood.
