---
title: Lower Code Quality
description: Burned-out or rushed developers are more likely to make mistakes, leading
  to an increase in defects.
category:
- Code
- Process
- Team
related_problems:
- slug: quality-compromises
  similarity: 0.65
- slug: quality-degradation
  similarity: 0.65
- slug: inadequate-code-reviews
  similarity: 0.65
- slug: insufficient-code-review
  similarity: 0.6
- slug: team-churn-impact
  similarity: 0.6
- slug: reduced-code-submission-frequency
  similarity: 0.55
solutions:
- code-reviews
- pair-and-mob-programming
- definition-of-done
- static-code-analysis
- code-metrics
- refactoring-katas
layout: problem
---

## Description

Lower code quality occurs when various pressures and circumstances cause developers to produce code that doesn't meet established standards for maintainability, reliability, or correctness. This degradation often results from burnout, time pressure, lack of motivation, or systemic issues that prevent developers from applying their best practices. Unlike isolated quality issues, this represents a systematic decline in the overall standard of code being produced by the team.

## Indicators ⟡
- Code review comments increasingly focus on basic quality issues
- Bug rates increase even for experienced developers
- Coding standards are frequently ignored or inconsistently applied
- Technical debt accumulates faster than it's addressed
- Developers express frustration about not having time to "do things right"

## Symptoms ▲

- [High Bug Introduction Rate](high-bug-introduction-rate.md)
<br/>  Lower quality code contains more defects, directly increasing the rate at which new bugs are introduced into the system.
- [Maintenance Overhead](maintenance-overhead.md)
<br/>  Poorly written code requires more effort to understand, modify, and fix, increasing ongoing maintenance burden.
- [High Technical Debt](high-technical-debt.md)
<br/>  Consistently lower code quality accumulates technical debt as shortcuts and poor implementations pile up.
- [Regression Bugs](regression-bugs.md)
<br/>  Code written without proper care, testing, or design is more fragile and likely to cause regressions when modified.
- [Increasing Brittleness](increasing-brittleness.md)
<br/>  Poor quality code with missing error handling, weak abstractions, and poor structure becomes increasingly fragile over time.
## Causes ▼

- [Developer Frustration and Burnout](developer-frustration-and-burnout.md)
<br/>  Burned-out developers lack the motivation and mental energy to write high-quality code, leading to shortcuts and mistakes.
- [Time Pressure](time-pressure.md)
<br/>  Pressure to deliver quickly forces developers to cut corners on code quality, testing, and design.
- [Deadline Pressure](deadline-pressure.md)
<br/>  Aggressive deadlines cause developers to skip best practices like code reviews, testing, and refactoring.
- [Team Churn Impact](team-churn-impact.md)
<br/>  Loss of experienced developers leaves less experienced team members producing lower quality code without adequate guidance.
- [Superficial Code Reviews](superficial-code-reviews.md)
<br/>  When code reviews fail to catch quality issues, lower quality code gets merged unchallenged, normalizing poor standards.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Inexperienced developers lacking knowledge of best practices naturally produce lower quality code with more defects a....
## Detection Methods ○
- **Static Code Analysis:** Use automated tools to measure code quality metrics over time
- **Code Review Metrics:** Track the number and types of issues found during code reviews
- **Bug Density Analysis:** Monitor defect rates and their correlation with code complexity
- **Technical Debt Tracking:** Measure the accumulation of technical debt over time
- **Developer Feedback:** Survey team members about their ability to maintain quality standards

## Examples

A software development team is under intense pressure to deliver a major feature before a competitor launches their version. Management repeatedly emphasizes that missing the deadline would be catastrophic for the business. Under this pressure, developers begin skipping unit tests, ignoring coding standards, and implementing quick fixes instead of proper solutions. Code reviews become perfunctory as everyone rushes to approve changes. The team delivers the feature on time, but the codebase is left with numerous quality issues: functions with no error handling, duplicated logic that should have been abstracted, and complex conditional statements that are difficult to understand. Over the following months, these quality compromises lead to production bugs, difficult maintenance, and slower development of subsequent features. Another example involves a team where several senior developers have left due to frustration with legacy system complexity. The remaining developers are overwhelmed and demoralized, leading them to implement features with minimal effort just to complete their assigned tasks. They stop writing comprehensive tests, skip refactoring opportunities, and copy-paste code rather than creating reusable components. The overall quality of new code additions steadily declines as the team loses both capacity and motivation to maintain their previous standards.
