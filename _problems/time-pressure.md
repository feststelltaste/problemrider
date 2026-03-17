---
title: Time Pressure
description: Teams are forced to take shortcuts to meet immediate deadlines, deferring
  proper solutions and rushing important tasks like code reviews.
category:
- Code
- Process
related_problems:
- slug: deadline-pressure
  similarity: 0.75
- slug: increased-stress-and-burnout
  similarity: 0.7
- slug: constant-firefighting
  similarity: 0.65
- slug: increased-technical-shortcuts
  similarity: 0.65
- slug: unrealistic-deadlines
  similarity: 0.65
- slug: constantly-shifting-deadlines
  similarity: 0.65
layout: problem
---

## Description
Time pressure is a pervasive problem in software development where the emphasis on speed and meeting deadlines leads to compromises in quality. When teams are constantly rushed, they are more likely to take shortcuts, skip important steps like testing and thorough code reviews, and make suboptimal design decisions. This can lead to an accumulation of technical debt, a decrease in code quality, and an increase in the number of bugs.

## Indicators ⟡
- The team is consistently working overtime to meet deadlines.
- Features are frequently descoped or rushed at the end of a release cycle.
- There is a general feeling of being in a constant state of "firefighting".

## Symptoms ▲

- [Increased Technical Shortcuts](increased-technical-shortcuts.md)
<br/>  Under time pressure, teams take quick fixes and workarounds instead of implementing proper solutions.
- [High Technical Debt](high-technical-debt.md)
<br/>  Shortcuts taken under time pressure accumulate as technical debt that becomes increasingly expensive to address.
- [Quality Compromises](quality-compromises.md)
<br/>  Time pressure forces teams to deliberately lower quality standards to meet deadlines.
- [Increased Stress and Burnout](increased-stress-and-burnout.md)
<br/>  Persistent time pressure leads to overwork, stress, and eventual burnout among team members.
- [Lower Code Quality](lower-code-quality.md)
<br/>  Rushed development under time pressure results in poorly designed, harder-to-maintain code.
- [Test Debt](test-debt.md)
<br/>  Testing is often the first activity sacrificed when teams are under time pressure, leading to accumulated test debt.
## Causes ▼

- [Unrealistic Deadlines](unrealistic-deadlines.md)
<br/>  Management setting deadlines that don't account for actual effort required is a primary driver of time pressure.
- [Market Pressure](market-pressure.md)
<br/>  External competitive forces drive organizations to push teams to deliver faster, creating time pressure.
- [Changing Project Scope](changing-project-scope.md)
<br/>  When scope expands without adjusting timelines, the same amount of time must cover more work, intensifying time pressure.
- [Constant Firefighting](constant-firefighting.md)
<br/>  When teams spend most of their time on urgent fixes, planned work gets squeezed into less time, creating time pressure for feature delivery.
## Detection Methods ○
- **Track Overtime Hours:** Monitor the number of hours the team is working beyond their normal schedule.
- **Analyze Bug Reports:** Look for an increase in the number of bugs, especially those that could have been prevented with more time for testing and review.
- **Team Retrospectives:** Discuss the impact of deadlines on the team's ability to produce high-quality work.

## Examples
A team is under pressure to deliver a new feature by the end of the quarter. To meet the deadline, they decide to skip writing unit tests and to perform only a cursory manual test. The feature is delivered on time, but it is full of bugs that are only discovered by users in production. The team then has to spend the next several weeks fixing the bugs, which ultimately delays the next feature release.
