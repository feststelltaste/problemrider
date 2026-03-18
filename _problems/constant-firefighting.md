---
title: Constant Firefighting
description: The development team is perpetually occupied with fixing bugs and addressing
  urgent issues, leaving little to no time for new feature development.
category:
- Code
- Process
related_problems:
- slug: development-disruption
  similarity: 0.75
- slug: time-pressure
  similarity: 0.65
- slug: uneven-work-flow
  similarity: 0.65
- slug: constantly-shifting-deadlines
  similarity: 0.65
- slug: operational-overhead
  similarity: 0.65
- slug: frequent-changes-to-requirements
  similarity: 0.65
layout: problem
---

## Description
Constant firefighting, also known as "reactive development," is a state where a development team is so consumed by urgent, unplanned work that they have little or no time for planned, proactive work. The team is constantly in a state of crisis, lurching from one emergency to the next. This is a highly inefficient and stressful way to work, and it is a clear sign that the system is unstable and the development process is broken.

## Indicators ⟡
- The majority of the team's time is spent on unplanned work.
- The team is frequently context-switching between different urgent tasks.
- There is a sense of chaos and urgency in the team's daily work.
- The team is consistently missing its deadlines for planned work.

## Symptoms ▲

- [Developer Frustration and Burnout](developer-frustration-and-burnout.md)
<br/>  Perpetual crisis mode exhausts developers emotionally and physically, leading directly to burnout.
- [Inability to Innovate](inability-to-innovate.md)
<br/>  When all time is consumed by urgent fixes, there is no capacity left for exploring improvements or new approaches.
- [Increased Technical Shortcuts](increased-technical-shortcuts.md)
<br/>  Under constant urgency, developers take shortcuts to resolve issues quickly, which creates more technical debt and future fires.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Planned feature work is consistently deprioritized in favor of urgent bug fixes, slowing the delivery of new functionality.
- [High Technical Debt](high-technical-debt.md)
<br/>  Constant firefighting prevents the team from addressing root causes, so technical debt accumulates as quick fixes pile up.
- [Quality Degradation](quality-degradation.md)
<br/>  Rushed fixes under crisis conditions often introduce new issues, causing overall system quality to decline over time.
## Causes ▼

- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  A high rate of production bugs generates the stream of urgent issues that keeps the team in constant firefighting mode.
- [Poor Test Coverage](poor-test-coverage.md)
<br/>  Without adequate test coverage, bugs slip into production frequently, creating the ongoing emergencies that drive firefighting.
- [Brittle Codebase](brittle-codebase.md)
<br/>  A fragile codebase where small changes break existing functionality generates a constant stream of production issues requiring urgent attention.
- [Monitoring Gaps](monitoring-gaps.md)
<br/>  Insufficient monitoring means issues are not caught early and escalate into emergencies requiring immediate firefighting response.
- [High Technical Debt](high-technical-debt.md)
<br/>  High technical debt causes systems to break more frequently, generating the stream of urgent issues that keeps teams ....
## Detection Methods ○
- **Track Unplanned Work:** Measure the percentage of the team's time that is spent on unplanned work. If this number is consistently high, it is a clear sign of a problem.
- **Analyze Bug Reports:** Look for patterns in the bug reports. Are the same problems recurring over and over again? This is a sign that the team is not addressing the root causes of the problems.
- **Team Retrospectives:** Ask the team about their experience with firefighting. Are they feeling overwhelmed? Do they feel like they are making progress?
- **Monitor Key Metrics:** Track metrics like mean time to recovery (MTTR) and change failure rate. A high MTTR and a high change failure rate are both indicators of a team that is struggling with firefighting.

## Examples
A team is responsible for maintaining a critical business application. The application is old and has a lot of technical debt. The team spends most of its time fixing production issues. They are constantly being pulled off of their planned work to deal with emergencies. As a result, they are never able to make any progress on the long-term improvements that would make the application more stable. The team is stuck in a vicious cycle of firefighting, and they are becoming increasingly burned out.
