---
title: Resistance to Change
description: Teams are hesitant to refactor or improve parts of the system due to
  perceived risk and effort, leading to stagnation.
category:
- Code
- Process
- Team
related_problems:
- slug: maintenance-paralysis
  similarity: 0.75
- slug: fear-of-change
  similarity: 0.75
- slug: refactoring-avoidance
  similarity: 0.75
- slug: fear-of-breaking-changes
  similarity: 0.7
- slug: system-stagnation
  similarity: 0.7
- slug: inability-to-innovate
  similarity: 0.65
layout: problem
---

## Description

Resistance to change occurs when development teams consistently avoid making necessary improvements, refactoring, or modernization efforts due to concerns about risk, effort, or disruption. This resistance can stem from past negative experiences, lack of confidence in the team's ability to manage change safely, or organizational culture that discourages taking risks. Over time, this resistance leads to system stagnation and accumulating technical debt.

## Indicators ⟡

- Improvement initiatives are consistently postponed or canceled
- Team discussions about refactoring focus primarily on risks rather than benefits
- Workarounds are preferred over fixing underlying problems
- New requirements are implemented as additions rather than improvements to existing code
- Proposals for system improvements receive skeptical or negative responses

## Symptoms ▲

- [Accumulation of Workarounds](accumulation-of-workarounds.md)
<br/>  When teams resist changing problematic code, they create workarounds instead of fixing root causes, adding complexity.
- [System Stagnation](system-stagnation.md)
<br/>  Persistent resistance to making improvements causes the system to remain unchanged while business needs and technology evolve.
- [High Technical Debt](high-technical-debt.md)
<br/>  Avoiding necessary refactoring and improvements allows technical debt to accumulate unchecked over time.
- [Inability to Innovate](inability-to-innovate.md)
<br/>  Resistance to changing the existing system prevents adoption of new approaches, technologies, or architectural improvements.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Unwillingness to improve the codebase forces developers to work around existing problems, slowing feature delivery.
## Causes ▼

- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  Legitimate fear of breaking existing functionality makes teams reluctant to touch working code, even when it needs improvement.
- [Poor Test Coverage](poor-test-coverage.md)
<br/>  Without tests to verify that changes don't break existing functionality, the perceived risk of any change is high, discouraging improvement efforts.
- [Blame Culture](blame-culture.md)
<br/>  When mistakes are punished, team members avoid taking the risk of making changes that could fail and draw blame.
- [Maintenance Paralysis](maintenance-paralysis.md)
<br/>  Teams that cannot verify changes don't break functionality become paralyzed and resist making any improvements.
- [Past Negative Experiences](past-negative-experiences.md)
<br/>  Past negative experiences with changes (failed deployments, broken systems) are a direct cause of teams becoming resi....
## Detection Methods ○

- **Improvement Proposal Tracking:** Monitor how many improvement initiatives are started vs. completed
- **Code Age Analysis:** Identify areas of code that haven't been improved despite known issues
- **Team Retrospectives:** Discuss attitudes toward change and improvement efforts
- **Technical Debt Trend Analysis:** Track whether technical debt is increasing or decreasing over time
- **Decision Pattern Analysis:** Look for patterns of choosing workarounds over fundamental fixes

## Examples

A development team identifies that their authentication system needs modernization to support new security requirements, but every discussion about updating it ends with concerns about breaking existing integrations. Instead of modernizing the system, they continue adding patch-like security measures that increase complexity while leaving fundamental vulnerabilities unaddressed. Another example involves a team that knows their database design is causing performance problems, but they resist redesigning the schema because they're afraid of data migration risks, instead implementing increasingly complex caching layers that add operational overhead without solving the root performance issues.
