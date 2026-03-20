---
title: Past Negative Experiences
description: A situation where developers are hesitant to make changes to the codebase
  because of negative experiences in the past.
category:
- Process
- Team
related_problems:
- slug: history-of-failed-changes
  similarity: 0.65
- slug: brittle-codebase
  similarity: 0.55
- slug: fear-of-breaking-changes
  similarity: 0.55
- slug: inconsistent-onboarding-experience
  similarity: 0.55
- slug: outdated-tests
  similarity: 0.55
- slug: inexperienced-developers
  similarity: 0.55
solutions:
- blameless-postmortems
layout: problem
---

## Description
Past negative experiences is a situation where developers are hesitant to make changes to the codebase because of negative experiences in the past. This is a common problem in teams that have a brittle codebase and a lack of automated tests. Past negative experiences can lead to a number of problems, including a fear of change, a slowdown in development velocity, and a general sense of stagnation.

## Indicators ⟡
- Developers are hesitant to make changes to the codebase.
- The team is not willing to take risks.
- The team is not innovating.
- The team is not learning from its mistakes.

## Symptoms ▲

- [Fear of Change](fear-of-change.md)
<br/>  Developers who have experienced production outages or blame from past changes become reluctant to modify the codebase.
- [Resistance to Change](resistance-to-change.md)
<br/>  Teams with past negative experiences actively resist refactoring or improvements due to perceived risk based on prior failures.
- [Refactoring Avoidance](refactoring-avoidance.md)
<br/>  Past incidents caused by changes make developers avoid refactoring even when they acknowledge it is necessary.
- [System Stagnation](system-stagnation.md)
<br/>  When developers are too cautious to make changes due to past failures, the system fails to evolve and stagnates.
- [Avoidance Behaviors](avoidance-behaviors.md)
<br/>  Developers postpone or avoid working on complex areas of the codebase where past changes have caused problems.
- [Large Estimates for Small Changes](large-estimates-for-small-changes.md)
<br/>  Teams provide inflated estimates for changes in areas where past modifications have caused problems, reflecting excessive caution.
## Causes ▼

- [Blame Culture](blame-culture.md)
<br/>  When mistakes are punished rather than treated as learning opportunities, developers internalize negative experiences and become risk-averse.
- [Brittle Codebase](brittle-codebase.md)
<br/>  A fragile codebase where changes frequently cause unexpected breakages creates repeated negative experiences for developers.
- [History of Failed Changes](history-of-failed-changes.md)
<br/>  A track record of failed deployments and problematic changes directly creates the negative experiences that make teams cautious.
- [Legacy Code Without Tests](legacy-code-without-tests.md)
<br/>  Without automated tests as a safety net, changes are risky and more likely to cause the production incidents that create negative experiences.
## Detection Methods ○
- **Developer Surveys:** Ask developers about their confidence level when making changes to different parts of the system.
- **Change Frequency Analysis:** Monitor how often different modules are modified; consistently avoided areas may indicate fear.
- **Estimation Patterns:** Look for patterns where similar changes have wildly different estimates based on the code area involved.
- **Code Review Comments:** Watch for excessive caution or lengthy discussions about potential risks during code reviews.

## Examples
A developer makes a change to the codebase that causes a major production outage. The developer is blamed for the outage, and they are hesitant to make changes to the codebase in the future. This is a common problem in companies that have a blame culture. It is important to create a culture where it is safe to fail. This will encourage developers to take risks and innovate.
