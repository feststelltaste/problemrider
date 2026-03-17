---
title: Incomplete Knowledge
description: Developers are unaware of all the locations where similar logic exists,
  which can lead to synchronization problems and other issues.
category:
- Communication
- Team
related_problems:
- slug: incomplete-projects
  similarity: 0.7
- slug: inconsistent-knowledge-acquisition
  similarity: 0.7
- slug: inexperienced-developers
  similarity: 0.7
- slug: information-fragmentation
  similarity: 0.65
- slug: insufficient-design-skills
  similarity: 0.65
- slug: poor-documentation
  similarity: 0.65
layout: problem
---

## Description
Incomplete knowledge is a common problem in software development. It occurs when developers are unaware of all the locations where similar logic exists. This can lead to a number of problems, including synchronization problems, code duplication, and a great deal of frustration for the development team. Incomplete knowledge is often a sign of a poorly documented system with a high degree of code duplication.

## Indicators ⟡
- The team is constantly reinventing the wheel.
- The team is not aware of all of the features in the system.
- The team is not sure how the system is supposed to behave.
- The team is not able to answer questions about the system.

## Symptoms ▲

- [Code Duplication](code-duplication.md)
<br/>  Developers unaware of existing implementations create duplicate code because they do not know solutions already exist.
- [Synchronization Problems](synchronization-problems.md)
<br/>  When developers do not know all locations where similar logic exists, updates to one copy miss others, causing divergent behavior.
- [Inconsistent Behavior](inconsistent-behavior.md)
<br/>  Developers modifying one instance of business logic are unaware of other instances, leading to inconsistent system behavior.
- [Regression Bugs](regression-bugs.md)
<br/>  Changes made without awareness of all affected locations inadvertently break functionality in unknown parts of the system.
- [Increased Risk of Bugs](increased-risk-of-bugs.md)
<br/>  Developers working with incomplete understanding of the system are more likely to introduce defects.

## Causes ▼
- [Information Decay](poor-documentation.md)
<br/>  Outdated or missing documentation prevents developers from learning about all relevant parts of the system.
- [Knowledge Silos](knowledge-silos.md)
<br/>  When knowledge is locked within individual team members, others cannot learn about system areas they have not personally worked on.
- [High Turnover](high-turnover.md)
<br/>  Frequent team member departures result in loss of institutional knowledge about system structure and logic locations.
- [Code Duplication](code-duplication.md)
<br/>  Extensive code duplication across the codebase makes it inherently difficult for any developer to know all locations where similar logic exists.

## Detection Methods ○
- **Developer Surveys:** Ask developers about their knowledge of the system.
- **Code Reviews:** Code reviews are a great way to identify knowledge gaps.
- **Pair Programming:** Pair programming is a great way to share knowledge between developers.
- **Knowledge Mapping:** Create a knowledge map of the system to identify areas where there are knowledge gaps.

## Examples
A company has a large, complex system. The system is not well-documented, and there is a high rate of turnover in the team. As a result, the team has a very incomplete knowledge of the system. The team is constantly reinventing the wheel, and they are not able to answer questions about the system. The company eventually has to hire a team of consultants to document the system and train the team.
