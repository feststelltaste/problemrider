---
title: History of Failed Changes
description: A past record of failed deployments or changes creates a culture of fear
  and resistance to future modifications.
category:
- Culture
- Process
related_problems:
- slug: fear-of-breaking-changes
  similarity: 0.65
- slug: past-negative-experiences
  similarity: 0.65
- slug: fear-of-change
  similarity: 0.65
- slug: resistance-to-change
  similarity: 0.65
- slug: fear-of-failure
  similarity: 0.6
- slug: perfectionist-culture
  similarity: 0.6
solutions:
- architecture-decision-records
- blameless-postmortems
- functional-spike
layout: problem
---

## Description
A history of failed changes can create a lasting negative impact on a team's culture and development velocity. When past deployments have resulted in significant outages or rollbacks, developers become hesitant to make further changes, leading to a culture of fear and risk aversion. This can stifle innovation and make it difficult to address technical debt or introduce new features.

## Indicators ⟡
- Developers are reluctant to take on tasks that involve modifying critical parts of the system.
- The team has a very slow and cumbersome change approval process.
- There is a general sentiment that "if it ain't broke, don't fix it."
- The team has a history of long and stressful release cycles.

## Symptoms ▲

- [Fear of Change](fear-of-change.md)
<br/>  Past deployment failures create a lasting emotional resistance to making modifications, even when changes are necessary.
- [Resistance to Change](resistance-to-change.md)
<br/>  Teams that have experienced failed changes develop organizational resistance to future modifications and modernization efforts.
- [Slow Development Velocity](slow-development-velocity.md)
<br/>  Excessive caution and bureaucratic approval processes born from past failures slow down the pace of development.
- [Inability to Innovate](inability-to-innovate.md)
<br/>  Fear stemming from past failures prevents teams from trying new approaches or technologies.
- [Stagnant Architecture](stagnant-architecture.md)
<br/>  Reluctance to change leads to architecture that remains frozen and unable to evolve with changing requirements.
## Causes ▼

- [Insufficient Testing](insufficient-testing.md)
<br/>  Inadequate testing allowed defects to reach production in past deployments, causing the failures that created this fear.
- [Large, Risky Releases](large-risky-releases.md)
<br/>  Large, infrequent releases carry higher risk of failure, and when they fail, the impact is severe enough to create lasting fear.
- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  Repeated production bugs from releases build a record of failed changes that reinforces risk-averse culture.
- [Missing Rollback Strategy](missing-rollback-strategy.md)
<br/>  Without rollback capability, failed deployments cause extended outages that amplify the negative impact and fear.
## Detection Methods ○
- **Deployment Frequency:** Track how often the team deploys changes to production. A low deployment frequency can be a sign of fear.
- **Lead Time for Changes:** Measure the time it takes from a code commit to a production deployment.
- **Change Failure Rate:** Track the percentage of deployments that result in a failure.
- **Developer Surveys:** Ask developers about their confidence in the deployment process and their willingness to make changes.

## Examples
A team at a financial services company experienced a major outage after a recent deployment. The incident caused significant financial losses and reputational damage. As a result, the company implemented a lengthy and bureaucratic change approval process. Now, even the smallest change requires multiple levels of approval and can take weeks to deploy. The developers are so afraid of causing another outage that they avoid making any changes unless they are absolutely necessary. This has led to a stagnant product and a frustrated development team.
