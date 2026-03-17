---
title: Review Process Avoidance
description: Team members actively seek ways to bypass or minimize code review requirements,
  undermining the quality assurance process.
category:
- Process
- Team
- Testing
related_problems:
- slug: reduced-review-participation
  similarity: 0.7
- slug: review-process-breakdown
  similarity: 0.7
- slug: inadequate-code-reviews
  similarity: 0.65
- slug: insufficient-code-review
  similarity: 0.65
- slug: rushed-approvals
  similarity: 0.65
- slug: reviewer-anxiety
  similarity: 0.65
layout: problem
---

## Description

Review process avoidance occurs when team members actively look for ways to bypass, minimize, or circumvent code review requirements due to frustration with the review process itself. This can include making changes directly in production, using emergency deployment procedures for non-urgent changes, committing directly to main branches, or finding technical loopholes to avoid review. This behavior undermines the quality assurance benefits that code reviews are meant to provide.

## Indicators ⟡

- Increased use of "hotfix" or emergency deployment procedures for non-critical changes
- Direct commits to main branches that bypass review requirements
- Changes made during off-hours to avoid review oversight
- Frequent requests to modify review requirements or make exceptions
- Team members express desire to "just skip the review this time"

## Symptoms ▲

- [High Defect Rate in Production](high-defect-rate-in-production.md)
<br/>  Changes that bypass review miss the quality gate, allowing more defects to reach production undetected.
- [Review Process Breakdown](review-process-breakdown.md)
<br/>  Widespread avoidance undermines the review process systematically, causing it to fail at its quality assurance purpose.
- [Inconsistent Coding Standards](inconsistent-coding-standards.md)
<br/>  Code that skips review doesn't get checked for standard compliance, leading to inconsistent coding practices across the codebase.
- [Knowledge Silos](knowledge-silos.md)
<br/>  Bypassing reviews eliminates a key knowledge-sharing mechanism, allowing code knowledge to remain siloed with the original author.

## Causes ▼
- [Review Bottlenecks](review-bottlenecks.md)
<br/>  When the review process is a significant bottleneck, developers are motivated to find ways around it to deliver their work.
- [Author Frustration](author-frustration.md)
<br/>  Frustration with conflicting or seemingly arbitrary review feedback drives developers to avoid the process altogether.
- [Time Pressure](time-pressure.md)
<br/>  Deadline pressure makes the review process feel like an unaffordable delay, motivating developers to bypass it.
- [Reviewer Anxiety](reviewer-anxiety.md)
<br/>  When reviewers are anxious and provide superficial or unhelpful feedback, authors see little value in the review process and avoid it.
- [Conflicting Reviewer Opinions](conflicting-reviewer-opinions.md)
<br/>  The frustration of dealing with conflicting opinions motivates developers to seek ways to bypass the review process entirely.
- [Extended Review Cycles](extended-review-cycles.md)
<br/>  Lengthy and painful review cycles motivate developers to find ways to bypass or minimize the review process.
- [Outdated Tests](outdated-tests.md)
<br/>  When tests are unreliable, teams start ignoring or bypassing test results, undermining the quality assurance process.

## Detection Methods ○

- **Review Bypass Tracking:** Monitor commits, deployments, or changes that circumvent normal review processes
- **Emergency Procedure Usage Analysis:** Track frequency and justification of emergency deployment usage
- **Process Compliance Assessment:** Measure what percentage of changes actually go through required review
- **Team Behavior Surveys:** Collect feedback on motivations for avoiding review processes  
- **Quality Impact Correlation:** Analyze whether bypassed changes have higher defect rates

## Examples

A developer becomes frustrated after spending three weeks getting a simple bug fix through the review process due to extensive style debates and conflicting feedback. When they encounter their next urgent bug, they deploy the fix using the emergency hotfix process to avoid review, even though the issue isn't actually critical. This sets a precedent, and soon multiple team members are using emergency procedures for convenience rather than true emergencies. Another example involves a team member who discovers they can make small changes directly in the deployment configuration that bypasses code review requirements. They begin making increasingly significant changes through this route, including business logic modifications that should receive review, because they want to avoid the time and frustration of the normal review process.
