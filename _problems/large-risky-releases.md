---
title: Large, Risky Releases
description: Infrequent releases lead to large, complex deployments that are difficult
  to test, prone to failure, and have a significant impact on users when they go wrong.
category:
- Code
- Operations
- Process
related_problems:
- slug: complex-deployment-process
  similarity: 0.7
- slug: long-release-cycles
  similarity: 0.65
- slug: release-instability
  similarity: 0.65
- slug: deployment-risk
  similarity: 0.65
- slug: frequent-hotfixes-and-rollbacks
  similarity: 0.65
- slug: release-anxiety
  similarity: 0.65
solutions:
- blue-green-canary-deployments
- ci-cd-pipeline
- feature-flags
layout: problem
---

## Description
Large, risky releases are a common problem in organizations with long release cycles. When releases are infrequent, they tend to be large and complex. This is because they contain a large number of changes, which can interact in unexpected ways. Large releases are difficult to test, and they are more likely to fail than small releases. When a large release fails, it can have a significant impact on users and the business. It can also be difficult and time-consuming to roll back a large release, which can prolong the outage.

## Indicators ⟡
- Releases are a major event that requires a lot of planning and coordination.
- The team is anxious and stressed about deployments.
- There is a high rate of post-deployment bugs and other issues.
- Rollbacks are a common occurrence.

## Symptoms ▲

- [Release Anxiety](release-anxiety.md)
<br/>  The high stakes of large, infrequent releases create stress and anxiety among the development team around deployment events.
- [Release Instability](release-instability.md)
<br/>  Large releases containing many changes are inherently less stable, with more unexpected interactions causing production issues.
- [Frequent Hotfixes and Rollbacks](frequent-hotfixes-and-rollbacks.md)
<br/>  Complex releases with many bundled changes are more likely to fail, requiring emergency hotfixes or complete rollbacks.
- [System Outages](system-outages.md)
<br/>  Failed large releases can cause significant service interruptions due to the complexity of the changes and difficulty of rollback.
- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Large risky releases that fail or introduce bugs directly impact users, causing frustration and dissatisfaction.
- [Increased Bug Count](increased-bug-count.md)
<br/>  Large releases with many bundled changes are more likely to introduce bugs due to complex interactions between change....
## Causes ▼

- [Long Release Cycles](long-release-cycles.md)
<br/>  Infrequent releases mean more changes accumulate between deployments, making each release larger and riskier.
- [Complex Deployment Process](complex-deployment-process.md)
<br/>  Manual, error-prone deployment processes discourage frequent releases, leading to batching of changes into larger releases.
- [Large Feature Scope](large-feature-scope.md)
<br/>  Features that cannot be broken down into incremental deliverables force multiple large changes to be released together.
- [Manual Deployment Processes](manual-deployment-processes.md)
<br/>  When deployments require manual intervention, teams avoid releasing frequently, causing changes to accumulate into risky batches.
## Detection Methods ○
- **Release Size:** Track the number of changes in each release.
- **Release Failure Rate:** Track the percentage of releases that result in a critical failure.
- **Mean Time to Recovery (MTTR):** Measure the average time it takes to recover from a failed release.
- **Post-Release Bug Count:** Count the number of bugs that are reported in the days and weeks following a release.

## Examples
A company releases a new version of its software once a year. The annual release is a major event that requires months of planning and coordination. The release contains a large number of new features and bug fixes. The testing process is long and arduous, but it is impossible to test every possible combination of changes. As a result, the release is always risky, and it often fails. When the release fails, it can take days to roll it back, which has a significant impact on the company's customers.
