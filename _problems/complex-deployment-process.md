---
title: Complex Deployment Process
description: The process of deploying software to production is manual, time-consuming,
  and error-prone, which contributes to long release cycles and a high risk of failure.
category:
- Operations
- Process
related_problems:
- slug: manual-deployment-processes
  similarity: 0.75
- slug: large-risky-releases
  similarity: 0.7
- slug: immature-delivery-strategy
  similarity: 0.7
- slug: deployment-risk
  similarity: 0.7
- slug: insufficient-testing
  similarity: 0.6
- slug: missing-rollback-strategy
  similarity: 0.6
layout: problem
---

## Description
A complex deployment process is a major obstacle to the continuous delivery of value. When the process of deploying software is manual, time-consuming, and error-prone, it is difficult to release new features quickly and safely. This can lead to long release cycles, large and risky releases, and a great deal of anxiety for the development team. A complex deployment process is often a sign of a legacy system that has not been designed for continuous delivery. It can also be a sign of a lack of investment in automation and tooling.

## Indicators ⟡
- The deployment process is not documented.
- The deployment process requires a lot of manual steps.
- The deployment process is different for different environments.
- The deployment process is not automated.

## Symptoms ▲

- [Long Release Cycles](long-release-cycles.md)
<br/>  Manual, time-consuming deployment processes directly extend the time between releases.
- [Release Anxiety](release-anxiety.md)
<br/>  The high failure risk of complex manual deployments creates stress and anxiety for the team.
- [Large, Risky Releases](large-risky-releases.md)
<br/>  Infrequent deployments due to process complexity lead to batching many changes into large, risky releases.
- [Delayed Value Delivery](delayed-value-delivery.md)
<br/>  Complex deployment processes delay getting completed features to users.
- [Deployment Risk](deployment-risk.md)
<br/>  Manual steps and inconsistent processes increase the likelihood of deployment failures.
- [Frequent Hotfixes and Rollbacks](frequent-hotfixes-and-rollbacks.md)
<br/>  Error-prone manual deployment processes lead to failed releases that require immediate hotfixes or rollbacks.
## Causes ▼

- [Manual Deployment Processes](manual-deployment-processes.md)
<br/>  Reliance on human intervention for deployment steps is the direct cause of deployment complexity and error-proneness.
- [Deployment Environment Inconsistencies](deployment-environment-inconsistencies.md)
<br/>  Different environment configurations require custom deployment steps for each environment, increasing process complexity.
- [Legacy Configuration Management Chaos](legacy-configuration-management-chaos.md)
<br/>  Hardcoded and undocumented configuration settings make automated deployment difficult, forcing manual processes.
- [Monolithic Architecture Constraints](monolithic-architecture-constraints.md)
<br/>  Large monolithic applications require deploying the entire system at once, making the deployment process inherently complex.
## Detection Methods ○
- **Deployment Time:** Measure the time it takes to deploy a new version of the software.
- **Deployment Frequency:** Measure how often the team deploys to production.
- **Deployment Failure Rate:** Track the percentage of deployments that fail.
- **Deployment Process Mapping:** Map out the steps in the deployment process to identify bottlenecks and areas for improvement.

## Examples
A company has a very complex and manual deployment process. It takes two days to deploy a new version of the software. The process is not documented, and it is different for every environment. The team is very anxious about deployments, and they often fail. When a deployment fails, it can take hours to roll it back. As a result, the company is only able to release new software once a month. This is a major competitive disadvantage, and it is a major source of frustration for the development team.
