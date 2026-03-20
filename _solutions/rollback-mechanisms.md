---
title: Rollback Mechanisms
description: Ability to revert changes and return to a previous stable state
category:
- Operations
- Process
quality_tactics_url: https://qualitytactics.de/en/reliability/rollback-mechanisms
problems:
- missing-rollback-strategy
- deployment-risk
- frequent-hotfixes-and-rollbacks
- large-risky-releases
- release-instability
- fear-of-change
- complex-deployment-process
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Design every deployment to be reversible by maintaining the previous version's artifacts and configuration
- Implement database migration rollback scripts alongside forward migrations
- Use blue-green or canary deployment strategies that enable instant traffic switching to the previous version
- Automate rollback procedures so they can be executed quickly under incident pressure
- Define rollback decision criteria (error rate thresholds, latency increases) and empower teams to act without management approval
- Test rollback procedures as part of the deployment pipeline, not just the forward deployment
- Keep rollback artifacts available for a defined retention period after each deployment

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically reduces the risk and impact of failed deployments
- Enables faster deployment cadence by providing a safety net
- Reduces incident duration by providing a quick path to a known-good state
- Builds team confidence to deploy changes to legacy systems more frequently

**Costs and Risks:**
- Database rollback scripts must be carefully designed to avoid data loss
- Some changes (data format migrations, API contract changes) are difficult to roll back
- Maintaining rollback capability adds effort to every deployment
- Frequent reliance on rollback can indicate deeper quality issues that need addressing

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A financial services company deployed updates to its legacy trading platform once a quarter because each deployment was risky and lacked rollback capability. After implementing automated rollback mechanisms including database migration reversal, artifact versioning, and load balancer traffic switching, the team could revert any deployment within five minutes. This safety net enabled the team to increase deployment frequency to weekly, catching and rolling back three problematic releases in the first quarter while reducing the average size and risk of each deployment.
