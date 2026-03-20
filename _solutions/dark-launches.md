---
title: Dark Launches
description: Limit blast radius of new features by deploying them hidden to a subset of users
category:
- Operations
- Process
quality_tactics_url: https://qualitytactics.de/en/reliability/dark-launches
problems:
- deployment-risk
- large-risky-releases
- release-anxiety
- fear-of-change
- release-instability
- high-defect-rate-in-production
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Implement a feature flag infrastructure that can enable or disable features without redeployment
- Deploy new code to production in a disabled state, then activate it selectively for internal users or a small test group
- Use shadow traffic to exercise new code paths with real production data without affecting user-visible responses
- Monitor the performance and correctness of dark-launched features through dedicated metrics and logging
- Gradually expand the user group as confidence grows, using percentage-based rollouts
- Establish kill-switch procedures that can disable a dark-launched feature instantly if problems are detected
- Clean up feature flags once a feature is fully rolled out to avoid flag accumulation

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Validates new features with real production traffic and data without exposing all users to risk
- Decouples deployment from feature release, enabling independent cadences
- Provides a rapid rollback mechanism through feature flag toggling
- Reduces anxiety around large feature launches by allowing incremental validation

**Costs and Risks:**
- Feature flag infrastructure adds complexity to the codebase and deployment process
- Accumulated feature flags create technical debt if not cleaned up after full rollout
- Dark-launched code still executes in production and can affect performance or cause side effects
- Shadow traffic approaches require careful handling to avoid unintended writes or state changes
- Testing becomes more complex with multiple feature flag combinations

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy banking application needed to replace its transaction processing engine but could not risk a big-bang cutover due to regulatory requirements. The team deployed the new engine alongside the old one and used dark launching to run both engines in parallel. Real transactions were processed by the old engine while the new engine received shadow copies and processed them independently. Results were compared automatically, and discrepancies were logged for investigation. Over eight weeks, the team resolved 12 edge cases that testing had not uncovered. The final cutover was a simple feature flag switch that took seconds to execute and could be reversed equally quickly.
