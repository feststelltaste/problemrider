---
title: Dark Launches
description: Limit blast radius of new features by deploying them hidden to a subset
  of users
category:
- Operations
- Process
problems:
- deployment-risk
- large-risky-releases
- release-anxiety
- fear-of-change
- release-instability
- high-defect-rate-in-production
layout: solution
related_solutions:
- slug: canary-releases
  similarity: 0.8
- slug: feature-toggles
  similarity: 0.75
- slug: chaos-engineering
  similarity: 0.75
- slug: rollback-mechanisms
  similarity: 0.7
- slug: restore-points
  similarity: 0.7
- slug: error-budgets
  similarity: 0.7
---

## Description

Dark launches deploy new code to production in a disabled or hidden state and then activate it selectively — for internal users, a small test group, or via shadow traffic that exercises the new path without affecting what users actually see — so that the new functionality is validated against real production conditions before it is exposed broadly. Legacy systems often carry a heightened fear of large releases precisely because past big-bang rollouts have gone wrong, and that fear in turn pushes releases to become even larger and further apart, since nobody wants to repeat the experience of a release that changed too much at once with no way to isolate what broke. By decoupling deployment from user-visible release through feature flags, dark launches let a team ship code continuously while controlling exposure independently, so problems can be caught and fixed — or the feature switched off instantly — before most users are ever affected. The same mechanism supports running an old and new implementation side by side in production, feeding both real inputs and comparing outputs automatically, which is particularly valuable when replacing a critical piece of legacy infrastructure under strict regulatory or reliability constraints that rule out a direct cutover. The cost of this safety is added complexity: the feature flag infrastructure itself needs to be built and maintained, dark-launched code still runs and consumes resources in production even while hidden, and flags that are not cleaned up after a full rollout accumulate as their own form of technical debt.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy banking application needed to replace its transaction processing engine but could not risk a big-bang cutover due to regulatory requirements. The team deployed the new engine alongside the old one and used dark launching to run both engines in parallel. Real transactions were processed by the old engine while the new engine received shadow copies and processed them independently. Results were compared automatically, and discrepancies were logged for investigation. Over eight weeks, the team resolved 12 edge cases that testing had not uncovered. The final cutover was a simple feature flag switch that took seconds to execute and could be reversed equally quickly.
