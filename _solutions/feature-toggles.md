---
title: Feature Toggles
description: Activating and deactivating features for flexible rollouts
category:
- Process
- Operations
problems:
- large-risky-releases
- deployment-risk
- fear-of-change
- feature-creep
- release-instability
- frequent-hotfixes-and-rollbacks
- long-release-cycles
- development-disruption
- increased-time-to-market
- large-pull-requests
- strangler-fig-pattern-failures
- excessive-customization
layout: solution
related_solutions:
- slug: feature-flags
  similarity: 0.9
- slug: dark-launches
  similarity: 0.75
- slug: rollback-mechanisms
  similarity: 0.75
- slug: canary-releases
  similarity: 0.7
- slug: restore-points
  similarity: 0.7
- slug: trunk-based-development
  similarity: 0.7
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Introduce a simple toggle mechanism (configuration file, database flags, or feature flag service) before adopting complex solutions
- Wrap new functionality in conditional blocks controlled by the toggle rather than maintaining separate code branches
- Use toggles to decouple deployment from release so code can ship to production in a disabled state
- Implement kill switches for risky legacy system changes that allow instant rollback without redeployment
- Establish a lifecycle for each toggle: define when it will be removed and clean up stale toggles regularly
- Use percentage-based rollouts or user-segment targeting to test changes with a subset of traffic first
- Monitor key metrics per toggle state to detect regressions before full activation

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables incremental rollout of changes in legacy systems with reduced blast radius
- Decouples deployment from release, lowering deployment anxiety
- Allows quick rollback of problematic features without code changes
- Supports A/B testing and canary releases in systems that lack modern deployment infrastructure

**Costs and Risks:**
- Toggle proliferation creates combinatorial testing challenges and code complexity
- Stale toggles left in the codebase become technical debt themselves
- Testing all toggle combinations is impractical, increasing the risk of untested paths
- Adds conditional branching that can make legacy code harder to understand

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An insurance company needed to migrate its claims processing logic from a legacy rules engine to a new implementation but could not afford extended downtime or a big-bang switch. The team introduced feature toggles that allowed both the old and new processing paths to coexist in production. They initially routed 5% of claims through the new path, compared outputs, and gradually increased the percentage over several weeks. When a subtle calculation difference was discovered, they toggled the new path off within minutes, fixed the issue, and resumed the rollout without any customer-facing impact.
