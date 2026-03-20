---
title: Feature Flags
description: Toggling feature availability at runtime per user segment
category:
- Operations
- Process
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/feature-flags
problems:
- deployment-risk
- large-risky-releases
- fear-of-change
- release-anxiety
- long-release-cycles
- strangler-fig-pattern-failures
- deployment-coupling
layout: solution
---

## How to Apply ◆

- Implement a feature flag system (LaunchDarkly, Unleash, or a simple configuration-based approach) in the legacy application.
- Use feature flags to decouple deployment from feature activation: deploy new code behind a disabled flag, then enable it gradually.
- Implement gradual rollouts: enable new features for a small percentage of users first, then increase as confidence grows.
- Use feature flags during legacy-to-modern migration to switch between old and new implementations at runtime.
- Establish lifecycle management for flags: review and remove flags once features are fully rolled out to prevent flag accumulation.
- Create kill switches for critical features that can be disabled instantly if problems arise in production.

## Tradeoffs ⇄

**Benefits:**
- Enables safe, gradual rollout of changes to legacy systems with instant rollback capability.
- Decouples deployment from release, reducing deployment risk.
- Allows A/B testing of legacy vs. modernized implementations.
- Provides a mechanism for safely introducing changes in systems where full regression testing is impractical.

**Costs:**
- Accumulated feature flags create conditional complexity and make code harder to understand.
- Flag management becomes its own maintenance burden if flags are not cleaned up after use.
- Testing combinatorial flag states can be challenging.
- Improperly managed flags can leave the system in inconsistent states.

## Examples

A legacy e-commerce platform is migrating its checkout flow to a new implementation but cannot afford downtime or a big-bang cutover. The team deploys the new checkout code behind a feature flag and enables it for 5% of users. Monitoring reveals a payment processing edge case that the new implementation handles differently from the legacy code. They disable the flag, fix the issue, redeploy, and resume the gradual rollout. Over three weeks, the new checkout is enabled for 100% of users with no customer-facing incidents. The legacy checkout code is removed in a subsequent cleanup release along with the feature flag. Without feature flags, the team would have faced a high-risk cutover with no easy rollback.
