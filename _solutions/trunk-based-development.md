---
title: Trunk-Based Development
description: Integrating short-lived branches continuously into main for rapid, safe modifications
category:
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/trunk-based-development
problems:
- long-lived-feature-branches
- merge-conflicts
- integration-difficulties
- large-pull-requests
- slow-development-velocity
- deployment-coupling
- large-risky-releases
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Keep branches short-lived: merge into the main branch at least daily, ideally multiple times per day
- Use feature flags to decouple deployment from feature release so incomplete work can be merged safely
- Invest in a robust CI pipeline that runs fast, comprehensive tests on every merge to main
- Break large changes into small, incremental commits that can each be merged independently
- Eliminate long-lived feature branches and replace them with techniques like branch by abstraction
- Ensure the main branch is always in a deployable state through automated quality gates
- Address flaky tests aggressively, as they undermine confidence in continuous integration

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically reduces merge conflicts by integrating changes frequently
- Provides fast feedback on integration issues rather than discovering them at merge time
- Enables continuous delivery by keeping the main branch always releasable
- Reduces code review burden because changes are small and focused

**Costs and Risks:**
- Requires mature CI infrastructure and fast test suites to support frequent merges
- Feature flags add complexity and must be cleaned up to avoid flag debt
- Teams must develop discipline to commit small, complete increments rather than large batches
- Partially complete features on main require careful management to avoid exposing them to users

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy enterprise application team had a practice of maintaining feature branches for weeks or months. Merge day was dreaded, often consuming an entire sprint. Integration bugs discovered during merges frequently required rework. The team transitioned to trunk-based development, starting by breaking their current long-lived branch into daily mergeable increments using feature flags. They invested in speeding up the test suite from 45 minutes to 8 minutes. Within three months, the team was merging to main multiple times per day. Merge conflicts became rare, integration bugs were caught immediately, and the team's velocity increased measurably because they spent far less time on merge-related rework.
