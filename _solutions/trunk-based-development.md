---
title: Trunk-Based Development
description: Integrating short-lived branches continuously into main for rapid, safe
  modifications
category:
- Process
problems:
- long-lived-feature-branches
- merge-conflicts
- integration-difficulties
- large-pull-requests
- slow-development-velocity
- deployment-coupling
- large-risky-releases
- extended-cycle-times
- extended-review-cycles
- increased-time-to-market
- review-bottlenecks
- reduced-code-submission-frequency
layout: solution
related_solutions:
- slug: continuous-integration
  similarity: 0.8
- slug: continuous-integration-and-delivery
  similarity: 0.8
- slug: rollback-mechanisms
  similarity: 0.75
- slug: continuous-delivery
  similarity: 0.75
- slug: canary-releases
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.7
---

## Description

Trunk-based development is a source control workflow in which developers integrate small, short-lived changes into the main branch at least daily, rather than working in isolation on long-lived feature branches that diverge from main for weeks or months before being merged. Incomplete work is made safe to merge continuously by hiding it behind feature flags, and a fast, comprehensive CI pipeline validates every integration so that main stays in a releasable state at essentially all times. The practice is a direct countermeasure to a pattern common in legacy codebases with entrenched branching habits: branches that live long enough to drift substantially from main, producing merge conflicts that consume days of rework and integration bugs that surface only when it is expensive to untangle which of many accumulated changes caused them. By collapsing the interval between writing code and integrating it, trunk-based development converts integration from a rare, high-stakes event into a routine, low-risk one, which is precisely the shift legacy teams need when large, risky, infrequent releases have historically made every change feel dangerous. Adopting it in a legacy context typically requires upfront investment in CI speed and test reliability, since the whole model depends on fast, trustworthy feedback on every small merge — without that investment, frequent integration just surfaces the same problems faster rather than actually resolving them.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy enterprise application team had a practice of maintaining feature branches for weeks or months. Merge day was dreaded, often consuming an entire sprint. Integration bugs discovered during merges frequently required rework. The team transitioned to trunk-based development, starting by breaking their current long-lived branch into daily mergeable increments using feature flags. They invested in speeding up the test suite from 45 minutes to 8 minutes. Within three months, the team was merging to main multiple times per day. Merge conflicts became rare, integration bugs were caught immediately, and the team's velocity increased measurably because they spent far less time on merge-related rework.
