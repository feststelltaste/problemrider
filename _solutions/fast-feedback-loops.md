---
title: Fast Feedback Loops
description: Treat the time from making a change to knowing whether it worked as the
  primary engineering metric, and attack whatever dominates it.
category:
- Code
- Process
- Testing
problems:
- long-build-and-test-times
- slow-development-velocity
- slow-feature-development
- inefficient-development-environment
- development-disruption
- reduced-code-submission-frequency
- extended-cycle-times
- flaky-tests
- context-switching-overhead
- tool-limitations
- increased-manual-work
- reduced-individual-productivity
- long-release-cycles
- automated-tooling-ineffectiveness
- delayed-bug-fixes
- excessive-logging
- extended-review-cycles
- fear-of-failure
- mental-fatigue
- reduced-review-participation
- review-bottlenecks
- review-process-avoidance
- testing-environment-fragility
layout: solution
related_solutions:
- slug: development-environment-optimization
  similarity: 0.75
- slug: small-change-batches
  similarity: 0.7
- slug: delivery-performance-metrics
  similarity: 0.7
- slug: regression-testing
  similarity: 0.7
- slug: development-workflow-automation
  similarity: 0.65
- slug: quality-ratchet
  similarity: 0.65
---

## Description

A fast feedback loop is the interval between making a change and finding out whether it did what you intended. It is the multiplier on everything else a team does, because every practice that depends on iteration — testing, refactoring, small batches, debugging — degrades in proportion to how long that interval is. A forty-minute build does not make a team forty minutes slower; it changes their behavior. They batch changes to avoid paying the cost, they stop running tests locally, they context-switch while waiting and lose the thread, and they debug by reasoning rather than by experiment. In legacy systems the loop is usually long and nobody has measured it, because it grew a minute at a time over years and each increment was too small to act on. Measuring and attacking it is frequently the highest-return engineering work available, and it is almost always undervalued because it delivers no feature.

## How to Apply ◆

> A team that cannot run a meaningful test in under a minute will not practice test-driven development, incremental refactoring, or small batches, regardless of how much they are encouraged to.

- **Measure the loops you actually have**, separately: time to compile, to run one test, to run the fast suite, to run everything, to get a working local environment, and to deploy to somewhere you can look at it. Teams routinely discover the dominant cost is not the one they complain about.
- **Attack the innermost loop first.** The compile-and-run-one-test cycle is used hundreds of times a day; the full pipeline a few times. An improvement in the inner loop is worth far more than the same improvement further out, even though the outer number is larger and more visible.
- **Split the test suite by speed**, not by type. A fast suite that runs in under two minutes on every change, and a slower one that runs on merge, gives most of the safety with a usable loop. The split is worth doing even if it is imperfect.
- **Eliminate flakiness aggressively.** A suite that fails randomly is not a slow feedback loop but a broken one: developers stop believing failures, which removes the value of running it at all. Quarantine flaky tests immediately and fix or delete them on a deadline.
- **Make the local environment reproducible and fast to create.** Where a developer needs a day to get a working setup, or shares one integration environment with five colleagues, that is the loop — and containerization or scripted provisioning usually addresses it faster than any test optimization.
- **Remove work from the loop rather than making it faster.** Test data created once and reused, dependencies not rebuilt when unchanged, and incremental compilation typically yield more than parallelizing what is already there.
- **Treat pipeline duration as a defect** with a stated budget. Without a stated limit, build time grows monotonically, because every individual addition is justified and no individual addition is refused.
- Give developers a way to **exercise a change against realistic behavior quickly** — a local stub of the external service, a recorded response set, a small anonymized dataset. Where the only way to know whether something works is to deploy and wait, that wait is the real cost.
- **Report the numbers alongside delivery metrics.** Build time is one of the few engineering measures whose improvement can be tied directly to throughput, which makes it unusually easy to justify — but only if someone is reporting it.

## Tradeoffs ⇄

> Shortening the loop compounds into everything else, but the work is invisible to stakeholders and the fast-suite split trades some safety for speed.

**Benefits:**

- Every iterative practice becomes viable. Test-driven development, incremental refactoring, and small batches are not disciplines a team lacks — they are disciplines a slow loop makes impractical.
- Batch sizes fall on their own, because the reason to batch was to amortize a cost that no longer exists.
- Debugging shifts from reasoning to experiment, which is faster and more reliable in a system nobody fully understands.
- Context switching declines, since the wait is short enough not to justify starting something else.
- The measured improvement is directly attributable, which makes it one of the easier engineering investments to defend.

**Costs and Risks:**

- The work delivers no feature, and in a delivery-pressured environment it needs deliberate protection to happen at all.
- Splitting the suite means the fast loop no longer covers everything, and a defect class that moves to the slow suite is caught later.
- Optimization can consume unbounded effort. Past a certain point the remaining time is structural, and further investment returns little.
- Test parallelization and shared fixtures introduce their own flakiness if isolation is imperfect, which can make the loop faster and less trustworthy.
- Fast local environments built on stubs can diverge from production behavior, moving defects from development into integration.

## How It Could Be

A team of eight maintaining a logistics platform had a 34-minute pipeline and a local test suite that took 11 minutes to start because it provisioned a full database schema each run. They measured the loops and found the innermost one — change one line, run the relevant test — averaged 13 minutes. Nobody ran tests locally as a result. Over one quarter they addressed three things: a reusable containerized database fixture, splitting 2,400 tests into a 90-second fast suite and a slower integration suite, and caching unchanged dependency builds. The inner loop dropped to under 40 seconds. Nothing was mandated about testing practice, but the number of tests written per month roughly tripled over the following two quarters, and the proportion of changes that broke the pipeline fell by more than half, because developers were now finding those breakages before pushing.

The flakiness work produced a separate effect the team had not expected. Their suite had 19 tests that failed intermittently, and the team's habit was to re-run the pipeline until it passed — a practice so normalized that nobody described it as a problem. They quarantined all 19 and set a two-week deadline to fix or delete them. Seven were genuinely broken tests and were deleted. Nine had real isolation problems that were fixed. Three turned out to be exposing an actual race condition in the application's cache invalidation, which had been producing rare, unexplained production inconsistencies for at least a year. The team had been suppressing a genuine defect signal by re-running the build.
