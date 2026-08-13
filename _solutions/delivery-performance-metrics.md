---
title: Delivery Performance Metrics
description: Track lead time, deployment frequency, change failure rate, and time
  to restore as a set, so that improvement claims and regressions both become visible.
category:
- Process
- Management
- Operations
problems:
- slow-development-velocity
- slow-feature-development
- long-release-cycles
- extended-cycle-times
- reduced-predictability
- planning-credibility-issues
- immature-delivery-strategy
- quality-degradation
- inefficient-processes
- high-defect-rate-in-production
- increased-time-to-market
- delayed-value-delivery
- approval-dependencies
- blame-culture
- bottleneck-formation
- extended-review-cycles
- feature-factory
- history-of-failed-changes
- micromanagement-culture
- modernization-roi-justification-failure
- negative-brand-perception
- poor-project-control
- process-design-flaws
- product-direction-chaos
- release-anxiety
- review-bottlenecks
- rushed-approvals
- short-term-focus
- uneven-work-flow
- user-trust-erosion
- difficulty-quantifying-benefits
layout: solution
related_solutions:
- slug: baseline-measurement
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: value-stream-mapping
  similarity: 0.7
- slug: development-environment-optimization
  similarity: 0.7
- slug: quality-ratchet
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
---

## Description

Delivery performance metrics are a small set of measures tracked together: how long a change takes from commit to production, how often changes are deployed, what proportion of them cause a problem, and how long recovery takes when one does. Their value comes from being a set rather than individual numbers, because each one is trivially gamed alone and the four constrain each other. Deploying more often looks good until the failure rate rises with it; shipping faster looks good until recovery time balloons. Together they describe throughput and stability at once, which is the tradeoff every delivery decision actually makes. For legacy teams the practical use is twofold: they give an honest baseline for how the current situation compares to what is claimed, and they make the effect of improvement work visible in terms management already understands.

## How to Apply ◆

> Legacy teams are routinely told to deliver faster without anyone establishing how fast they currently deliver or what the constraint is.

- **Establish the baseline before improving anything**, from data you already have: version control timestamps, deployment records, incident tickets. Reconstructing three to six months of history is usually possible in a few days and is worth far more than starting to measure from today.
- **Measure lead time from commit to production**, not from ticket creation. The commit-to-production interval is what the delivery process controls; including the time an idea sat in a backlog measures prioritization, which is a different problem with a different fix.
- **Track all four together and report them together.** A dashboard showing only deployment frequency invites optimizing it in isolation, which is how a team ships more often and breaks more often and calls it improvement.
- **Use distributions, not averages.** A median lead time of two days with a ninety-fifth percentile of six weeks describes a process with a serious problem that the average conceals. The tail is usually where the interesting cause lives.
- **Define change failure rate concretely** — a deployment requiring a hotfix, rollback, or causing an incident — and apply the definition consistently. Precision matters less than stability, because the trend is what is informative.
- **Report at team level and never at individual level.** These measures describe a delivery system, and applied to individuals they measure commit granularity and produce immediate gaming.
- **Pair the metrics with the value stream map.** The metrics tell you the process is slow; the map tells you where. Metrics without the map produce exhortation, and maps without metrics produce improvements nobody can confirm.
- **Attribute improvement work against them.** A build-time reduction, an environment automation, or a review-process change should show up in lead time or deployment frequency. Improvement work that moves none of the four deserves an explanation.
- **Do not chase industry benchmarks.** A team maintaining a mainframe batch system will not reach the deployment frequency of a web service, and pursuing the benchmark rather than the trend produces demoralization and distortion.
- **Add stability measures for legacy contexts**: the share of capacity going to unplanned work, and incident hours. In maintenance-dominated environments these often describe the constraint better than throughput does.

## Tradeoffs ⇄

> The four measures give an honest, comparable picture of delivery and make improvement demonstrable, but any measure that determines how a team is judged will eventually be optimized directly.

**Benefits:**

- Improvement work becomes demonstrable in terms management already accepts, which is usually the missing ingredient in funding it.
- Regressions surface early. A slowly rising lead time or failure rate is visible in the trend long before it becomes an obvious crisis.
- The throughput-stability tradeoff is made explicit, which is what prevents each being pursued at the other's expense.
- Claims about performance — from the team, from management, from vendors — become checkable rather than rhetorical.
- Percentile reporting reveals the long tail, which is where the structural blockers in a legacy delivery process usually hide.

**Costs and Risks:**

- Measures become targets and are then gamed, typically by splitting changes finer or reclassifying failures. This is not preventable, only detectable.
- Instrumentation takes real effort where deployments are manual and incidents are tracked informally, which is common in legacy environments.
- Applied to individuals or used comparatively between teams with very different systems, the metrics do active harm.
- Benchmarks invite inappropriate comparison, and a team held to figures achievable only with a different architecture will optimize the number rather than the system.
- The four measures say nothing about whether what is delivered is worth delivering, and a team can improve all of them while shipping features nobody uses.

## How It Could Be

A team maintaining an insurance underwriting platform was under sustained pressure to deliver faster, with no agreement about how fast they currently were. They reconstructed six months of history from version control and deployment logs: median commit-to-production lead time of 19 days, ninety-fifth percentile of 71 days, deployments twice a month, change failure rate around 22 percent, and median time to restore of just over five hours. The ninety-fifth percentile was the finding. Investigation showed the tail consisted almost entirely of changes touching a subsystem whose deployment required coordination with a partner's release schedule. That was a specific, addressable constraint that no amount of general pressure to work faster would ever have found. Negotiating an independent deployment path for that subsystem took a quarter and pulled the ninety-fifth percentile down to 12 days.

The four-measure discipline prevented a mistake the following year. A push toward more frequent deployment moved the team from twice-monthly to twice-weekly over two quarters, which looked like clear success. The change failure rate, tracked alongside, rose from 22 percent to 34 percent in the same period, and median time to restore lengthened. Reading the set together showed that the team had increased deployment frequency by deploying the same large changes more often rather than by making changes smaller, so each deployment carried comparable risk and there were now more of them. They paused the frequency push, spent a quarter on batch size and on the test suite, and then resumed. By the end of the following quarter they were deploying three times a week with a failure rate of 11 percent — an outcome that reporting deployment frequency alone would have declared achieved a year earlier and incorrectly.
