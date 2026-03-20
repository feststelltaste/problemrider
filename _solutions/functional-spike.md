---
title: Functional Spike
description: Investigate business risks through time-limited experiments
category:
- Process
- Requirements
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/functional-spike
problems:
- fear-of-change
- analysis-paralysis
- modernization-strategy-paralysis
- difficulty-quantifying-benefits
- assumption-based-development
- implementation-rework
- fear-of-breaking-changes
- history-of-failed-changes
layout: solution
---

## How to Apply ◆

> In legacy systems, functional spikes help teams reduce uncertainty before committing to costly changes by running short, focused experiments.

- Identify the highest-risk assumptions in a planned legacy modernization effort — for example, whether a critical business rule can be extracted from a monolith without breaking dependent workflows.
- Time-box each spike strictly (one to five days) and define a clear question it must answer, such as "Can we replace the batch pricing calculation with a real-time service without exceeding latency thresholds?"
- Build the simplest possible implementation that answers the question — throwaway code is acceptable and expected, because the goal is learning, not production readiness.
- Involve domain experts during the spike to validate that the experiment addresses actual business behavior, especially when legacy logic is undocumented or only exists in tribal knowledge.
- Document the findings immediately after the spike concludes, including what worked, what failed, and what new risks were discovered.
- Use spike results to update estimates and plans for the actual implementation, replacing guesswork with evidence from the experiment.
- If a spike reveals that the original approach is not viable, treat that as a success — the team avoided weeks or months of wasted effort.

## Tradeoffs ⇄

> Spikes trade a small amount of time for significantly reduced risk, but they require discipline to keep focused and time-limited.

**Benefits:**

- Reduces the risk of committing to expensive modernization paths that turn out to be infeasible, by surfacing technical and business blockers early.
- Builds team confidence in proposed changes by providing concrete evidence rather than theoretical arguments.
- Helps justify modernization investments to stakeholders by demonstrating feasibility with minimal upfront cost.
- Uncovers hidden dependencies and undocumented behavior in legacy systems before they derail full implementation efforts.

**Costs and Risks:**

- Teams may struggle to discard spike code and insist on evolving it into production code, undermining quality.
- Without strict time-boxing, spikes can expand into mini-projects that consume resources without delivering production value.
- Stakeholders unfamiliar with the practice may view spikes as wasted time rather than risk reduction.
- Repeated spikes without follow-through on actual implementation can erode team motivation and stakeholder trust.

## How It Could Be

> The following scenarios illustrate how functional spikes reduce risk in legacy modernization contexts.

A financial services company needed to determine whether its 15-year-old order management system could be decomposed into microservices. Rather than committing to a six-month migration plan, the team ran a three-day spike to extract a single order validation rule from the monolith and deploy it as a standalone service. The spike revealed that the validation logic depended on seven undocumented database views and two stored procedures, making extraction far more complex than initially estimated. This finding led the team to adopt a strangler fig approach instead, saving months of rework.

An e-commerce platform was considering replacing its legacy search engine with a modern alternative. A two-day spike comparing search result quality between the old and new engines on production data revealed that the legacy system had accumulated years of hand-tuned relevance boosting rules that the new engine could not replicate out of the box. The team used this finding to plan a phased migration with explicit relevance tuning milestones rather than a big-bang replacement.
