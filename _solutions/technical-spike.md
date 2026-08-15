---
title: Technical Spike
description: Validate that an architecture will remain maintainable under expected
  growth
category:
- Architecture
- Process
problems:
- analysis-paralysis
- implementation-starts-without-design
- modernization-strategy-paralysis
- fear-of-change
- assumption-based-development
- premature-technology-introduction
- decision-avoidance
- cv-driven-development
- decision-paralysis
- delayed-decision-making
- extended-research-time
- inability-to-innovate
- procrastination-on-complex-tasks
- reduced-innovation
- complex-implementation-paths
layout: solution
related_solutions:
- slug: functional-spike
  similarity: 0.8
- slug: walking-skeleton
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.65
- slug: living-documentation
  similarity: 0.65
- slug: pattern-language
  similarity: 0.65
- slug: risk-analysis
  similarity: 0.65
---

## Description

A technical spike is a strictly time-boxed investigation — typically one to three days — built to answer a single, specific architectural question through the simplest possible prototype, with the code itself discarded once the answer is captured. Legacy modernization decisions are especially prone to stalling in unresolved debate precisely because they often hinge on unknowns that no amount of discussion can settle — whether a migration approach will actually perform under load, whether a new framework integrates cleanly with a legacy API — and a spike replaces that debate with empirical evidence gathered directly against the real system. Discarding the prototype code afterward, rather than letting it slip toward production, keeps the exercise honest: the value is the answer to the question, not a head start on the implementation, and a spike whose scope is not held tightly can quietly turn into an open-ended side project instead of a fast, decisive input to the actual decision.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define a clear question or hypothesis the spike should answer before starting
- Time-box the spike strictly (typically one to three days) to prevent it from becoming an open-ended project
- Build the simplest possible prototype that validates or invalidates the hypothesis
- Focus on the riskiest unknowns: integration with legacy APIs, performance under load, or migration feasibility
- Document findings and decisions regardless of whether the spike succeeds or fails
- Discard spike code after capturing learnings; do not let prototype code slip into production
- Present spike results to the team to inform collective decision-making

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces risk by validating assumptions before committing to expensive implementation
- Provides concrete evidence to support or challenge architectural decisions
- Breaks analysis paralysis by turning theoretical debates into empirical investigations
- Builds team confidence in the chosen approach

**Costs and Risks:**
- Time spent on spikes does not directly produce production-ready code
- Poorly scoped spikes can drag on and become mini-projects
- Spike results may be misinterpreted if the prototype conditions do not match production reality
- Teams may become dependent on spikes and reluctant to commit without one

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A team was debating whether to migrate a legacy monolith's data access layer from raw JDBC to an ORM framework. Opinions were divided, and the discussion had stalled for weeks. The architect proposed a two-day spike where one developer migrated a single, representative module to the ORM and measured the impact on performance, code complexity, and test writability. The spike revealed that the ORM handled 90% of queries well but struggled with the system's complex reporting queries. This evidence led the team to adopt the ORM for standard CRUD operations while keeping optimized SQL for reporting, ending the debate with a pragmatic, evidence-based decision.
