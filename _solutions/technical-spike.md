---
title: Technical Spike
description: Validate that an architecture will remain maintainable under expected growth
category:
- Architecture
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/technical-spike
problems:
- analysis-paralysis
- implementation-starts-without-design
- modernization-strategy-paralysis
- fear-of-change
- assumption-based-development
- premature-technology-introduction
- decision-avoidance
layout: solution
---

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
