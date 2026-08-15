---
title: Error Budgets
description: Quantifying acceptable unreliability as balance between feature velocity
  and reliability
category:
- Management
- Process
problems:
- quality-compromises
- short-term-focus
- deployment-risk
- high-defect-rate-in-production
- competing-priorities
- constant-firefighting
- blame-culture
- micromanagement-culture
layout: solution
related_solutions:
- slug: chaos-engineering
  similarity: 0.8
- slug: service-level-indicators
  similarity: 0.8
- slug: incident-management
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
- slug: site-reliability-engineering-sre
  similarity: 0.8
- slug: error-reporting-and-analysis
  similarity: 0.75
---

## Description

An error budget quantifies the maximum amount of unreliability a system is allowed to accumulate over a period, calculated as the complement of a service level objective — for example, a 99.9% availability SLO leaves a monthly error budget of roughly 43 minutes of downtime. Rather than treating every incident as a unilateral failure to be minimized at all costs, the error budget reframes reliability as a resource that can be spent: as long as consumption stays within the budget, the team is free to prioritize feature velocity, and once it is exhausted, policy dictates that work shifts to stability. This is especially useful in legacy systems trapped in a cycle of shipping features, causing incidents, and then firefighting reactively, because it replaces subjective, often political arguments about "how reliable is reliable enough" with an objective, pre-agreed threshold that both engineering and product stakeholders can see depleting in real time. The mechanism only works, though, if the underlying monitoring can measure reliability accurately and the SLOs are calibrated realistically; done well, it turns the tension between velocity and stability into a managed trade-off instead of a source of blame.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define Service Level Objectives (SLOs) for critical system functions based on business requirements (e.g., 99.9% availability)
- Calculate the error budget as the inverse of the SLO (e.g., 0.1% allowed downtime per month)
- Implement monitoring that tracks actual reliability against the SLO and shows remaining error budget in real time
- Establish policies for what happens when the error budget is exhausted (e.g., freeze feature releases, focus on reliability work)
- Use error budget consumption rate to make data-driven decisions about release velocity vs. stability investment
- Review error budgets monthly with both engineering and product stakeholders to maintain alignment
- Start with a few critical services and expand the practice as the team gains experience

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides an objective framework for balancing feature development against reliability work
- Eliminates subjective arguments about when to invest in stability vs. features
- Makes reliability costs visible to product and business stakeholders
- Creates natural incentives for teams to invest in reliability before error budget is exhausted
- Acknowledges that perfect reliability is neither achievable nor desirable

**Costs and Risks:**
- Requires mature monitoring and observability to measure reliability accurately
- Error budget policies can feel punitive if not framed constructively
- SLOs may be set incorrectly, either too strict (blocking all development) or too lenient (no real constraint)
- Cultural resistance from teams not accustomed to quantified reliability targets
- Gaming the metrics is possible if measurement points are not comprehensive

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy SaaS platform was caught in a cycle where the team shipped features rapidly, causing production incidents, then spent weeks firefighting before returning to features. The team adopted error budgets with a 99.9% availability SLO for their API. In the first month, a series of incidents consumed 80% of the monthly error budget by the second week. Per policy, the team halted feature work and spent the remaining two weeks on reliability improvements: adding circuit breakers, fixing connection pool leaks, and improving deployment rollback speed. The next month, the error budget was barely touched, and the team delivered more features than in any previous month because they were not interrupted by incidents.
