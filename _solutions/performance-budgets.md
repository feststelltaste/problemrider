---
title: Performance Budgets
description: Defining performance indicators as part of the requirements
category:
- Performance
- Requirements
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/performance-budgets
problems:
- gradual-performance-degradation
- slow-application-performance
- quality-blind-spots
- inadequate-requirements-gathering
- feature-creep-without-refactoring
- high-client-side-resource-consumption
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Establish measurable performance targets for key user-facing operations (e.g., page load under 2 seconds, API response under 500 ms)
- Baseline current performance metrics to understand the gap between current state and desired targets
- Integrate performance budget checks into the CI/CD pipeline so regressions are caught before deployment
- Define budgets for bundle size, time to interactive, memory consumption, and API response times
- Assign performance budgets to teams or components so ownership of performance is distributed
- Review and adjust budgets quarterly as the system evolves and user expectations change
- Make performance budget violations as visible as failing tests to prevent gradual degradation

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Prevents the gradual performance decay that plagues long-lived legacy systems
- Creates shared accountability for performance across the development team
- Provides objective criteria for evaluating the performance impact of new features
- Makes performance a first-class requirement rather than an afterthought

**Costs and Risks:**
- Setting budgets too aggressively can slow feature development and frustrate teams
- Budgets require ongoing calibration as the system and usage patterns evolve
- Legacy systems may be far from any reasonable budget, making the initial gap demoralizing
- Measuring performance accurately requires investment in monitoring infrastructure
- Teams may optimize for the metric rather than actual user experience

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A SaaS platform had seen its main dashboard load time increase from 1.5 seconds to 8 seconds over three years as features were added without performance oversight. The team established a performance budget of 3 seconds for initial dashboard load and 200 ms for subsequent interactions. They added Lighthouse CI checks to their build pipeline that failed the build if budgets were exceeded. Within six months, the team had reduced load time to 2.8 seconds through incremental optimizations, and the budget checks prevented several regressions from reaching production. The budgets also gave the team a clear, non-confrontational way to push back on feature requests that would have blown the budget.
