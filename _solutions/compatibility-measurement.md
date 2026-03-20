---
title: Compatibility Measurement
description: Quantify compatibility status through metrics, audits, and risk assessments
category:
- Process
- Testing
quality_tactics_url: https://qualitytactics.de/en/compatibility/compatibility-measurement
problems:
- quality-blind-spots
- invisible-nature-of-technical-debt
- monitoring-gaps
- difficulty-quantifying-benefits
- integration-difficulties
- breaking-changes
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Define measurable compatibility metrics such as API contract violation rate, consumer migration percentage, and integration test pass rate
- Instrument API gateways and integration points to track compatibility incidents in production
- Conduct periodic compatibility audits that assess all integration points against current standards
- Create dashboards that show compatibility health across the system landscape
- Include compatibility metrics in release readiness reviews
- Track the age and usage of deprecated interfaces to prioritize retirement efforts

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Makes compatibility status visible and actionable for management and engineering alike
- Enables data-driven prioritization of compatibility improvements
- Provides early warning when compatibility is degrading before incidents occur

**Costs and Risks:**
- Defining meaningful metrics requires domain knowledge and cross-team agreement
- Measurement infrastructure adds operational complexity
- Metrics can be gamed or misinterpreted if not carefully designed
- Over-measurement can create dashboard fatigue without driving action

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An enterprise platform team introduced a compatibility dashboard tracking three metrics: percentage of API consumers on the latest version, contract test pass rate across services, and count of deprecated endpoints still receiving traffic. The dashboard revealed that 40% of consumers were still using an API version scheduled for removal in two months. This early visibility triggered a targeted outreach campaign, and consumer migration reached 95% before the deadline, avoiding what would have been a significant production disruption.
