---
title: Service Level Indicators
description: Tracking key metrics of software reliability and performance
category:
- Operations
- Management
quality_tactics_url: https://qualitytactics.de/en/reliability/service-level-indicators
problems:
- monitoring-gaps
- gradual-performance-degradation
- constant-firefighting
- slow-application-performance
- poor-operational-concept
- difficulty-quantifying-benefits
- unpredictable-system-behavior
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify the metrics that best represent the user experience for each legacy system service (latency, error rate, throughput)
- Instrument legacy applications to emit SLI data through metrics collection, log aggregation, or synthetic monitoring
- Define measurement boundaries clearly (e.g., latency measured from load balancer receipt to response, excluding client network time)
- Establish baselines from historical data before setting targets
- Create dashboards that display SLI trends over time and highlight deviations from expected behavior
- Use SLIs to derive error budgets that balance reliability investment with feature development velocity
- Review SLIs in regular operations meetings to maintain awareness of legacy system health trends

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides objective, quantitative visibility into legacy system reliability
- Enables trend-based early warning before users experience problems
- Creates a common language for discussing system health across technical and business teams
- Supports data-driven decisions about when legacy systems need investment versus when they are stable enough

**Costs and Risks:**
- Choosing the wrong SLIs can provide a misleading picture of system health
- Instrumenting legacy systems to emit reliable metrics may require significant effort
- Focusing solely on measurable indicators can neglect important qualitative aspects
- SLI data without context can lead to misguided optimization efforts

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce platform's operations team relied on anecdotal reports to assess system health. By implementing SLIs tracking p50 and p99 request latency, error rates per endpoint, and checkout completion rates, the team discovered that while average performance was acceptable, the p99 latency had been steadily increasing for six months due to growing database table sizes. This data-driven insight led to a targeted database optimization effort that reduced p99 latency by 70% and improved checkout completion rates by 8%, directly demonstrating the business value of reliability investment in the legacy system.
