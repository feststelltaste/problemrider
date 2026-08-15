---
title: Service Level Indicators
description: Tracking key metrics of software reliability and performance
category:
- Operations
- Management
problems:
- monitoring-gaps
- gradual-performance-degradation
- constant-firefighting
- slow-application-performance
- poor-operational-concept
- difficulty-quantifying-benefits
- unpredictable-system-behavior
layout: solution
related_solutions:
- slug: service-level-agreements
  similarity: 0.85
- slug: service-level-objectives
  similarity: 0.85
- slug: monitoring
  similarity: 0.8
- slug: transparent-performance-metrics
  similarity: 0.8
- slug: error-budgets
  similarity: 0.8
- slug: continuous-performance-monitoring
  similarity: 0.8
---

## Description

A service level indicator is a directly measured, quantitative signal of user-facing behavior — latency, error rate, throughput, or a similar metric — captured continuously from the running system rather than inferred or reported anecdotally. SLIs are the raw measurement layer beneath service level objectives and agreements: without a reliable SLI, an SLO target is unenforceable and an SLA commitment is unverifiable. In legacy systems, this measurement layer is often the missing piece, because components were built before observability was a design concern and expose no natural hooks for capturing request timing, success rates, or queue depth. Defining SLIs therefore starts with deciding what "good" looks like from the user's perspective and then instrumenting, retrofitting, or externally probing the legacy system until that signal can be captured reliably and attributed to a specific boundary (such as load-balancer-to-response latency, excluding client network time). The resulting data replaces guesswork and tribal knowledge about system health with a continuous, trendable record, which is what makes it possible to notice slow degradation — a creeping p99 latency increase, a shrinking error margin — long before it manifests as an outage. Because SLIs are the foundation for error budgets and burn-rate alerting, getting the indicator definition right is a prerequisite for every downstream reliability practice built on top of it.

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
