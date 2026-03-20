---
title: Continuous Performance Monitoring
description: Ongoing monitoring and analysis of application performance in production
category:
- Performance
- Operations
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/continuous-performance-monitoring
problems:
- monitoring-gaps
- gradual-performance-degradation
- slow-application-performance
- slow-incident-resolution
- unpredictable-system-behavior
- system-outages
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Instrument the legacy application with APM agents or metrics libraries to collect response times, error rates, and throughput
- Define performance baselines and set alerts for deviations from normal behavior
- Monitor at multiple levels: infrastructure (CPU, memory, disk), application (response times, error rates), and business (transaction volumes, conversion rates)
- Implement real user monitoring (RUM) to capture actual end-user experience rather than relying only on synthetic tests
- Create dashboards that visualize performance trends over time to detect gradual degradation
- Integrate performance monitoring into the deployment pipeline to detect regressions immediately after releases
- Conduct regular performance review sessions where the team examines trends and plans optimizations

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Detects performance degradation before it impacts users, enabling proactive intervention
- Provides evidence-based data for prioritizing performance optimization work
- Reduces mean time to resolution by pointing directly to the source of slowdowns
- Creates accountability for performance by making it visible and measurable

**Costs and Risks:**
- Monitoring infrastructure adds cost and operational overhead
- Instrumentation can itself impact performance if not implemented carefully
- Alert fatigue from poorly tuned thresholds can cause teams to ignore real issues
- Large volumes of monitoring data require storage and management

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy e-commerce platform experienced a gradual increase in page load times over six months, but because there was no systematic performance monitoring, the degradation went unnoticed until customers started complaining. The team deployed an APM solution and established baselines for key transactions. Within the first week, the monitoring revealed that a specific database query used by the product search had degraded from 50ms to 800ms as the product catalog grew. After adding a missing index, search performance returned to normal. The team then set alerts for any transaction exceeding twice its baseline, catching two more performance regressions in the following month before users noticed them.
