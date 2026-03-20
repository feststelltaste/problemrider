---
title: Performance Measurements
description: Continuous measurement and storage of performance metrics in production
category:
- Performance
- Operations
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/performance-measurements
problems:
- gradual-performance-degradation
- monitoring-gaps
- slow-application-performance
- slow-incident-resolution
- quality-blind-spots
- capacity-mismatch
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Instrument key code paths with timing metrics, starting with the most user-visible operations
- Deploy a metrics collection system (e.g., Prometheus, Datadog, StatsD) that stores time-series performance data
- Create dashboards that visualize performance trends over time, making degradation immediately visible
- Set up alerts for performance threshold violations so issues are detected before users report them
- Capture percentile distributions (p50, p95, p99) rather than just averages to understand the full performance picture
- Correlate performance metrics with deployment events to identify regressions introduced by specific changes
- Retain historical data long enough to observe seasonal patterns and long-term trends

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Makes gradual performance degradation visible before it reaches crisis levels
- Provides evidence-based data for prioritizing performance improvements
- Reduces mean time to resolution for performance incidents through faster root cause identification
- Creates accountability by linking performance changes to specific deployments

**Costs and Risks:**
- Instrumentation adds a small overhead to request processing
- Legacy systems without standardized instrumentation points require significant initial effort
- Metrics infrastructure requires its own maintenance, storage, and monitoring
- Too many metrics can create noise and alert fatigue
- Teams may over-index on measurable metrics while missing user-perceived issues

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A banking application experienced intermittent slowdowns that were reported by customers but could never be reproduced in testing. The team added distributed tracing and response time metrics to all API endpoints, storing the data in Prometheus with Grafana dashboards. Within two weeks, the dashboards revealed that the p99 response time for account balance queries spiked to 15 seconds every day between 2 and 3 PM, correlating with an automated reconciliation batch job that competed for database connections. This insight, invisible without continuous measurement, led to rescheduling the batch job to off-peak hours and implementing connection pool isolation.
