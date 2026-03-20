---
title: Monitoring
description: Continuous monitoring of system states, performance, and errors
category:
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/monitoring
problems:
- monitoring-gaps
- slow-incident-resolution
- constant-firefighting
- system-outages
- gradual-performance-degradation
- unpredictable-system-behavior
- high-defect-rate-in-production
- poor-operational-concept
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Instrument legacy applications with metrics collection for key business and technical indicators
- Deploy centralized log aggregation to consolidate logs from all legacy system components
- Create dashboards that display system health, error rates, response times, and resource utilization
- Set up alerting rules with appropriate severity levels and notification channels
- Monitor both infrastructure metrics (CPU, memory, disk) and application metrics (request rates, error rates, latency)
- Add distributed tracing to track requests across legacy system boundaries
- Review and tune alert thresholds regularly to reduce noise and prevent alert fatigue
- Include business metrics (order counts, transaction values) alongside technical monitoring

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables proactive detection of problems before they become user-facing incidents
- Provides data for root cause analysis and trend identification
- Reduces mean time to detection and resolution for production issues
- Supports capacity planning with historical utilization data
- Creates visibility into legacy system behavior that may have been opaque for years

**Costs and Risks:**
- Monitoring infrastructure requires its own maintenance and capacity planning
- Excessive monitoring can create alert fatigue, causing teams to ignore warnings
- Instrumenting legacy applications may require code changes or wrapper scripts
- Storage costs for metrics and logs can grow significantly over time
- Poorly configured monitoring provides false confidence

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company operated a legacy warehouse management system with no monitoring beyond checking if the process was running. Issues were discovered only when warehouse workers reported errors or missing data. After deploying monitoring that tracked order processing rates, database query latencies, and error logs, the team gained visibility into a slow memory leak that had been causing weekly restarts and a database query that degraded as inventory grew. With this data, they fixed both issues proactively and established alerting that caught future problems minutes after they appeared rather than hours later.
