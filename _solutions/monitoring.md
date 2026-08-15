---
title: Monitoring
description: Continuous monitoring of system states, performance, and errors
category:
- Operations
problems:
- monitoring-gaps
- slow-incident-resolution
- constant-firefighting
- system-outages
- gradual-performance-degradation
- unpredictable-system-behavior
- high-defect-rate-in-production
- poor-operational-concept
- cache-invalidation-problems
- database-connection-leaks
- deadlock-conditions
- index-fragmentation
- inefficient-database-indexing
- load-balancing-problems
- poor-caching-strategy
- synchronization-problems
- unused-indexes
- upstream-timeouts
- log-spam
- long-running-database-transactions
- race-conditions
- dma-coherency-issues
- excessive-logging
- lock-contention
- long-running-transactions
- rate-limiting-issues
- service-discovery-failures
layout: solution
related_solutions:
- slug: continuous-performance-monitoring
  similarity: 0.85
- slug: monitoring-system-utilization
  similarity: 0.85
- slug: logging
  similarity: 0.8
- slug: performance-measurements
  similarity: 0.8
- slug: security-monitoring
  similarity: 0.8
- slug: monitoring-system-integrity
  similarity: 0.8
---

## Description

Monitoring is the continuous collection and observation of a system's technical and business signals — metrics, logs, traces, error rates, response times, and resource utilization — surfaced through dashboards and alerting so that problems can be detected and diagnosed proactively rather than discovered only when a user reports them. Implementing it means instrumenting applications to emit metrics, aggregating logs centrally across all components, adding distributed tracing so a single request can be followed across service boundaries, and tuning alert thresholds and severities so the signal reaches the right people at the right urgency. Legacy systems are frequently operated with monitoring that amounts to little more than confirming a process is still running, which means the team has no visibility into gradual degradation, resource exhaustion, or intermittent errors until they escalate into an incident severe enough for someone downstream to notice and report it. Establishing real monitoring over such a system is often the single highest-leverage first step in any modernization effort, because it converts years of opaque, undocumented runtime behavior into observable data — memory leaks, slow queries that degrade with data growth, race conditions — that can then be diagnosed and fixed with evidence instead of guesswork. The risk of doing this poorly is that monitoring, once in place, can just as easily produce false confidence or alert fatigue as it can produce insight, so the instrumentation needs to be paired with disciplined threshold review to keep signal-to-noise workable as the legacy system and its failure modes continue to evolve.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company operated a legacy warehouse management system with no monitoring beyond checking if the process was running. Issues were discovered only when warehouse workers reported errors or missing data. After deploying monitoring that tracked order processing rates, database query latencies, and error logs, the team gained visibility into a slow memory leak that had been causing weekly restarts and a database query that degraded as inventory grew. With this data, they fixed both issues proactively and established alerting that caught future problems minutes after they appeared rather than hours later.
