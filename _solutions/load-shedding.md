---
title: Load Shedding
description: Deliberately dropping low-priority requests under overload, preserving critical capacity
category:
- Architecture
- Performance
quality_tactics_url: https://qualitytactics.de/en/reliability/load-shedding
problems:
- capacity-mismatch
- slow-application-performance
- system-outages
- cascade-failures
- rate-limiting-issues
- task-queues-backing-up
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Classify all request types in the legacy system by business priority (critical, important, best-effort)
- Implement admission control that measures current system load and rejects low-priority requests when thresholds are exceeded
- Return appropriate HTTP status codes (503 with Retry-After) so clients can back off and retry
- Ensure critical paths such as payments, authentication, and core transactions are always served first
- Configure queue-based systems to drop or defer low-priority messages when queue depth exceeds limits
- Monitor shed load volume and alert when shedding frequency indicates a need for capacity expansion

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Keeps critical system functions available during overload situations
- Prevents total system collapse by proactively managing demand
- Provides a controlled response to traffic spikes rather than unpredictable failures
- Buys time for auto-scaling or manual intervention

**Costs and Risks:**
- Dropped requests degrade user experience for low-priority operations
- Priority classification requires careful business input and ongoing maintenance
- Incorrect priority assignments can shed important traffic
- Legacy systems may lack the instrumentation needed to measure load accurately

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A ticket sales platform built on a legacy stack experienced complete failures during high-demand events when all users competed for limited inventory. The team implemented load shedding that prioritized checkout and payment requests while rejecting or queuing search and browsing requests when system load exceeded 80% capacity. During the next major sale event, the checkout flow remained responsive while some users experienced temporary delays on search results. Overall successful transactions increased by 35% compared to previous events where the entire system had collapsed under load.
