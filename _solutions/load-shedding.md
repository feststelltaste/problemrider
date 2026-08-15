---
title: Load Shedding
description: Deliberately dropping low-priority requests under overload, preserving
  critical capacity
category:
- Architecture
- Performance
problems:
- capacity-mismatch
- slow-application-performance
- system-outages
- cascade-failures
- rate-limiting-issues
- task-queues-backing-up
- unbounded-data-structures
- insufficient-worker-capacity
- work-queue-buildup
layout: solution
related_solutions:
- slug: rate-limiting
  similarity: 0.8
- slug: graceful-degradation
  similarity: 0.75
- slug: load-balancing
  similarity: 0.7
- slug: backpressure
  similarity: 0.7
- slug: distributed-caching
  similarity: 0.65
- slug: lazy-loading
  similarity: 0.65
---

## Description

Load shedding is the deliberate, controlled rejection of a portion of incoming traffic when a system is overloaded, so that the capacity that remains is preserved for the requests that matter most rather than being spread so thin across all requests that everything fails. It works by classifying requests into priority tiers ahead of time, measuring current load against defined thresholds, and having the system actively refuse or defer low-priority work — typically returning an explicit status such as 503 with a retry hint — while critical paths like authentication or payment continue to be served normally. Legacy systems are especially exposed to overload collapse because they were often built without any admission control at all: every request is treated identically, resources are consumed on a first-come basis, and once demand exceeds capacity the system does not degrade gracefully but grinds to a halt for every user simultaneously, including the ones performing the most business-critical actions. Introducing load shedding turns an uncontrolled, all-or-nothing failure mode into a designed, partial one, which is a meaningful improvement for legacy applications that cannot be easily redesigned for elastic scaling and must instead survive demand spikes with roughly fixed capacity. The approach depends on an accurate, continuously maintained understanding of which requests are actually low priority from the business's perspective, which is often the hardest part to establish in a legacy system where that classification was never made explicit in the first place.

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
