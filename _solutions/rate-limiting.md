---
title: Rate Limiting
description: Controlling incoming request rates against system overload during traffic
  spikes
category:
- Architecture
- Performance
problems:
- rate-limiting-issues
- capacity-mismatch
- system-outages
- cascade-failures
- slow-application-performance
- high-api-latency
- graphql-complexity-issues
- unbounded-data-structures
- work-queue-buildup
- task-queues-backing-up
layout: solution
related_solutions:
- slug: load-shedding
  similarity: 0.8
- slug: retry
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: load-balancing
  similarity: 0.8
- slug: connection-pooling
  similarity: 0.75
- slug: circuit-breaker
  similarity: 0.75
---

## Description

Rate limiting caps the number of requests a client, API key, or endpoint may make within a given time window, typically enforced at an API gateway or reverse proxy using algorithms such as token bucket or sliding window, with requests beyond the limit rejected with an informative 429 response rather than allowed to overwhelm the backend. This is especially relevant for legacy systems because they were frequently designed and sized for a fixed, bounded set of consumers and a load profile that has since grown well beyond the original assumptions, leaving no architectural headroom to absorb an unexpected surge from a single misbehaving client or integration. A single poorly implemented downstream integration hammering a legacy endpoint can exhaust a shared resource — a database connection pool, for instance — and degrade the experience for every other consumer of that same legacy backend, a failure mode that rate limiting converts from an uncontrolled, system-wide outage into a predictable, isolated rejection of the offending traffic alone. Placed at the gateway, rate limiting protects the legacy system without requiring any change to the legacy code itself, which matters because that code is frequently the part of the system least safe or least understood well enough to modify directly. The tradeoff is that setting effective limits requires an accurate understanding of the legacy system's actual sustainable throughput, and limits set incorrectly either fail to protect the backend or needlessly throttle legitimate high-volume usage during genuine business peaks.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify the maximum sustainable throughput for legacy system endpoints through load testing
- Implement rate limits at the API gateway or reverse proxy layer to protect legacy backends
- Use token bucket or sliding window algorithms for smooth rate enforcement
- Configure different rate limits per client, API key, or endpoint based on business priority
- Return informative 429 (Too Many Requests) responses with Retry-After headers
- Implement rate limiting for internal service-to-service calls to prevent noisy neighbor problems
- Monitor rate limit hits to distinguish between abuse and legitimate demand that needs capacity expansion

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Protects legacy systems from traffic spikes that exceed their capacity
- Prevents a single client or integration from monopolizing system resources
- Provides a predictable, controlled response to overload rather than unpredictable failures
- Enables fair resource sharing across multiple consumers of legacy services

**Costs and Risks:**
- Legitimate high-volume users may be throttled during peak business periods
- Rate limit configuration requires understanding of actual system capacity
- Incorrectly set limits can either fail to protect the system or unnecessarily reject valid traffic
- Rate limiting at the edge does not protect against internal amplification patterns

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy ERP system exposed APIs consumed by multiple internal applications and third-party integrations. A poorly implemented integration from a partner repeatedly hammered the order lookup endpoint with thousands of requests per minute, causing database connection pool exhaustion that affected all users. By deploying rate limiting at the API gateway with per-client quotas, the team protected the legacy backend from individual consumer overload. The partner was given clear rate limit documentation and adjusted their integration to use batch queries, reducing their request volume by 95% while retrieving the same data.
