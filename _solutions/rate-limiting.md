---
title: Rate Limiting
description: Controlling incoming request rates against system overload during traffic spikes
category:
- Architecture
- Performance
quality_tactics_url: https://qualitytactics.de/en/reliability/rate-limiting
problems:
- rate-limiting-issues
- capacity-mismatch
- system-outages
- cascade-failures
- slow-application-performance
- high-api-latency
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy ERP system exposed APIs consumed by multiple internal applications and third-party integrations. A poorly implemented integration from a partner repeatedly hammered the order lookup endpoint with thousands of requests per minute, causing database connection pool exhaustion that affected all users. By deploying rate limiting at the API gateway with per-client quotas, the team protected the legacy backend from individual consumer overload. The partner was given clear rate limit documentation and adjusted their integration to use batch queries, reducing their request volume by 95% while retrieving the same data.
