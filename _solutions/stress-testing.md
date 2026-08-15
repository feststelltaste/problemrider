---
title: Stress Testing
description: Testing the software under extreme load conditions
category:
- Testing
- Performance
problems:
- system-outages
- cascade-failures
- capacity-mismatch
- unpredictable-system-behavior
- scaling-inefficiencies
- slow-incident-resolution
- missing-rollback-strategy
- deadlock-conditions
- stack-overflow-errors
- race-conditions
- dma-coherency-issues
- incorrect-max-connection-pool-size
- lock-contention
- misconfigured-connection-pools
layout: solution
related_solutions:
- slug: load-testing
  similarity: 0.9
- slug: chaos-engineering
  similarity: 0.85
- slug: rate-limiting
  similarity: 0.75
- slug: integration-tests
  similarity: 0.75
- slug: cross-version-testing
  similarity: 0.75
- slug: resilience
  similarity: 0.75
---

## Description

Stress testing deliberately pushes a system beyond its expected peak load — gradually increasing traffic, exhausting resources such as database connections or memory, or injecting failures like process kills and network partitions — until it degrades or breaks, in order to discover its actual capacity ceiling and failure modes rather than assuming they are understood. This differs from ordinary load testing in intent: the goal is not to confirm the system handles expected traffic but to deliberately find where and how it stops handling traffic, which is information that can only be obtained by actually causing the failure under controlled conditions. Legacy systems are especially prone to failing in ways nobody anticipated, because their original capacity assumptions were set long ago against traffic patterns that have since changed, and the components involved were often never designed with graceful degradation in mind — a connection pool exhaustion might trigger an unhandled crash rather than a controlled backpressure response, for instance. Running stress tests surfaces these failure modes — a queue's overflow mechanism silently dropping messages instead of applying backpressure, a crash instead of a degraded response — while the system is under observation in a controlled environment, rather than during an actual production incident when the same discovery is far more costly and far less calm. The results directly inform where circuit breakers, auto-scaling rules, and alerting thresholds should be set, but the practice requires an isolated environment to avoid corrupting real data or state, and it consumes significant infrastructure resources to execute meaningfully.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Design stress tests that push the system beyond expected peak load to find breaking points and failure modes
- Gradually increase load until the system degrades or fails, recording metrics at each level to build a capacity profile
- Test failure and recovery behaviors: what happens when the database runs out of connections, memory is exhausted, or disks fill up
- Include chaos engineering elements such as killing processes, introducing network partitions, or degrading dependencies
- Run stress tests against a production-like environment with representative data volumes
- Document observed failure modes and their symptoms to improve incident response playbooks
- Use stress test results to establish and validate alerting thresholds

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reveals how the system fails, enabling proactive hardening before production incidents
- Identifies the actual capacity ceiling, not just the comfortable operating range
- Validates that graceful degradation and circuit breakers work under extreme conditions
- Improves team confidence in handling production emergencies

**Costs and Risks:**
- Stress tests can cause data corruption or state inconsistencies in the test environment
- Requires isolated environments to prevent impact on other systems
- Legacy systems may fail in destructive ways during stress tests, requiring careful preparation
- Results may be alarming to stakeholders if not communicated with context
- Running stress tests requires significant infrastructure resources

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A payment processing system had experienced two major outages in the past year during unexpected traffic surges, but the team had no understanding of the system's actual limits. They ran a series of stress tests that gradually increased transaction volume from normal levels to 5x peak. At 2.5x peak, they discovered that the legacy message queue's disk-based overflow mechanism had a bug that caused message loss rather than backpressure. At 4x peak, the database's connection pool exhaustion triggered an unhandled exception that crashed the application server rather than degrading gracefully. Both issues were fixed, and the stress test results informed the deployment of auto-scaling rules that activated before the system reached its breaking point.
