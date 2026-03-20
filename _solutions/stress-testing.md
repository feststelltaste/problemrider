---
title: Stress Testing
description: Testing the software under extreme load conditions
category:
- Testing
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/stress-testing
problems:
- system-outages
- cascade-failures
- capacity-mismatch
- unpredictable-system-behavior
- scaling-inefficiencies
- slow-incident-resolution
- missing-rollback-strategy
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A payment processing system had experienced two major outages in the past year during unexpected traffic surges, but the team had no understanding of the system's actual limits. They ran a series of stress tests that gradually increased transaction volume from normal levels to 5x peak. At 2.5x peak, they discovered that the legacy message queue's disk-based overflow mechanism had a bug that caused message loss rather than backpressure. At 4x peak, the database's connection pool exhaustion triggered an unhandled exception that crashed the application server rather than degrading gracefully. Both issues were fixed, and the stress test results informed the deployment of auto-scaling rules that activated before the system reached its breaking point.
