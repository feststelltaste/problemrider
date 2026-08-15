---
title: Heartbeat
description: Regular transmission of a component's heartbeat to a monitoring instance
category:
- Operations
problems:
- monitoring-gaps
- single-points-of-failure
- slow-incident-resolution
- system-outages
- unpredictable-system-behavior
- constant-firefighting
layout: solution
related_solutions:
- slug: ping
  similarity: 0.85
- slug: watchdog
  similarity: 0.85
- slug: monitoring
  similarity: 0.8
- slug: failover-mechanisms
  similarity: 0.7
- slug: health-check-endpoints
  similarity: 0.7
- slug: self-monitoring-and-diagnosis
  similarity: 0.7
---

## Description

A heartbeat is a signal that a component sends at regular intervals to a monitoring system purely to assert that it is still running, distinct from a health check endpoint in that it is pushed by the component itself rather than pulled on demand by an external caller. This push-based model is particularly well suited to legacy background processes and batch jobs that have no request-driven interface to poll in the first place — a nightly reconciliation job or a long-running queue consumer can report "I am still alive and at this point in my work" without anyone needing to ask. The absence of an expected heartbeat, rather than the presence of an error, becomes the actionable signal: when a legacy job silently hangs with no exception and no log output, which is a common failure mode in older systems with minimal instrumentation, the missing heartbeat is often the only sign that anything is wrong at all. This shifts detection from passive discovery — a downstream team eventually noticing missing data hours later — to active alerting within seconds of the expected interval elapsing, which is frequently one of the cheapest monitoring improvements available for legacy systems that predate modern observability tooling. Because a component can keep sending heartbeats while behaving incorrectly, the mechanism proves only that a process is alive, not that it is doing its job correctly, so it complements rather than replaces deeper functional monitoring.

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Instrument legacy components to send periodic heartbeat signals to a central monitoring system
- Define appropriate heartbeat intervals based on the criticality and expected response time of each component
- Configure alerting rules that trigger when heartbeats are missed for a defined number of consecutive intervals
- Include basic health metadata in heartbeat payloads (memory usage, queue depth, last processed timestamp)
- Implement heartbeat receivers that aggregate status across all monitored components into a dashboard
- Use heartbeat absence as a trigger for automated recovery actions such as process restarts

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Detects component failures within seconds rather than waiting for user reports
- Provides continuous proof-of-life for background processes and batch jobs
- Simple to implement even in legacy systems with limited monitoring infrastructure
- Enables automated failover by detecting unresponsive components

**Costs and Risks:**
- Network issues can cause false-positive failure detections
- Heartbeat mechanisms add minor network and processing overhead
- A component can send heartbeats while being functionally broken (alive but not working correctly)
- The monitoring system itself becomes a dependency that must be kept available

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A manufacturing company ran critical batch processing jobs on a legacy system that occasionally hung without any error output. The operations team often discovered stalled jobs only when downstream systems reported missing data hours later. By adding a simple heartbeat mechanism where each batch job reported progress every 30 seconds, the monitoring system could detect stalled jobs within a minute and automatically restart them. This reduced the average detection time from four hours to under two minutes.
