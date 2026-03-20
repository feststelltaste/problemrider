---
title: Self-Monitoring and Diagnosis
description: A system's ability to monitor its own state and detect issues
category:
- Operations
- Architecture
quality_tactics_url: https://qualitytactics.de/en/reliability/self-monitoring-and-diagnosis
problems:
- monitoring-gaps
- slow-incident-resolution
- unpredictable-system-behavior
- gradual-performance-degradation
- constant-firefighting
- system-outages
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Embed diagnostic capabilities within legacy components that continuously check their own operational health
- Implement internal consistency checks that verify data invariants and processing correctness
- Add automatic detection of resource leaks (memory, connections, file handles) within the application
- Create diagnostic endpoints that expose internal state for troubleshooting without external tooling
- Implement automatic remediation for known self-diagnosable issues (connection pool refresh, cache clearing)
- Log diagnostic findings with structured data to enable automated analysis and alerting
- Design self-monitoring to degrade gracefully so monitoring failures do not impact core functionality

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables faster problem detection by monitoring from within the application's own context
- Catches internal issues that external monitoring cannot observe (logic errors, data inconsistencies)
- Reduces dependency on external monitoring infrastructure
- Can trigger automated self-healing for known issue patterns

**Costs and Risks:**
- Self-monitoring code adds complexity and must itself be correct to avoid false diagnostics
- Monitoring overhead in the application process can affect performance
- Self-monitoring has blind spots for issues that affect the monitoring code itself
- Legacy systems may lack extensibility points for adding internal monitoring

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy payment processing system experienced intermittent failures that external monitoring could not explain because all health metrics appeared normal. By adding self-monitoring that tracked internal queue depths, transaction processing rates, and data consistency checksums, the system detected a subtle issue where a background thread was silently dying after processing a specific malformed message type. The self-monitoring system automatically restarted the thread and logged the problematic message for investigation, preventing payment processing delays that had previously gone undetected for hours.
