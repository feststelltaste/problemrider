---
title: Watchdog
description: Monitoring component for detecting and handling system errors or failures
category:
- Operations
quality_tactics_url: https://qualitytactics.de/en/reliability/watchdog
problems:
- system-outages
- cascade-failures
- slow-incident-resolution
- monitoring-gaps
- unpredictable-system-behavior
- single-points-of-failure
- constant-firefighting
- silent-data-corruption
layout: solution
---

## How to Apply ◆

> Legacy systems often fail silently, with no mechanism to detect when a component has stopped functioning correctly. Watchdog processes provide automated detection and recovery for failures that would otherwise go unnoticed until users report symptoms.

- Identify all critical processes in the legacy system that must remain running continuously — application servers, batch processors, message consumers, scheduled jobs, and background workers. Each of these needs watchdog supervision.
- Implement heartbeat-based monitoring where each critical process periodically signals its liveness to a watchdog service. If the watchdog does not receive a heartbeat within the expected interval, it triggers an alert and optionally initiates a recovery action.
- Use operating system-level process supervisors (systemd, supervisord, Windows Service Control Manager) as a first line of defense to automatically restart crashed processes. Configure restart limits to prevent restart loops when a process crashes repeatedly due to a persistent issue.
- Implement application-level health checks that go beyond process liveness. A process may be running but stuck (deadlocked threads, exhausted connection pool, infinite loop). Watchdog checks should verify that the process is actually making progress — processing messages, completing requests, advancing batch jobs.
- Add stuck-state detection for batch processes and long-running operations. If a batch job has not advanced its progress marker within a configurable time window, the watchdog should alert and optionally terminate the stuck process so it can be restarted.
- Configure escalation policies: first attempt automatic recovery (restart the process), then alert the on-call engineer if automatic recovery fails or if the same process requires recovery more than a threshold number of times within a time window.
- Monitor the watchdog itself — a watchdog that fails silently is worse than no watchdog at all. Use a secondary monitoring system or mutual watchdog arrangements where two watchdog processes monitor each other.

## Tradeoffs ⇄

> Watchdog processes provide automated failure detection and recovery, reducing downtime and the need for manual intervention, but they add operational complexity and can mask underlying problems if not properly managed.

**Benefits:**

- Detects failures within seconds or minutes rather than hours, dramatically reducing the impact of component failures on users.
- Enables automatic recovery from transient failures (process crashes, resource exhaustion) without requiring human intervention.
- Provides a consistent monitoring layer for legacy components that cannot be easily instrumented with modern observability tools.
- Reduces the operational burden on on-call engineers by handling common failure scenarios automatically.

**Costs and Risks:**

- Automatic restarts can mask persistent underlying problems by continuously restarting a failing process rather than addressing the root cause.
- Watchdog-triggered restarts may cause data loss or corruption if the failing process holds uncommitted state that is not properly cleaned up on termination.
- The watchdog itself is a component that must be monitored, maintained, and kept highly available — a failed watchdog creates a false sense of safety.
- Overly aggressive restart policies can cause thrashing where a process repeatedly starts, fails, restarts, and fails again, potentially worsening the situation.

## Examples

> The following scenarios illustrate how watchdog processes detect and recover from failures in legacy systems.

A legacy messaging system runs a Java-based message consumer that processes incoming orders from a queue. Approximately once a week, the consumer encounters a rare deserialization error that causes an unhandled exception, terminating the JVM process. Without a watchdog, the dead consumer goes unnoticed for hours until the queue depth alarm triggers, by which time thousands of orders have accumulated. The team configures a systemd service unit for the consumer with automatic restart on failure and a restart limit of 5 attempts within 10 minutes. They also deploy a custom watchdog that monitors the consumer's processing rate — if zero messages are processed for 3 consecutive minutes while the queue is non-empty, the watchdog kills the process to force a restart and alerts the on-call engineer. After implementation, consumer failures are recovered automatically within 30 seconds, and the on-call team is notified only when repeated failures suggest a systemic problem requiring investigation.

A legacy financial reporting system runs nightly batch jobs that generate regulatory reports. Occasionally, a batch job hangs indefinitely due to a database lock contention issue, blocking all subsequent jobs in the pipeline. The operations team discovers the stuck job the next morning when reports are missing. The team implements a watchdog that monitors each batch job's progress by checking a progress table that the jobs update periodically. If a job has not updated its progress marker within 15 minutes, the watchdog terminates the job, releases its database locks, and marks it for retry. An alert is sent to the operations team, but the rest of the batch pipeline continues. The nightly batch suite now completes reliably every night, and stuck-job incidents that previously required 2-3 hours of manual intervention are resolved automatically.
