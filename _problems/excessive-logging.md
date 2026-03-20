---
title: Excessive Logging
description: Applications generate a very high volume of logs, consuming excessive
  disk space and potentially impacting performance.
category:
- Code
- Performance
related_problems:
- slug: log-spam
  similarity: 0.65
- slug: lazy-loading
  similarity: 0.65
- slug: excessive-disk-io
  similarity: 0.6
- slug: logging-configuration-issues
  similarity: 0.6
- slug: imperative-data-fetching-logic
  similarity: 0.6
- slug: excessive-object-allocation
  similarity: 0.6
solutions:
- observability-and-monitoring
layout: problem
---

## Description
Excessive logging can have a significant impact on application performance and maintainability. When an application logs too much information, it can consume a large amount of disk space, slow down the application, and make it difficult to find important information in the logs. A well-designed logging strategy should be focused on logging only the information that is necessary for debugging and monitoring. This requires a deep understanding of the application and its use cases.

## Indicators ⟡
- Log files are growing at an unexpectedly high rate.
- You are paying a lot of money for log storage and analysis.
- It is difficult to find important information in your logs because of the noise.
- Your application is slow, and you suspect that logging is a contributing factor.

## Symptoms ▲

- [Excessive Disk I/O](excessive-disk-io.md)
<br/>  High-volume logging generates constant disk write operations, contributing significantly to overall disk I/O.
- [Slow Application Performance](slow-application-performance.md)
<br/>  Writing large volumes of logs, especially synchronously, consumes CPU and I/O resources that slow down the main application.
- [Resource Contention](resource-contention.md)
<br/>  Log writing competes with application processing for disk I/O bandwidth and CPU cycles.
- [Debugging Difficulties](debugging-difficulties.md)
<br/>  When logs contain too much noise, finding the relevant information for debugging becomes like searching for a needle in a haystack.
- [Gradual Performance Degradation](gradual-performance-degradation.md)
<br/>  As log volumes accumulate over time, disk space fills up and I/O overhead grows, progressively degrading system performance.
## Causes ▼

- [Logging Configuration Issues](logging-configuration-issues.md)
<br/>  Misconfigured log levels, such as leaving DEBUG enabled in production, directly cause excessive log output.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Less experienced developers tend to add excessive logging statements as a debugging aid without considering production impact.
- [Inadequate Error Handling](inadequate-error-handling.md)
<br/>  Poor error handling that logs full stack traces for every exception contributes to log volume explosion.
- [Tangled Cross-Cutting Concerns](tangled-cross-cutting-concerns.md)
<br/>  When logging is interleaved throughout business logic rather than managed as a cross-cutting concern, logging statements proliferate uncontrollably.
## Detection Methods ○

- **Disk Usage Monitoring:** Monitor disk space consumption on servers where logs are stored.
- **I/O Monitoring:** Use system monitoring tools to track disk write operations related to logging.
- **Log Volume Analysis:** Use log aggregation tools to analyze the volume of logs generated per application or service.
- **Code Review:** Look for logging statements that are overly verbose or placed in performance-critical sections.
- **Configuration Review:** Check logging configurations to ensure appropriate logging levels are set for different environments.

## Examples
A microservice processes millions of events per day. A developer, while debugging an issue, sets the logging level to `DEBUG` and forgets to revert it before deploying to production. Within hours, the server's disk space is completely consumed by log files, causing the service to crash. In another case, an application logs the entire JSON payload of every incoming API request at an `INFO` level. This leads to massive log files and significant network traffic when these logs are shipped to a centralized logging system, even though only a small part of the payload is relevant for most debugging. While logging is crucial for observability, excessive logging can become a performance and cost burden, requiring a balance between providing enough information and avoiding unnecessary overhead.
