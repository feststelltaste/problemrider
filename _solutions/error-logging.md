---
title: Error Logging
description: Capturing and storing errors and exceptions
category:
- Operations
- Code
quality_tactics_url: https://qualitytactics.de/en/reliability/error-logging
problems:
- monitoring-gaps
- debugging-difficulties
- slow-incident-resolution
- inadequate-error-handling
- excessive-logging
- logging-configuration-issues
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Standardize error log format to include timestamp, severity, error type, message, stack trace, and correlation ID
- Implement structured logging (JSON format) to enable automated parsing and analysis
- Configure appropriate log levels so errors stand out from informational noise
- Centralize log collection using tools like ELK stack, Splunk, or Datadog for cross-service visibility
- Add contextual data to error logs: user ID, request ID, affected entity, and relevant parameters
- Implement log rotation and retention policies to manage storage while preserving necessary history
- Set up alerts on error log patterns to detect issues proactively rather than reactively

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides the forensic data needed to diagnose and fix production issues
- Enables pattern detection across error logs to identify systemic problems
- Supports compliance and audit requirements through comprehensive error records
- Reduces mean time to resolution by giving responders the context they need

**Costs and Risks:**
- Excessive logging can degrade application performance and consume significant storage
- Sensitive data in error logs creates security and privacy risks if not handled carefully
- Poorly structured logs are difficult to search and analyze at scale
- Log infrastructure requires its own monitoring and maintenance
- Teams may rely on logs as a substitute for proper monitoring and alerting

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy payment system logged errors inconsistently: some modules used log4j, others wrote to flat files, and some printed to stderr. When a transaction processing bug caused silent failures, the team spent three days correlating information from six different log sources to diagnose the issue. They standardized on SLF4J with a JSON appender, centralized all logs in Elasticsearch, and established a logging guideline that required correlation IDs and transaction context in every error log entry. The next time a similar issue occurred, the team identified the root cause in 20 minutes by searching for the affected transaction ID across all services in Kibana.
