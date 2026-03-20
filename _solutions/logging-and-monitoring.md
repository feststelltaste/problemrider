---
title: Logging and Monitoring
description: Log and monitor security-related events
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/logging-and-monitoring
problems:
- monitoring-gaps
- insufficient-audit-logging
- slow-incident-resolution
- debugging-difficulties
- logging-configuration-issues
- log-spam
- excessive-logging
- log-injection-vulnerabilities
layout: solution
---

## How to Apply ◆

> Legacy systems often log too little for security purposes (missing authentication events, access decisions) while logging too much noise (verbose debug output, redundant health checks). Security-focused logging and monitoring captures the right events and makes them actionable.

- Define a security logging policy that specifies which events must be logged: authentication attempts (success and failure), authorization decisions, data access and modifications, administrative actions, configuration changes, and security-relevant errors.
- Implement structured logging with consistent fields across all legacy system components: timestamp (UTC), event type, severity, user identity, source IP, resource accessed, action performed, and outcome (success/failure).
- Forward all security logs to a centralized Security Information and Event Management (SIEM) system in near-real-time. Centralized aggregation enables correlation across components and prevents log tampering on compromised systems.
- Create detection rules and alerts for security-relevant patterns: multiple failed login attempts, access to sensitive resources outside business hours, privilege escalation, unusual data export volumes, and configuration changes by unexpected users.
- Implement log protection to prevent tampering: store logs in append-only storage, hash log entries for integrity verification, and restrict write access to the logging service account only.
- Address log noise by filtering or separating operational logs (health checks, routine status updates) from security logs (authentication, authorization, data access). Security logs must be reliably searchable without being buried in operational noise.
- Ensure sensitive data (passwords, tokens, credit card numbers, personal data) is never written to logs. Implement log sanitization filters that mask or redact sensitive fields before they reach the logging pipeline.

## Tradeoffs ⇄

> Security logging and monitoring provide visibility into threats and enable rapid incident response, but they require investment in infrastructure, tuning, and skilled analysts.

**Benefits:**

- Enables detection of security incidents that preventive controls fail to block, reducing the time between breach and discovery.
- Provides forensic evidence for incident investigation, supporting root cause analysis and legal proceedings.
- Meets compliance requirements for security event logging and monitoring (PCI DSS, HIPAA, SOX, GDPR).
- Creates accountability by recording who did what and when, deterring unauthorized actions.

**Costs and Risks:**

- Security logging generates significant data volumes requiring storage, retention management, and processing infrastructure.
- Without proper tuning, monitoring systems produce alert fatigue from false positives, causing analysts to miss genuine threats.
- Retrofitting security logging into legacy systems requires code changes at many points, with risk of missing critical event types.
- Logs containing sensitive data (if sanitization is incomplete) create a secondary data protection risk.

## How It Could Be

> The following scenarios illustrate how security logging and monitoring detect threats in legacy systems.

A legacy banking application logs only successful transactions but does not log failed login attempts or authorization denials. An attacker conducts a credential stuffing attack against the login page, testing 50,000 username/password combinations over two days. Because failed logins are not logged, the attack is invisible to the operations team. After implementing comprehensive security logging, the team configures alerts for patterns such as more than 10 failed login attempts from a single IP within 5 minutes and more than 5 failed attempts for a single username within an hour. The next credential stuffing attempt triggers an alert within 3 minutes, and the attacking IP addresses are automatically blocked at the WAF. The security team identifies 12 accounts where the attacker guessed correctly and forces password resets before any unauthorized transactions occur.

A legacy document management system has verbose application logging that writes 50GB of logs per day, making it impossible to search for security-relevant events. Failed access attempts, administrative actions, and data exports are mixed with thousands of debug messages about image rendering and document formatting. The team implements a structured logging strategy that separates security events into a dedicated log stream forwarded to the SIEM, while operational logs continue to the existing log store. Security events are tagged with standardized event types, enabling the SIEM to apply detection rules. Within the first week, the SIEM detects that a terminated employee's account is still active and accessing confidential documents — an event that was occurring for months but was invisible in the 50GB daily log volume.
