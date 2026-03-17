---
title: Log Injection Vulnerabilities
description: Unsanitized user input in log messages allows attackers to inject malicious
  content that can compromise log integrity or exploit log processing systems.
category:
- Code
- Security
related_problems:
- slug: sql-injection-vulnerabilities
  similarity: 0.65
- slug: cross-site-scripting-vulnerabilities
  similarity: 0.6
- slug: insufficient-audit-logging
  similarity: 0.6
- slug: logging-configuration-issues
  similarity: 0.6
- slug: log-spam
  similarity: 0.55
- slug: authentication-bypass-vulnerabilities
  similarity: 0.5
layout: problem
---

## Description

Log injection vulnerabilities occur when applications include unsanitized user input in log messages, allowing attackers to inject malicious content that can corrupt log files, exploit log processing systems, or create false log entries. This can lead to log tampering, denial of service attacks on logging systems, or exploitation of log analysis tools.

## Indicators ⟡

- User input directly included in log messages without sanitization
- Log entries containing unexpected formatting characters or escape sequences
- Log processing systems experiencing errors when parsing certain log entries
- Log files containing suspicious or malformed entries
- Users able to influence log content through input fields

## Symptoms ▲

- [Insufficient Audit Logging](insufficient-audit-logging.md)
<br/>  Injected fake log entries corrupt audit trails, making legitimate audit logging unreliable and untrustworthy.
- [Data Protection Risk](data-protection-risk.md)
<br/>  Log injection can be used to hide attack traces or inject content that compromises systems processing those logs, creating data protection risks.
- [System Outages](system-outages.md)
<br/>  Injected format strings or malicious content can crash log processing systems, causing service disruptions.
- [Debugging Difficulties](debugging-difficulties.md)
<br/>  Corrupted or tampered log files make it extremely difficult to diagnose real issues when fake entries obscure genuine log data.
## Causes ▼

- [Inadequate Error Handling](inadequate-error-handling.md)
<br/>  Poor error handling practices that dump unsanitized user input into log messages create injection opportunities.
- [Logging Configuration Issues](logging-configuration-issues.md)
<br/>  Improperly configured logging that doesn't enforce input sanitization or structured logging formats enables injection attacks.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers unfamiliar with security best practices may not realize that user input in log messages needs sanitization.
## Detection Methods ○

- **Log Content Analysis:** Regularly analyze log files for suspicious or malformed entries
- **Input Validation Testing:** Test logging behavior with various malicious input patterns
- **Log Processing Security Testing:** Test log analysis tools with potentially malicious log entries
- **Log Integrity Monitoring:** Monitor logs for signs of tampering or corruption
- **User Input Sanitization Review:** Review how user input is handled in logging code

## Examples

A web application logs failed login attempts including the username provided: `Log.info("Failed login attempt for user: " + username)`. An attacker enters a username containing newline characters and fake log entries: `"admin\n[INFO] Successful login for user: admin\n[INFO] Admin privileges granted"`. This creates false log entries that make it appear as if the admin user successfully logged in and gained privileges, potentially hiding the actual attack. Another example involves an e-commerce application that logs user search queries. An attacker injects log format strings into the search field, causing the logging system to crash when it tries to process format specifiers like `%s%s%s%n` that expect additional parameters, effectively creating a denial of service against the logging infrastructure.