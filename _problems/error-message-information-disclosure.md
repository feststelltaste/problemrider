---
title: Error Message Information Disclosure
description: Error messages reveal sensitive system information that can be exploited
  by attackers to understand system architecture and vulnerabilities.
category:
- Code
- Communication
- Security
related_problems:
- slug: buffer-overflow-vulnerabilities
  similarity: 0.6
- slug: logging-configuration-issues
  similarity: 0.6
- slug: log-injection-vulnerabilities
  similarity: 0.55
- slug: inadequate-error-handling
  similarity: 0.55
- slug: silent-data-corruption
  similarity: 0.55
layout: problem
---

## Description

Error message information disclosure occurs when applications reveal sensitive technical information through error messages, stack traces, or debug output that can help attackers understand system architecture, database schemas, file paths, or internal application logic. This information can be used to craft more targeted attacks or identify specific vulnerabilities.

## Indicators ⟡

- Database error messages revealing table names, column names, or query structure
- Stack traces exposing internal file paths, class names, or system architecture
- Error messages containing system configuration details or version information
- Debug information displayed to end users in production environments
- Error responses revealing existence or non-existence of resources

## Symptoms ▲

- [SQL Injection Vulnerabilities](sql-injection-vulnerabilities.md)
<br/>  Detailed database error messages reveal schema information that attackers use to craft targeted SQL injection attacks.
- [Authentication Bypass Vulnerabilities](authentication-bypass-vulnerabilities.md)
<br/>  Error messages revealing authentication logic details help attackers identify weaknesses in the authentication mechanism.
- [Data Protection Risk](data-protection-risk.md)
<br/>  Disclosure of system internals through error messages creates legal and compliance risks around data protection.
- [Cross-Site Scripting Vulnerabilities](cross-site-scripting-vulnerabilities.md)
<br/>  Stack traces and debug output revealing application structure help attackers identify injection points for XSS attacks.
## Causes ▼

- [Inadequate Error Handling](inadequate-error-handling.md)
<br/>  Poor error handling passes raw exceptions and stack traces to users instead of displaying sanitized error messages.
- [Logging Configuration Issues](logging-configuration-issues.md)
<br/>  Misconfigured logging levels in production environments cause debug-level information to be displayed to end users.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers without security awareness may not realize that detailed error messages in production pose a security risk.
- [Insufficient Testing](insufficient-testing.md)
<br/>  Lack of security testing for error conditions means information disclosure through error messages goes undetected before production.
## Detection Methods ○

- **Error Message Security Review:** Review all error messages for sensitive information disclosure
- **Production Error Testing:** Test error conditions in production-like environments
- **Error Response Analysis:** Analyze error responses for information that could aid attackers
- **Security Testing for Information Disclosure:** Test various error conditions for information leakage
- **Error Handling Code Audit:** Review error handling code for appropriate information filtering

## Examples

A web application's login form displays detailed database error messages when SQL queries fail, revealing the complete database schema including table names like "users", "admin_accounts", and "payment_info" along with column names like "password_hash" and "credit_card_number". Attackers can use this information to craft SQL injection attacks targeting specific tables and columns. Another example involves a file upload service that displays full Java stack traces when file processing fails, revealing internal application architecture, library versions, and file system paths like "/opt/app/uploads/processing/temp/". This information helps attackers understand the system structure and identify potential attack vectors like directory traversal or dependency-specific vulnerabilities.