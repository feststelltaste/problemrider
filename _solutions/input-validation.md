---
title: Input Validation
description: Validate all inputs from users and external systems
category:
- Security
- Code
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- inadequate-error-handling
- log-injection-vulnerabilities
- integer-overflow-underflow
- silent-data-corruption
- rest-api-design-issues
- null-pointer-dereferences
- entity-attribute-value-overuse
layout: solution
related_solutions:
- slug: canonicalization
  similarity: 0.8
- slug: authentication
  similarity: 0.75
- slug: negative-testing
  similarity: 0.75
- slug: output-encoding
  similarity: 0.75
- slug: value-range-definition
  similarity: 0.75
- slug: data-flow-control
  similarity: 0.7
---

## Description

Input validation checks that data entering a system — through web forms, API calls, file uploads, or messages from other systems — conforms to an expected type, length, range, and format before that data is acted upon, ideally using an allowlist approach that defines what is accepted rather than a denylist that tries to enumerate what is rejected, since denylists are structurally incomplete against new encoding tricks. Legacy systems are disproportionately exposed here because many entry points were built at a time when trusting input from users and other systems was the default assumption rather than an explicit design decision, leaving string concatenation into SQL queries, unchecked file uploads, and unvalidated numeric fields scattered across dozens or hundreds of endpoints that accumulated over the system's lifetime. Retrofitting validation onto such a system is necessarily incremental and entry-point by entry-point, and it must be enforced server-side regardless of any client-side checks that exist, since client-side validation is a convenience that any attacker can simply bypass. Input validation is also explicitly a complement to, not a substitute for, structural defenses such as parameterized queries — the two together provide defense in depth, where parameterized queries eliminate SQL injection at the architectural level and validation catches malformed or malicious input at the boundary before it reaches any downstream logic at all. The ongoing cost is that validation rules must evolve alongside business requirements, since overly strict rules reject legitimate edge cases like valid international characters, while stale rules fail to catch newly discovered attack patterns.

> Legacy systems frequently trust input from users and external systems without validation, creating vulnerabilities ranging from injection attacks to data corruption. Comprehensive input validation ensures that all data entering the system conforms to expected formats, types, and ranges.

- Identify all input entry points: web forms, API endpoints, file uploads, command-line arguments, environment variables, database inputs from other systems, and message queue payloads. Each entry point is a potential attack vector.
- Implement allowlist validation (define what is accepted) rather than denylist validation (define what is rejected). Denylists are inherently incomplete and can be bypassed with new encoding tricks, while allowlists explicitly define the acceptable input space.
- Validate input type, length, range, and format at every entry point. Numeric fields should reject non-numeric input, string fields should enforce maximum lengths, date fields should verify valid date formats, and enumerated fields should accept only valid values.
- Apply validation on the server side, even if client-side validation exists. Client-side validation is a user experience convenience that can be trivially bypassed — server-side validation is the security control.
- Use parameterized queries or prepared statements for all database operations to prevent SQL injection. This is the most effective defense regardless of input validation, as it structurally separates code from data.
- Validate file uploads by checking file type (magic bytes, not just extension), enforcing size limits, and scanning for malicious content. Store uploaded files outside the web root with randomized names.
- Implement structured logging that prevents log injection by encoding special characters in log entries. Attackers who can inject newlines and control characters into logs can forge log entries and obscure their activities.

## Tradeoffs ⇄

> Input validation prevents a wide range of injection and data corruption attacks at the system boundary, but it requires comprehensive coverage and ongoing maintenance as input requirements evolve.

**Benefits:**

- Prevents injection attacks (SQL, XSS, command injection, LDAP injection) by ensuring that input cannot contain executable code or control characters.
- Catches malformed data at the system boundary before it causes errors, corruption, or unexpected behavior in downstream processing.
- Improves data quality by enforcing format and range constraints that legacy systems often lack.
- Reduces the attack surface by rejecting input that does not conform to known-good patterns before it reaches application logic.

**Costs and Risks:**

- Comprehensive input validation across all entry points of a legacy system requires significant development effort, especially when entry points are numerous and scattered.
- Overly strict validation can reject legitimate input, particularly for international characters, unusual but valid formats, and edge cases not anticipated during implementation.
- Validation rules must be maintained as business requirements change — outdated rules may block new valid inputs or fail to catch new invalid ones.
- Input validation alone does not prevent all injection attacks — it must be combined with output encoding, parameterized queries, and other defense-in-depth measures.

## How It Could Be

> The following scenarios illustrate how input validation prevents attacks and data corruption in legacy systems.

A legacy web application constructs SQL queries by concatenating user input directly into query strings. An attacker enters `' OR 1=1 --` in the username field and gains access to all user accounts. The immediate fix replaces string concatenation with parameterized queries throughout the data access layer. Additionally, the team implements input validation that restricts usernames to alphanumeric characters and underscores with a maximum length of 50 characters. The combination of parameterized queries (which prevent SQL injection structurally) and input validation (which rejects obviously malicious input at the boundary) provides defense in depth. The team extends this pattern to all 87 form fields in the legacy application, defining validation rules for each based on the expected data type and format.

A legacy order processing system accepts XML files from suppliers via FTP. A malformed XML file containing an extremely large element (5GB of repeated characters) causes the XML parser to allocate all available memory, crashing the order processing service. The team implements input validation at the file upload boundary: files are limited to 100MB, XML structure is validated against a schema before full parsing, element and attribute values are limited to defined maximum lengths, and entity expansion is disabled to prevent XML bomb attacks. These boundary validations are implemented in a preprocessing step that runs before the legacy XML parser, protecting it from inputs that would trigger crashes or resource exhaustion.
