---
title: Input Validation
description: Validate all inputs from users and external systems
category:
- Security
- Code
quality_tactics_url: https://qualitytactics.de/en/security/input-validation
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- inadequate-error-handling
- log-injection-vulnerabilities
- integer-overflow-underflow
- silent-data-corruption
- rest-api-design-issues
layout: solution
---

## How to Apply ◆

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

## Examples

> The following scenarios illustrate how input validation prevents attacks and data corruption in legacy systems.

A legacy web application constructs SQL queries by concatenating user input directly into query strings. An attacker enters `' OR 1=1 --` in the username field and gains access to all user accounts. The immediate fix replaces string concatenation with parameterized queries throughout the data access layer. Additionally, the team implements input validation that restricts usernames to alphanumeric characters and underscores with a maximum length of 50 characters. The combination of parameterized queries (which prevent SQL injection structurally) and input validation (which rejects obviously malicious input at the boundary) provides defense in depth. The team extends this pattern to all 87 form fields in the legacy application, defining validation rules for each based on the expected data type and format.

A legacy order processing system accepts XML files from suppliers via FTP. A malformed XML file containing an extremely large element (5GB of repeated characters) causes the XML parser to allocate all available memory, crashing the order processing service. The team implements input validation at the file upload boundary: files are limited to 100MB, XML structure is validated against a schema before full parsing, element and attribute values are limited to defined maximum lengths, and entity expansion is disabled to prevent XML bomb attacks. These boundary validations are implemented in a preprocessing step that runs before the legacy XML parser, protecting it from inputs that would trigger crashes or resource exhaustion.
