---
title: Negative Testing
description: Deliberately test invalid inputs and edge cases to check error handling
category:
- Security
- Testing
problems:
- inadequate-error-handling
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- insufficient-testing
- legacy-code-without-tests
- error-message-information-disclosure
- null-pointer-dereferences
- integer-overflow-underflow
- stack-overflow-errors
- race-conditions
layout: solution
related_solutions:
- slug: fuzz-testing
  similarity: 0.8
- slug: test-coverage-strategy
  similarity: 0.75
- slug: input-validation
  similarity: 0.75
- slug: functional-tests
  similarity: 0.75
- slug: penetration-tests
  similarity: 0.75
- slug: automated-tests
  similarity: 0.75
---

## Description

Negative testing is the practice of deliberately feeding a system invalid, malformed, boundary, or malicious input and asserting that it fails safely, rather than exercising only the "happy path" that valid inputs traverse. Where functional testing confirms that a feature does what it should when given reasonable input, negative testing probes what the system does when it is given input it was never designed to handle — empty values, oversized payloads, wrong types, out-of-range numbers, and known attack patterns such as injection strings. In legacy systems, this distinction matters because decades of incremental feature work tend to concentrate testing effort on demonstrating that new functionality works, while error paths accumulate silently and are rarely exercised until an attacker or an unusual production event finally reaches them. Because legacy error handling is frequently implemented ad hoc — string concatenation into queries, raw exception messages surfaced to users, missing bounds checks — these paths are disproportionately likely to contain the injection, disclosure, and crash vulnerabilities that negative testing is designed to surface. The technique is comparatively cheap to retrofit: it does not require rewriting business logic, only adding test cases that submit adversarial input to interfaces that already exist. Once discovered, findings are typically fixed locally (parameterization, validation, generic error messages) and then locked in as automated regression tests so the same defect cannot silently return as the legacy system continues to change.

## How to Apply ◆

> Legacy system tests typically verify that the system works correctly with valid inputs but never test what happens with invalid, unexpected, or malicious inputs. Negative testing deliberately provides bad inputs to verify that error handling is correct and secure.

- For each input field and API parameter, define and test boundary values: maximum length + 1, minimum value - 1, empty strings, null values, negative numbers where positive are expected, and values outside enumerated sets.
- Test with inputs that are known attack patterns: SQL injection payloads, XSS scripts, path traversal sequences, command injection strings, and format string specifiers. The system should handle these gracefully without executing the injected content.
- Verify error responses: the system should return appropriate error messages that help the user correct their input without revealing implementation details (stack traces, database errors, file paths, version numbers) that would aid an attacker.
- Test authentication and authorization negatively: attempt access without credentials, with expired credentials, with another user's credentials, and with modified tokens. Verify that each scenario is properly rejected.
- Test concurrent and out-of-order operations: submit requests in unexpected sequences, send duplicate requests, and test race conditions by submitting competing modifications simultaneously.
- Test resource limits: upload files that exceed size limits, submit requests at rates above rate limits, create objects that exceed quantity limits. Verify that limits are enforced and exceeded requests are handled gracefully.
- Implement negative test cases as automated tests that run in the CI/CD pipeline, ensuring that error handling remains correct as the legacy system evolves.

## Tradeoffs ⇄

> Negative testing verifies that error handling is correct and secure, preventing attackers from exploiting error conditions, but it requires creative test design and comprehensive coverage.

**Benefits:**

- Discovers error handling failures that expose the system to injection attacks, information disclosure, and denial of service.
- Verifies that the system fails safely rather than failing open when presented with unexpected input.
- Catches security-relevant error paths that functional testing never exercises, closing gaps in test coverage.
- Provides assurance that input validation and error handling changes do not introduce regressions.

**Costs and Risks:**

- Designing comprehensive negative test cases requires creativity and security knowledge to anticipate the range of invalid inputs an attacker might use.
- Negative tests can be fragile if they depend on specific error message text or error code values that change across versions.
- Running aggressive negative tests against legacy systems can cause crashes or data corruption in test environments, requiring careful isolation.
- The space of possible invalid inputs is infinite; negative testing reduces risk but cannot eliminate all error handling vulnerabilities.

## How It Could Be

> The following scenarios illustrate how negative testing uncovers security weaknesses in legacy systems.

A legacy web application's registration form validates that email addresses contain an "@" character but performs no other validation. Negative testing reveals that submitting an email address of `admin'--@example.com` causes a SQL error because the value is concatenated into a query without parameterization, and the error message displays the full SQL query including table and column names. Further negative tests discover that submitting a 10,000-character email address causes a buffer overflow in the email validation function, and submitting an email containing `<script>alert(1)</script>` results in the script executing when the administrator views the user list. Each finding leads to a specific fix: parameterized queries, input length validation, and output encoding. The negative test cases are automated and run on every build to prevent regression.

A legacy REST API accepts JSON payloads for order creation. Functional tests verify that well-formed orders are processed correctly, but negative testing reveals multiple issues: sending a JSON payload with a negative quantity results in the system creating a credit instead of a charge, sending a price field with 15 decimal places causes a floating-point precision error that can be exploited to underpay by fractions of a cent at scale, and sending a product ID that does not exist returns a 500 error with a stack trace containing database connection string details. The team fixes each issue (quantity validation, decimal precision handling, generic error responses) and adds 35 negative test cases to the automated test suite covering invalid types, out-of-range values, missing required fields, and known attack patterns for each API parameter.
