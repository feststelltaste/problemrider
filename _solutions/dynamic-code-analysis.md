---
title: Dynamic Code Analysis
description: Testing security properties by executing and observing program behavior
category:
- Security
- Testing
quality_tactics_url: https://qualitytactics.de/en/security/dynamic-code-analysis
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- authentication-bypass-vulnerabilities
- memory-leaks
- inadequate-error-handling
- legacy-code-without-tests
- insufficient-testing
layout: solution
---

## How to Apply ◆

> Legacy systems often cannot be fully analyzed through static code review alone due to complex runtime behavior, dynamic configuration, and third-party dependencies. Dynamic analysis tests security properties by executing the application and observing its behavior under various conditions.

- Deploy Dynamic Application Security Testing (DAST) tools that interact with the running legacy application as an external attacker would — sending malicious payloads, testing authentication bypasses, and probing for injection vulnerabilities through the application's interfaces.
- Implement Interactive Application Security Testing (IAST) by instrumenting the legacy application runtime to observe how it handles input internally. IAST detects vulnerabilities that DAST might miss by tracking data flow through the application code.
- Run memory analysis tools (Valgrind, AddressSanitizer, or platform-specific equivalents) against legacy applications written in memory-unsafe languages (C, C++) to detect buffer overflows, use-after-free errors, and memory leaks that create security vulnerabilities.
- Configure dynamic analysis to run against a staging environment that mirrors production, using realistic test data and configuration. Testing against a minimal development environment may miss vulnerabilities that only manifest under production-like conditions.
- Integrate dynamic security testing into the CI/CD pipeline so that new deployments are automatically scanned before reaching production. Start with a focused set of high-priority tests to keep pipeline execution time manageable.
- Complement automated dynamic analysis with manual exploratory security testing for complex business logic vulnerabilities that automated tools cannot detect, such as authorization bypasses in multi-step workflows.

## Tradeoffs ⇄

> Dynamic analysis detects runtime security vulnerabilities that static analysis cannot find, but it requires a running application environment and may not achieve complete code coverage.

**Benefits:**

- Discovers vulnerabilities in the running application's actual behavior, including issues caused by runtime configuration, third-party libraries, and environmental factors.
- Tests the application as an attacker would interact with it, finding vulnerabilities that are actually exploitable rather than theoretical.
- Detects memory safety issues in native code that are invisible to source code review.
- Requires no access to source code, making it applicable to legacy systems with lost or unavailable source.

**Costs and Risks:**

- Dynamic analysis can only test code paths that are actually executed during the test, potentially missing vulnerabilities in untested paths.
- Running security tests against production-like environments requires careful isolation to prevent test activities from affecting real data or services.
- DAST tools can generate significant load and may trigger security controls (WAF, IDS) that interfere with testing.
- Legacy systems may be fragile and could crash or behave unpredictably under the unusual inputs that dynamic analysis generates.

## How It Could Be

> The following scenarios illustrate how dynamic code analysis uncovers security vulnerabilities in legacy systems.

A legacy Java web application has been in production for 12 years with minimal security testing. The team deploys a DAST scanner against a staging instance and discovers 14 reflected XSS vulnerabilities, 3 SQL injection points, and an authentication bypass in the password reset flow. The SQL injection vulnerabilities are in legacy JSP pages that construct queries by string concatenation. The authentication bypass occurs because the password reset token is a predictable sequential number rather than a cryptographic random value. Static analysis had previously flagged the string concatenation as a code quality issue but could not confirm exploitability — the dynamic test proves that these are actively exploitable vulnerabilities and prioritizes remediation by severity.

A legacy C++ trading system processes market data through a high-performance parsing pipeline. The team runs the application under AddressSanitizer during their integration test suite and discovers a heap buffer overflow in the market data parser that occurs when processing a specific malformed message format. The overflow allows an attacker who can inject crafted market data messages to execute arbitrary code on the trading server. This vulnerability existed for 7 years but was never triggered by normal market data, making it invisible to functional testing and code review. The dynamic analysis catch leads to a targeted fix in the parser's bounds checking, and the team adds the malformed message as a permanent regression test case.
