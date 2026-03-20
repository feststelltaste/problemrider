---
title: Security Tests
description: Verify security properties through specialized testing methods
category:
- Security
- Testing
quality_tactics_url: https://qualitytactics.de/en/security/security-tests
problems:
- insufficient-testing
- poor-test-coverage
- legacy-code-without-tests
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- authentication-bypass-vulnerabilities
- high-defect-rate-in-production
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Implement static application security testing (SAST) to scan source code for vulnerability patterns
- Deploy dynamic application security testing (DAST) to probe running applications for exploitable weaknesses
- Add interactive application security testing (IAST) for runtime analysis during functional test execution
- Create security-focused unit tests for authentication, authorization, input validation, and cryptographic functions
- Integrate security tests into the CI/CD pipeline to catch vulnerabilities before deployment
- Maintain a library of security test cases based on OWASP Top 10 and findings from past incidents
- Schedule periodic comprehensive security test runs beyond what the CI pipeline covers

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Catches security vulnerabilities before they reach production environments
- Provides repeatable, automated verification of security properties
- Builds developer awareness of security issues through immediate feedback
- Creates a safety net during legacy code refactoring and modernization

**Costs and Risks:**
- Security testing tools produce false positives that require expert triage
- Legacy codebases may be difficult to instrument for dynamic testing
- Security tests add to build pipeline execution time
- Tool licenses and maintenance represent ongoing costs
- Tests verify known vulnerability patterns but cannot guarantee absence of novel attacks

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A retail company integrated OWASP ZAP into their CI pipeline for their legacy e-commerce application. During the first full scan, the tool identified 23 potential vulnerabilities including reflected XSS in the search function, missing security headers, and information disclosure through verbose error messages. After triaging false positives, the team confirmed 15 genuine issues and fixed them over two sprints. The automated security tests then prevented three similar vulnerabilities from being reintroduced during subsequent development, each caught at the pull request stage before reaching the main branch.
