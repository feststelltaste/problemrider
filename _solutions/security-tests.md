---
title: Security Tests
description: Verify security properties through specialized testing methods
category:
- Security
- Testing
problems:
- insufficient-testing
- poor-test-coverage
- legacy-code-without-tests
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- authentication-bypass-vulnerabilities
- high-defect-rate-in-production
- session-management-issues
layout: solution
related_solutions:
- slug: regression-tests
  similarity: 0.85
- slug: secure-software-development
  similarity: 0.85
- slug: static-code-analysis
  similarity: 0.85
- slug: secure-coding-guidelines
  similarity: 0.8
- slug: security-tests-by-external-parties
  similarity: 0.8
- slug: vulnerability-scans
  similarity: 0.8
---

## Description

Security tests verify specific security properties — authentication, authorization, input validation, cryptographic correctness — through specialized methods such as static analysis (SAST), dynamic scanning (DAST), and targeted unit tests, rather than treating security as something checked only occasionally through a separate audit. Integrating these tests into the CI/CD pipeline catches vulnerabilities at the point they are introduced, which matters enormously for legacy codebases being actively modified during modernization, since every refactor is an opportunity to reintroduce a vulnerability pattern the code has already been fixed for once. Automated security tests inevitably surface false positives that require expert triage to distinguish from genuine findings, and they verify known vulnerability patterns rather than guaranteeing the absence of novel attacks, but the repeatable safety net they provide is what makes it possible to touch legacy security-sensitive code with any confidence at all.

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
