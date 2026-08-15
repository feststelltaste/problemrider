---
title: Security Regression Tests
description: Retest previously fixed security vulnerabilities to prevent their recurrence
category:
- Security
- Testing
problems:
- regression-bugs
- insufficient-testing
- legacy-code-without-tests
- high-bug-introduction-rate
- fear-of-breaking-changes
- partial-bug-fixes
- test-debt
- poor-test-coverage
layout: solution
related_solutions:
- slug: security-tests
  similarity: 0.85
- slug: security-tests-by-external-parties
  similarity: 0.8
- slug: vulnerability-scans
  similarity: 0.8
- slug: security-audits
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
- slug: secure-software-development
  similarity: 0.8
---

## Description

Security regression tests are automated test cases that specifically reproduce a previously discovered and fixed vulnerability, so that the exact attack vector which once succeeded is retried against the system on every build to confirm the fix still holds. Unlike general functional regression tests, they are written directly from a vulnerability report or penetration test finding, encoding the specific payload, input, or sequence of requests that previously bypassed a security control. This matters in legacy systems because security fixes there are often applied as narrow, localized patches to code that is otherwise poorly understood, poorly tested, and subject to frequent, uncoordinated changes by different teams — conditions under which a fixed vulnerability is highly prone to reappearing, whether through a reversion, a parallel code path that received the same flaw independently, or a refactor that inadvertently undoes the original fix. By running these tests continuously in the CI/CD pipeline rather than relying on periodic manual penetration tests, teams get immediate feedback the moment a change reintroduces a known weakness, closing the gap between when a regression is introduced and when it is caught. Over time, the growing suite of security regression tests becomes an executable record of the system's security history, capturing institutional knowledge that would otherwise depend on individuals remembering incidents from years past.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Create a dedicated security regression test suite that includes a test case for every previously identified and fixed vulnerability
- Integrate security regression tests into the CI/CD pipeline so they run on every build
- Write tests that specifically reproduce the original attack vector to confirm the fix remains effective
- Maintain a vulnerability registry that maps each finding to its corresponding regression test
- Extend regression tests when new attack variants or bypass techniques are discovered
- Review and update security regression tests when the affected code undergoes refactoring

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Prevents reintroduction of previously fixed security vulnerabilities during refactoring or feature development
- Builds institutional memory of past security issues in executable form
- Provides confidence that legacy system changes do not silently degrade security posture
- Complements manual security testing by automating verification of known issues

**Costs and Risks:**
- Writing meaningful security regression tests requires understanding of both the vulnerability and the fix
- Test suite maintenance grows over time and can slow CI pipelines if not managed
- Tests may give false confidence if they do not accurately reproduce the original attack conditions
- Legacy systems with poor testability may require significant refactoring before tests can be added

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A banking application had experienced three separate recurrences of a cross-site scripting vulnerability over two years, each time in a slightly different input field. After the third occurrence, the team created a security regression test suite that automated browser-based injection attempts against all user-input fields. Each new vulnerability finding was immediately added as a regression test. Over the following year, the regression suite caught two additional instances where developers inadvertently introduced similar vulnerabilities during feature development, preventing them from reaching production.
