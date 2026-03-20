---
title: Static Code Analysis
description: Automatically check source code for programming errors and security vulnerabilities
category:
- Security
- Code
- Testing
quality_tactics_url: https://qualitytactics.de/en/security/static-code-analysis
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- lower-code-quality
- inconsistent-coding-standards
- high-bug-introduction-rate
- legacy-code-without-tests
- inadequate-code-reviews
- inefficient-code
- gradual-performance-degradation
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Select static analysis tools that support the languages and frameworks used in the legacy codebase (e.g., SonarQube, ESLint, PMD, FindBugs)
- Configure tool rules to focus on high-severity security findings before expanding to style and quality rules
- Integrate static analysis into the CI/CD pipeline as a required check for pull requests
- Establish a baseline of existing findings and create a plan to reduce them incrementally rather than fixing all at once
- Tune rules to minimize false positives, which erode developer trust in the tooling
- Use incremental analysis to check only changed files, reducing scan time for large legacy codebases
- Train developers to interpret and act on static analysis findings effectively
- Track finding trends over time to measure the impact of the static analysis program

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Catches common vulnerability patterns and performance anti-patterns automatically without manual review effort
- Provides consistent, objective code quality feedback regardless of reviewer expertise
- Scales to large legacy codebases where manual security review is impractical
- Creates a continuous feedback loop that educates developers about secure coding patterns

**Costs and Risks:**
- Legacy codebases often produce overwhelming numbers of initial findings that require triage
- False positives can lead to alert fatigue and developers ignoring genuine findings
- Static analysis cannot detect runtime vulnerabilities, business logic flaws, or data-dependent performance issues
- Tool configuration and maintenance requires ongoing effort and expertise
- Some legacy languages or frameworks may have limited static analysis tool support

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A financial services company deployed SonarQube with security-focused rules on their 500,000-line legacy Java codebase. The initial scan produced over 3,000 findings, which the team triaged into 180 genuine security issues, 800 quality improvements, and the rest as false positives or low-priority items. They configured the tool to enforce a "zero new findings" policy on all new code while creating a quarterly sprint to reduce the legacy backlog. After one year, the legacy finding count had dropped by 65%, and no new critical security findings were introduced in code that had passed the static analysis gate.
