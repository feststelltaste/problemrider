---
title: Secure Programming Interfaces
description: Using Libraries and Frameworks with Security Features
category:
- Security
- Code
quality_tactics_url: https://qualitytactics.de/en/security/secure-programming-interfaces
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- obsolete-technologies
- technology-lock-in
- inadequate-error-handling
- dependency-version-conflicts
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Evaluate current libraries and frameworks for built-in security features such as input sanitization, CSRF protection, and secure session handling
- Replace custom security implementations with well-tested library functions where available
- Upgrade legacy frameworks to versions that include modern security features by default
- Configure framework security features to be enabled by default rather than requiring opt-in
- Establish a list of approved libraries and frameworks that meet security requirements
- Remove or replace libraries that have known unpatched vulnerabilities or have reached end of life

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Leverages community-tested security implementations rather than custom code
- Reduces the likelihood of common vulnerabilities through built-in protections
- Keeps security capabilities current through library and framework updates
- Decreases the amount of security-specific code the team must maintain

**Costs and Risks:**
- Upgrading frameworks in legacy systems can introduce breaking changes
- Relying on framework defaults requires understanding what those defaults actually do
- Library vulnerabilities can affect all applications that depend on them
- Migration from custom security code to framework features requires careful testing to ensure equivalent protection

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A media company's legacy Python web application used custom-built HTML escaping functions that had been written in 2010. A security review found that these functions missed several edge cases that modern templating engines handle automatically. The team migrated from raw string rendering to Jinja2 with auto-escaping enabled, eliminating an entire class of XSS vulnerabilities. The migration also removed approximately 800 lines of custom security code that no longer needed maintenance.
