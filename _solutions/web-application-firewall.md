---
title: Web Application Firewall
description: Filtering HTTP traffic at application layer against web attacks
category:
- Security
- Operations
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- authentication-bypass-vulnerabilities
- rate-limiting-issues
- system-outages
- legacy-code-without-tests
layout: solution
related_solutions:
- slug: security-monitoring
  similarity: 0.75
- slug: vulnerability-scans
  similarity: 0.75
- slug: honeypots
  similarity: 0.75
- slug: secure-protocols
  similarity: 0.75
- slug: secure-programming-interfaces
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.7
---

## Description

A Web Application Firewall is a filtering layer placed in front of a web application that inspects incoming HTTP traffic and blocks requests matching known attack patterns, such as SQL injection payloads, cross-site scripting attempts, or malformed authentication traffic, before those requests ever reach the application code. It operates as a reverse proxy or inline appliance, evaluating each request against a rule set — typically derived from the OWASP Top 10 and refined with application-specific patterns — and either passing, blocking, or flagging it for review. For legacy systems, this mechanism matters because the underlying application code is often riddled with vulnerabilities that are expensive and risky to fix directly: raw SQL concatenation scattered across thousands of call sites, unescaped output in templates written before secure coding practices were standard, or authentication logic too brittle to touch without extensive regression testing. A WAF provides a compensating control that shrinks the exploitable attack surface immediately, without requiring any change to the legacy codebase itself, buying the time needed to remediate the actual vulnerabilities safely. It is deployed at the network edge rather than within the application, which makes it one of the few security measures that can be applied to legacy systems with no access to source code or no appetite for redeployment risk. Because it depends on pattern matching rather than fixing the underlying flaw, it is best understood as a mitigating shield rather than a cure, and its effectiveness has to be continuously tuned against both false positives and evolving attacker techniques.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Deploy a WAF in front of legacy web applications as a protective layer that does not require application code changes
- Start in monitoring mode to understand traffic patterns and baseline legitimate requests before enabling blocking
- Configure rules targeting the OWASP Top 10 vulnerability categories most relevant to the legacy application
- Create custom rules for application-specific attack patterns discovered through penetration testing or incident analysis
- Implement rate limiting and bot detection to protect legacy applications from abuse and denial-of-service attacks
- Integrate WAF logs with the security monitoring system for correlation with other security events
- Regularly review and tune WAF rules to balance protection with false positive rates

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides immediate protection for legacy applications without requiring code changes
- Acts as a compensating control for vulnerabilities that cannot be quickly fixed in legacy code
- Offers visibility into attack attempts and patterns targeting the application
- Can be deployed quickly relative to the time needed to fix underlying application vulnerabilities

**Costs and Risks:**
- WAFs can be bypassed by sophisticated attackers who craft payloads to evade detection rules
- False positives can block legitimate traffic and create user-facing issues
- WAFs add latency to every request, which may affect performance-sensitive legacy applications
- Over-reliance on WAFs as a substitute for fixing underlying vulnerabilities creates a false sense of security
- WAF rules require ongoing tuning and maintenance as attack techniques evolve

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A travel booking company discovered multiple SQL injection vulnerabilities in their legacy booking engine during a penetration test. Fixing the vulnerabilities in the 12-year-old codebase was estimated at three months of work due to the deeply embedded raw SQL patterns. The team deployed a cloud-based WAF within one week, configured with SQL injection detection rules, and immediately began blocking exploitation attempts. WAF logs showed over 500 blocked SQL injection attempts in the first month alone. The WAF served as a protective layer while the development team methodically replaced raw SQL queries with parameterized statements over the following quarter.
