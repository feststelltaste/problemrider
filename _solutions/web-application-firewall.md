---
title: Web Application Firewall
description: Filtering HTTP traffic at application layer against web attacks
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/web-application-firewall
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- authentication-bypass-vulnerabilities
- rate-limiting-issues
- system-outages
- legacy-code-without-tests
layout: solution
---

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
