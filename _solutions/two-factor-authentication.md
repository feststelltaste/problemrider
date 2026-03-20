---
title: Two-Factor Authentication
description: Verify identity using two independent factors
category:
- Security
quality_tactics_url: https://qualitytactics.de/en/security/two-factor-authentication
problems:
- authentication-bypass-vulnerabilities
- password-security-weaknesses
- data-protection-risk
- session-management-issues
- authorization-flaws
- regulatory-compliance-drift
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Evaluate 2FA methods suitable for the legacy system's user base: TOTP apps, hardware tokens, SMS codes, or push notifications
- Implement 2FA for the most privileged accounts first (administrators, database access, deployment credentials)
- Add 2FA support to the legacy authentication flow without disrupting the existing login experience
- Provide fallback recovery mechanisms such as backup codes for users who lose access to their second factor
- Integrate with existing identity providers or implement a standalone 2FA service that the legacy application calls
- Offer a migration period where 2FA is encouraged before it becomes mandatory
- Log all 2FA events for audit purposes and monitor for anomalous authentication patterns

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically reduces the risk of account compromise from stolen or weak passwords
- Provides a strong additional layer of defense for critical system access
- Satisfies regulatory and compliance requirements for strong authentication
- Deters automated credential stuffing and brute force attacks

**Costs and Risks:**
- Adds friction to the login process, which can frustrate users and reduce productivity
- Lost or malfunctioning second factors can lock users out, requiring support processes
- SMS-based 2FA is vulnerable to SIM swapping and interception attacks
- Retrofitting 2FA into legacy authentication systems may require significant code changes
- Service accounts and automated processes may not easily support 2FA workflows

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A law firm experienced a breach when an attacker used credentials obtained from a data leak on another site to access their legacy case management system. After the incident, the firm implemented TOTP-based two-factor authentication for all users. For the legacy application, which did not natively support 2FA, they deployed an authentication proxy that handled the second factor before forwarding authenticated sessions to the legacy system. This approach required no changes to the legacy codebase. Within three months, two additional credential stuffing attempts were blocked by the 2FA requirement, confirming its effectiveness.
