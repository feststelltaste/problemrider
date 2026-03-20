---
title: Secure by Default
description: Align default settings and delivery state for maximum security
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/secure-by-default
problems:
- configuration-chaos
- configuration-drift
- secret-management-problems
- password-security-weaknesses
- authentication-bypass-vulnerabilities
- error-message-information-disclosure
- inadequate-configuration-management
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Audit all default configurations in the legacy system for insecure settings such as default passwords, open ports, and verbose error messages
- Change default settings to the most restrictive options that still allow the system to function
- Disable unnecessary features, services, and debug endpoints in production deployments
- Ensure error messages do not leak internal system details such as stack traces, version numbers, or file paths
- Ship configuration templates with security-hardened defaults and require explicit opt-in for relaxed settings
- Document the security rationale for each default setting so future maintainers understand why it was chosen

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces the attack surface without requiring ongoing user action
- Prevents common misconfigurations that lead to security incidents
- Lowers the barrier for secure deployment by making security the path of least resistance
- Catches oversights where administrators forget to harden non-default installations

**Costs and Risks:**
- Overly restrictive defaults may break existing integrations or workflows that depend on relaxed settings
- Users may work around secure defaults rather than understanding why they exist
- Legacy systems often have undocumented dependencies on insecure default behaviors
- Changing defaults in production systems requires careful rollout and testing

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A SaaS company discovered that their legacy application server shipped with debug mode enabled by default, exposing detailed stack traces and database connection strings in error responses. A security audit also found that the default admin account used a well-known password. The team created a hardened configuration profile that disabled debug mode, enforced strong initial passwords, and closed unnecessary network ports. After deploying the new defaults across all environments, the number of information disclosure findings in subsequent penetration tests dropped from 14 to zero.
