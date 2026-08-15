---
title: Secure by Default
description: Align default settings and delivery state for maximum security
category:
- Security
- Operations
problems:
- configuration-chaos
- configuration-drift
- secret-management-problems
- password-security-weaknesses
- authentication-bypass-vulnerabilities
- error-message-information-disclosure
- inadequate-configuration-management
layout: solution
related_solutions:
- slug: secure-configuration
  similarity: 0.85
- slug: secure-protocols
  similarity: 0.8
- slug: secure-software-development
  similarity: 0.75
- slug: secure-programming-interfaces
  similarity: 0.75
- slug: secure-coding-guidelines
  similarity: 0.75
- slug: secure-software
  similarity: 0.75
---

## Description

Secure by default means shipping and configuring a system so that its out-of-the-box settings are already the most restrictive ones consistent with the system functioning, rather than requiring an administrator to actively harden it after installation — disabling unnecessary services and debug endpoints, using strong default credentials or none at all, and ensuring error messages never leak internal details such as stack traces or connection strings. The underlying principle is that security should not depend on every operator remembering and correctly performing a hardening step, since in practice, some fraction of deployments will always skip that step. Legacy systems are especially exposed here because many of them were built or configured at a time when insecure defaults — verbose debug output, default admin passwords, open diagnostic ports — were the industry norm rather than the exception, and those defaults have often persisted untouched for years simply because nobody revisited the original installation. Retrofitting secure-by-default settings into such a system means auditing what the current defaults actually are, which frequently surfaces forgotten configurations nobody would have knowingly approved, and then building a hardened configuration profile that becomes the new baseline for every environment going forward. Because legacy systems can have undocumented dependencies on the very insecure behaviors being removed, changing defaults must be rolled out carefully, but the resulting reduction in attack surface applies automatically to every future deployment without relying on continued administrator vigilance.

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
