---
title: Secure Protocols
description: Use only secure and current versions of network protocols
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/secure-protocols
problems:
- insecure-data-transmission
- obsolete-technologies
- regulatory-compliance-drift
- data-protection-risk
- poor-system-environment
- technology-lock-in
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Inventory all network protocols and their versions used by the legacy system, including TLS, SSH, SMTP, and database protocols
- Disable deprecated protocols such as SSLv3, TLS 1.0, and TLS 1.1 across all system components
- Configure servers and clients to use only current, secure protocol versions with strong cipher suites
- Update legacy integrations that rely on outdated protocols, providing migration paths for third-party partners
- Implement automated scanning to detect any insecure protocol usage across the network
- Plan and execute certificate rotation procedures for all TLS endpoints

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Protects data in transit from eavesdropping and tampering
- Satisfies compliance requirements that mandate current protocol versions
- Reduces the attack surface by eliminating known protocol vulnerabilities
- Improves overall security posture with minimal application code changes

**Costs and Risks:**
- Legacy clients or integrations may not support modern protocol versions, requiring coordinated upgrades
- Protocol upgrades can cause service disruptions if not thoroughly tested
- Some legacy hardware or embedded devices may be incapable of supporting current protocols
- Cipher suite configuration requires expertise to avoid both insecure and incompatible choices

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company discovered during a compliance audit that their legacy shipping integration still used TLS 1.0 for communicating with carrier APIs. Several carriers had already begun rejecting TLS 1.0 connections, causing intermittent shipping label generation failures. The team upgraded all outbound connections to TLS 1.2, updated the internal certificate authority, and implemented a protocol version monitoring dashboard. The upgrade resolved both the compliance finding and the intermittent failures, and the monitoring system caught two additional legacy services still using outdated protocols within the first week.
