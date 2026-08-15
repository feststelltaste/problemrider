---
title: Secure Protocols
description: Use only secure and current versions of network protocols
category:
- Security
- Operations
problems:
- insecure-data-transmission
- obsolete-technologies
- regulatory-compliance-drift
- data-protection-risk
- poor-system-environment
- technology-lock-in
layout: solution
related_solutions:
- slug: cryptographic-methods
  similarity: 0.85
- slug: secure-by-default
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
- slug: encryption
  similarity: 0.75
- slug: vulnerability-scans
  similarity: 0.75
- slug: security-hardening-process
  similarity: 0.75
---

## Description

Secure protocols means restricting all network communication — client-server traffic, service-to-service calls, and administrative access — to protocol versions and cipher suites that are currently considered cryptographically sound, and retiring everything else. In practice this covers transport-layer protocols such as TLS, remote access protocols such as SSH, and application-layer protocols like SMTP or database wire protocols, all of which accumulate deprecated versions over a system's lifetime as new vulnerabilities are found and old ones are patched only by version replacement rather than in place. Legacy systems are particularly prone to running outdated protocol versions because upgrading them was never anyone's assigned responsibility, external integration partners froze on whatever version existed when the connection was first built, and the operational risk of touching a working network configuration discouraged proactive upgrades. The mechanism is comparatively simple — protocol version negotiation is a configuration setting on servers and clients rather than an application code change — but its effect is disproportionate: eliminating an entire class of known cryptographic weaknesses at once, rather than patching individual exploits as they surface. Because protocol enforcement sits below the application layer, it can often be rolled out independently of a broader modernization effort, making it one of the more tractable security improvements available for a legacy estate. The main constraint in legacy contexts is compatibility: aging clients, embedded devices, or third-party integrations may simply be unable to negotiate a current protocol version, which turns a configuration change into a coordination and migration problem.

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

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A logistics company discovered during a compliance audit that their legacy shipping integration still used TLS 1.0 for communicating with carrier APIs. Several carriers had already begun rejecting TLS 1.0 connections, causing intermittent shipping label generation failures. The team upgraded all outbound connections to TLS 1.2, updated the internal certificate authority, and implemented a protocol version monitoring dashboard. The upgrade resolved both the compliance finding and the intermittent failures, and the monitoring system caught two additional legacy services still using outdated protocols within the first week.
