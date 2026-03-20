---
title: Zero Trust Architecture
description: '"Never trust, always verify" — verifying every request regardless of network location'
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/zero-trust-architecture
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- monolithic-architecture-constraints
- system-integration-blindness
- configuration-drift
- poor-interfaces-between-applications
- insecure-data-transmission
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Eliminate implicit trust based on network location by requiring authentication and authorization for every request
- Implement identity-based access controls that verify the user, device, and context for each access attempt
- Introduce micro-segmentation to restrict lateral movement between legacy system components
- Deploy an identity-aware proxy or API gateway in front of legacy applications that lack native zero trust capabilities
- Encrypt all communication channels, including internal traffic between legacy components
- Implement continuous monitoring and logging of all access decisions for anomaly detection
- Apply least privilege access principles to all service-to-service communication in the legacy environment

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Dramatically reduces the impact of network breaches by eliminating implicit trust zones
- Provides granular access control that adapts to context rather than relying on static network boundaries
- Improves security visibility through comprehensive access logging and monitoring
- Supports modern hybrid and cloud deployment models for legacy system migration

**Costs and Risks:**
- Retrofitting zero trust into legacy systems that assume trusted networks requires significant architectural changes
- Performance overhead from verifying every request can affect latency-sensitive legacy applications
- Operational complexity increases substantially with per-request authentication and authorization
- Legacy protocols and integrations may not support the identity and encryption requirements of zero trust
- Full zero trust implementation is a multi-year journey, not a single project

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A large enterprise began their zero trust journey after a breach where an attacker used a compromised VPN connection to move freely across their internal network, accessing legacy systems that trusted all internal traffic by default. They started by deploying an identity-aware proxy in front of their most critical legacy applications, requiring per-request authentication even from internal users. They then added mutual TLS between the legacy application servers and the database tier. Within a year, the internal network was segmented into zones with explicit access policies. A subsequent red team exercise confirmed that compromising a single internal host no longer provided access to other systems, a stark contrast to the pre-zero-trust posture.
