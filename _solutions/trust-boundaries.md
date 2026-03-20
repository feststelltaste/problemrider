---
title: Trust Boundaries
description: Define boundaries between systems and components with different trust levels
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/trust-boundaries
problems:
- architectural-mismatch
- monolithic-architecture-constraints
- system-integration-blindness
- authentication-bypass-vulnerabilities
- authorization-flaws
- poor-interfaces-between-applications
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Map the legacy system's component topology and identify where different trust levels exist or should exist
- Define explicit trust boundaries between internal and external components, between user-facing and backend services, and between legacy and modern systems
- Implement validation, authentication, and authorization at every trust boundary crossing
- Ensure that data crossing trust boundaries is validated and sanitized regardless of its source
- Use network segmentation to enforce trust boundaries at the infrastructure level
- Document trust assumptions for each boundary so they can be reviewed as the system evolves
- Apply the principle of least privilege at trust boundaries, granting only the minimum access required

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Contains the blast radius of security breaches by preventing lateral movement across boundaries
- Makes implicit trust assumptions explicit and reviewable
- Provides clear points for implementing security controls and monitoring
- Enables independent security assessment and hardening of each trust zone

**Costs and Risks:**
- Legacy systems often evolved without trust boundaries, making retrofitting complex
- Adding authentication and validation at internal boundaries introduces latency and complexity
- Over-segmentation can create operational overhead and complicate legitimate cross-boundary communication
- Maintaining consistent trust boundary enforcement requires ongoing governance

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy enterprise application had grown from a single-server deployment to a distributed system over 15 years, but all internal communication still used unauthenticated, unencrypted connections because the original design assumed a trusted network. After a security incident where an attacker used a compromised web server to access the database directly, the team defined three trust zones: public-facing, application tier, and data tier. They implemented mutual TLS between zones, added input validation at each boundary, and deployed network policies restricting cross-zone communication to only necessary paths. The compartmentalization ensured that a subsequent web application vulnerability could not be leveraged to reach the database tier.
