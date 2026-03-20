---
title: Security by Design
description: Consider security already in the design of the architecture and implementation
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/security-by-design
problems:
- implementation-starts-without-design
- stagnant-architecture
- architectural-mismatch
- authentication-bypass-vulnerabilities
- authorization-flaws
- quality-blind-spots
- technical-architecture-limitations
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Incorporate security requirements as first-class architectural drivers alongside performance and scalability
- Apply defense-in-depth principles by layering security controls at network, application, and data levels
- Design for least privilege by default in all new components and interfaces
- Include security considerations in architecture decision records and design reviews
- Use threat modeling to identify and address security concerns before implementation begins
- Establish secure design patterns as reusable templates for common architectural decisions
- Review legacy architectural decisions against current security best practices during modernization planning

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Prevents entire categories of vulnerabilities through architectural choices rather than patches
- Reduces long-term security maintenance costs by addressing root causes in design
- Creates architectures that are inherently more resilient to evolving threats
- Makes security an enabler rather than a blocker of feature delivery

**Costs and Risks:**
- Retrofitting security-by-design principles into existing legacy architectures can be prohibitively expensive
- Requires architects who understand both security and system design deeply
- Can lead to over-engineering if every design decision is treated as security-critical
- Initial design phases take longer when security is a primary concern

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A payment processing company was planning to extract a microservice from their legacy monolith to handle sensitive card data. Instead of replicating the monolith's flat security model, they designed the new service with security as a primary architectural driver: mutual TLS for all communication, encrypted data at rest with per-tenant keys, no direct database access from other services, and a dedicated audit log stream. While the initial development took three weeks longer than a naive extraction, the service passed PCI DSS assessment on the first attempt, whereas the legacy monolith had required three remediation cycles in its last audit.
