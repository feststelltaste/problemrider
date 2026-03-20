---
title: Security Architecture Analysis
description: Examine architecture and design for conceptual security gaps
category:
- Security
- Architecture
quality_tactics_url: https://qualitytactics.de/en/security/security-architecture-analysis
problems:
- stagnant-architecture
- architectural-mismatch
- monolithic-architecture-constraints
- single-points-of-failure
- system-integration-blindness
- quality-blind-spots
- technical-architecture-limitations
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Document the current system architecture including all components, data flows, trust boundaries, and external integrations
- Identify security-relevant architectural decisions and evaluate whether they still hold under current threat models
- Analyze the architecture for common weaknesses such as missing authentication between internal services, unencrypted internal communications, and excessive trust
- Review the separation of concerns to ensure that security-critical components are properly isolated
- Evaluate the architecture's resilience to common attack patterns like lateral movement and privilege escalation
- Compare the legacy architecture against security reference architectures and industry standards
- Produce a findings report with prioritized recommendations mapped to specific architectural components

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Identifies systemic security weaknesses that code-level reviews miss
- Provides strategic direction for security improvements during modernization
- Reveals hidden trust assumptions and implicit security dependencies in legacy designs
- Informs decisions about which components to prioritize for refactoring or replacement

**Costs and Risks:**
- Requires architects with both security expertise and understanding of the legacy system
- Legacy systems often lack up-to-date architecture documentation, requiring discovery effort
- Findings may reveal fundamental design issues that are expensive to remediate
- Analysis results can become outdated quickly if the system undergoes rapid changes

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A telecommunications company commissioned a security architecture analysis of their legacy billing system. The analysis revealed that all 14 internal microservices communicated over unencrypted HTTP with no mutual authentication, meaning any compromised service could impersonate any other. The architecture also lacked network segmentation, so the customer-facing web tier had direct database access. Based on these findings, the team implemented mutual TLS between services, introduced an API gateway, and segmented the network into trust zones. These architectural changes addressed the root causes that individual vulnerability patches could not.
