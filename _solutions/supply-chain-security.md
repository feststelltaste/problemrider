---
title: Supply Chain Security
description: Securing the software supply chain through SBOMs and provenance verification
category:
- Security
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/security/supply-chain-security
problems:
- dependency-version-conflicts
- vendor-dependency
- vendor-lock-in
- obsolete-technologies
- shared-dependencies
- technology-lock-in
- regulatory-compliance-drift
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Generate Software Bills of Materials (SBOMs) for all legacy applications to create a complete inventory of dependencies
- Implement provenance verification to ensure dependencies come from trusted sources with verified integrity
- Establish a process for monitoring dependency vulnerability disclosures and applying patches promptly
- Pin dependency versions and use lock files to prevent unexpected changes in the supply chain
- Evaluate alternative dependencies for components that are no longer maintained or have a history of security issues
- Implement artifact signing and verification for internal build outputs
- Create a dependency governance policy defining criteria for adopting, maintaining, and retiring third-party components

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Provides complete visibility into the software components and their known vulnerabilities
- Enables rapid response when supply chain attacks or dependency vulnerabilities are disclosed
- Supports regulatory compliance requirements for software transparency
- Reduces risk of unknowingly incorporating compromised or malicious dependencies

**Costs and Risks:**
- Legacy systems may use dependencies that are no longer tracked in modern vulnerability databases
- SBOM generation for legacy builds with custom or vendored dependencies requires manual effort
- Strict supply chain controls can slow down dependency updates and development velocity
- Provenance verification may not be feasible for older dependencies that predate modern signing practices

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

After a major open-source supply chain attack made headlines, a healthcare company conducted an emergency audit of their legacy systems and discovered they had no inventory of third-party components. Generating SBOMs for their five legacy Java applications revealed 847 transitive dependencies, 23 of which had known critical vulnerabilities and four of which were no longer maintained. The team established a quarterly dependency review process and integrated automated SBOM generation and vulnerability scanning into their build pipeline. Within six months, all critical dependency vulnerabilities were remediated, and the team could respond to new vulnerability disclosures within 48 hours.
