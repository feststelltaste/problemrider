---
title: Dependency on Supplier
description: External vendors control critical parts of the system, reducing organizational
  flexibility and increasing lock-in risk.
category:
- Architecture
- Dependencies
- Management
related_problems:
- slug: vendor-dependency
  similarity: 0.9
- slug: vendor-dependency-entrapment
  similarity: 0.75
- slug: vendor-lock-in
  similarity: 0.7
- slug: vendor-relationship-strain
  similarity: 0.6
- slug: tight-coupling-issues
  similarity: 0.6
- slug: knowledge-dependency
  similarity: 0.55
layout: problem
---

## Description

Dependency on supplier occurs when an organization becomes overly reliant on external vendors for critical system components, services, or expertise, creating strategic vulnerabilities and reducing autonomy. This dependency can manifest as technical lock-in, knowledge dependency, or operational reliance that makes it difficult or expensive to change suppliers or bring capabilities in-house.

## Indicators ⟡

- Critical system functionality depends on vendor-specific technologies or services
- Organization lacks internal expertise to maintain or modify vendor-supplied components
- Switching costs to alternative suppliers are prohibitively high
- Vendor has significant control over roadmap, pricing, or service levels
- Organization cannot operate effectively if vendor relationship ends

## Symptoms ▲

- [Technology Lock-In](technology-lock-in.md)
<br/>  Reliance on a supplier's proprietary technology makes it prohibitively expensive to switch, locking the organization into their ecosystem.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Vendor-controlled components often come with escalating licensing and support costs that the organization cannot negotiate away.
- [Reduced Team Flexibility](reduced-team-flexibility.md)
<br/>  Dependency on a supplier limits the team's ability to choose technologies or approaches that best fit their needs.
- [Knowledge Dependency](knowledge-dependency.md)
<br/>  When critical knowledge resides with the supplier rather than the organization, internal teams cannot independently maintain or troubleshoot the system.
- [Delayed Value Delivery](delayed-value-delivery.md)
<br/>  Changes and improvements must wait on the supplier's schedule and priorities, delaying delivery of value to users.

## Causes ▼
- [Short-Term Focus](short-term-focus.md)
<br/>  Choosing vendor solutions for short-term convenience without evaluating long-term lock-in risks creates supplier dependency.
- [Knowledge Gaps](knowledge-gaps.md)
<br/>  Lack of internal expertise in critical technology areas forces reliance on external suppliers.
- [Poor Contract Design](poor-contract-design.md)
<br/>  Contracts that fail to protect against lock-in or ensure knowledge transfer create conditions for deep supplier dependency.
- [Accumulated Decision Debt](accumulated-decision-debt.md)
<br/>  Deferring build-vs-buy decisions and not investing in internal capabilities compounds supplier dependency over time.

## Detection Methods ○

- **Vendor Dependency Mapping:** Identify and assess all critical vendor dependencies
- **Switching Cost Analysis:** Calculate costs and effort required to change vendors for critical services
- **Vendor Performance Monitoring:** Track vendor performance and relationship health over time
- **Alternative Supplier Assessment:** Evaluate availability and viability of alternative suppliers
- **Internal Capability Gap Analysis:** Assess organization's ability to reduce vendor dependencies

## Examples

A company builds their entire customer management system on a proprietary platform from a specific vendor. Over five years, they develop hundreds of custom integrations and workflows specific to that platform. When the vendor significantly increases licensing costs and reduces support quality, the company discovers that migrating to an alternative would require rebuilding most of their system at a cost of millions of dollars and years of effort. Another example involves an organization that outsources all database administration to a vendor, failing to maintain any internal database expertise. When performance problems arise, they cannot diagnose issues independently and must rely entirely on the vendor's availability and expertise, leading to extended downtime and high support costs.
