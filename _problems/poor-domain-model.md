---
title: Poor Domain Model
description: Core business concepts are poorly understood or reflected in the system,
  leading to fragile logic and miscommunication
category:
- Architecture
- Code
- Communication
related_problems:
- slug: complex-domain-model
  similarity: 0.75
- slug: poorly-defined-responsibilities
  similarity: 0.55
- slug: poor-interfaces-between-applications
  similarity: 0.55
- slug: legacy-business-logic-extraction-difficulty
  similarity: 0.5
- slug: database-schema-design-problems
  similarity: 0.5
layout: problem
---

## Description

A poor domain model occurs when the software system fails to accurately represent the real-world business concepts, relationships, and rules it is meant to support. This leads to a fundamental disconnect between how the business operates and how the software models that operation. The resulting system becomes fragile, difficult to modify, and prone to bugs because the code structure doesn't align with business reality. This problem is especially critical in legacy modernization where existing poor models often get replicated rather than improved.

## Indicators ⟡

- Business stakeholders and developers frequently talk past each other using different terminology
- Database schemas that don't reflect natural business relationships
- Business rules scattered throughout the codebase rather than centralized in domain logic
- Frequent requests for "simple" changes that require touching many unrelated parts of the system
- Domain experts expressing confusion about how the system represents their work
- New team members struggling to understand the connection between code and business processes

## Symptoms ▲

- [Accumulation of Workarounds](accumulation-of-workarounds.md)
<br/>  When the domain model doesn't match business reality, developers create workarounds to compensate for the mismatch.
- [Ripple Effect of Changes](ripple-effect-of-changes.md)
<br/>  Business rule changes require touching many unrelated parts of the system because business logic is scattered rather than centralized.
- [Large Estimates for Small Changes](large-estimates-for-small-changes.md)
<br/>  Simple business changes require disproportionate effort because the code structure doesn't align with business concepts.
- [Regression Bugs](regression-bugs.md)
<br/>  Scattered business logic means changes in one area inadvertently break business rules enforced elsewhere.
- [Database Schema Design Problems](database-schema-design-problems.md)
<br/>  A poor domain model leads to database schemas that don't reflect natural business relationships.

## Causes ▼
- [Stakeholder-Developer Communication Gap](stakeholder-developer-communication-gap.md)
<br/>  When developers and business experts don't communicate effectively, the software model diverges from business reality.
- [Insufficient Design Skills](insufficient-design-skills.md)
<br/>  Lack of domain modeling expertise leads to naive representations of complex business concepts.
- [Incomplete Knowledge](incomplete-knowledge.md)
<br/>  Incomplete understanding of business processes results in a domain model that misses critical concepts and relationships.
- [Convenience-Driven Development](convenience-driven-development.md)
<br/>  Developers model the domain based on what's easy to implement rather than what accurately represents the business.

## Detection Methods ○

- Conduct domain modeling workshops with business experts and development teams
- Review code for business logic scattered across multiple layers or modules
- Analyze bug patterns to identify areas where business rules are poorly implemented
- Map business processes to code structures to identify misalignments
- Interview domain experts about how well the system reflects their mental models
- Review database schemas for tables and relationships that don't map to business concepts
- Examine integration points where domain model mismatches cause translation complexity

## Examples

An e-commerce company's order management system treats "Order" as a simple data structure with status fields, rather than modeling the complex business reality where orders go through distinct states (placed, confirmed, fulfilled, shipped, delivered) with specific business rules governing transitions. This leads to scenarios where orders can be marked as "shipped" before being "confirmed," or "delivered" without being "fulfilled." Business users constantly encounter data that doesn't make sense, requiring manual intervention. When the company tries to add new features like partial shipments or order modifications, they discover that the poor domain model makes these changes extremely difficult, requiring extensive refactoring across multiple systems rather than simple business rule additions.
