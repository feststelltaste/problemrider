---
title: Domain-Driven Design
description: Structuring software architecture based on the business domain
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/domain-driven-design
problems:
- poor-domain-model
- complex-domain-model
- architectural-mismatch
- legacy-business-logic-extraction-difficulty
- monolithic-architecture-constraints
- high-coupling-low-cohesion
- stakeholder-developer-communication-gap
layout: solution
---

## How to Apply ◆

- Develop a ubiquitous language shared between developers and domain experts, replacing the technical jargon embedded in legacy code.
- Identify bounded contexts within the legacy system and define explicit boundaries between them.
- Refactor core domain logic using DDD tactical patterns (entities, value objects, aggregates, domain events) to replace procedural or anemic domain models.
- Use context mapping to document how the legacy system's modules relate to each other and to external systems.
- Prioritize DDD efforts on the core domain (the part that gives the business competitive advantage) rather than trying to apply it everywhere.
- Introduce anti-corruption layers to protect new domain models from being contaminated by legacy system structures.

## Tradeoffs ⇄

**Benefits:**
- Aligns code structure with business concepts, making the system more intuitive for developers and stakeholders.
- Reduces the gap between business requirements and their implementation.
- Provides a principled approach to decomposing monolithic legacy systems.
- Creates a shared vocabulary that improves communication between technical and business teams.

**Costs:**
- Requires significant investment in understanding the business domain, which takes time away from feature delivery.
- DDD concepts have a steep learning curve and can be applied incorrectly without experienced guidance.
- Retrofitting DDD onto a legacy system is a gradual process that may take years.
- Over-applying DDD to simple or generic subdomains wastes effort without proportional benefit.

## Examples

A legacy insurance company has a core policy management system where business logic is scattered across stored procedures, service classes, and UI code. The term "policy" means different things in different parts of the system, leading to frequent misunderstandings between underwriters and developers. The team engages domain experts in workshops to establish a ubiquitous language and identify bounded contexts: underwriting, claims, and billing each have their own notion of a policy. Within the underwriting context, they refactor the anemic data model into rich domain objects with behavior, replacing hundreds of lines of procedural validation code with expressive domain rules. The resulting code reads like business documentation, and new underwriting features that previously took weeks to implement can now be delivered in days because the code structure matches how the business thinks about the domain.
