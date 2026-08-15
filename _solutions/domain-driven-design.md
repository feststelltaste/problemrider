---
title: Domain-Driven Design
description: Structuring software architecture based on the business domain
category:
- Architecture
- Code
problems:
- poor-domain-model
- complex-domain-model
- architectural-mismatch
- legacy-business-logic-extraction-difficulty
- monolithic-architecture-constraints
- high-coupling-low-cohesion
- stakeholder-developer-communication-gap
- inconsistent-naming-conventions
- over-reliance-on-utility-classes
- procedural-background
- god-object-anti-pattern
- poor-naming-conventions
- insufficient-design-skills
- procedural-programming-in-oop-languages
- entity-attribute-value-overuse
layout: solution
related_solutions:
- slug: domain-modeling
  similarity: 0.8
- slug: domain-aligned-architecture
  similarity: 0.8
- slug: bounded-contexts
  similarity: 0.75
- slug: domain-patterns
  similarity: 0.75
- slug: business-event-processing
  similarity: 0.75
- slug: hexagonal-architecture
  similarity: 0.7
---

## Description

Domain-Driven Design is an approach to structuring software so that its code directly mirrors the concepts, language, and boundaries of the business domain it serves, using a shared ubiquitous language between developers and domain experts, explicit bounded contexts, and tactical patterns such as entities, value objects, aggregates, and domain events in place of procedural or anemic data models. Legacy systems frequently drift far from this ideal over time: business logic accretes across stored procedures, service classes, and UI code in whatever location was convenient when each feature was added, and a single term like "policy" can end up meaning subtly different things in different parts of the system, leading to a persistent stakeholder-developer communication gap. Applying DDD to such a system means deliberately identifying where these bounded contexts actually lie, often via workshops with domain experts, and then refactoring the corresponding logic so the code's structure and vocabulary match how the business actually talks and thinks about that part of the domain. Because the effort of building this shared understanding is substantial, it pays off best when concentrated on the core domain — the part of the system that gives the business its actual competitive differentiation — rather than spread evenly across every subdomain including generic, undifferentiated ones. Retrofitting DDD onto an established legacy codebase is necessarily gradual and carries real risk of being misapplied without experienced guidance, but done well it collapses the translation distance between what the business needs and what the code expresses, which shows up concretely as faster, less error-prone delivery of domain-specific features.

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

## How It Could Be

A legacy insurance company has a core policy management system where business logic is scattered across stored procedures, service classes, and UI code. The term "policy" means different things in different parts of the system, leading to frequent misunderstandings between underwriters and developers. The team engages domain experts in workshops to establish a ubiquitous language and identify bounded contexts: underwriting, claims, and billing each have their own notion of a policy. Within the underwriting context, they refactor the anemic data model into rich domain objects with behavior, replacing hundreds of lines of procedural validation code with expressive domain rules. The resulting code reads like business documentation, and new underwriting features that previously took weeks to implement can now be delivered in days because the code structure matches how the business thinks about the domain.
