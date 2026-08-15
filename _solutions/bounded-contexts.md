---
title: Bounded Contexts
description: Separate business areas with different terms and rules from each other
category:
- Architecture
problems:
- monolithic-architecture-constraints
- complex-domain-model
- poor-domain-model
- tight-coupling-issues
- high-coupling-low-cohesion
- spaghetti-code
- ripple-effect-of-changes
- shared-database
layout: solution
related_solutions:
- slug: domain-driven-design
  similarity: 0.75
- slug: domain-aligned-architecture
  similarity: 0.75
- slug: domain-modeling
  similarity: 0.7
- slug: modularization-and-bounded-contexts
  similarity: 0.7
- slug: high-cohesion
  similarity: 0.7
- slug: separation-of-concerns
  similarity: 0.65
---

## Description

A bounded context is an explicitly drawn boundary around a portion of a system within which a particular domain model, its terminology, and its business rules apply consistently, with translation happening deliberately at the boundary whenever information crosses into a different context. The mechanism works by accepting that a single universal model that means the same thing everywhere is unrealistic for any system of real size — the same word, such as "customer," legitimately means different things to billing and to support — and instead of forcing one shared definition, it partitions the system so each part can define its own model without corrupting the others. Legacy systems are frequently the opposite of this: a single entity or table accreted fields and conditional logic from every department that ever needed something slightly different from it, producing a bloated, deeply coupled model that no one team can change without affecting every other team that also depends on it. Introducing bounded contexts into such a system means identifying where these implicit boundaries already exist in practice, formalizing them with explicit interfaces and anti-corruption layers, and giving each context ownership of its own data rather than leaving them to share tables directly. The result is that each context can evolve, deploy, and be reasoned about independently, which is also what makes bounded contexts the natural decomposition boundary when a monolith is eventually being broken apart into separate services.

## How to Apply ◆

- Identify distinct business domains within the legacy system where the same terms have different meanings or where business rules differ (e.g., "customer" in billing vs. support).
- Draw explicit boundaries around these domains and define how they communicate through well-specified interfaces.
- Map existing legacy code modules to bounded contexts to understand where boundaries are violated.
- Introduce anti-corruption layers at context boundaries to translate between different domain models.
- Refactor shared database tables that span multiple contexts by giving each context ownership of its own data.
- Use context maps to document relationships between bounded contexts (shared kernel, customer-supplier, conformist).

## Tradeoffs ⇄

**Benefits:**
- Each context can evolve independently with its own domain model and rules.
- Reduces cognitive load by scoping complexity to a manageable boundary.
- Prevents terminology confusion that leads to bugs when different domains share the same codebase.
- Creates natural decomposition boundaries for breaking apart monoliths.

**Costs:**
- Identifying correct boundaries requires deep domain knowledge that may be partially lost in legacy systems.
- Introducing boundaries into a tightly coupled monolith is a gradual, effortful process.
- Data duplication across contexts requires synchronization mechanisms.
- Over-decomposition can lead to excessive inter-context communication overhead.

## How It Could Be

A legacy university management system uses a single "Student" entity across enrollment, grading, financial aid, and housing. Each department has different rules and attributes for what a "student" means, leading to a bloated model with hundreds of fields and complex conditional logic. The team identifies four bounded contexts and creates separate student models for each, connected through a shared student identifier. An anti-corruption layer translates between contexts when they need to exchange information. The enrollment context can now add new registration workflows without affecting the financial aid module's complex eligibility calculations, and each team can reason about their domain model independently.
