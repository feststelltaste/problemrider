---
title: Bounded Contexts
description: Separate business areas with different terms and rules from each other
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/bounded-contexts
problems:
- monolithic-architecture-constraints
- complex-domain-model
- poor-domain-model
- tight-coupling-issues
- high-coupling-low-cohesion
- spaghetti-code
- ripple-effect-of-changes
layout: solution
---

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

## Examples

A legacy university management system uses a single "Student" entity across enrollment, grading, financial aid, and housing. Each department has different rules and attributes for what a "student" means, leading to a bloated model with hundreds of fields and complex conditional logic. The team identifies four bounded contexts and creates separate student models for each, connected through a shared student identifier. An anti-corruption layer translates between contexts when they need to exchange information. The enrollment context can now add new registration workflows without affecting the financial aid module's complex eligibility calculations, and each team can reason about their domain model independently.
